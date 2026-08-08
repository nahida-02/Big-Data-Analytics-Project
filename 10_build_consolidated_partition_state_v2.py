import argparse
import time
from functools import reduce
from typing import Any, Dict, List, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


STATE_CONFIGS = [
    ("S01", "row_count", "R01", "LONG"),
    ("S02", "null_passenger_count", "R02", "LONG"),
    ("S03", "null_fare_count", "R03", "LONG"),
    ("S04", "invalid_fare_count", "R04", "LONG"),
    ("S05", "invalid_distance_count", "R05", "LONG"),
    ("S06", "invalid_passenger_count", "R06", "LONG"),
    ("S07", "invalid_pickup_time_count", "R07", "LONG"),
    ("S08", "minimum_fare", "R08", "DOUBLE"),
    ("S09", "maximum_fare", "R09", "DOUBLE"),
    ("S10", "minimum_distance", "R10", "DOUBLE"),
    ("S11", "maximum_distance", "R11", "DOUBLE"),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate StateGuard S01-S11 into one exact 64-row "
            "partition-state table to reduce small-file and multi-table "
            "startup overhead during incremental validation."
        )
    )
    parser.add_argument("--initial-state-root", required=True)
    parser.add_argument("--baseline-rules", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected-delta-version", type=int, default=0)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--double-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def delta_metrics(
    spark: SparkSession,
    path: str,
) -> Dict[str, int]:
    detail = DeltaTable.forPath(spark, path).detail().collect()[0]
    return {
        "num_files": int(detail["numFiles"]),
        "size_bytes": int(detail["sizeInBytes"]),
    }


def state_path(
    root: str,
    state_id: str,
    state_name: str,
) -> str:
    return f"{root}/{state_id.lower()}_{state_name}"


def parse_baseline_value(row: Row) -> Any:
    metric_type = str(row["metric_type"])

    if metric_type == "LONG":
        return int(row["long_value"])

    if metric_type == "DOUBLE":
        return float(row["double_value"])

    raise RuntimeError(
        f"Unsupported baseline metric type: {metric_type}"
    )


def assert_equal(
    rule_id: str,
    actual: Any,
    expected: Any,
    tolerance: float,
) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if abs(float(actual) - float(expected)) > tolerance:
            raise RuntimeError(
                f"{rule_id} mismatch: expected={expected}, actual={actual}"
            )
        return

    if actual != expected:
        raise RuntimeError(
            f"{rule_id} mismatch: expected={expected}, actual={actual}"
        )


def main() -> None:
    args = parse_arguments()

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    if args.expected_delta_version < 0:
        raise ValueError("--expected-delta-version must be non-negative.")

    if args.double_tolerance < 0:
        raise ValueError("--double-tolerance cannot be negative.")

    spark = (
        SparkSession.builder
        .appName("StateGuardConsolidatedPartitionStateV2")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.shuffle.partitions", "64")

    initial_root = args.initial_state_root.rstrip("/")
    output_path = args.output_path.rstrip("/")
    result_root = args.result_root.rstrip("/")

    baseline_rows = (
        spark.read.option("header", "true")
        .csv(args.baseline_rules)
        .filter(
            F.col("rule_id").isin(
                [f"R{index:02d}" for index in range(1, 12)]
            )
        )
        .select(
            "rule_id",
            "metric_type",
            "long_value",
            "double_value",
        )
        .collect()
    )

    if len(baseline_rows) != 11:
        raise RuntimeError(
            f"Expected 11 baseline rules, found {len(baseline_rows)}."
        )

    baseline = {
        str(row["rule_id"]): parse_baseline_value(row)
        for row in baseline_rows
    }

    state_frames: List[DataFrame] = []
    source_catalog_rows: List[Dict[str, Any]] = []
    source_total_files = 0
    source_total_bytes = 0
    observed_versions = set()

    read_start = time.perf_counter()

    for state_id, state_name, rule_id, metric_type in STATE_CONFIGS:
        path = state_path(initial_root, state_id, state_name)

        if not DeltaTable.isDeltaTable(spark, path):
            raise RuntimeError(f"Missing Delta state table: {path}")

        delta_table = DeltaTable.forPath(spark, path)
        version = int(delta_table.history(1).collect()[0]["version"])
        observed_versions.add(version)

        metrics = delta_metrics(spark, path)
        source_total_files += metrics["num_files"]
        source_total_bytes += metrics["size_bytes"]

        cast_type = "long" if metric_type == "LONG" else "double"

        frame = (
            spark.read.format("delta")
            .load(path)
            .select(
                F.col("state_partition_id").cast("int"),
                F.col("value").cast(cast_type).alias(state_name),
            )
        )

        state_frames.append(frame)
        source_catalog_rows.append(
            {
                "state_id": state_id,
                "state_name": state_name,
                "rule_id": rule_id,
                "metric_type": metric_type,
                "delta_version": version,
                "num_files": metrics["num_files"],
                "size_bytes": metrics["size_bytes"],
                "source_path": path,
            }
        )

    source_read_seconds = time.perf_counter() - read_start

    if observed_versions != {args.expected_delta_version}:
        raise RuntimeError(
            "Input state versions differ from the expected baseline: "
            f"{sorted(observed_versions)}"
        )

    consolidated = reduce(
        lambda left, right: left.join(
            right,
            on="state_partition_id",
            how="inner",
        ),
        state_frames,
    )

    verification_start = time.perf_counter()

    partition_count = consolidated.count()
    distinct_partition_count = (
        consolidated.select("state_partition_id").distinct().count()
    )

    if partition_count != args.state_partitions:
        raise RuntimeError(
            f"Expected {args.state_partitions} rows, found {partition_count}."
        )

    if distinct_partition_count != args.state_partitions:
        raise RuntimeError(
            "state_partition_id is not unique in the consolidated table."
        )

    required_value_columns = [
        state_name
        for _, state_name, _, _ in STATE_CONFIGS
    ]

    null_expression = reduce(
        lambda left, right: left | right,
        [F.col(column_name).isNull() for column_name in required_value_columns],
    )

    null_rows = consolidated.filter(null_expression).count()

    if null_rows != 0:
        raise RuntimeError(
            f"Consolidated state contains {null_rows} rows with null values."
        )

    global_metrics = (
        consolidated.agg(
            F.sum("row_count").cast("long").alias("R01"),
            F.sum("null_passenger_count").cast("long").alias("R02"),
            F.sum("null_fare_count").cast("long").alias("R03"),
            F.sum("invalid_fare_count").cast("long").alias("R04"),
            F.sum("invalid_distance_count").cast("long").alias("R05"),
            F.sum("invalid_passenger_count").cast("long").alias("R06"),
            F.sum("invalid_pickup_time_count").cast("long").alias("R07"),
            F.min("minimum_fare").cast("double").alias("R08"),
            F.max("maximum_fare").cast("double").alias("R09"),
            F.min("minimum_distance").cast("double").alias("R10"),
            F.max("maximum_distance").cast("double").alias("R11"),
        )
        .collect()[0]
    )

    verification_rows: List[Dict[str, Any]] = []

    for rule_number in range(1, 12):
        rule_id = f"R{rule_number:02d}"
        actual = global_metrics[rule_id]
        expected = baseline[rule_id]

        assert_equal(
            rule_id,
            actual,
            expected,
            args.double_tolerance,
        )

        verification_rows.append(
            {
                "rule_id": rule_id,
                "expected_value": str(expected),
                "actual_value": str(actual),
                "exact_match": True,
            }
        )

    verification_seconds = time.perf_counter() - verification_start

    write_start = time.perf_counter()

    (
        consolidated.select(
            "state_partition_id",
            *required_value_columns,
        )
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_path)
    )

    write_seconds = time.perf_counter() - write_start
    output_metrics = delta_metrics(spark, output_path)

    output_version = int(
        DeltaTable.forPath(spark, output_path)
        .history(1)
        .collect()[0]["version"]
    )

    if output_version != 0:
        raise RuntimeError(
            f"Expected new consolidated Delta version 0, found {output_version}."
        )

    catalog_schema = T.StructType(
        [
            T.StructField("state_id", T.StringType(), False),
            T.StructField("state_name", T.StringType(), False),
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("metric_type", T.StringType(), False),
            T.StructField("delta_version", T.LongType(), False),
            T.StructField("num_files", T.LongType(), False),
            T.StructField("size_bytes", T.LongType(), False),
            T.StructField("source_path", T.StringType(), False),
        ]
    )

    verification_schema = T.StructType(
        [
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("expected_value", T.StringType(), False),
            T.StructField("actual_value", T.StringType(), False),
            T.StructField("exact_match", T.BooleanType(), False),
        ]
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("baseline_delta_version", T.LongType(), False),
            T.StructField("output_delta_version", T.LongType(), False),
            T.StructField("state_count", T.LongType(), False),
            T.StructField("partition_count", T.LongType(), False),
            T.StructField("source_total_files", T.LongType(), False),
            T.StructField("source_total_size_bytes", T.LongType(), False),
            T.StructField("output_num_files", T.LongType(), False),
            T.StructField("output_size_bytes", T.LongType(), False),
            T.StructField("source_read_seconds", T.DoubleType(), False),
            T.StructField("verification_seconds", T.DoubleType(), False),
            T.StructField("write_seconds", T.DoubleType(), False),
            T.StructField("exact_match_count", T.LongType(), False),
            T.StructField("output_path", T.StringType(), False),
        ]
    )

    catalog_df = spark.createDataFrame(
        source_catalog_rows,
        schema=catalog_schema,
    ).orderBy("state_id")

    verification_df = spark.createDataFrame(
        verification_rows,
        schema=verification_schema,
    ).orderBy("rule_id")

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                args.expected_delta_version,
                output_version,
                len(STATE_CONFIGS),
                partition_count,
                source_total_files,
                source_total_bytes,
                output_metrics["num_files"],
                output_metrics["size_bytes"],
                float(source_read_seconds),
                float(verification_seconds),
                float(write_seconds),
                len(verification_rows),
                output_path,
            )
        ],
        schema=summary_schema,
    )

    write_csv(catalog_df, f"{result_root}/source_catalog_csv")
    write_csv(
        verification_df,
        f"{result_root}/verification_csv",
    )
    write_csv(summary_df, f"{result_root}/summary_csv")

    (
        catalog_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/source_catalog_json")
    )
    (
        verification_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/verification_json")
    )
    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("STATEGUARD CONSOLIDATED PARTITION STATE V2")
    print("=" * 78)
    print(f"Input state tables: {len(STATE_CONFIGS)}")
    print(f"Input total files: {source_total_files}")
    print(f"Input total bytes: {source_total_bytes}")
    print(f"Output rows: {partition_count}")
    print(f"Output files: {output_metrics['num_files']}")
    print(f"Output bytes: {output_metrics['size_bytes']}")
    print("=" * 78)
    print("STATEGUARD_CONSOLIDATED_STATE_BEGIN")
    print("CONSOLIDATED_STATE_STATUS=PASS")
    print(f"BASELINE_DELTA_VERSION={args.expected_delta_version}")
    print(f"OUTPUT_DELTA_VERSION={output_version}")
    print(f"STATE_COUNT={len(STATE_CONFIGS)}")
    print(f"PARTITION_COUNT={partition_count}")
    print(f"EXACT_MATCH_COUNT={len(verification_rows)}")
    print("RULE_COUNT=11")
    print(f"SOURCE_TOTAL_FILES={source_total_files}")
    print(f"SOURCE_TOTAL_SIZE_BYTES={source_total_bytes}")
    print(f"OUTPUT_NUM_FILES={output_metrics['num_files']}")
    print(f"OUTPUT_SIZE_BYTES={output_metrics['size_bytes']}")
    print(f"SOURCE_READ_SECONDS={source_read_seconds:.3f}")
    print(f"VERIFICATION_SECONDS={verification_seconds:.3f}")
    print(f"WRITE_SECONDS={write_seconds:.3f}")
    print(f"OUTPUT_PATH={output_path}")
    print(f"SUMMARY_PATH={result_root}/summary_csv")
    print("STATEGUARD_CONSOLIDATED_STATE_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
