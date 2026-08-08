import argparse
import time
from typing import Any, Dict, List, Optional

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T


SCALAR_STATES = [
    {
        "state_id": "S01",
        "state_name": "row_count",
        "state_family": "ADDITIVE",
        "column": "row_count",
        "supported_rules": "R01",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S02",
        "state_name": "null_passenger_count",
        "state_family": "ADDITIVE",
        "column": "null_passenger_count",
        "supported_rules": "R02",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S03",
        "state_name": "null_fare_count",
        "state_family": "ADDITIVE",
        "column": "null_fare_count",
        "supported_rules": "R03",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S04",
        "state_name": "invalid_fare_count",
        "state_family": "ADDITIVE",
        "column": "invalid_fare_count",
        "supported_rules": "R04",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S05",
        "state_name": "invalid_distance_count",
        "state_family": "ADDITIVE",
        "column": "invalid_distance_count",
        "supported_rules": "R05",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S06",
        "state_name": "invalid_passenger_count",
        "state_family": "ADDITIVE",
        "column": "invalid_passenger_count",
        "supported_rules": "R06",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S07",
        "state_name": "invalid_pickup_time_count",
        "state_family": "ADDITIVE",
        "column": "invalid_pickup_time_count",
        "supported_rules": "R07",
        "maintenance_mode": "CDF arithmetic",
    },
    {
        "state_id": "S08",
        "state_name": "minimum_fare",
        "state_family": "PARTITION_EXTREMA",
        "column": "minimum_fare",
        "supported_rules": "R08",
        "maintenance_mode": "CDF update or affected-partition recompute",
    },
    {
        "state_id": "S09",
        "state_name": "maximum_fare",
        "state_family": "PARTITION_EXTREMA",
        "column": "maximum_fare",
        "supported_rules": "R09",
        "maintenance_mode": "CDF update or affected-partition recompute",
    },
    {
        "state_id": "S10",
        "state_name": "minimum_distance",
        "state_family": "PARTITION_EXTREMA",
        "column": "minimum_distance",
        "supported_rules": "R10",
        "maintenance_mode": "CDF update or affected-partition recompute",
    },
    {
        "state_id": "S11",
        "state_name": "maximum_distance",
        "state_family": "PARTITION_EXTREMA",
        "column": "maximum_distance",
        "supported_rules": "R11",
        "maintenance_mode": "CDF update or affected-partition recompute",
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the initial exact StateGuard auxiliary-state catalog "
            "from the canonical Delta table."
        )
    )
    parser.add_argument(
        "--delta-path",
        required=True,
        help="GCS path of the canonical Delta table.",
    )
    parser.add_argument(
        "--baseline-rules",
        required=True,
        help="GCS path of full_validation_v1/rules_csv.",
    )
    parser.add_argument(
        "--baseline-summary",
        required=True,
        help="GCS path of full_validation_v1/summary_csv.",
    )
    parser.add_argument(
        "--state-root",
        required=True,
        help="GCS root where auxiliary Delta states will be written.",
    )
    parser.add_argument(
        "--result-root",
        required=True,
        help="GCS root where state catalog and verification results are written.",
    )
    parser.add_argument(
        "--min-valid-pickup",
        default="2024-12-31 00:00:00",
        help="Inclusive lower bound used by R07.",
    )
    parser.add_argument(
        "--max-valid-pickup",
        default="2026-06-01 23:59:59",
        help="Inclusive upper bound used by R07.",
    )
    parser.add_argument(
        "--max-passengers",
        type=int,
        default=8,
        help="Maximum valid non-null passenger count for R06.",
    )
    parser.add_argument(
        "--key-buckets",
        type=int,
        default=256,
        help="Physical partition count for the exact key-frequency state.",
    )
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


def parse_optional_long(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def parse_optional_double(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value.strip().lower() == "true"


def assert_equal(
    name: str,
    actual: Any,
    expected: Any,
    tolerance: float = 0.0,
) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None:
            if actual != expected:
                raise RuntimeError(
                    f"{name} mismatch: expected={expected}, actual={actual}"
                )
            return
        if abs(float(actual) - float(expected)) > tolerance:
            raise RuntimeError(
                f"{name} mismatch: expected={expected}, actual={actual}"
            )
        return

    if actual != expected:
        raise RuntimeError(
            f"{name} mismatch: expected={expected}, actual={actual}"
        )


def main() -> None:
    args = parse_arguments()

    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")

    if args.key_buckets <= 0:
        raise ValueError("--key-buckets must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardInitialAuxiliaryStates")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", "64")

    delta_path = args.delta_path.rstrip("/")
    state_root = args.state_root.rstrip("/")
    result_root = args.result_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, delta_path):
        raise RuntimeError(f"Not a Delta table: {delta_path}")

    baseline_rules_rows = (
        spark.read.option("header", "true")
        .csv(args.baseline_rules)
        .select(
            "rule_id",
            "metric_type",
            "long_value",
            "double_value",
            "boolean_value",
        )
        .collect()
    )

    if len(baseline_rules_rows) != 13:
        raise RuntimeError(
            f"Expected 13 baseline rules, found {len(baseline_rules_rows)}."
        )

    baseline: Dict[str, Any] = {}

    for row in baseline_rules_rows:
        rule_id = row["rule_id"]
        metric_type = row["metric_type"]

        if metric_type == "LONG":
            baseline[rule_id] = parse_optional_long(row["long_value"])
        elif metric_type == "DOUBLE":
            baseline[rule_id] = parse_optional_double(row["double_value"])
        elif metric_type == "BOOLEAN":
            baseline[rule_id] = parse_optional_bool(row["boolean_value"])
        else:
            raise RuntimeError(
                f"Unsupported baseline metric type for {rule_id}: {metric_type}"
            )

    baseline_summary_rows = (
        spark.read.option("header", "true")
        .csv(args.baseline_summary)
        .select(
            F.col("delta_version").cast("long").alias("delta_version"),
            F.col("row_count").cast("long").alias("row_count"),
            F.col("distinct_trip_keys")
            .cast("long")
            .alias("distinct_trip_keys"),
            F.col("duplicate_key_groups")
            .cast("long")
            .alias("duplicate_key_groups"),
            F.col("duplicate_extra_rows")
            .cast("long")
            .alias("duplicate_extra_rows"),
            F.col("maximum_key_multiplicity")
            .cast("long")
            .alias("maximum_key_multiplicity"),
            F.col("uniqueness_pass")
            .cast("boolean")
            .alias("uniqueness_pass"),
        )
        .collect()
    )

    if len(baseline_summary_rows) != 1:
        raise RuntimeError(
            "Expected exactly one row in the baseline summary."
        )

    baseline_summary = baseline_summary_rows[0]

    delta_table = DeltaTable.forPath(spark, delta_path)
    history = delta_table.history(1).collect()[0]
    delta_version = int(history["version"])

    if delta_version != int(baseline_summary["delta_version"]):
        raise RuntimeError(
            "Canonical Delta version differs from the full-validation baseline."
        )

    df = spark.read.format("delta").load(delta_path)

    required_columns = {
        "trip_key",
        "state_partition_id",
        "passenger_count",
        "fare_amount",
        "trip_distance",
        "tpep_pickup_datetime",
    }

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise RuntimeError(
            "Canonical table is missing required columns: "
            + ", ".join(missing)
        )

    min_pickup = F.lit(args.min_valid_pickup).cast("timestamp_ntz")
    max_pickup = F.lit(args.max_valid_pickup).cast("timestamp_ntz")

    print("=" * 78)
    print("STATEGUARD INITIAL AUXILIARY-STATE BUILD")
    print("=" * 78)
    print(f"Canonical Delta version: {delta_version}")
    print(f"Canonical path: {delta_path}")
    print(f"State root: {state_root}")
    print(f"Key-frequency buckets: {args.key_buckets}")
    print("=" * 78)

    catalog_rows: List[Dict[str, Any]] = []

    partition_build_start = time.perf_counter()

    partition_metrics = (
        df.groupBy("state_partition_id")
        .agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.sum(
                F.when(F.col("passenger_count").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_passenger_count"),
            F.sum(
                F.when(F.col("fare_amount").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_fare_count"),
            F.sum(
                F.when(
                    F.col("fare_amount").isNotNull()
                    & (F.col("fare_amount") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_fare_count"),
            F.sum(
                F.when(
                    F.col("trip_distance").isNotNull()
                    & (F.col("trip_distance") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_distance_count"),
            F.sum(
                F.when(
                    F.col("passenger_count").isNotNull()
                    & (
                        (F.col("passenger_count") < 1)
                        | (
                            F.col("passenger_count")
                            > F.lit(args.max_passengers)
                        )
                    ),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_passenger_count"),
            F.sum(
                F.when(
                    F.col("tpep_pickup_datetime").isNull()
                    | (F.col("tpep_pickup_datetime") < min_pickup)
                    | (F.col("tpep_pickup_datetime") > max_pickup),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_pickup_time_count"),
            F.min("fare_amount").cast("double").alias("minimum_fare"),
            F.max("fare_amount").cast("double").alias("maximum_fare"),
            F.min("trip_distance").cast("double").alias("minimum_distance"),
            F.max("trip_distance").cast("double").alias("maximum_distance"),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    partition_count = partition_metrics.count()

    if partition_count != 64:
        raise RuntimeError(
            f"Expected 64 logical partitions, found {partition_count}."
        )

    partition_global = (
        partition_metrics.agg(
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

    for state in SCALAR_STATES:
        state_id = state["state_id"]
        state_name = state["state_name"]
        column_name = state["column"]
        state_path = f"{state_root}/{state_id.lower()}_{state_name}"

        state_start = time.perf_counter()

        state_df = partition_metrics.select(
            "state_partition_id",
            F.col(column_name).alias("value"),
        )

        (
            state_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(state_path)
        )

        build_seconds = time.perf_counter() - state_start
        metrics = delta_metrics(spark, state_path)

        catalog_rows.append(
            {
                "state_id": state_id,
                "state_name": state_name,
                "state_family": state["state_family"],
                "supported_rules": state["supported_rules"],
                "maintenance_mode": state["maintenance_mode"],
                "row_count": partition_count,
                "num_files": metrics["num_files"],
                "size_bytes": metrics["size_bytes"],
                "build_seconds": float(build_seconds),
                "path": state_path,
            }
        )

    partition_build_seconds = time.perf_counter() - partition_build_start

    for rule_number in range(1, 12):
        rule_id = f"R{rule_number:02d}"
        actual = partition_global[rule_id]
        expected = baseline[rule_id]
        assert_equal(rule_id, actual, expected, tolerance=1e-9)

    key_state_path = f"{state_root}/s12_exact_key_frequency"

    key_build_start = time.perf_counter()

    key_frequency = (
        df.groupBy("trip_key")
        .agg(F.count(F.lit(1)).cast("long").alias("frequency"))
        .withColumn(
            "key_bucket",
            F.pmod(
                F.xxhash64("trip_key"),
                F.lit(args.key_buckets),
            ).cast("int"),
        )
        .select("key_bucket", "trip_key", "frequency")
    )

    (
        key_frequency.repartition(args.key_buckets, "key_bucket")
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("key_bucket")
        .save(key_state_path)
    )

    key_build_seconds = time.perf_counter() - key_build_start
    key_metrics = delta_metrics(spark, key_state_path)

    key_verify_start = time.perf_counter()

    key_verification = (
        spark.read.format("delta")
        .load(key_state_path)
        .agg(
            F.count(F.lit(1))
            .cast("long")
            .alias("distinct_trip_keys"),
            F.sum("frequency").cast("long").alias("frequency_sum"),
            F.sum(
                F.when(F.col("frequency") > 1, 1).otherwise(0)
            ).cast("long").alias("duplicate_key_groups"),
            F.sum(
                F.when(
                    F.col("frequency") > 1,
                    F.col("frequency") - F.lit(1),
                ).otherwise(F.lit(0))
            ).cast("long").alias("duplicate_extra_rows"),
            F.max("frequency")
            .cast("long")
            .alias("maximum_key_multiplicity"),
        )
        .collect()[0]
    )

    key_verify_seconds = time.perf_counter() - key_verify_start

    distinct_trip_keys = int(key_verification["distinct_trip_keys"])
    frequency_sum = int(key_verification["frequency_sum"])
    duplicate_key_groups = int(
        key_verification["duplicate_key_groups"]
    )
    duplicate_extra_rows = int(
        key_verification["duplicate_extra_rows"]
    )
    maximum_key_multiplicity = int(
        key_verification["maximum_key_multiplicity"]
    )
    uniqueness_pass = duplicate_extra_rows == 0

    assert_equal(
        "frequency_sum",
        frequency_sum,
        int(baseline_summary["row_count"]),
    )
    assert_equal(
        "distinct_trip_keys",
        distinct_trip_keys,
        int(baseline_summary["distinct_trip_keys"]),
    )
    assert_equal(
        "duplicate_key_groups",
        duplicate_key_groups,
        int(baseline_summary["duplicate_key_groups"]),
    )
    assert_equal(
        "R12",
        duplicate_extra_rows,
        baseline["R12"],
    )
    assert_equal(
        "maximum_key_multiplicity",
        maximum_key_multiplicity,
        int(baseline_summary["maximum_key_multiplicity"]),
    )
    assert_equal(
        "R13",
        uniqueness_pass,
        baseline["R13"],
    )

    catalog_rows.append(
        {
            "state_id": "S12",
            "state_name": "exact_key_frequency",
            "state_family": "KEY_FREQUENCY",
            "supported_rules": "R12,R13",
            "maintenance_mode": "Delta MERGE by trip_key using CDF deltas",
            "row_count": distinct_trip_keys,
            "num_files": key_metrics["num_files"],
            "size_bytes": key_metrics["size_bytes"],
            "build_seconds": float(key_build_seconds),
            "path": key_state_path,
        }
    )

    partition_metrics.unpersist()

    catalog_schema = T.StructType(
        [
            T.StructField("state_id", T.StringType(), False),
            T.StructField("state_name", T.StringType(), False),
            T.StructField("state_family", T.StringType(), False),
            T.StructField("supported_rules", T.StringType(), False),
            T.StructField("maintenance_mode", T.StringType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("num_files", T.LongType(), False),
            T.StructField("size_bytes", T.LongType(), False),
            T.StructField("build_seconds", T.DoubleType(), False),
            T.StructField("path", T.StringType(), False),
        ]
    )

    catalog_df = (
        spark.createDataFrame(catalog_rows, schema=catalog_schema)
        .orderBy("state_id")
    )

    catalog_summary = (
        catalog_df.agg(
            F.count(F.lit(1)).cast("long").alias("state_count"),
            F.sum("size_bytes")
            .cast("long")
            .alias("total_state_size_bytes"),
            F.sum(
                F.when(
                    F.col("state_family") == "ADDITIVE",
                    F.col("size_bytes"),
                ).otherwise(F.lit(0))
            ).cast("long").alias("additive_state_size_bytes"),
            F.sum(
                F.when(
                    F.col("state_family") == "PARTITION_EXTREMA",
                    F.col("size_bytes"),
                ).otherwise(F.lit(0))
            ).cast("long").alias("extrema_state_size_bytes"),
            F.sum(
                F.when(
                    F.col("state_family") == "KEY_FREQUENCY",
                    F.col("size_bytes"),
                ).otherwise(F.lit(0))
            ).cast("long").alias("key_frequency_size_bytes"),
        )
        .collect()[0]
    )

    total_state_size_bytes = int(
        catalog_summary["total_state_size_bytes"]
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("delta_version", T.LongType(), False),
            T.StructField("state_count", T.LongType(), False),
            T.StructField("logical_partition_count", T.LongType(), False),
            T.StructField("key_bucket_count", T.LongType(), False),
            T.StructField("distinct_trip_keys", T.LongType(), False),
            T.StructField("frequency_sum", T.LongType(), False),
            T.StructField("duplicate_key_groups", T.LongType(), False),
            T.StructField("duplicate_extra_rows", T.LongType(), False),
            T.StructField("uniqueness_pass", T.BooleanType(), False),
            T.StructField(
                "total_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "additive_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "extrema_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "key_frequency_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "partition_state_build_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "key_frequency_build_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "key_frequency_verify_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField("spark_version", T.StringType(), False),
            T.StructField("state_root", T.StringType(), False),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                delta_version,
                int(catalog_summary["state_count"]),
                partition_count,
                args.key_buckets,
                distinct_trip_keys,
                frequency_sum,
                duplicate_key_groups,
                duplicate_extra_rows,
                uniqueness_pass,
                total_state_size_bytes,
                int(catalog_summary["additive_state_size_bytes"]),
                int(catalog_summary["extrema_state_size_bytes"]),
                int(catalog_summary["key_frequency_size_bytes"]),
                float(partition_build_seconds),
                float(key_build_seconds),
                float(key_verify_seconds),
                spark.version,
                state_root,
            )
        ],
        schema=summary_schema,
    )

    write_csv(catalog_df, f"{result_root}/state_catalog_csv")
    write_csv(summary_df, f"{result_root}/summary_csv")

    (
        catalog_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/state_catalog_json")
    )

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("AUXILIARY STATE CATALOG")
    print("=" * 78)

    catalog_df.select(
        "state_id",
        "state_name",
        "state_family",
        "supported_rules",
        "row_count",
        "size_bytes",
    ).show(20, truncate=False)

    print("=" * 78)
    print("STATEGUARD_STATE_BUILD_BEGIN")
    print("STATE_BUILD_STATUS=PASS")
    print(f"DELTA_VERSION={delta_version}")
    print(f"STATE_COUNT={int(catalog_summary['state_count'])}")
    print(f"LOGICAL_PARTITION_COUNT={partition_count}")
    print(f"KEY_BUCKET_COUNT={args.key_buckets}")
    print(f"DISTINCT_TRIP_KEYS={distinct_trip_keys}")
    print(f"FREQUENCY_SUM={frequency_sum}")
    print(f"DUPLICATE_KEY_GROUPS={duplicate_key_groups}")
    print(f"DUPLICATE_EXTRA_ROWS={duplicate_extra_rows}")
    print(f"UNIQUENESS_PASS={str(uniqueness_pass).lower()}")
    print(f"TOTAL_STATE_SIZE_BYTES={total_state_size_bytes}")
    print(
        "ADDITIVE_STATE_SIZE_BYTES="
        f"{int(catalog_summary['additive_state_size_bytes'])}"
    )
    print(
        "EXTREMA_STATE_SIZE_BYTES="
        f"{int(catalog_summary['extrema_state_size_bytes'])}"
    )
    print(
        "KEY_FREQUENCY_SIZE_BYTES="
        f"{int(catalog_summary['key_frequency_size_bytes'])}"
    )
    print(
        "PARTITION_STATE_BUILD_SECONDS="
        f"{partition_build_seconds:.3f}"
    )
    print(
        "KEY_FREQUENCY_BUILD_SECONDS="
        f"{key_build_seconds:.3f}"
    )
    print(
        "KEY_FREQUENCY_VERIFY_SECONDS="
        f"{key_verify_seconds:.3f}"
    )
    print(f"STATE_ROOT={state_root}")
    print(f"STATE_CATALOG_PATH={result_root}/state_catalog_csv")
    print(f"SUMMARY_PATH={result_root}/summary_csv")
    print("STATEGUARD_STATE_BUILD_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
