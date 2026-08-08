import argparse
import time
from datetime import datetime
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


STATE_CONFIGS = [
    ("S01", "row_count", "ADDITIVE", "R01", "long"),
    ("S02", "null_passenger_count", "ADDITIVE", "R02", "long"),
    ("S03", "null_fare_count", "ADDITIVE", "R03", "long"),
    ("S04", "invalid_fare_count", "ADDITIVE", "R04", "long"),
    ("S05", "invalid_distance_count", "ADDITIVE", "R05", "long"),
    ("S06", "invalid_passenger_count", "ADDITIVE", "R06", "long"),
    ("S07", "invalid_pickup_time_count", "ADDITIVE", "R07", "long"),
    ("S08", "minimum_fare", "PARTITION_EXTREMA", "R08", "double"),
    ("S09", "maximum_fare", "PARTITION_EXTREMA", "R09", "double"),
    ("S10", "minimum_distance", "PARTITION_EXTREMA", "R10", "double"),
    ("S11", "maximum_distance", "PARTITION_EXTREMA", "R11", "double"),
]

EXTREMA_CONFIGS = [
    {
        "state_id": "S08",
        "state_column": "minimum_fare",
        "source_column": "fare_amount",
        "rule_id": "R08",
        "direction": "MIN",
    },
    {
        "state_id": "S09",
        "state_column": "maximum_fare",
        "source_column": "fare_amount",
        "rule_id": "R09",
        "direction": "MAX",
    },
    {
        "state_id": "S10",
        "state_column": "minimum_distance",
        "source_column": "trip_distance",
        "rule_id": "R10",
        "direction": "MIN",
    },
    {
        "state_id": "S11",
        "state_column": "maximum_distance",
        "source_column": "trip_distance",
        "rule_id": "R11",
        "direction": "MAX",
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact CDF-based StateGuard validation from an initial "
            "auxiliary-state portfolio, maintain delete-sensitive extrema "
            "with affected-partition recomputation, maintain uniqueness "
            "through a compact key-frequency overlay, and compare all "
            "13 results against full-table ground truth."
        )
    )
    parser.add_argument("--working-path", required=True)
    parser.add_argument("--consolidated-partition-state", required=True)
    parser.add_argument("--key-frequency-state", required=True)
    parser.add_argument("--initial-state-summary", required=True)
    parser.add_argument("--baseline-full-summary", required=True)
    parser.add_argument("--ground-truth-rules", required=True)
    parser.add_argument("--ground-truth-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-version", type=int, default=0)
    parser.add_argument("--end-version", type=int, default=12)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--key-buckets", type=int, default=256)
    parser.add_argument(
        "--min-valid-pickup",
        default="2024-12-31 00:00:00",
    )
    parser.add_argument(
        "--max-valid-pickup",
        default="2026-06-01 23:59:59",
    )
    parser.add_argument("--max-passengers", type=int, default=8)
    parser.add_argument("--double-tolerance", type=float, default=1e-9)
    parser.add_argument("--small-cdf-threshold", type=int, default=10000)
    parser.add_argument("--small-overlay-threshold", type=int, default=10000)
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


def read_single_csv_row(
    spark: SparkSession,
    path: str,
) -> Row:
    rows = spark.read.option("header", "true").csv(path).collect()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one CSV row at {path}; found {len(rows)}."
        )

    return rows[0]


def required_int(row: Row, column_name: str) -> int:
    value = row[column_name]

    if value is None or value == "":
        raise RuntimeError(f"Missing integer field: {column_name}")

    return int(value)


def required_float(row: Row, column_name: str) -> float:
    value = row[column_name]

    if value is None or value == "":
        raise RuntimeError(f"Missing floating-point field: {column_name}")

    return float(value)


def required_bool(row: Row, column_name: str) -> bool:
    value = row[column_name]

    if value is None or value == "":
        raise RuntimeError(f"Missing boolean field: {column_name}")

    return str(value).strip().lower() == "true"


def load_consolidated_partition_state(
    spark: SparkSession,
    path: str,
    expected_version: int,
) -> DataFrame:
    if not DeltaTable.isDeltaTable(spark, path):
        raise RuntimeError(
            f"Missing consolidated partition state: {path}"
        )

    delta_table = DeltaTable.forPath(spark, path)
    version = int(delta_table.history(1).collect()[0]["version"])

    if version != expected_version:
        raise RuntimeError(
            "Consolidated partition-state version mismatch: "
            f"expected={expected_version}, actual={version}"
        )

    required_columns = [
        "state_partition_id",
        "row_count",
        "null_passenger_count",
        "null_fare_count",
        "invalid_fare_count",
        "invalid_distance_count",
        "invalid_passenger_count",
        "invalid_pickup_time_count",
        "minimum_fare",
        "maximum_fare",
        "minimum_distance",
        "maximum_distance",
    ]

    dataframe = spark.read.format("delta").load(path)
    missing = sorted(set(required_columns).difference(dataframe.columns))

    if missing:
        raise RuntimeError(
            "Consolidated partition state is missing columns: "
            + ", ".join(missing)
        )

    return dataframe.select(
        F.col("state_partition_id").cast("int"),
        F.col("row_count").cast("long"),
        F.col("null_passenger_count").cast("long"),
        F.col("null_fare_count").cast("long"),
        F.col("invalid_fare_count").cast("long"),
        F.col("invalid_distance_count").cast("long"),
        F.col("invalid_passenger_count").cast("long"),
        F.col("invalid_pickup_time_count").cast("long"),
        F.col("minimum_fare").cast("double"),
        F.col("maximum_fare").cast("double"),
        F.col("minimum_distance").cast("double"),
        F.col("maximum_distance").cast("double"),
    )


def compact_cdf_locally(
    spark: SparkSession,
    raw_cdf: DataFrame,
) -> Tuple[DataFrame, Dict[str, int], int, int]:
    rows = raw_cdf.collect()

    change_counts = {
        "insert": 0,
        "delete": 0,
        "update_preimage": 0,
        "update_postimage": 0,
    }

    by_row_id: Dict[int, List[Row]] = {}

    for row in rows:
        change_type = str(row["_change_type"])
        change_counts[change_type] = (
            change_counts.get(change_type, 0) + 1
        )
        by_row_id.setdefault(int(row["row_id"]), []).append(row)

    negative_types = {"delete", "update_preimage"}
    positive_types = {"insert", "update_postimage"}
    net_records: List[Dict[str, Any]] = []

    for row_id, history in by_row_id.items():
        first_version = min(
            int(row["_commit_version"]) for row in history
        )
        last_version = max(
            int(row["_commit_version"]) for row in history
        )

        base_candidates = [
            row
            for row in history
            if str(row["_change_type"]) in negative_types
            and int(row["_commit_version"]) == first_version
        ]
        final_candidates = [
            row
            for row in history
            if str(row["_change_type"]) in positive_types
            and int(row["_commit_version"]) == last_version
        ]

        if len(base_candidates) > 1:
            raise RuntimeError(
                f"Multiple base images for row_id={row_id}."
            )

        if len(final_candidates) > 1:
            raise RuntimeError(
                f"Multiple final images for row_id={row_id}."
            )

        if base_candidates:
            record = base_candidates[0].asDict(recursive=True)
            record["_net_role"] = "BASE_IMAGE"
            record["_sign"] = -1
            net_records.append(record)

        if final_candidates:
            record = final_candidates[0].asDict(recursive=True)
            record["_net_role"] = "FINAL_IMAGE"
            record["_sign"] = 1
            net_records.append(record)

    net_schema = T.StructType(
        list(raw_cdf.schema.fields)
        + [
            T.StructField("_net_role", T.StringType(), False),
            T.StructField("_sign", T.LongType(), False),
        ]
    )

    net_images = spark.createDataFrame(
        net_records,
        schema=net_schema,
    )

    return (
        net_images,
        change_counts,
        len(by_row_id),
        len(net_records),
    )



def null_safe_min(
    left: Column,
    right: Column,
) -> Column:
    return (
        F.when(left.isNull(), right)
        .when(right.isNull(), left)
        .otherwise(F.least(left, right))
    )


def null_safe_max(
    left: Column,
    right: Column,
) -> Column:
    return (
        F.when(left.isNull(), right)
        .when(right.isNull(), left)
        .otherwise(F.greatest(left, right))
    )


def violation_status(value: int) -> str:
    return "PASS" if value == 0 else "FAIL"


def make_rule(
    order: int,
    rule_id: str,
    rule_name: str,
    state_kind: str,
    metric_type: str,
    metric_display: str,
    status: str,
    definition: str,
    planner_mode: str,
    long_value: Optional[int] = None,
    double_value: Optional[float] = None,
    boolean_value: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "rule_order": order,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "state_kind": state_kind,
        "metric_type": metric_type,
        "metric_display": metric_display,
        "status": status,
        "definition": definition,
        "planner_mode": planner_mode,
        "long_value": long_value,
        "double_value": double_value,
        "boolean_value": boolean_value,
    }


def main() -> None:
    args = parse_arguments()

    if args.start_version < 0:
        raise ValueError("--start-version must be non-negative.")

    if args.end_version <= args.start_version:
        raise ValueError("--end-version must be greater than --start-version.")

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    if args.key_buckets <= 0:
        raise ValueError("--key-buckets must be positive.")

    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")

    if args.double_tolerance < 0:
        raise ValueError("--double-tolerance cannot be negative.")

    if args.small_cdf_threshold < 0:
        raise ValueError("--small-cdf-threshold cannot be negative.")

    if args.small_overlay_threshold < 0:
        raise ValueError("--small-overlay-threshold cannot be negative.")

    datetime.strptime(args.min_valid_pickup, "%Y-%m-%d %H:%M:%S")
    datetime.strptime(args.max_valid_pickup, "%Y-%m-%d %H:%M:%S")

    spark = (
        SparkSession.builder
        .appName("StateGuardExactIncrementalValidationV2")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.state_partitions))
    spark.conf.set(
        "spark.sql.optimizer.dynamicPartitionPruning.enabled",
        "true",
    )

    working_path = args.working_path.rstrip("/")
    consolidated_partition_state = (
        args.consolidated_partition_state.rstrip("/")
    )
    key_state_path = args.key_frequency_state.rstrip("/")
    output_root = args.output_root.rstrip("/")

    partition_state_output = (
        f"{output_root}/updated_partition_state_delta"
    )
    key_overlay_output = (
        f"{output_root}/key_frequency_overlay_delta"
    )

    if not DeltaTable.isDeltaTable(spark, working_path):
        raise RuntimeError(f"Not a Delta table: {working_path}")

    working_table = DeltaTable.forPath(spark, working_path)
    working_detail = working_table.detail().collect()[0]
    working_version = int(
        working_table.history(1).collect()[0]["version"]
    )

    if working_version < args.end_version:
        raise RuntimeError(
            f"Working table is only at version {working_version}; "
            f"requested end version is {args.end_version}."
        )

    properties = working_detail["properties"] or {}
    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    if not cdf_enabled:
        raise RuntimeError("Working table does not have CDF enabled.")

    initial_state_summary = read_single_csv_row(
        spark,
        args.initial_state_summary,
    )
    baseline_full_summary = read_single_csv_row(
        spark,
        args.baseline_full_summary,
    )
    ground_truth_summary = read_single_csv_row(
        spark,
        args.ground_truth_summary,
    )

    if required_int(
        initial_state_summary,
        "delta_version",
    ) != args.start_version:
        raise RuntimeError(
            "Initial state version does not match --start-version."
        )

    if required_int(
        initial_state_summary,
        "logical_partition_count",
    ) != args.state_partitions:
        raise RuntimeError(
            "Initial logical partition count does not match the request."
        )

    if required_int(
        initial_state_summary,
        "key_bucket_count",
    ) != args.key_buckets:
        raise RuntimeError(
            "Initial key-bucket count does not match the request."
        )

    if required_int(
        baseline_full_summary,
        "delta_version",
    ) != args.start_version:
        raise RuntimeError(
            "Baseline full-validation version does not match start version."
        )

    if required_int(
        ground_truth_summary,
        "delta_version",
    ) != args.end_version:
        raise RuntimeError(
            "Ground-truth version does not match --end-version."
        )

    baseline_row_count = required_int(
        baseline_full_summary,
        "row_count",
    )
    ground_truth_row_count = required_int(
        ground_truth_summary,
        "row_count",
    )
    full_validation_seconds = required_float(
        ground_truth_summary,
        "total_seconds",
    )

    baseline_distinct_keys = required_int(
        initial_state_summary,
        "distinct_trip_keys",
    )
    baseline_duplicate_groups = required_int(
        initial_state_summary,
        "duplicate_key_groups",
    )
    baseline_duplicate_extra = required_int(
        initial_state_summary,
        "duplicate_extra_rows",
    )
    baseline_frequency_sum = required_int(
        initial_state_summary,
        "frequency_sum",
    )
    baseline_max_multiplicity = required_int(
        baseline_full_summary,
        "maximum_key_multiplicity",
    )

    if baseline_frequency_sum != baseline_row_count:
        raise RuntimeError(
            "Initial key-frequency sum does not equal baseline row count."
        )

    print("=" * 78)
    print("STATEGUARD EXACT INCREMENTAL VALIDATION V2")
    print("=" * 78)
    print(f"Working path: {working_path}")
    print(
        f"Version interval: "
        f"{args.start_version} -> {args.end_version}"
    )
    print(
        "Consolidated partition state: "
        f"{consolidated_partition_state}"
    )
    print(f"Exact key-frequency state: {key_state_path}")
    print(
        "Correctness target: exact agreement with 13-rule full validation"
    )
    print("=" * 78)

    algorithm_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase A: Adaptively compact CDF into exact base/final row images.
    # Tiny change feeds are collected once and compacted on the driver
    # to avoid unnecessary Spark shuffles. Larger feeds retain the
    # distributed compaction path.
    # ------------------------------------------------------------------
    cdf_start = time.perf_counter()

    raw_cdf_unpersist_required = False
    row_bounds: Optional[DataFrame] = None

    raw_cdf_source = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", args.start_version + 1)
        .option("endingVersion", args.end_version)
        .load(working_path)
    )

    required_cdf_columns = {
        "row_id",
        "trip_key",
        "state_partition_id",
        "passenger_count",
        "fare_amount",
        "trip_distance",
        "tpep_pickup_datetime",
        "_change_type",
        "_commit_version",
    }

    missing_cdf_columns = sorted(
        required_cdf_columns.difference(raw_cdf_source.columns)
    )

    if missing_cdf_columns:
        raise RuntimeError(
            "CDF is missing required columns: "
            + ", ".join(missing_cdf_columns)
        )

    probe_rows = raw_cdf_source.limit(
        args.small_cdf_threshold + 1
    ).collect()

    if len(probe_rows) <= args.small_cdf_threshold:
        cdf_execution_mode = "DRIVER_LOCAL_COMPACTION"
        raw_cdf_rows = len(probe_rows)
        raw_cdf = spark.createDataFrame(
            probe_rows,
            schema=raw_cdf_source.schema,
        )

        (
            net_images,
            change_counts,
            net_changed_row_ids,
            net_image_rows,
        ) = compact_cdf_locally(
            spark,
            raw_cdf,
        )

        net_images = net_images.persist(
            StorageLevel.MEMORY_AND_DISK
        )
        net_images.count()
    else:
        cdf_execution_mode = "DISTRIBUTED_COMPACTION"
        raw_cdf = raw_cdf_source.persist(
            StorageLevel.MEMORY_AND_DISK
        )
        raw_cdf_unpersist_required = True
        raw_cdf_rows = raw_cdf.count()

        change_counts_rows = (
            raw_cdf.groupBy("_change_type")
            .count()
            .collect()
        )

        change_counts = {
            "insert": 0,
            "delete": 0,
            "update_preimage": 0,
            "update_postimage": 0,
        }

        for row in change_counts_rows:
            change_counts[str(row["_change_type"])] = int(
                row["count"]
            )

        row_bounds = (
            raw_cdf.groupBy("row_id")
            .agg(
                F.min("_commit_version")
                .cast("long")
                .alias("first_version"),
                F.max("_commit_version")
                .cast("long")
                .alias("last_version"),
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        net_changed_row_ids = row_bounds.count()

        negative_types = ["delete", "update_preimage"]
        positive_types = ["insert", "update_postimage"]

        base_images = (
            raw_cdf.join(row_bounds, on="row_id", how="inner")
            .filter(
                F.col("_change_type").isin(negative_types)
                & (
                    F.col("_commit_version")
                    == F.col("first_version")
                )
            )
            .drop("first_version", "last_version")
            .withColumn("_net_role", F.lit("BASE_IMAGE"))
            .withColumn("_sign", F.lit(-1).cast("long"))
        )

        final_images = (
            raw_cdf.join(row_bounds, on="row_id", how="inner")
            .filter(
                F.col("_change_type").isin(positive_types)
                & (
                    F.col("_commit_version")
                    == F.col("last_version")
                )
            )
            .drop("first_version", "last_version")
            .withColumn("_net_role", F.lit("FINAL_IMAGE"))
            .withColumn("_sign", F.lit(1).cast("long"))
        )

        net_images = (
            base_images.unionByName(final_images)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        net_image_rows = net_images.count()

        duplicate_net_roles = (
            net_images.groupBy("row_id", "_net_role")
            .count()
            .filter(F.col("count") != 1)
            .limit(1)
            .count()
        )

        if duplicate_net_roles != 0:
            raise RuntimeError(
                "CDF compaction produced duplicate base/final images."
            )

    cdf_seconds = time.perf_counter() - cdf_start

    # ------------------------------------------------------------------
    # Phase B: Load the initial exact state portfolio and update R01-R07
    # using signed arithmetic over the compact net row images.
    # ------------------------------------------------------------------
    additive_start = time.perf_counter()

    partition_state_input_metrics = delta_metrics(
        spark,
        consolidated_partition_state,
    )

    base_partition_state = (
        load_consolidated_partition_state(
            spark,
            consolidated_partition_state,
            args.start_version,
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    base_partition_count = base_partition_state.count()

    if base_partition_count != args.state_partitions:
        raise RuntimeError(
            f"Expected {args.state_partitions} initial partitions; "
            f"found {base_partition_count}."
        )

    min_pickup = F.lit(
        args.min_valid_pickup
    ).cast("timestamp_ntz")
    max_pickup = F.lit(
        args.max_valid_pickup
    ).cast("timestamp_ntz")

    signed_metrics = (
        net_images.select(
            F.col("state_partition_id").cast("int"),
            F.col("_sign").cast("long").alias("sign"),
            (
                F.col("_sign")
                * F.when(
                    F.col("passenger_count").isNull(),
                    1,
                ).otherwise(0)
            ).cast("long").alias("null_passenger_delta"),
            (
                F.col("_sign")
                * F.when(
                    F.col("fare_amount").isNull(),
                    1,
                ).otherwise(0)
            ).cast("long").alias("null_fare_delta"),
            (
                F.col("_sign")
                * F.when(
                    F.col("fare_amount").isNotNull()
                    & (F.col("fare_amount") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_fare_delta"),
            (
                F.col("_sign")
                * F.when(
                    F.col("trip_distance").isNotNull()
                    & (F.col("trip_distance") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_distance_delta"),
            (
                F.col("_sign")
                * F.when(
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
            ).cast("long").alias("invalid_passenger_delta"),
            (
                F.col("_sign")
                * F.when(
                    F.col("tpep_pickup_datetime").isNull()
                    | (
                        F.col("tpep_pickup_datetime")
                        < min_pickup
                    )
                    | (
                        F.col("tpep_pickup_datetime")
                        > max_pickup
                    ),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_pickup_delta"),
        )
    )

    additive_deltas = (
        signed_metrics.groupBy("state_partition_id")
        .agg(
            F.sum("sign").cast("long").alias("row_count_delta"),
            F.sum("null_passenger_delta")
            .cast("long")
            .alias("null_passenger_delta"),
            F.sum("null_fare_delta")
            .cast("long")
            .alias("null_fare_delta"),
            F.sum("invalid_fare_delta")
            .cast("long")
            .alias("invalid_fare_delta"),
            F.sum("invalid_distance_delta")
            .cast("long")
            .alias("invalid_distance_delta"),
            F.sum("invalid_passenger_delta")
            .cast("long")
            .alias("invalid_passenger_delta"),
            F.sum("invalid_pickup_delta")
            .cast("long")
            .alias("invalid_pickup_delta"),
        )
    )

    updated_partition_state = (
        base_partition_state.join(
            additive_deltas,
            on="state_partition_id",
            how="left",
        )
        .fillna(
            {
                "row_count_delta": 0,
                "null_passenger_delta": 0,
                "null_fare_delta": 0,
                "invalid_fare_delta": 0,
                "invalid_distance_delta": 0,
                "invalid_passenger_delta": 0,
                "invalid_pickup_delta": 0,
            }
        )
        .withColumn(
            "row_count",
            F.col("row_count") + F.col("row_count_delta"),
        )
        .withColumn(
            "null_passenger_count",
            F.col("null_passenger_count")
            + F.col("null_passenger_delta"),
        )
        .withColumn(
            "null_fare_count",
            F.col("null_fare_count")
            + F.col("null_fare_delta"),
        )
        .withColumn(
            "invalid_fare_count",
            F.col("invalid_fare_count")
            + F.col("invalid_fare_delta"),
        )
        .withColumn(
            "invalid_distance_count",
            F.col("invalid_distance_count")
            + F.col("invalid_distance_delta"),
        )
        .withColumn(
            "invalid_passenger_count",
            F.col("invalid_passenger_count")
            + F.col("invalid_passenger_delta"),
        )
        .withColumn(
            "invalid_pickup_time_count",
            F.col("invalid_pickup_time_count")
            + F.col("invalid_pickup_delta"),
        )
        .drop(
            "row_count_delta",
            "null_passenger_delta",
            "null_fare_delta",
            "invalid_fare_delta",
            "invalid_distance_delta",
            "invalid_passenger_delta",
            "invalid_pickup_delta",
        )
    )

    additive_seconds = time.perf_counter() - additive_start

    # ------------------------------------------------------------------
    # Phase C: Detect delete-sensitive extrema invalidation. Only the
    # partitions whose stored extrema were removed are read from the
    # final table. All other extrema are updated from final positive
    # images without rescanning base data.
    # ------------------------------------------------------------------
    extrema_start = time.perf_counter()

    invalidated_by_metric: Dict[str, Set[int]] = {}
    positive_candidates: Dict[str, DataFrame] = {}

    for config in EXTREMA_CONFIGS:
        state_column = str(config["state_column"])
        source_column = str(config["source_column"])
        direction = str(config["direction"])

        removed = (
            net_images.filter(
                (F.col("_sign") == -1)
                & F.col(source_column).isNotNull()
            )
            .select(
                F.col("state_partition_id").cast("int"),
                F.col(source_column)
                .cast("double")
                .alias("removed_value"),
            )
        )

        invalidated_rows = (
            removed.join(
                base_partition_state.select(
                    "state_partition_id",
                    F.col(state_column)
                    .cast("double")
                    .alias("stored_extreme"),
                ),
                on="state_partition_id",
                how="inner",
            )
            .filter(
                F.col("removed_value").eqNullSafe(
                    F.col("stored_extreme")
                )
            )
            .select("state_partition_id")
            .distinct()
            .collect()
        )

        invalidated_by_metric[state_column] = {
            int(row["state_partition_id"])
            for row in invalidated_rows
        }

        positive = net_images.filter(
            (F.col("_sign") == 1)
            & F.col(source_column).isNotNull()
        )

        if direction == "MIN":
            candidate = positive.groupBy(
                "state_partition_id"
            ).agg(
                F.min(source_column)
                .cast("double")
                .alias(f"{state_column}_positive_candidate")
            )
        else:
            candidate = positive.groupBy(
                "state_partition_id"
            ).agg(
                F.max(source_column)
                .cast("double")
                .alias(f"{state_column}_positive_candidate")
            )

        positive_candidates[state_column] = candidate

    all_invalidated_partitions = sorted(
        set().union(*invalidated_by_metric.values())
        if invalidated_by_metric
        else set()
    )

    recompute_row_count = 0

    if all_invalidated_partitions:
        final_table = (
            spark.read.format("delta")
            .option("versionAsOf", args.end_version)
            .load(working_path)
        )

        recomputed_partitions = (
            final_table.filter(
                F.col("state_partition_id").isin(
                    all_invalidated_partitions
                )
            )
            .groupBy("state_partition_id")
            .agg(
                F.count(F.lit(1))
                .cast("long")
                .alias("recomputed_row_count"),
                F.min("fare_amount")
                .cast("double")
                .alias("recomputed_minimum_fare"),
                F.max("fare_amount")
                .cast("double")
                .alias("recomputed_maximum_fare"),
                F.min("trip_distance")
                .cast("double")
                .alias("recomputed_minimum_distance"),
                F.max("trip_distance")
                .cast("double")
                .alias("recomputed_maximum_distance"),
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        recomputed_rows = recomputed_partitions.collect()

        if len(recomputed_rows) != len(
            all_invalidated_partitions
        ):
            raise RuntimeError(
                "One or more invalidated partitions disappeared."
            )

        recompute_row_count = sum(
            int(row["recomputed_row_count"])
            for row in recomputed_rows
        )
    else:
        recomputed_schema = T.StructType(
            [
                T.StructField(
                    "state_partition_id",
                    T.IntegerType(),
                    False,
                ),
                T.StructField(
                    "recomputed_row_count",
                    T.LongType(),
                    False,
                ),
                T.StructField(
                    "recomputed_minimum_fare",
                    T.DoubleType(),
                    True,
                ),
                T.StructField(
                    "recomputed_maximum_fare",
                    T.DoubleType(),
                    True,
                ),
                T.StructField(
                    "recomputed_minimum_distance",
                    T.DoubleType(),
                    True,
                ),
                T.StructField(
                    "recomputed_maximum_distance",
                    T.DoubleType(),
                    True,
                ),
            ]
        )
        recomputed_partitions = spark.createDataFrame(
            [],
            schema=recomputed_schema,
        )

    updated_partition_state = updated_partition_state.join(
        recomputed_partitions,
        on="state_partition_id",
        how="left",
    )

    recomputed_column_map = {
        "minimum_fare": "recomputed_minimum_fare",
        "maximum_fare": "recomputed_maximum_fare",
        "minimum_distance": "recomputed_minimum_distance",
        "maximum_distance": "recomputed_maximum_distance",
    }

    for config in EXTREMA_CONFIGS:
        state_column = str(config["state_column"])
        direction = str(config["direction"])
        candidate_column = (
            f"{state_column}_positive_candidate"
        )
        invalidated_ids = sorted(
            invalidated_by_metric[state_column]
        )

        updated_partition_state = updated_partition_state.join(
            positive_candidates[state_column],
            on="state_partition_id",
            how="left",
        )

        if direction == "MIN":
            state_assisted_value = null_safe_min(
                F.col(state_column),
                F.col(candidate_column),
            )
        else:
            state_assisted_value = null_safe_max(
                F.col(state_column),
                F.col(candidate_column),
            )

        if invalidated_ids:
            exact_value = F.when(
                F.col("state_partition_id").isin(
                    invalidated_ids
                ),
                F.col(recomputed_column_map[state_column]),
            ).otherwise(state_assisted_value)
        else:
            exact_value = state_assisted_value

        updated_partition_state = (
            updated_partition_state.withColumn(
                state_column,
                exact_value.cast("double"),
            )
            .drop(candidate_column)
        )

    updated_partition_state = (
        updated_partition_state.drop(
            "recomputed_row_count",
            "recomputed_minimum_fare",
            "recomputed_maximum_fare",
            "recomputed_minimum_distance",
            "recomputed_maximum_distance",
        )
        .select(
            "state_partition_id",
            "row_count",
            "null_passenger_count",
            "null_fare_count",
            "invalid_fare_count",
            "invalid_distance_count",
            "invalid_passenger_count",
            "invalid_pickup_time_count",
            "minimum_fare",
            "maximum_fare",
            "minimum_distance",
            "maximum_distance",
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    updated_partition_count = updated_partition_state.count()

    if updated_partition_count != args.state_partitions:
        raise RuntimeError(
            "Updated partition-state row count is incorrect."
        )

    negative_partition_states = (
        updated_partition_state.filter(
            (F.col("row_count") < 0)
            | (F.col("null_passenger_count") < 0)
            | (F.col("null_fare_count") < 0)
            | (F.col("invalid_fare_count") < 0)
            | (F.col("invalid_distance_count") < 0)
            | (F.col("invalid_passenger_count") < 0)
            | (F.col("invalid_pickup_time_count") < 0)
        )
        .limit(1)
        .count()
    )

    if negative_partition_states != 0:
        raise RuntimeError(
            "Incremental arithmetic produced a negative state value."
        )

    partition_global = (
        updated_partition_state.agg(
            F.sum("row_count").cast("long").alias("R01"),
            F.sum("null_passenger_count")
            .cast("long")
            .alias("R02"),
            F.sum("null_fare_count")
            .cast("long")
            .alias("R03"),
            F.sum("invalid_fare_count")
            .cast("long")
            .alias("R04"),
            F.sum("invalid_distance_count")
            .cast("long")
            .alias("R05"),
            F.sum("invalid_passenger_count")
            .cast("long")
            .alias("R06"),
            F.sum("invalid_pickup_time_count")
            .cast("long")
            .alias("R07"),
            F.min("minimum_fare")
            .cast("double")
            .alias("R08"),
            F.max("maximum_fare")
            .cast("double")
            .alias("R09"),
            F.min("minimum_distance")
            .cast("double")
            .alias("R10"),
            F.max("maximum_distance")
            .cast("double")
            .alias("R11"),
        )
        .collect()[0]
    )

    if int(partition_global["R01"]) != ground_truth_row_count:
        raise RuntimeError(
            "Incremental partition states do not match final row count."
        )

    extrema_seconds = time.perf_counter() - extrema_start

    # ------------------------------------------------------------------
    # Phase D: Maintain exact uniqueness using a copy-on-write overlay.
    # Only affected key buckets are read from the 2.2-GiB base state.
    # ------------------------------------------------------------------
    key_start = time.perf_counter()

    affected_key_deltas = (
        net_images.groupBy("trip_key")
        .agg(
            F.sum("_sign")
            .cast("long")
            .alias("delta_frequency")
        )
        .filter(F.col("delta_frequency") != 0)
        .withColumn(
            "key_bucket",
            F.pmod(
                F.xxhash64("trip_key"),
                F.lit(args.key_buckets),
            ).cast("int"),
        )
        .select(
            "key_bucket",
            "trip_key",
            "delta_frequency",
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    affected_key_count = affected_key_deltas.count()

    affected_key_buckets = sorted(
        {
            int(row["key_bucket"])
            for row in affected_key_deltas.select(
                "key_bucket"
            ).distinct().collect()
        }
    )

    if not DeltaTable.isDeltaTable(spark, key_state_path):
        raise RuntimeError(
            f"Missing exact key-frequency state: {key_state_path}"
        )

    base_key_state = spark.read.format("delta").load(
        key_state_path
    )

    if affected_key_buckets:
        base_key_subset = base_key_state.filter(
            F.col("key_bucket").isin(affected_key_buckets)
        )
    else:
        base_key_subset = base_key_state.limit(0)

    key_overlay = (
        affected_key_deltas.join(
            base_key_subset.select(
                "key_bucket",
                "trip_key",
                F.col("frequency")
                .cast("long")
                .alias("base_frequency"),
            ),
            on=["key_bucket", "trip_key"],
            how="left",
        )
        .fillna({"base_frequency": 0})
        .withColumn(
            "final_frequency",
            F.col("base_frequency")
            + F.col("delta_frequency"),
        )
        .withColumn(
            "overlay_action",
            F.when(
                F.col("final_frequency") == 0,
                F.lit("TOMBSTONE"),
            ).otherwise(F.lit("UPSERT")),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    invalid_key_frequency = (
        key_overlay.filter(F.col("final_frequency") < 0)
        .limit(1)
        .count()
    )

    if invalid_key_frequency != 0:
        raise RuntimeError(
            "Key-frequency overlay produced a negative frequency."
        )

    key_adjustments = (
        key_overlay.agg(
            F.sum(
                F.when(
                    (F.col("base_frequency") == 0)
                    & (F.col("final_frequency") > 0),
                    1,
                )
                .when(
                    (F.col("base_frequency") > 0)
                    & (F.col("final_frequency") == 0),
                    -1,
                )
                .otherwise(0)
            ).cast("long").alias("distinct_key_delta"),
            F.sum(
                (
                    F.when(
                        F.col("final_frequency") > 1,
                        1,
                    ).otherwise(0)
                    - F.when(
                        F.col("base_frequency") > 1,
                        1,
                    ).otherwise(0)
                )
            ).cast("long").alias("duplicate_group_delta"),
            F.sum(
                (
                    F.greatest(
                        F.col("final_frequency") - F.lit(1),
                        F.lit(0),
                    )
                    - F.greatest(
                        F.col("base_frequency") - F.lit(1),
                        F.lit(0),
                    )
                )
            ).cast("long").alias("duplicate_extra_delta"),
            F.sum("delta_frequency")
            .cast("long")
            .alias("frequency_sum_delta"),
            F.max("final_frequency")
            .cast("long")
            .alias("max_final_affected"),
            F.sum(
                F.when(
                    (
                        F.col("base_frequency")
                        == F.lit(baseline_max_multiplicity)
                    )
                    & (
                        F.col("final_frequency")
                        < F.lit(baseline_max_multiplicity)
                    ),
                    1,
                ).otherwise(0)
            ).cast("long").alias(
                "invalidated_baseline_max_keys"
            ),
        )
        .collect()[0]
    )

    distinct_key_delta = int(
        key_adjustments["distinct_key_delta"] or 0
    )
    duplicate_group_delta = int(
        key_adjustments["duplicate_group_delta"] or 0
    )
    duplicate_extra_delta = int(
        key_adjustments["duplicate_extra_delta"] or 0
    )
    frequency_sum_delta = int(
        key_adjustments["frequency_sum_delta"] or 0
    )
    max_final_affected = int(
        key_adjustments["max_final_affected"] or 0
    )
    invalidated_baseline_max_keys = int(
        key_adjustments["invalidated_baseline_max_keys"] or 0
    )

    final_distinct_keys = (
        baseline_distinct_keys + distinct_key_delta
    )
    final_duplicate_groups = (
        baseline_duplicate_groups + duplicate_group_delta
    )
    final_duplicate_extra = (
        baseline_duplicate_extra + duplicate_extra_delta
    )
    final_frequency_sum = (
        baseline_frequency_sum + frequency_sum_delta
    )

    maximum_frequency_fallback_scan = False

    if (
        invalidated_baseline_max_keys > 0
        and max_final_affected < baseline_max_multiplicity
    ):
        maximum_frequency_fallback_scan = True

        unaffected_max_row = (
            base_key_state.join(
                key_overlay.select(
                    "trip_key"
                ).distinct(),
                on="trip_key",
                how="left_anti",
            )
            .agg(
                F.max("frequency")
                .cast("long")
                .alias("unaffected_max")
            )
            .collect()[0]
        )

        unaffected_max = int(
            unaffected_max_row["unaffected_max"] or 0
        )
        final_max_multiplicity = max(
            unaffected_max,
            max_final_affected,
        )
    else:
        final_max_multiplicity = max(
            baseline_max_multiplicity,
            max_final_affected,
        )

    final_uniqueness_pass = final_duplicate_extra == 0

    if final_frequency_sum != int(partition_global["R01"]):
        raise RuntimeError(
            "Updated key frequencies do not sum to final row count."
        )

    key_seconds = time.perf_counter() - key_start
    validation_ready_seconds = time.perf_counter() - algorithm_start

    # ------------------------------------------------------------------
    # Phase E: Persist the maintained state snapshot and compact overlay.
    # State writes are included in incremental runtime because they are
    # required for future exact validation, unlike result-report writes.
    # ------------------------------------------------------------------
    state_persist_start = time.perf_counter()

    (
        updated_partition_state.coalesce(1)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(partition_state_output)
    )

    key_overlay_to_write = key_overlay.select(
        "key_bucket",
        "trip_key",
        "base_frequency",
        "delta_frequency",
        "final_frequency",
        "overlay_action",
    )

    if affected_key_count <= args.small_overlay_threshold:
        overlay_layout = "COMPACT_SINGLE_FILE"
        (
            key_overlay_to_write.coalesce(1)
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(key_overlay_output)
        )
    else:
        overlay_layout = "BUCKET_PARTITIONED"
        (
            key_overlay_to_write.repartition(
                max(1, len(affected_key_buckets)),
                "key_bucket",
            )
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("key_bucket")
            .save(key_overlay_output)
        )

    partition_state_metrics = delta_metrics(
        spark,
        partition_state_output,
    )
    key_overlay_metrics = delta_metrics(
        spark,
        key_overlay_output,
    )

    state_persist_seconds = (
        time.perf_counter() - state_persist_start
    )
    maintenance_ready_seconds = (
        time.perf_counter() - algorithm_start
    )

    # ------------------------------------------------------------------
    # Phase F: Materialize 13 rule results and prove exact equivalence
    # to the independent full-table validation at the same Delta version.
    # ------------------------------------------------------------------
    comparison_start = time.perf_counter()

    planner_modes: Dict[str, str] = {
        "R01": "STATE_ASSISTED",
        "R02": "STATE_ASSISTED",
        "R03": "STATE_ASSISTED",
        "R04": "STATE_ASSISTED",
        "R05": "STATE_ASSISTED",
        "R06": "STATE_ASSISTED",
        "R07": "STATE_ASSISTED",
        "R12": "STATE_ASSISTED_KEY_OVERLAY",
        "R13": "STATE_ASSISTED_KEY_OVERLAY",
    }

    for config in EXTREMA_CONFIGS:
        state_column = str(config["state_column"])
        rule_id = str(config["rule_id"])

        planner_modes[rule_id] = (
            "PARTITION_RECOMPUTE"
            if invalidated_by_metric[state_column]
            else "STATE_ASSISTED"
        )

    rule_values = {
        f"R{index:02d}": partition_global[f"R{index:02d}"]
        for index in range(1, 12)
    }

    rules: List[Dict[str, Any]] = [
        make_rule(
            1,
            "R01",
            "Row count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R01"])),
            "OBSERVED",
            "Total number of records.",
            planner_modes["R01"],
            long_value=int(rule_values["R01"]),
        ),
        make_rule(
            2,
            "R02",
            "Null passenger count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R02"])),
            violation_status(int(rule_values["R02"])),
            "Rows where passenger_count is NULL.",
            planner_modes["R02"],
            long_value=int(rule_values["R02"]),
        ),
        make_rule(
            3,
            "R03",
            "Null fare count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R03"])),
            violation_status(int(rule_values["R03"])),
            "Rows where fare_amount is NULL.",
            planner_modes["R03"],
            long_value=int(rule_values["R03"]),
        ),
        make_rule(
            4,
            "R04",
            "Invalid fare count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R04"])),
            violation_status(int(rule_values["R04"])),
            "Non-null rows where fare_amount is negative.",
            planner_modes["R04"],
            long_value=int(rule_values["R04"]),
        ),
        make_rule(
            5,
            "R05",
            "Invalid distance count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R05"])),
            violation_status(int(rule_values["R05"])),
            "Non-null rows where trip_distance is negative.",
            planner_modes["R05"],
            long_value=int(rule_values["R05"]),
        ),
        make_rule(
            6,
            "R06",
            "Invalid passenger count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R06"])),
            violation_status(int(rule_values["R06"])),
            (
                "Non-null passenger_count outside project policy "
                f"range 1..{args.max_passengers}."
            ),
            planner_modes["R06"],
            long_value=int(rule_values["R06"]),
        ),
        make_rule(
            7,
            "R07",
            "Invalid pickup-time count",
            "ADDITIVE",
            "LONG",
            str(int(rule_values["R07"])),
            violation_status(int(rule_values["R07"])),
            (
                "NULL pickup timestamps or values outside "
                f"[{args.min_valid_pickup}, {args.max_valid_pickup}]."
            ),
            planner_modes["R07"],
            long_value=int(rule_values["R07"]),
        ),
        make_rule(
            8,
            "R08",
            "Minimum fare",
            "EXTREMA",
            "DOUBLE",
            str(float(rule_values["R08"])),
            "OBSERVED",
            "Minimum non-null fare_amount.",
            planner_modes["R08"],
            double_value=float(rule_values["R08"]),
        ),
        make_rule(
            9,
            "R09",
            "Maximum fare",
            "EXTREMA",
            "DOUBLE",
            str(float(rule_values["R09"])),
            "OBSERVED",
            "Maximum non-null fare_amount.",
            planner_modes["R09"],
            double_value=float(rule_values["R09"]),
        ),
        make_rule(
            10,
            "R10",
            "Minimum distance",
            "EXTREMA",
            "DOUBLE",
            str(float(rule_values["R10"])),
            "OBSERVED",
            "Minimum non-null trip_distance.",
            planner_modes["R10"],
            double_value=float(rule_values["R10"]),
        ),
        make_rule(
            11,
            "R11",
            "Maximum distance",
            "EXTREMA",
            "DOUBLE",
            str(float(rule_values["R11"])),
            "OBSERVED",
            "Maximum non-null trip_distance.",
            planner_modes["R11"],
            double_value=float(rule_values["R11"]),
        ),
        make_rule(
            12,
            "R12",
            "Duplicate key count",
            "KEY_FREQUENCY",
            "LONG",
            str(final_duplicate_extra),
            violation_status(final_duplicate_extra),
            (
                "Extra rows beyond the first occurrence of each "
                "exact 20-column trip_key."
            ),
            planner_modes["R12"],
            long_value=final_duplicate_extra,
        ),
        make_rule(
            13,
            "R13",
            "Uniqueness pass",
            "KEY_FREQUENCY",
            "BOOLEAN",
            str(final_uniqueness_pass).lower(),
            "PASS" if final_uniqueness_pass else "FAIL",
            "True only when R12 equals zero.",
            planner_modes["R13"],
            boolean_value=final_uniqueness_pass,
        ),
    ]

    rules_schema = T.StructType(
        [
            T.StructField("rule_order", T.IntegerType(), False),
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("rule_name", T.StringType(), False),
            T.StructField("state_kind", T.StringType(), False),
            T.StructField("metric_type", T.StringType(), False),
            T.StructField("metric_display", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("definition", T.StringType(), False),
            T.StructField("planner_mode", T.StringType(), False),
            T.StructField("long_value", T.LongType(), True),
            T.StructField("double_value", T.DoubleType(), True),
            T.StructField("boolean_value", T.BooleanType(), True),
        ]
    )

    incremental_rules = (
        spark.createDataFrame(rules, schema=rules_schema)
        .orderBy("rule_order")
    )

    ground_truth_rules = (
        spark.read.option("header", "true")
        .csv(args.ground_truth_rules)
        .select(
            F.col("rule_id"),
            F.col("metric_type").alias("ground_truth_metric_type"),
            F.col("metric_display")
            .alias("ground_truth_display"),
            F.col("status").alias("ground_truth_status"),
            F.col("long_value")
            .cast("long")
            .alias("ground_truth_long"),
            F.col("double_value")
            .cast("double")
            .alias("ground_truth_double"),
            F.col("boolean_value")
            .cast("boolean")
            .alias("ground_truth_boolean"),
        )
    )

    comparison = (
        incremental_rules.alias("i")
        .join(
            ground_truth_rules.alias("g"),
            on="rule_id",
            how="inner",
        )
        .withColumn(
            "metric_exact_match",
            F.when(
                F.col("i.metric_type") == "LONG",
                F.col("i.long_value").eqNullSafe(
                    F.col("g.ground_truth_long")
                ),
            )
            .when(
                F.col("i.metric_type") == "DOUBLE",
                F.abs(
                    F.col("i.double_value")
                    - F.col("g.ground_truth_double")
                )
                <= F.lit(args.double_tolerance),
            )
            .when(
                F.col("i.metric_type") == "BOOLEAN",
                F.col("i.boolean_value").eqNullSafe(
                    F.col("g.ground_truth_boolean")
                ),
            )
            .otherwise(F.lit(False)),
        )
        .withColumn(
            "status_exact_match",
            F.col("i.status").eqNullSafe(
                F.col("g.ground_truth_status")
            ),
        )
        .withColumn(
            "exact_match",
            F.col("metric_exact_match")
            & F.col("status_exact_match"),
        )
        .withColumn(
            "absolute_error",
            F.when(
                F.col("i.metric_type") == "LONG",
                F.abs(
                    F.col("i.long_value").cast("double")
                    - F.col("g.ground_truth_long").cast("double")
                ),
            )
            .when(
                F.col("i.metric_type") == "DOUBLE",
                F.abs(
                    F.col("i.double_value")
                    - F.col("g.ground_truth_double")
                ),
            )
            .otherwise(F.lit(None).cast("double")),
        )
        .select(
            F.col("i.rule_order").alias("rule_order"),
            "rule_id",
            F.col("i.rule_name").alias("rule_name"),
            F.col("i.state_kind").alias("state_kind"),
            F.col("i.planner_mode").alias("planner_mode"),
            F.col("i.metric_type").alias("metric_type"),
            F.col("i.metric_display").alias("incremental_value"),
            F.col("g.ground_truth_display")
            .alias("full_validation_value"),
            F.col("i.status").alias("incremental_status"),
            F.col("g.ground_truth_status")
            .alias("full_validation_status"),
            "absolute_error",
            "metric_exact_match",
            "status_exact_match",
            "exact_match",
        )
        .orderBy("rule_order")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    comparison_rows = comparison.count()

    if comparison_rows != 13:
        raise RuntimeError(
            f"Expected 13 comparison rows; found {comparison_rows}."
        )

    mismatch_count = comparison.filter(
        ~F.col("exact_match")
    ).count()

    exact_match_count = comparison_rows - mismatch_count
    exact_agreement_rate = (
        exact_match_count / comparison_rows
    )

    if mismatch_count != 0:
        comparison.filter(~F.col("exact_match")).show(
            20,
            truncate=False,
        )
        raise RuntimeError(
            f"Incremental validation disagreed on {mismatch_count} rules."
        )

    comparison_seconds = (
        time.perf_counter() - comparison_start
    )

    incremental_total_seconds = (
        time.perf_counter() - algorithm_start
    )

    validation_speedup_vs_full = (
        full_validation_seconds / validation_ready_seconds
        if validation_ready_seconds > 0
        else None
    )
    end_to_end_speedup_vs_full = (
        full_validation_seconds / incremental_total_seconds
        if incremental_total_seconds > 0
        else None
    )

    affected_partition_count = (
        net_images.select("state_partition_id")
        .distinct()
        .count()
    )
    cdf_fraction_percent = (
        100.0 * raw_cdf_rows / ground_truth_row_count
    )

    planner_rows: List[Tuple[Any, ...]] = []

    for rule_number in range(1, 14):
        rule_id = f"R{rule_number:02d}"
        state_id = (
            f"S{rule_number:02d}"
            if rule_number <= 12
            else "S12"
        )

        if rule_number <= 7:
            invalidated_partitions = 0
            reason = (
                "Signed arithmetic over compact base/final CDF images."
            )
        elif rule_number <= 11:
            state_column = {
                "R08": "minimum_fare",
                "R09": "maximum_fare",
                "R10": "minimum_distance",
                "R11": "maximum_distance",
            }[rule_id]
            invalidated_partitions = len(
                invalidated_by_metric[state_column]
            )
            reason = (
                "Stored partition extreme was removed; recompute only "
                "invalidated partitions."
                if invalidated_partitions > 0
                else (
                    "No stored extreme was removed; merge positive "
                    "final images with retained partition extrema."
                )
            )
        else:
            invalidated_partitions = 0
            reason = (
                "Update affected exact key frequencies through a "
                "copy-on-write bucket-pruned overlay."
            )

        planner_rows.append(
            (
                rule_number,
                rule_id,
                state_id,
                planner_modes[rule_id],
                invalidated_partitions,
                reason,
            )
        )

    planner_schema = T.StructType(
        [
            T.StructField("rule_order", T.IntegerType(), False),
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("state_id", T.StringType(), False),
            T.StructField("planner_mode", T.StringType(), False),
            T.StructField(
                "invalidated_partition_count",
                T.IntegerType(),
                False,
            ),
            T.StructField("decision_reason", T.StringType(), False),
        ]
    )

    planner_df = (
        spark.createDataFrame(
            planner_rows,
            schema=planner_schema,
        )
        .orderBy("rule_order")
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("start_version", T.LongType(), False),
            T.StructField("end_version", T.LongType(), False),
            T.StructField("baseline_rows", T.LongType(), False),
            T.StructField("final_rows", T.LongType(), False),
            T.StructField("raw_cdf_rows", T.LongType(), False),
            T.StructField(
                "net_changed_row_ids",
                T.LongType(),
                False,
            ),
            T.StructField("net_image_rows", T.LongType(), False),
            T.StructField(
                "cdf_fraction_percent",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "affected_partition_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "invalidated_partition_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "recomputed_partition_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "affected_key_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "affected_key_bucket_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "maximum_frequency_fallback_scan",
                T.BooleanType(),
                False,
            ),
            T.StructField(
                "exact_match_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "rule_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "exact_agreement_rate",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "cdf_execution_mode",
                T.StringType(),
                False,
            ),
            T.StructField(
                "partition_state_input_num_files",
                T.LongType(),
                False,
            ),
            T.StructField(
                "partition_state_input_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "overlay_layout",
                T.StringType(),
                False,
            ),
            T.StructField(
                "cdf_compaction_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "additive_update_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "extrema_update_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "key_frequency_update_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "state_persist_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "comparison_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "validation_ready_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "maintenance_ready_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "incremental_total_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "full_validation_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "validation_speedup_vs_full",
                T.DoubleType(),
                True,
            ),
            T.StructField(
                "end_to_end_speedup_vs_full",
                T.DoubleType(),
                True,
            ),
            T.StructField(
                "initial_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "updated_partition_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "key_overlay_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "change_insert_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "change_delete_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "change_update_preimage_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "change_update_postimage_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "distinct_trip_keys",
                T.LongType(),
                False,
            ),
            T.StructField(
                "duplicate_key_groups",
                T.LongType(),
                False,
            ),
            T.StructField(
                "duplicate_extra_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "maximum_key_multiplicity",
                T.LongType(),
                False,
            ),
            T.StructField(
                "uniqueness_pass",
                T.BooleanType(),
                False,
            ),
            T.StructField("working_path", T.StringType(), False),
            T.StructField(
                "partition_state_path",
                T.StringType(),
                False,
            ),
            T.StructField(
                "key_overlay_path",
                T.StringType(),
                False,
            ),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                args.start_version,
                args.end_version,
                baseline_row_count,
                int(partition_global["R01"]),
                raw_cdf_rows,
                net_changed_row_ids,
                net_image_rows,
                float(cdf_fraction_percent),
                int(affected_partition_count),
                len(all_invalidated_partitions),
                int(recompute_row_count),
                int(affected_key_count),
                len(affected_key_buckets),
                maximum_frequency_fallback_scan,
                exact_match_count,
                comparison_rows,
                float(exact_agreement_rate),
                cdf_execution_mode,
                int(partition_state_input_metrics["num_files"]),
                int(partition_state_input_metrics["size_bytes"]),
                overlay_layout,
                float(cdf_seconds),
                float(additive_seconds),
                float(extrema_seconds),
                float(key_seconds),
                float(state_persist_seconds),
                float(comparison_seconds),
                float(validation_ready_seconds),
                float(maintenance_ready_seconds),
                float(incremental_total_seconds),
                float(full_validation_seconds),
                (
                    float(validation_speedup_vs_full)
                    if validation_speedup_vs_full is not None
                    else None
                ),
                (
                    float(end_to_end_speedup_vs_full)
                    if end_to_end_speedup_vs_full is not None
                    else None
                ),
                required_int(
                    initial_state_summary,
                    "total_state_size_bytes",
                ),
                int(partition_state_metrics["size_bytes"]),
                int(key_overlay_metrics["size_bytes"]),
                change_counts["insert"],
                change_counts["delete"],
                change_counts["update_preimage"],
                change_counts["update_postimage"],
                final_distinct_keys,
                final_duplicate_groups,
                final_duplicate_extra,
                final_max_multiplicity,
                final_uniqueness_pass,
                working_path,
                partition_state_output,
                key_overlay_output,
            )
        ],
        schema=summary_schema,
    )

    # Result-report persistence is deliberately excluded from algorithm time.
    write_csv(
        incremental_rules,
        f"{output_root}/incremental_rules_csv",
    )
    write_csv(
        comparison,
        f"{output_root}/correctness_comparison_csv",
    )
    write_csv(
        planner_df,
        f"{output_root}/planner_decisions_csv",
    )
    write_csv(
        summary_df,
        f"{output_root}/summary_csv",
    )

    (
        incremental_rules.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/incremental_rules_json")
    )
    (
        comparison.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/correctness_comparison_json")
    )
    (
        planner_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/planner_decisions_json")
    )
    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("EXACT INCREMENTAL RULE RESULTS")
    print("=" * 78)

    incremental_rules.select(
        "rule_id",
        "rule_name",
        "metric_display",
        "status",
        "planner_mode",
    ).show(20, truncate=False)

    print("=" * 78)
    print("INCREMENTAL VS FULL-VALIDATION CORRECTNESS")
    print("=" * 78)

    comparison.select(
        "rule_id",
        "incremental_value",
        "full_validation_value",
        "planner_mode",
        "exact_match",
    ).show(20, truncate=False)

    print("=" * 78)
    print("STATEGUARD_INCREMENTAL_VALIDATION_BEGIN")
    print("INCREMENTAL_VALIDATION_STATUS=PASS")
    print(f"START_VERSION={args.start_version}")
    print(f"END_VERSION={args.end_version}")
    print(f"BASELINE_ROWS={baseline_row_count}")
    print(f"FINAL_ROWS={int(partition_global['R01'])}")
    print(f"RAW_CDF_ROWS={raw_cdf_rows}")
    print(f"NET_CHANGED_ROW_IDS={net_changed_row_ids}")
    print(f"NET_IMAGE_ROWS={net_image_rows}")
    print(
        f"CDF_FRACTION_PERCENT={cdf_fraction_percent:.10f}"
    )
    print(
        f"AFFECTED_PARTITION_COUNT={affected_partition_count}"
    )
    print(
        "INVALIDATED_PARTITION_COUNT="
        f"{len(all_invalidated_partitions)}"
    )
    print(
        f"RECOMPUTED_PARTITION_ROWS={recompute_row_count}"
    )
    print(f"AFFECTED_KEY_COUNT={affected_key_count}")
    print(
        "AFFECTED_KEY_BUCKET_COUNT="
        f"{len(affected_key_buckets)}"
    )
    print(f"EXACT_MATCH_COUNT={exact_match_count}")
    print("RULE_COUNT=13")
    print(
        f"EXACT_AGREEMENT_RATE={exact_agreement_rate:.6f}"
    )
    print(f"CDF_EXECUTION_MODE={cdf_execution_mode}")
    print(
        "PARTITION_STATE_INPUT_NUM_FILES="
        f"{partition_state_input_metrics['num_files']}"
    )
    print(
        "PARTITION_STATE_INPUT_SIZE_BYTES="
        f"{partition_state_input_metrics['size_bytes']}"
    )
    print(f"OVERLAY_LAYOUT={overlay_layout}")
    print(f"CDF_COMPACTION_SECONDS={cdf_seconds:.3f}")
    print(f"ADDITIVE_UPDATE_SECONDS={additive_seconds:.3f}")
    print(f"EXTREMA_UPDATE_SECONDS={extrema_seconds:.3f}")
    print(f"KEY_FREQUENCY_UPDATE_SECONDS={key_seconds:.3f}")
    print(f"STATE_PERSIST_SECONDS={state_persist_seconds:.3f}")
    print(f"COMPARISON_SECONDS={comparison_seconds:.3f}")
    print(
        "VALIDATION_READY_SECONDS="
        f"{validation_ready_seconds:.3f}"
    )
    print(
        "MAINTENANCE_READY_SECONDS="
        f"{maintenance_ready_seconds:.3f}"
    )
    print(
        "INCREMENTAL_TOTAL_SECONDS="
        f"{incremental_total_seconds:.3f}"
    )
    print(
        "FULL_VALIDATION_SECONDS="
        f"{full_validation_seconds:.3f}"
    )
    print(
        "VALIDATION_SPEEDUP_VS_FULL="
        f"{validation_speedup_vs_full:.6f}"
        if validation_speedup_vs_full is not None
        else "VALIDATION_SPEEDUP_VS_FULL="
    )
    print(
        "END_TO_END_SPEEDUP_VS_FULL="
        f"{end_to_end_speedup_vs_full:.6f}"
        if end_to_end_speedup_vs_full is not None
        else "END_TO_END_SPEEDUP_VS_FULL="
    )
    print(
        "UPDATED_PARTITION_STATE_SIZE_BYTES="
        f"{partition_state_metrics['size_bytes']}"
    )
    print(
        "KEY_OVERLAY_SIZE_BYTES="
        f"{key_overlay_metrics['size_bytes']}"
    )
    print(
        f"DISTINCT_TRIP_KEYS={final_distinct_keys}"
    )
    print(
        f"DUPLICATE_KEY_GROUPS={final_duplicate_groups}"
    )
    print(
        f"DUPLICATE_EXTRA_ROWS={final_duplicate_extra}"
    )
    print(
        "MAXIMUM_KEY_MULTIPLICITY="
        f"{final_max_multiplicity}"
    )
    print(
        "UNIQUENESS_PASS="
        f"{str(final_uniqueness_pass).lower()}"
    )
    print(
        "CORRECTNESS_PATH="
        f"{output_root}/correctness_comparison_csv"
    )
    print(
        "PLANNER_PATH="
        f"{output_root}/planner_decisions_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_INCREMENTAL_VALIDATION_END")
    print("=" * 78)

    comparison.unpersist()
    key_overlay.unpersist()
    affected_key_deltas.unpersist()
    updated_partition_state.unpersist()
    base_partition_state.unpersist()
    net_images.unpersist()
    if row_bounds is not None:
        row_bounds.unpersist()
    if raw_cdf_unpersist_required:
        raw_cdf.unpersist()

    if all_invalidated_partitions:
        recomputed_partitions.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
