import argparse
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from delta.tables import DeltaTable
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


RULE_DEFINITIONS = [
    ("R01", "Row count", "LONG"),
    ("R02", "Null passenger count", "LONG"),
    ("R03", "Null fare count", "LONG"),
    ("R04", "Invalid fare count", "LONG"),
    ("R05", "Invalid distance count", "LONG"),
    ("R06", "Invalid passenger count", "LONG"),
    ("R07", "Invalid pickup-time count", "LONG"),
    ("R08", "Minimum fare", "DOUBLE"),
    ("R09", "Maximum fare", "DOUBLE"),
    ("R10", "Minimum distance", "DOUBLE"),
    ("R11", "Maximum distance", "DOUBLE"),
    ("R12", "Duplicate key count", "LONG"),
    ("R13", "Uniqueness pass", "BOOLEAN"),
]

PARTITION_STATE_COLUMNS = [
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact full-validation ground truth and transparent "
            "comparison baselines over all 16 StateGuard workload versions."
        )
    )
    parser.add_argument("--research-matrix-root", required=True)
    parser.add_argument("--performance-workload-result", required=True)
    parser.add_argument("--consolidated-partition-state", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--expected-baseline-version", type=int, default=0)
    parser.add_argument("--expected-baseline-rows", type=int, default=67721884)
    parser.add_argument("--min-valid-pickup", default="2024-12-31 00:00:00")
    parser.add_argument("--max-valid-pickup", default="2026-06-01 23:59:59")
    parser.add_argument("--max-passengers", type=int, default=8)
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def latest_version(path: str, spark: SparkSession) -> int:
    return int(DeltaTable.forPath(spark, path).history(1).collect()[0]["version"])


def dataframe_at_version(
    spark: SparkSession,
    path: str,
    version: int,
) -> DataFrame:
    return (
        spark.read.format("delta")
        .option("versionAsOf", version)
        .load(path)
    )


def scalar_partition_aggregation(
    dataframe: DataFrame,
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
) -> DataFrame:
    return dataframe.groupBy("state_partition_id").agg(
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
                    | (F.col("passenger_count") > F.lit(max_passengers))
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


def aggregate_partition_state(partition_state: DataFrame) -> Dict[str, Any]:
    row = partition_state.agg(
        F.coalesce(F.sum("row_count"), F.lit(0)).cast("long").alias("R01"),
        F.coalesce(F.sum("null_passenger_count"), F.lit(0)).cast("long").alias("R02"),
        F.coalesce(F.sum("null_fare_count"), F.lit(0)).cast("long").alias("R03"),
        F.coalesce(F.sum("invalid_fare_count"), F.lit(0)).cast("long").alias("R04"),
        F.coalesce(F.sum("invalid_distance_count"), F.lit(0)).cast("long").alias("R05"),
        F.coalesce(F.sum("invalid_passenger_count"), F.lit(0)).cast("long").alias("R06"),
        F.coalesce(F.sum("invalid_pickup_time_count"), F.lit(0)).cast("long").alias("R07"),
        F.min("minimum_fare").cast("double").alias("R08"),
        F.max("maximum_fare").cast("double").alias("R09"),
        F.min("minimum_distance").cast("double").alias("R10"),
        F.max("maximum_distance").cast("double").alias("R11"),
    ).collect()[0]

    return {f"R{index:02d}": row[f"R{index:02d}"] for index in range(1, 12)}


def duplicate_metrics(dataframe: DataFrame) -> Dict[str, Any]:
    row = (
        dataframe.groupBy("trip_key")
        .count()
        .agg(
            F.count(F.lit(1)).cast("long").alias("distinct_trip_keys"),
            F.coalesce(
                F.sum(F.when(F.col("count") > 1, 1).otherwise(0)),
                F.lit(0),
            ).cast("long").alias("duplicate_key_groups"),
            F.coalesce(
                F.sum(
                    F.when(
                        F.col("count") > 1,
                        F.col("count") - F.lit(1),
                    ).otherwise(F.lit(0))
                ),
                F.lit(0),
            ).cast("long").alias("duplicate_extra_rows"),
            F.coalesce(F.max("count"), F.lit(0)).cast("long").alias("maximum_key_multiplicity"),
        )
        .collect()[0]
    )

    duplicate_extra_rows = int(row["duplicate_extra_rows"])
    return {
        "distinct_trip_keys": int(row["distinct_trip_keys"]),
        "duplicate_key_groups": int(row["duplicate_key_groups"]),
        "duplicate_extra_rows": duplicate_extra_rows,
        "maximum_key_multiplicity": int(row["maximum_key_multiplicity"]),
        "R12": duplicate_extra_rows,
        "R13": duplicate_extra_rows == 0,
    }


def full_validation_once(
    dataframe: DataFrame,
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_start = time.perf_counter()

    scalar_start = time.perf_counter()
    partition_state = scalar_partition_aggregation(
        dataframe,
        min_pickup,
        max_pickup,
        max_passengers,
    )
    metrics = aggregate_partition_state(partition_state)
    scalar_seconds = time.perf_counter() - scalar_start

    duplicate_start = time.perf_counter()
    duplicates = duplicate_metrics(dataframe)
    duplicate_seconds = time.perf_counter() - duplicate_start

    metrics["R12"] = duplicates["R12"]
    metrics["R13"] = duplicates["R13"]

    timing = {
        "cdf_discovery_seconds": 0.0,
        "changed_image_seconds": 0.0,
        "partition_scan_seconds": 0.0,
        "scalar_seconds": float(scalar_seconds),
        "duplicate_seconds": float(duplicate_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
        "affected_partition_count": 64,
        "cdf_row_count": 0,
        "changed_final_row_count": 0,
        **{
            key: duplicates[key]
            for key in [
                "distinct_trip_keys",
                "duplicate_key_groups",
                "duplicate_extra_rows",
                "maximum_key_multiplicity",
            ]
        },
    }
    return metrics, timing


def load_cdf(
    spark: SparkSession,
    path: str,
    end_version: int,
) -> DataFrame:
    return (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", 1)
        .option("endingVersion", end_version)
        .load(path)
    )


def changed_final_images(cdf: DataFrame) -> DataFrame:
    change_priority = (
        F.when(F.col("_change_type") == "delete", 4)
        .when(F.col("_change_type") == "update_postimage", 3)
        .when(F.col("_change_type") == "insert", 3)
        .when(F.col("_change_type") == "update_preimage", 2)
        .otherwise(0)
    )
    window = Window.partitionBy("row_id").orderBy(
        F.col("_commit_version").desc(),
        change_priority.desc(),
    )
    return (
        cdf.withColumn("_sg_last_rank", F.row_number().over(window))
        .filter(F.col("_sg_last_rank") == 1)
        .filter(F.col("_change_type").isin(["insert", "update_postimage"]))
        .drop(
            "_sg_last_rank",
            "_change_type",
            "_commit_version",
            "_commit_timestamp",
        )
    )


def changed_rows_only_once(
    spark: SparkSession,
    path: str,
    end_version: int,
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_start = time.perf_counter()

    cdf_start = time.perf_counter()
    cdf = load_cdf(spark, path, end_version)
    cdf_row_count = cdf.count()
    cdf_seconds = time.perf_counter() - cdf_start

    image_start = time.perf_counter()
    final_images = changed_final_images(cdf)
    changed_final_row_count = final_images.count()
    image_seconds = time.perf_counter() - image_start

    scalar_start = time.perf_counter()
    metrics = aggregate_partition_state(
        scalar_partition_aggregation(
            final_images,
            min_pickup,
            max_pickup,
            max_passengers,
        )
    )
    scalar_seconds = time.perf_counter() - scalar_start

    duplicate_start = time.perf_counter()
    duplicates = duplicate_metrics(final_images)
    duplicate_seconds = time.perf_counter() - duplicate_start

    metrics["R12"] = duplicates["R12"]
    metrics["R13"] = duplicates["R13"]

    affected_partition_count = cdf.select("state_partition_id").distinct().count()

    timing = {
        "cdf_discovery_seconds": float(cdf_seconds),
        "changed_image_seconds": float(image_seconds),
        "partition_scan_seconds": 0.0,
        "scalar_seconds": float(scalar_seconds),
        "duplicate_seconds": float(duplicate_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
        "affected_partition_count": int(affected_partition_count),
        "cdf_row_count": int(cdf_row_count),
        "changed_final_row_count": int(changed_final_row_count),
        **{
            key: duplicates[key]
            for key in [
                "distinct_trip_keys",
                "duplicate_key_groups",
                "duplicate_extra_rows",
                "maximum_key_multiplicity",
            ]
        },
    }
    return metrics, timing


def differential_partition_once(
    spark: SparkSession,
    path: str,
    end_version: int,
    baseline_partition_state: DataFrame,
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_start = time.perf_counter()

    cdf_start = time.perf_counter()
    cdf = load_cdf(spark, path, end_version)
    cdf_row_count = cdf.count()
    affected_partitions = (
        cdf.select(F.col("state_partition_id").cast("int").alias("state_partition_id"))
        .distinct()
    )
    affected_partition_count = affected_partitions.count()
    cdf_seconds = time.perf_counter() - cdf_start

    final_df = dataframe_at_version(spark, path, end_version)

    partition_start = time.perf_counter()
    recomputed = scalar_partition_aggregation(
        final_df.join(
            F.broadcast(affected_partitions),
            on="state_partition_id",
            how="inner",
        ),
        min_pickup,
        max_pickup,
        max_passengers,
    )
    unchanged = baseline_partition_state.join(
        F.broadcast(affected_partitions),
        on="state_partition_id",
        how="left_anti",
    )
    metrics = aggregate_partition_state(unchanged.unionByName(recomputed))
    partition_seconds = time.perf_counter() - partition_start

    duplicate_start = time.perf_counter()
    duplicates = duplicate_metrics(final_df)
    duplicate_seconds = time.perf_counter() - duplicate_start

    metrics["R12"] = duplicates["R12"]
    metrics["R13"] = duplicates["R13"]

    timing = {
        "cdf_discovery_seconds": float(cdf_seconds),
        "changed_image_seconds": 0.0,
        "partition_scan_seconds": float(partition_seconds),
        "scalar_seconds": float(partition_seconds),
        "duplicate_seconds": float(duplicate_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
        "affected_partition_count": int(affected_partition_count),
        "cdf_row_count": int(cdf_row_count),
        "changed_final_row_count": 0,
        **{
            key: duplicates[key]
            for key in [
                "distinct_trip_keys",
                "duplicate_key_groups",
                "duplicate_extra_rows",
                "maximum_key_multiplicity",
            ]
        },
    }
    return metrics, timing


def values_equal(left: Any, right: Any, rule_id: str) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if rule_id in {"R08", "R09", "R10", "R11"}:
        return abs(float(left) - float(right)) <= 1e-9
    if rule_id == "R13":
        return bool(left) == bool(right)
    return int(left) == int(right)


def metric_display(value: Any, rule_id: str) -> str:
    if value is None:
        return "NULL"
    if rule_id == "R13":
        return str(bool(value)).lower()
    if rule_id in {"R08", "R09", "R10", "R11"}:
        return format(float(value), ".12g")
    return str(int(value))


def rule_status(rule_id: str, value: Any) -> str:
    if rule_id in {"R01", "R08", "R09", "R10", "R11"}:
        return "OBSERVED"
    if rule_id == "R13":
        return "PASS" if bool(value) else "FAIL"
    return "PASS" if int(value) == 0 else "FAIL"


def metrics_to_rule_rows(
    workload_id: str,
    family_id: str,
    version: int,
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rule_order, (rule_id, rule_name, metric_type) in enumerate(
        RULE_DEFINITIONS,
        start=1,
    ):
        value = metrics[rule_id]
        rows.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": version,
                "rule_order": rule_order,
                "rule_id": rule_id,
                "rule_name": rule_name,
                "metric_type": metric_type,
                "metric_display": metric_display(value, rule_id),
                "long_value": int(value) if metric_type == "LONG" and value is not None else None,
                "double_value": float(value) if metric_type == "DOUBLE" and value is not None else None,
                "boolean_value": bool(value) if metric_type == "BOOLEAN" and value is not None else None,
                "status": rule_status(rule_id, value),
            }
        )
    return rows


def read_workloads(
    spark: SparkSession,
    research_matrix_root: str,
    performance_workload_result: str,
) -> List[Dict[str, Any]]:
    matrix = (
        spark.read.option("header", "true")
        .csv(f"{research_matrix_root}/mutation_matrix_csv")
        .filter(F.col("workload_class") == "PERFORMANCE_MATRIX")
        .select(
            "workload_id",
            "family_id",
            F.col("target_version").cast("int").alias("target_version"),
            F.col("cumulative_operations").cast("long").alias("cumulative_operations"),
            F.col("expected_rows_after").cast("long").alias("expected_rows_after"),
            F.col("expected_cumulative_cdf_rows").cast("long").alias("expected_cumulative_cdf_rows"),
        )
    )
    manifest = (
        spark.read.option("header", "true")
        .csv(f"{performance_workload_result}/table_manifest_csv")
        .select(
            "family_id",
            "working_path",
            F.col("final_version").cast("int").alias("final_version"),
        )
    )
    rows = (
        matrix.join(manifest, on="family_id", how="inner")
        .orderBy("family_id", "target_version")
        .collect()
    )
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 workload rows, found {len(rows)}.")
    return [row.asDict() for row in rows]


def trial_record(
    workload: Dict[str, Any],
    method: str,
    trial_number: int,
    is_warmup: bool,
    timing: Dict[str, Any],
    exact_match_count: int,
) -> Dict[str, Any]:
    return {
        "workload_id": str(workload["workload_id"]),
        "family_id": str(workload["family_id"]),
        "target_version": int(workload["target_version"]),
        "cumulative_operations": int(workload["cumulative_operations"]),
        "method": method,
        "trial_number": trial_number,
        "is_warmup": is_warmup,
        "cdf_row_count": int(timing["cdf_row_count"]),
        "changed_final_row_count": int(timing["changed_final_row_count"]),
        "affected_partition_count": int(timing["affected_partition_count"]),
        "cdf_discovery_seconds": float(timing["cdf_discovery_seconds"]),
        "changed_image_seconds": float(timing["changed_image_seconds"]),
        "partition_scan_seconds": float(timing["partition_scan_seconds"]),
        "scalar_seconds": float(timing["scalar_seconds"]),
        "duplicate_seconds": float(timing["duplicate_seconds"]),
        "total_seconds": float(timing["total_seconds"]),
        "exact_match_count": exact_match_count,
        "rule_count": 13,
        "exact_agreement_rate": exact_match_count / 13.0,
        "status": "PASS" if method == "CHANGED_ROWS_ONLY" or exact_match_count == 13 else "FAIL",
    }


def median_or_zero(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    args = parse_arguments()
    if args.timed_repeats < 1:
        raise ValueError("--timed-repeats must be at least 1.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")
    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")

    spark = SparkSession.builder.appName("StateGuardGroundTruthAndBaselines").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.state_partitions))
    spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

    research_matrix_root = args.research_matrix_root.rstrip("/")
    performance_result = args.performance_workload_result.rstrip("/")
    partition_state_path = args.consolidated_partition_state.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, partition_state_path):
        raise RuntimeError("Consolidated baseline partition state is missing.")
    if latest_version(partition_state_path, spark) != args.expected_baseline_version:
        raise RuntimeError("Baseline partition-state version mismatch.")

    baseline_partition_state = (
        spark.read.format("delta")
        .load(partition_state_path)
        .select(*PARTITION_STATE_COLUMNS)
    )
    if baseline_partition_state.count() != args.state_partitions:
        raise RuntimeError("Baseline partition state must contain 64 rows.")

    baseline_row_count = int(
        baseline_partition_state.agg(F.sum("row_count").alias("row_count"))
        .collect()[0]["row_count"]
    )
    if baseline_row_count != args.expected_baseline_rows:
        raise RuntimeError("Baseline row count differs from the expected table.")

    workloads = read_workloads(spark, research_matrix_root, performance_result)
    min_pickup = F.lit(args.min_valid_pickup).cast("timestamp_ntz")
    max_pickup = F.lit(args.max_valid_pickup).cast("timestamp_ntz")

    ground_truth_rule_records: List[Dict[str, Any]] = []
    ground_truth_summary_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    trial_records: List[Dict[str, Any]] = []

    print("=" * 78)
    print("STATEGUARD FULL GROUND TRUTH AND BASELINE EXPERIMENTS")
    print("=" * 78)
    print(f"Workload conditions: {len(workloads)}")
    print(f"Warm-up runs per exact method: {args.warmup_runs}")
    print(f"Timed repetitions per exact method: {args.timed_repeats}")
    print("Methods: FULL_VALIDATION, DIFFERENTIAL_PARTITION_WITH_UNIQUENESS_FALLBACK, CHANGED_ROWS_ONLY")
    print("=" * 78)

    for workload_index, workload in enumerate(workloads, start=1):
        workload_id = str(workload["workload_id"])
        family_id = str(workload["family_id"])
        target_version = int(workload["target_version"])
        working_path = str(workload["working_path"])
        final_version = int(workload["final_version"])

        if not DeltaTable.isDeltaTable(spark, working_path):
            raise RuntimeError(f"Missing performance table: {working_path}")
        if final_version < target_version:
            raise RuntimeError(f"{workload_id} requests unavailable version {target_version}.")
        if latest_version(working_path, spark) != final_version:
            raise RuntimeError(f"{workload_id} table manifest version is stale.")

        final_df = dataframe_at_version(spark, working_path, target_version)
        expected_rows = int(workload["expected_rows_after"])
        observed_rows = final_df.count()
        if observed_rows != expected_rows:
            raise RuntimeError(
                f"{workload_id} row mismatch: expected={expected_rows}, actual={observed_rows}"
            )

        print()
        print(f"[{workload_index:02d}/16] {workload_id} rows={observed_rows}")

        full_reference: Optional[Dict[str, Any]] = None
        full_reference_timing: Optional[Dict[str, Any]] = None
        full_timed_seconds: List[float] = []
        total_full_runs = args.warmup_runs + args.timed_repeats

        for run_index in range(total_full_runs):
            is_warmup = run_index < args.warmup_runs
            trial_number = 0 if is_warmup else run_index - args.warmup_runs + 1
            metrics, timing = full_validation_once(
                final_df,
                min_pickup,
                max_pickup,
                args.max_passengers,
            )
            if full_reference is None:
                full_reference = metrics
                full_reference_timing = timing
            else:
                for rule_id, _, _ in RULE_DEFINITIONS:
                    if not values_equal(full_reference[rule_id], metrics[rule_id], rule_id):
                        raise RuntimeError(
                            f"{workload_id} full-validation trial disagreement on {rule_id}."
                        )
            trial_records.append(
                trial_record(
                    workload,
                    "FULL_VALIDATION",
                    trial_number,
                    is_warmup,
                    timing,
                    13,
                )
            )
            if not is_warmup:
                full_timed_seconds.append(float(timing["total_seconds"]))

        if full_reference is None or full_reference_timing is None:
            raise RuntimeError(f"{workload_id} did not produce full ground truth.")

        ground_truth_rule_records.extend(
            metrics_to_rule_rows(
                workload_id,
                family_id,
                target_version,
                full_reference,
            )
        )
        ground_truth_summary_records.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": target_version,
                "cumulative_operations": int(workload["cumulative_operations"]),
                "row_count": int(full_reference["R01"]),
                "rule_count": 13,
                "distinct_trip_keys": int(full_reference_timing["distinct_trip_keys"]),
                "duplicate_key_groups": int(full_reference_timing["duplicate_key_groups"]),
                "duplicate_extra_rows": int(full_reference["R12"]),
                "maximum_key_multiplicity": int(full_reference_timing["maximum_key_multiplicity"]),
                "uniqueness_pass": bool(full_reference["R13"]),
                "timed_repeats": args.timed_repeats,
                "median_full_seconds": median_or_zero(full_timed_seconds),
                "minimum_full_seconds": float(min(full_timed_seconds)),
                "maximum_full_seconds": float(max(full_timed_seconds)),
                "working_path": working_path,
                "status": "PASS",
            }
        )

        differential_timed_seconds: List[float] = []
        differential_reference: Optional[Dict[str, Any]] = None
        total_differential_runs = args.warmup_runs + args.timed_repeats

        for run_index in range(total_differential_runs):
            is_warmup = run_index < args.warmup_runs
            trial_number = 0 if is_warmup else run_index - args.warmup_runs + 1
            metrics, timing = differential_partition_once(
                spark,
                working_path,
                target_version,
                baseline_partition_state,
                min_pickup,
                max_pickup,
                args.max_passengers,
            )
            exact_matches = sum(
                1
                for rule_id, _, _ in RULE_DEFINITIONS
                if values_equal(metrics[rule_id], full_reference[rule_id], rule_id)
            )
            if exact_matches != 13:
                raise RuntimeError(
                    f"{workload_id} differential baseline matched only {exact_matches}/13 rules."
                )
            if differential_reference is None:
                differential_reference = metrics
            trial_records.append(
                trial_record(
                    workload,
                    "DIFFERENTIAL_PARTITION_WITH_UNIQUENESS_FALLBACK",
                    trial_number,
                    is_warmup,
                    timing,
                    exact_matches,
                )
            )
            if not is_warmup:
                differential_timed_seconds.append(float(timing["total_seconds"]))

        changed_metrics, changed_timing = changed_rows_only_once(
            spark,
            working_path,
            target_version,
            min_pickup,
            max_pickup,
            args.max_passengers,
        )

        expected_cdf_rows = int(
            workload["expected_cumulative_cdf_rows"]
        )
        if int(changed_timing["cdf_row_count"]) != expected_cdf_rows:
            raise RuntimeError(
                f"{workload_id} cumulative CDF mismatch: "
                f"expected={expected_cdf_rows}, "
                f"actual={changed_timing['cdf_row_count']}"
            )

        if differential_reference is None:
            raise RuntimeError(f"{workload_id} did not produce a differential result.")

        changed_exact_matches = 0
        for rule_id, rule_name, _ in RULE_DEFINITIONS:
            differential_match = values_equal(
                differential_reference[rule_id],
                full_reference[rule_id],
                rule_id,
            )
            changed_match = values_equal(
                changed_metrics[rule_id],
                full_reference[rule_id],
                rule_id,
            )
            changed_exact_matches += int(changed_match)
            comparison_records.append(
                {
                    "workload_id": workload_id,
                    "family_id": family_id,
                    "target_version": target_version,
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "full_value": metric_display(full_reference[rule_id], rule_id),
                    "differential_value": metric_display(differential_reference[rule_id], rule_id),
                    "changed_rows_value": metric_display(changed_metrics[rule_id], rule_id),
                    "differential_exact_match": differential_match,
                    "changed_rows_exact_match": changed_match,
                }
            )

        trial_records.append(
            trial_record(
                workload,
                "CHANGED_ROWS_ONLY",
                1,
                False,
                changed_timing,
                changed_exact_matches,
            )
        )

        print(f"  FULL median={median_or_zero(full_timed_seconds):.3f}s")
        print(
            "  DIFFERENTIAL median="
            f"{median_or_zero(differential_timed_seconds):.3f}s agreement=13/13"
        )
        print(
            "  CHANGED_ROWS_ONLY agreement="
            f"{changed_exact_matches}/13 time={changed_timing['total_seconds']:.3f}s"
        )

    rule_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("rule_order", T.IntegerType(), False),
        T.StructField("rule_id", T.StringType(), False),
        T.StructField("rule_name", T.StringType(), False),
        T.StructField("metric_type", T.StringType(), False),
        T.StructField("metric_display", T.StringType(), False),
        T.StructField("long_value", T.LongType(), True),
        T.StructField("double_value", T.DoubleType(), True),
        T.StructField("boolean_value", T.BooleanType(), True),
        T.StructField("status", T.StringType(), False),
    ])

    ground_truth_summary_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("row_count", T.LongType(), False),
        T.StructField("rule_count", T.LongType(), False),
        T.StructField("distinct_trip_keys", T.LongType(), False),
        T.StructField("duplicate_key_groups", T.LongType(), False),
        T.StructField("duplicate_extra_rows", T.LongType(), False),
        T.StructField("maximum_key_multiplicity", T.LongType(), False),
        T.StructField("uniqueness_pass", T.BooleanType(), False),
        T.StructField("timed_repeats", T.IntegerType(), False),
        T.StructField("median_full_seconds", T.DoubleType(), False),
        T.StructField("minimum_full_seconds", T.DoubleType(), False),
        T.StructField("maximum_full_seconds", T.DoubleType(), False),
        T.StructField("working_path", T.StringType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    comparison_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("rule_id", T.StringType(), False),
        T.StructField("rule_name", T.StringType(), False),
        T.StructField("full_value", T.StringType(), False),
        T.StructField("differential_value", T.StringType(), False),
        T.StructField("changed_rows_value", T.StringType(), False),
        T.StructField("differential_exact_match", T.BooleanType(), False),
        T.StructField("changed_rows_exact_match", T.BooleanType(), False),
    ])

    trial_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("method", T.StringType(), False),
        T.StructField("trial_number", T.IntegerType(), False),
        T.StructField("is_warmup", T.BooleanType(), False),
        T.StructField("cdf_row_count", T.LongType(), False),
        T.StructField("changed_final_row_count", T.LongType(), False),
        T.StructField("affected_partition_count", T.LongType(), False),
        T.StructField("cdf_discovery_seconds", T.DoubleType(), False),
        T.StructField("changed_image_seconds", T.DoubleType(), False),
        T.StructField("partition_scan_seconds", T.DoubleType(), False),
        T.StructField("scalar_seconds", T.DoubleType(), False),
        T.StructField("duplicate_seconds", T.DoubleType(), False),
        T.StructField("total_seconds", T.DoubleType(), False),
        T.StructField("exact_match_count", T.IntegerType(), False),
        T.StructField("rule_count", T.IntegerType(), False),
        T.StructField("exact_agreement_rate", T.DoubleType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    ground_truth_rules_df = spark.createDataFrame(
        ground_truth_rule_records,
        schema=rule_schema,
    ).orderBy("family_id", "target_version", "rule_order")

    ground_truth_summary_df = spark.createDataFrame(
        ground_truth_summary_records,
        schema=ground_truth_summary_schema,
    ).orderBy("family_id", "target_version")

    comparison_df = spark.createDataFrame(
        comparison_records,
        schema=comparison_schema,
    ).orderBy("family_id", "target_version", "rule_id")

    trials_df = spark.createDataFrame(
        trial_records,
        schema=trial_schema,
    ).orderBy("family_id", "target_version", "method", "is_warmup", "trial_number")

    measured_trials = trials_df.filter(~F.col("is_warmup"))
    method_summary_df = (
        measured_trials.groupBy(
            "workload_id",
            "family_id",
            "target_version",
            "cumulative_operations",
            "method",
        )
        .agg(
            F.count(F.lit(1)).cast("long").alias("trial_count"),
            F.expr("percentile_approx(total_seconds, 0.5, 10000)").cast("double").alias("median_total_seconds"),
            F.min("total_seconds").cast("double").alias("minimum_total_seconds"),
            F.max("total_seconds").cast("double").alias("maximum_total_seconds"),
            F.avg("affected_partition_count").cast("double").alias("mean_affected_partition_count"),
            F.max("exact_match_count").cast("long").alias("exact_match_count"),
            F.max("rule_count").cast("long").alias("rule_count"),
            F.max("exact_agreement_rate").cast("double").alias("exact_agreement_rate"),
        )
        .orderBy("family_id", "target_version", "method")
    )

    differential_mismatches = comparison_df.filter(~F.col("differential_exact_match")).count()
    changed_rows_mismatches = comparison_df.filter(~F.col("changed_rows_exact_match")).count()
    full_trial_failures = measured_trials.filter(
        (F.col("method") == "FULL_VALIDATION")
        & (F.col("exact_match_count") != 13)
    ).count()
    differential_trial_failures = measured_trials.filter(
        F.col("method").startswith("DIFFERENTIAL_PARTITION")
        & (F.col("exact_match_count") != 13)
    ).count()

    summary_schema = T.StructType([
        T.StructField("status", T.StringType(), False),
        T.StructField("workload_condition_count", T.LongType(), False),
        T.StructField("ground_truth_rule_rows", T.LongType(), False),
        T.StructField("timed_repeats", T.LongType(), False),
        T.StructField("warmup_runs", T.LongType(), False),
        T.StructField("full_validation_trial_failures", T.LongType(), False),
        T.StructField("differential_trial_failures", T.LongType(), False),
        T.StructField("differential_rule_mismatches", T.LongType(), False),
        T.StructField("changed_rows_rule_mismatches", T.LongType(), False),
        T.StructField("exact_differential_agreement_rate", T.DoubleType(), False),
        T.StructField("changed_rows_agreement_rate", T.DoubleType(), False),
        T.StructField("output_root", T.StringType(), False),
    ])

    total_rule_comparisons = 16 * 13
    summary_df = spark.createDataFrame([
        (
            "PASS",
            16,
            len(ground_truth_rule_records),
            args.timed_repeats,
            args.warmup_runs,
            full_trial_failures,
            differential_trial_failures,
            differential_mismatches,
            changed_rows_mismatches,
            (total_rule_comparisons - differential_mismatches) / total_rule_comparisons,
            (total_rule_comparisons - changed_rows_mismatches) / total_rule_comparisons,
            output_root,
        )
    ], schema=summary_schema)

    write_csv(ground_truth_rules_df, f"{output_root}/ground_truth_rules_csv")
    write_csv(ground_truth_summary_df, f"{output_root}/ground_truth_summary_csv")
    write_csv(comparison_df, f"{output_root}/rule_comparison_csv")
    write_csv(trials_df, f"{output_root}/trial_results_csv")
    write_csv(method_summary_df, f"{output_root}/method_summary_csv")
    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        ground_truth_rules_df.repartition(4)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{output_root}/ground_truth_rules_delta")
    )
    (
        ground_truth_summary_df.coalesce(1)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{output_root}/ground_truth_summary_delta")
    )
    (
        trials_df.repartition(4)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{output_root}/trial_results_delta")
    )

    print()
    print("=" * 78)
    print("BASELINE METHOD SUMMARY")
    print("=" * 78)
    method_summary_df.select(
        "workload_id",
        "method",
        "trial_count",
        "median_total_seconds",
        "mean_affected_partition_count",
        "exact_match_count",
        "rule_count",
    ).show(60, truncate=False)

    print("=" * 78)
    print("STATEGUARD_BASELINE_EXPERIMENTS_BEGIN")
    print("BASELINE_EXPERIMENT_STATUS=PASS")
    print("WORKLOAD_CONDITION_COUNT=16")
    print(f"GROUND_TRUTH_RULE_ROWS={len(ground_truth_rule_records)}")
    print(f"TIMED_REPEATS={args.timed_repeats}")
    print(f"WARMUP_RUNS={args.warmup_runs}")
    print(f"FULL_VALIDATION_TRIAL_FAILURES={full_trial_failures}")
    print(f"DIFFERENTIAL_TRIAL_FAILURES={differential_trial_failures}")
    print(f"DIFFERENTIAL_RULE_MISMATCHES={differential_mismatches}")
    print(f"CHANGED_ROWS_RULE_MISMATCHES={changed_rows_mismatches}")
    print(
        "EXACT_DIFFERENTIAL_AGREEMENT_RATE="
        f"{(total_rule_comparisons - differential_mismatches) / total_rule_comparisons:.6f}"
    )
    print(
        "CHANGED_ROWS_AGREEMENT_RATE="
        f"{(total_rule_comparisons - changed_rows_mismatches) / total_rule_comparisons:.6f}"
    )
    print(f"GROUND_TRUTH_RULES_PATH={output_root}/ground_truth_rules_csv")
    print(f"GROUND_TRUTH_SUMMARY_PATH={output_root}/ground_truth_summary_csv")
    print(f"TRIAL_RESULTS_PATH={output_root}/trial_results_csv")
    print(f"METHOD_SUMMARY_PATH={output_root}/method_summary_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_BASELINE_EXPERIMENTS_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
