import argparse
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Row, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


STATE_RULE_IDS = [f"R{index:02d}" for index in range(1, 12)]
ALL_RULE_IDS = [f"R{index:02d}" for index in range(1, 14)]

EXTREMA_CONFIGS = [
    {
        "rule_id": "R08",
        "state_column": "minimum_fare",
        "source_column": "fare_amount",
        "direction": "MIN",
    },
    {
        "rule_id": "R09",
        "state_column": "maximum_fare",
        "source_column": "fare_amount",
        "direction": "MAX",
    },
    {
        "rule_id": "R10",
        "state_column": "minimum_distance",
        "source_column": "trip_distance",
        "direction": "MIN",
    },
    {
        "rule_id": "R11",
        "state_column": "maximum_distance",
        "source_column": "trip_distance",
        "direction": "MAX",
    },
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
            "Run the final all-rule StateGuard experiment on all 16 mutable Delta "
            "workloads. S01-S11 use exact CDF-based state maintenance; R12-R13 "
            "use an explicit exact full-snapshot uniqueness fallback when the "
            "2.36 GiB S12 state is outside budget. Every condition is checked "
            "against independent R01-R13 ground truth. The workload sizes selected by "
            "--timing-operation-counts receive three measured repetitions for "
            "fair comparisons with full Spark and official Deequ."
        )
    )
    parser.add_argument("--research-matrix-root", required=True)
    parser.add_argument("--performance-workload-result", required=True)
    parser.add_argument("--consolidated-partition-state", required=True)
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--deequ-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument(
        "--timing-operation-counts",
        default="1000,100000",
        help=(
            "Comma-separated cumulative mutation counts selected for repeated "
            "timing. Every one of the 16 conditions still receives one exact run."
        ),
    )
    parser.add_argument("--expected-baseline-version", type=int, default=0)
    parser.add_argument("--expected-baseline-rows", type=int, default=67721884)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--logical-portfolio-bytes", type=int, default=318331)
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
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def latest_version(path: str, spark: SparkSession) -> int:
    return int(
        DeltaTable.forPath(spark, path)
        .history(1)
        .select("version")
        .collect()[0]["version"]
    )


def delta_metrics(spark: SparkSession, path: str) -> Dict[str, int]:
    detail = DeltaTable.forPath(spark, path).detail().collect()[0]
    return {
        "num_files": int(detail["numFiles"]),
        "size_bytes": int(detail["sizeInBytes"]),
    }


def parse_int_set(text: str, argument_name: str) -> Set[int]:
    try:
        result = {
            int(value.strip())
            for value in text.split(",")
            if value.strip()
        }
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} must contain only integers."
        ) from exc

    if not result:
        raise ValueError(f"{argument_name} cannot be empty.")
    if any(value <= 0 for value in result):
        raise ValueError(
            f"{argument_name} values must be positive."
        )
    return result


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
            F.col("cumulative_operations")
            .cast("long")
            .alias("cumulative_operations"),
            F.col("expected_rows_after")
            .cast("long")
            .alias("expected_rows_after"),
            F.col("expected_cumulative_cdf_rows")
            .cast("long")
            .alias("expected_cumulative_cdf_rows"),
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
        raise RuntimeError(
            f"Expected 16 workload conditions; found {len(rows)}."
        )

    return [row.asDict(recursive=True) for row in rows]


def load_ground_truth(
    spark: SparkSession,
    ground_truth_root: str,
) -> Dict[str, Dict[str, Any]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{ground_truth_root}/ground_truth_rules_csv")
        .filter(F.col("rule_id").isin(ALL_RULE_IDS))
        .select(
            "workload_id",
            "rule_id",
            "metric_type",
            F.col("long_value").cast("long").alias("long_value"),
            F.col("double_value").cast("double").alias("double_value"),
            F.col("boolean_value").cast("boolean").alias("boolean_value"),
        )
        .collect()
    )

    expected_rows = 16 * len(ALL_RULE_IDS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} all-rule ground-truth rows; "
            f"found {len(rows)}."
        )

    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        workload_id = str(row["workload_id"])
        rule_id = str(row["rule_id"])
        metric_type = str(row["metric_type"])

        if metric_type == "LONG":
            value: Any = int(row["long_value"])
        elif metric_type == "DOUBLE":
            value = float(row["double_value"])
        elif metric_type == "BOOLEAN":
            value = bool(row["boolean_value"])
        else:
            raise RuntimeError(
                f"Unsupported ground-truth metric type: {metric_type}"
            )

        result.setdefault(workload_id, {})[rule_id] = value

    for workload_id, metrics in result.items():
        if set(metrics) != set(ALL_RULE_IDS):
            raise RuntimeError(
                f"{workload_id} does not contain R01-R13 ground truth."
            )

    return result

def load_baseline_medians(
    spark: SparkSession,
    baseline_root: str,
) -> Dict[str, Dict[str, float]]:
    trials = (
        spark.read.option("header", "true")
        .csv(f"{baseline_root}/trial_results_csv")
        .filter(
            (F.col("method") == "FULL_VALIDATION")
            & (F.lower(F.col("is_warmup")) == F.lit("false"))
        )
        .select(
            "workload_id",
            F.col("scalar_seconds").cast("double").alias("scalar_seconds"),
            F.col("duplicate_seconds")
            .cast("double")
            .alias("duplicate_seconds"),
            F.col("total_seconds").cast("double").alias("total_seconds"),
        )
    )

    rows = (
        trials.groupBy("workload_id")
        .agg(
            F.count(F.lit(1)).cast("long").alias("trial_count"),
            F.expr(
                "percentile_approx(scalar_seconds, 0.5, 10000)"
            ).cast("double").alias("median_scalar_seconds"),
            F.expr(
                "percentile_approx(duplicate_seconds, 0.5, 10000)"
            ).cast("double").alias("median_duplicate_seconds"),
            F.expr(
                "percentile_approx(total_seconds, 0.5, 10000)"
            ).cast("double").alias("median_total_seconds"),
        )
        .collect()
    )

    if len(rows) != 16:
        raise RuntimeError(
            f"Expected baseline medians for 16 workloads; found {len(rows)}."
        )

    result: Dict[str, Dict[str, float]] = {}

    for row in rows:
        if int(row["trial_count"]) != 3:
            raise RuntimeError(
                f"{row['workload_id']} does not have three baseline trials."
            )

        result[str(row["workload_id"])] = {
            "median_scalar_seconds": float(
                row["median_scalar_seconds"]
            ),
            "median_duplicate_seconds": float(
                row["median_duplicate_seconds"]
            ),
            "median_total_seconds": float(
                row["median_total_seconds"]
            ),
        }

    return result


def load_deequ_medians(
    spark: SparkSession,
    deequ_root: str,
) -> Dict[str, float]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{deequ_root}/method_summary_csv")
        .filter(F.col("median_total_seconds").isNotNull())
        .select(
            "workload_id",
            F.col("median_total_seconds")
            .cast("double")
            .alias("median_total_seconds"),
        )
        .collect()
    )

    if len(rows) != 8:
        raise RuntimeError(
            f"Expected eight timed Deequ endpoint rows; found {len(rows)}."
        )

    return {
        str(row["workload_id"]): float(row["median_total_seconds"])
        for row in rows
    }


def load_baseline_partition_state(
    spark: SparkSession,
    path: str,
    expected_version: int,
    expected_partitions: int,
    expected_rows: int,
) -> DataFrame:
    if not DeltaTable.isDeltaTable(spark, path):
        raise RuntimeError(
            f"Missing consolidated partition state: {path}"
        )

    version = latest_version(path, spark)
    if version != expected_version:
        raise RuntimeError(
            "Consolidated state version mismatch: "
            f"expected={expected_version}, actual={version}"
        )

    dataframe = (
        spark.read.format("delta")
        .load(path)
        .select(
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
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    partition_count = dataframe.count()
    if partition_count != expected_partitions:
        raise RuntimeError(
            f"Expected {expected_partitions} state rows; "
            f"found {partition_count}."
        )

    row_count = int(
        dataframe.agg(F.sum("row_count").alias("row_count"))
        .collect()[0]["row_count"]
    )
    if row_count != expected_rows:
        raise RuntimeError(
            "Baseline partition state row count mismatch: "
            f"expected={expected_rows}, actual={row_count}"
        )

    return dataframe


def null_safe_min(left: Column, right: Column) -> Column:
    return (
        F.when(left.isNull(), right)
        .when(right.isNull(), left)
        .otherwise(F.least(left, right))
    )


def null_safe_max(left: Column, right: Column) -> Column:
    return (
        F.when(left.isNull(), right)
        .when(right.isNull(), left)
        .otherwise(F.greatest(left, right))
    )


def compact_net_cdf(
    spark: SparkSession,
    working_path: str,
    target_version: int,
) -> Tuple[DataFrame, Dict[str, int], int]:
    raw = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", 1)
        .option("endingVersion", target_version)
        .load(working_path)
        .select(
            "row_id",
            "trip_key",
            "state_partition_id",
            "passenger_count",
            "fare_amount",
            "trip_distance",
            "tpep_pickup_datetime",
            "_change_type",
            "_commit_version",
        )
        .withColumn(
            "_type_order",
            F.when(
                F.col("_change_type").isin(
                    "delete",
                    "update_preimage",
                ),
                F.lit(0),
            ).otherwise(F.lit(1)),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    change_count_rows = (
        raw.groupBy("_change_type")
        .count()
        .collect()
    )
    change_counts = {
        "insert": 0,
        "delete": 0,
        "update_preimage": 0,
        "update_postimage": 0,
    }
    for row in change_count_rows:
        change_counts[str(row["_change_type"])] = int(row["count"])

    raw_count = int(sum(change_counts.values()))

    ascending = Window.partitionBy("row_id").orderBy(
        F.col("_commit_version").asc(),
        F.col("_type_order").asc(),
    )
    descending = Window.partitionBy("row_id").orderBy(
        F.col("_commit_version").desc(),
        F.col("_type_order").desc(),
    )

    first_images = (
        raw.withColumn("_rn", F.row_number().over(ascending))
        .filter(F.col("_rn") == 1)
        .filter(
            F.col("_change_type").isin(
                "delete",
                "update_preimage",
            )
        )
        .drop("_rn", "_type_order")
        .withColumn("_net_role", F.lit("BASE_IMAGE"))
        .withColumn("_sign", F.lit(-1).cast("long"))
    )

    final_images = (
        raw.withColumn("_rn", F.row_number().over(descending))
        .filter(F.col("_rn") == 1)
        .filter(
            F.col("_change_type").isin(
                "insert",
                "update_postimage",
            )
        )
        .drop("_rn", "_type_order")
        .withColumn("_net_role", F.lit("FINAL_IMAGE"))
        .withColumn("_sign", F.lit(1).cast("long"))
    )

    net_images = (
        first_images.unionByName(final_images)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    net_image_count = net_images.count()
    raw.unpersist()

    return net_images, change_counts, net_image_count


def values_equal(
    observed: Any,
    expected: Any,
    rule_id: str,
    tolerance: float,
) -> bool:
    if rule_id in {"R08", "R09", "R10", "R11"}:
        return abs(float(observed) - float(expected)) <= tolerance
    if rule_id == "R13":
        return bool(observed) == bool(expected)
    return int(observed) == int(expected)


def run_compact_state_once(
    spark: SparkSession,
    workload: Dict[str, Any],
    baseline_partition_state: DataFrame,
    ground_truth: Dict[str, Any],
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
    double_tolerance: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    workload_id = str(workload["workload_id"])
    working_path = str(workload["working_path"])
    target_version = int(workload["target_version"])
    expected_cdf_rows = int(workload["expected_cumulative_cdf_rows"])
    expected_rows = int(workload["expected_rows_after"])

    algorithm_start = time.perf_counter()

    cdf_start = time.perf_counter()
    net_images, change_counts, net_image_count = compact_net_cdf(
        spark,
        working_path,
        target_version,
    )
    raw_cdf_rows = int(sum(change_counts.values()))
    if raw_cdf_rows != expected_cdf_rows:
        raise RuntimeError(
            f"{workload_id} CDF row mismatch: "
            f"expected={expected_cdf_rows}, actual={raw_cdf_rows}"
        )
    cdf_seconds = time.perf_counter() - cdf_start

    state_start = time.perf_counter()

    partition_deltas = (
        net_images.groupBy("state_partition_id")
        .agg(
            F.sum("_sign").cast("long").alias("row_count_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("passenger_count").isNull(),
                    1,
                ).otherwise(0)
            ).cast("long").alias("null_passenger_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("fare_amount").isNull(),
                    1,
                ).otherwise(0)
            ).cast("long").alias("null_fare_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("fare_amount").isNotNull()
                    & (F.col("fare_amount") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_fare_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("trip_distance").isNotNull()
                    & (F.col("trip_distance") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_distance_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("passenger_count").isNotNull()
                    & (
                        (F.col("passenger_count") < 1)
                        | (
                            F.col("passenger_count")
                            > F.lit(max_passengers)
                        )
                    ),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_passenger_delta"),
            F.sum(
                F.col("_sign")
                * F.when(
                    F.col("tpep_pickup_datetime").isNull()
                    | (F.col("tpep_pickup_datetime") < min_pickup)
                    | (F.col("tpep_pickup_datetime") > max_pickup),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_pickup_delta"),
            F.min(
                F.when(
                    (F.col("_sign") == 1)
                    & F.col("fare_amount").isNotNull(),
                    F.col("fare_amount"),
                )
            ).cast("double").alias("positive_minimum_fare"),
            F.max(
                F.when(
                    (F.col("_sign") == 1)
                    & F.col("fare_amount").isNotNull(),
                    F.col("fare_amount"),
                )
            ).cast("double").alias("positive_maximum_fare"),
            F.min(
                F.when(
                    (F.col("_sign") == 1)
                    & F.col("trip_distance").isNotNull(),
                    F.col("trip_distance"),
                )
            ).cast("double").alias("positive_minimum_distance"),
            F.max(
                F.when(
                    (F.col("_sign") == 1)
                    & F.col("trip_distance").isNotNull(),
                    F.col("trip_distance"),
                )
            ).cast("double").alias("positive_maximum_distance"),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    affected_partition_count = partition_deltas.count()

    invalidated_by_metric: Dict[str, Set[int]] = {}

    negative_images = net_images.filter(F.col("_sign") == -1)

    for config in EXTREMA_CONFIGS:
        state_column = str(config["state_column"])
        source_column = str(config["source_column"])

        invalidated_rows = (
            negative_images.filter(F.col(source_column).isNotNull())
            .select(
                F.col("state_partition_id").cast("int"),
                F.col(source_column)
                .cast("double")
                .alias("removed_value"),
            )
            .join(
                baseline_partition_state.select(
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

    all_invalidated_partitions = sorted(
        set().union(*invalidated_by_metric.values())
        if invalidated_by_metric
        else set()
    )

    recomputed_partitions: Optional[DataFrame] = None
    recomputed_row_count = 0

    if all_invalidated_partitions:
        recomputed_partitions = (
            spark.read.format("delta")
            .option("versionAsOf", target_version)
            .load(working_path)
            .filter(
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
        recomputed_row_count = sum(
            int(row["recomputed_row_count"])
            for row in recomputed_rows
        )

        if len(recomputed_rows) != len(all_invalidated_partitions):
            raise RuntimeError(
                f"{workload_id} lost an invalidated partition."
            )

    updated = (
        baseline_partition_state.join(
            partition_deltas,
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
        .withColumn(
            "minimum_fare",
            null_safe_min(
                F.col("minimum_fare"),
                F.col("positive_minimum_fare"),
            ),
        )
        .withColumn(
            "maximum_fare",
            null_safe_max(
                F.col("maximum_fare"),
                F.col("positive_maximum_fare"),
            ),
        )
        .withColumn(
            "minimum_distance",
            null_safe_min(
                F.col("minimum_distance"),
                F.col("positive_minimum_distance"),
            ),
        )
        .withColumn(
            "maximum_distance",
            null_safe_max(
                F.col("maximum_distance"),
                F.col("positive_maximum_distance"),
            ),
        )
    )

    if recomputed_partitions is not None:
        updated = updated.join(
            recomputed_partitions,
            on="state_partition_id",
            how="left",
        )

        for config in EXTREMA_CONFIGS:
            state_column = str(config["state_column"])
            invalidated = sorted(
                invalidated_by_metric[state_column]
            )
            recomputed_column = f"recomputed_{state_column}"

            if invalidated:
                updated = updated.withColumn(
                    state_column,
                    F.when(
                        F.col("state_partition_id").isin(invalidated),
                        F.col(recomputed_column),
                    ).otherwise(F.col(state_column)),
                )

    global_row = (
        updated.agg(
            F.sum("row_count").cast("long").alias("R01"),
            F.sum("null_passenger_count")
            .cast("long")
            .alias("R02"),
            F.sum("null_fare_count").cast("long").alias("R03"),
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
            F.min("minimum_fare").cast("double").alias("R08"),
            F.max("maximum_fare").cast("double").alias("R09"),
            F.min("minimum_distance")
            .cast("double")
            .alias("R10"),
            F.max("maximum_distance")
            .cast("double")
            .alias("R11"),
            F.min("row_count").cast("long").alias("minimum_partition_rows"),
            F.min("null_passenger_count")
            .cast("long")
            .alias("minimum_null_passenger"),
            F.min("null_fare_count")
            .cast("long")
            .alias("minimum_null_fare"),
            F.min("invalid_fare_count")
            .cast("long")
            .alias("minimum_invalid_fare"),
            F.min("invalid_distance_count")
            .cast("long")
            .alias("minimum_invalid_distance"),
            F.min("invalid_passenger_count")
            .cast("long")
            .alias("minimum_invalid_passenger"),
            F.min("invalid_pickup_time_count")
            .cast("long")
            .alias("minimum_invalid_pickup"),
        )
        .collect()[0]
    )

    negative_state_fields = [
        "minimum_partition_rows",
        "minimum_null_passenger",
        "minimum_null_fare",
        "minimum_invalid_fare",
        "minimum_invalid_distance",
        "minimum_invalid_passenger",
        "minimum_invalid_pickup",
    ]
    if any(int(global_row[field]) < 0 for field in negative_state_fields):
        raise RuntimeError(
            f"{workload_id} produced a negative compact state."
        )

    metrics = {
        rule_id: global_row[rule_id]
        for rule_id in STATE_RULE_IDS
    }

    if int(metrics["R01"]) != expected_rows:
        raise RuntimeError(
            f"{workload_id} row-count mismatch: "
            f"expected={expected_rows}, actual={metrics['R01']}"
        )

    state_seconds = time.perf_counter() - state_start
    validation_ready_seconds = time.perf_counter() - algorithm_start

    comparison_records: List[Dict[str, Any]] = []
    mismatch_rules: List[str] = []

    for rule_id in STATE_RULE_IDS:
        observed = metrics[rule_id]
        expected = ground_truth[rule_id]
        exact = values_equal(
            observed,
            expected,
            rule_id,
            double_tolerance,
        )
        if not exact:
            mismatch_rules.append(rule_id)

        planner_mode = "STATE_ASSISTED"
        if rule_id in {"R08", "R09", "R10", "R11"}:
            state_column = {
                "R08": "minimum_fare",
                "R09": "maximum_fare",
                "R10": "minimum_distance",
                "R11": "maximum_distance",
            }[rule_id]
            if invalidated_by_metric[state_column]:
                planner_mode = "AFFECTED_PARTITION_RECOMPUTE"

        comparison_records.append(
            {
                "rule_id": rule_id,
                "incremental_value": str(observed),
                "ground_truth_value": str(expected),
                "planner_mode": planner_mode,
                "exact_match": exact,
            }
        )

    if mismatch_rules:
        raise RuntimeError(
            f"{workload_id} compact StateGuard mismatches: "
            f"{mismatch_rules}"
        )

    diagnostics = {
        "raw_cdf_rows": raw_cdf_rows,
        "net_image_rows": net_image_count,
        "affected_partition_count": affected_partition_count,
        "invalidated_partition_count": len(
            all_invalidated_partitions
        ),
        "recomputed_partition_rows": recomputed_row_count,
        "cdf_seconds": float(cdf_seconds),
        "state_update_seconds": float(state_seconds),
        "validation_ready_seconds": float(
            validation_ready_seconds
        ),
        "insert_cdf_rows": int(change_counts["insert"]),
        "delete_cdf_rows": int(change_counts["delete"]),
        "update_preimage_cdf_rows": int(
            change_counts["update_preimage"]
        ),
        "update_postimage_cdf_rows": int(
            change_counts["update_postimage"]
        ),
    }

    partition_deltas.unpersist()
    net_images.unpersist()
    if recomputed_partitions is not None:
        recomputed_partitions.unpersist()

    return metrics, diagnostics, comparison_records




def exact_uniqueness_fallback(
    spark: SparkSession,
    working_path: str,
    target_version: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start = time.perf_counter()

    key_counts = (
        spark.read.format("delta")
        .option("versionAsOf", target_version)
        .load(working_path)
        .select("trip_key")
        .groupBy("trip_key")
        .count()
    )

    row = (
        key_counts.agg(
            F.count(F.lit(1))
            .cast("long")
            .alias("distinct_trip_keys"),
            F.sum(
                F.when(
                    F.col("count") > 1,
                    F.col("count") - F.lit(1),
                ).otherwise(F.lit(0))
            )
            .cast("long")
            .alias("duplicate_extra_rows"),
            F.sum(
                F.when(
                    F.col("count") > 1,
                    F.lit(1),
                ).otherwise(F.lit(0))
            )
            .cast("long")
            .alias("duplicate_key_groups"),
            F.max("count")
            .cast("long")
            .alias("maximum_key_multiplicity"),
        )
        .collect()[0]
    )

    duplicate_extra_rows = int(row["duplicate_extra_rows"] or 0)

    metrics = {
        "R12": duplicate_extra_rows,
        "R13": duplicate_extra_rows == 0,
    }
    diagnostics = {
        "uniqueness_fallback_seconds": float(
            time.perf_counter() - start
        ),
        "distinct_trip_keys": int(row["distinct_trip_keys"]),
        "duplicate_key_groups": int(
            row["duplicate_key_groups"] or 0
        ),
        "duplicate_extra_rows": duplicate_extra_rows,
        "maximum_key_multiplicity": int(
            row["maximum_key_multiplicity"] or 0
        ),
    }
    return metrics, diagnostics


def run_all_rule_state_guard_once(
    spark: SparkSession,
    workload: Dict[str, Any],
    baseline_partition_state: DataFrame,
    ground_truth: Dict[str, Any],
    min_pickup: Column,
    max_pickup: Column,
    max_passengers: int,
    double_tolerance: float,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    total_start = time.perf_counter()

    state_ground_truth = {
        rule_id: ground_truth[rule_id]
        for rule_id in STATE_RULE_IDS
    }

    (
        state_metrics,
        state_diagnostics,
        state_comparisons,
    ) = run_compact_state_once(
        spark,
        workload,
        baseline_partition_state,
        state_ground_truth,
        min_pickup,
        max_pickup,
        max_passengers,
        double_tolerance,
    )

    uniqueness_metrics, uniqueness_diagnostics = (
        exact_uniqueness_fallback(
            spark,
            str(workload["working_path"]),
            int(workload["target_version"]),
        )
    )

    metrics = {
        **state_metrics,
        **uniqueness_metrics,
    }

    comparison_records = list(state_comparisons)
    mismatch_rules: List[str] = []

    for rule_id in ["R12", "R13"]:
        exact = values_equal(
            metrics[rule_id],
            ground_truth[rule_id],
            rule_id,
            double_tolerance,
        )
        if not exact:
            mismatch_rules.append(rule_id)

        comparison_records.append(
            {
                "rule_id": rule_id,
                "incremental_value": str(metrics[rule_id]),
                "ground_truth_value": str(
                    ground_truth[rule_id]
                ),
                "planner_mode": "EXACT_FULL_SCAN_FALLBACK",
                "exact_match": exact,
            }
        )

    if mismatch_rules:
        raise RuntimeError(
            f"{workload['workload_id']} uniqueness fallback "
            f"mismatches: {mismatch_rules}"
        )

    exact_total_seconds = float(
        time.perf_counter() - total_start
    )
    compact_seconds = float(
        state_diagnostics["validation_ready_seconds"]
    )
    uniqueness_seconds = float(
        uniqueness_diagnostics[
            "uniqueness_fallback_seconds"
        ]
    )

    diagnostics = {
        **state_diagnostics,
        **uniqueness_diagnostics,
        "compact_validation_seconds": compact_seconds,
        "exact_total_seconds": exact_total_seconds,
        "uniqueness_fraction_of_exact_total": (
            uniqueness_seconds / exact_total_seconds
            if exact_total_seconds > 0
            else 0.0
        ),
    }

    return metrics, diagnostics, comparison_records


def main() -> None:
    args = parse_arguments()

    if args.timed_repeats < 1:
        raise ValueError("--timed-repeats must be at least 1.")
    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")
    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")
    if args.double_tolerance < 0:
        raise ValueError("--double-tolerance cannot be negative.")
    if args.logical_portfolio_bytes <= 0:
        raise ValueError(
            "--logical-portfolio-bytes must be positive."
        )

    timing_operation_counts = parse_int_set(
        args.timing_operation_counts,
        "--timing-operation-counts",
    )

    datetime.strptime(
        args.min_valid_pickup,
        "%Y-%m-%d %H:%M:%S",
    )
    datetime.strptime(
        args.max_valid_pickup,
        "%Y-%m-%d %H:%M:%S",
    )

    spark = (
        SparkSession.builder
        .appName("StateGuardStrongAllRuleBenchmark")
        .getOrCreate()
    )
    spark.conf.set(
        "spark.sql.session.timeZone",
        "America/New_York",
    )
    spark.conf.set(
        "spark.sql.shuffle.partitions",
        str(args.state_partitions),
    )
    spark.conf.set(
        "spark.sql.optimizer.dynamicPartitionPruning.enabled",
        "true",
    )
    spark.conf.set("spark.sql.adaptive.enabled", "true")

    research_root = args.research_matrix_root.rstrip("/")
    performance_root = (
        args.performance_workload_result.rstrip("/")
    )
    state_path = args.consolidated_partition_state.rstrip("/")
    ground_truth_root = args.ground_truth_root.rstrip("/")
    baseline_root = args.baseline_root.rstrip("/")
    deequ_root = args.deequ_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    state_metrics = delta_metrics(spark, state_path)
    baseline_state = load_baseline_partition_state(
        spark,
        state_path,
        args.expected_baseline_version,
        args.state_partitions,
        args.expected_baseline_rows,
    )
    workloads = read_workloads(
        spark,
        research_root,
        performance_root,
    )
    ground_truth = load_ground_truth(
        spark,
        ground_truth_root,
    )
    baseline_medians = load_baseline_medians(
        spark,
        baseline_root,
    )
    deequ_medians = load_deequ_medians(
        spark,
        deequ_root,
    )

    min_pickup = F.lit(
        args.min_valid_pickup
    ).cast("timestamp_ntz")
    max_pickup = F.lit(
        args.max_valid_pickup
    ).cast("timestamp_ntz")

    trial_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("STATEGUARD STRONG ALL-RULE EXPERIMENT")
    print("=" * 80)
    print("Correctness sweep: 16 workload conditions")
    print("Exact rule scope: R01-R13")
    print("S01-S11: CDF-maintained compact state")
    print("R12-R13: explicit exact uniqueness fallback")
    print(
        "Repeated timing endpoints: "
        + ",".join(
            str(value)
            for value in sorted(timing_operation_counts)
        )
    )
    print(f"Timed repetitions: {args.timed_repeats}")
    print(
        "Primary fair comparisons: compact R01-R11 vs full Spark "
        "R01-R11; all-rule planner vs full Spark and Deequ R01-R13"
    )
    print("=" * 80)

    for workload_index, workload in enumerate(
        workloads,
        start=1,
    ):
        workload_id = str(workload["workload_id"])
        family_id = str(workload["family_id"])
        working_path = str(workload["working_path"])
        target_version = int(workload["target_version"])
        final_version = int(workload["final_version"])
        operations = int(workload["cumulative_operations"])
        is_timing_condition = (
            operations in timing_operation_counts
        )

        if not DeltaTable.isDeltaTable(spark, working_path):
            raise RuntimeError(
                f"Missing workload table: {working_path}"
            )
        actual_version = latest_version(
            working_path,
            spark,
        )
        if actual_version != final_version:
            raise RuntimeError(
                f"{workload_id} manifest is stale: "
                f"manifest={final_version}, actual={actual_version}"
            )
        if target_version > final_version:
            raise RuntimeError(
                f"{workload_id} requests unavailable version "
                f"{target_version}."
            )

        print()
        print(
            f"[{workload_index:02d}/16] {workload_id} "
            f"operations={operations}"
        )

        (
            reference_metrics,
            reference_diagnostics,
            comparisons,
        ) = run_all_rule_state_guard_once(
            spark,
            workload,
            baseline_state,
            ground_truth[workload_id],
            min_pickup,
            max_pickup,
            args.max_passengers,
            args.double_tolerance,
        )

        for comparison in comparisons:
            comparison_records.append(
                {
                    "workload_id": workload_id,
                    "family_id": family_id,
                    "target_version": target_version,
                    "cumulative_operations": operations,
                    **comparison,
                }
            )

        trial_records.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": target_version,
                "cumulative_operations": operations,
                "trial_number": 0,
                "run_role": (
                    "CORRECTNESS_AND_WARMUP"
                    if is_timing_condition
                    else "CORRECTNESS_ONLY"
                ),
                "is_timed": False,
                **reference_diagnostics,
                "exact_match_count": 13,
                "rule_count": 13,
                "status": "PASS",
            }
        )

        compact_times: List[float] = []
        uniqueness_times: List[float] = []
        exact_times: List[float] = []
        cdf_times: List[float] = []
        state_times: List[float] = []

        if is_timing_condition:
            for trial_number in range(
                1,
                args.timed_repeats + 1,
            ):
                (
                    trial_metrics,
                    diagnostics,
                    _,
                ) = run_all_rule_state_guard_once(
                    spark,
                    workload,
                    baseline_state,
                    ground_truth[workload_id],
                    min_pickup,
                    max_pickup,
                    args.max_passengers,
                    args.double_tolerance,
                )

                for rule_id in ALL_RULE_IDS:
                    if not values_equal(
                        trial_metrics[rule_id],
                        reference_metrics[rule_id],
                        rule_id,
                        args.double_tolerance,
                    ):
                        raise RuntimeError(
                            f"{workload_id} timed trial "
                            f"{trial_number} disagrees on {rule_id}."
                        )

                compact_times.append(
                    float(
                        diagnostics[
                            "compact_validation_seconds"
                        ]
                    )
                )
                uniqueness_times.append(
                    float(
                        diagnostics[
                            "uniqueness_fallback_seconds"
                        ]
                    )
                )
                exact_times.append(
                    float(diagnostics["exact_total_seconds"])
                )
                cdf_times.append(
                    float(diagnostics["cdf_seconds"])
                )
                state_times.append(
                    float(
                        diagnostics[
                            "state_update_seconds"
                        ]
                    )
                )

                trial_records.append(
                    {
                        "workload_id": workload_id,
                        "family_id": family_id,
                        "target_version": target_version,
                        "cumulative_operations": operations,
                        "trial_number": trial_number,
                        "run_role": "TIMED",
                        "is_timed": True,
                        **diagnostics,
                        "exact_match_count": 13,
                        "rule_count": 13,
                        "status": "PASS",
                    }
                )

        baseline = baseline_medians[workload_id]

        compact_median = (
            float(statistics.median(compact_times))
            if compact_times
            else None
        )
        uniqueness_median = (
            float(statistics.median(uniqueness_times))
            if uniqueness_times
            else None
        )
        exact_median = (
            float(statistics.median(exact_times))
            if exact_times
            else None
        )
        exact_minimum = (
            float(min(exact_times))
            if exact_times
            else None
        )
        exact_maximum = (
            float(max(exact_times))
            if exact_times
            else None
        )

        compact_speedup = (
            float(baseline["median_scalar_seconds"])
            / compact_median
            if compact_median is not None
            and compact_median > 0
            else None
        )
        all_rule_speedup_vs_full = (
            float(baseline["median_total_seconds"])
            / exact_median
            if exact_median is not None
            and exact_median > 0
            else None
        )
        deequ_seconds = deequ_medians.get(workload_id)
        all_rule_speedup_vs_deequ = (
            float(deequ_seconds) / exact_median
            if deequ_seconds is not None
            and exact_median is not None
            and exact_median > 0
            else None
        )
        uniqueness_fraction = (
            uniqueness_median / exact_median
            if uniqueness_median is not None
            and exact_median is not None
            and exact_median > 0
            else None
        )

        summary_records.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": target_version,
                "cumulative_operations": operations,
                "expected_rows": int(
                    workload["expected_rows_after"]
                ),
                "is_timing_condition": is_timing_condition,
                "timed_repeats": (
                    args.timed_repeats
                    if is_timing_condition
                    else 0
                ),
                "median_compact_seconds": compact_median,
                "median_uniqueness_fallback_seconds": (
                    uniqueness_median
                ),
                "median_exact_all_rule_seconds": exact_median,
                "minimum_exact_all_rule_seconds": exact_minimum,
                "maximum_exact_all_rule_seconds": exact_maximum,
                "median_cdf_seconds": (
                    float(statistics.median(cdf_times))
                    if cdf_times
                    else None
                ),
                "median_state_update_seconds": (
                    float(statistics.median(state_times))
                    if state_times
                    else None
                ),
                "correctness_compact_seconds": float(
                    reference_diagnostics[
                        "compact_validation_seconds"
                    ]
                ),
                "correctness_exact_all_rule_seconds": float(
                    reference_diagnostics[
                        "exact_total_seconds"
                    ]
                ),
                "full_scalar_median_seconds": float(
                    baseline["median_scalar_seconds"]
                ),
                "full_all_rule_median_seconds": float(
                    baseline["median_total_seconds"]
                ),
                "deequ_all_rule_median_seconds": (
                    float(deequ_seconds)
                    if deequ_seconds is not None
                    else None
                ),
                "compact_speedup_vs_full_scalar": (
                    compact_speedup
                ),
                "exact_speedup_vs_full_all_rule": (
                    all_rule_speedup_vs_full
                ),
                "exact_speedup_vs_deequ_all_rule": (
                    all_rule_speedup_vs_deequ
                ),
                "uniqueness_fraction_of_exact_total": (
                    uniqueness_fraction
                ),
                "raw_cdf_rows": int(
                    reference_diagnostics["raw_cdf_rows"]
                ),
                "net_image_rows": int(
                    reference_diagnostics["net_image_rows"]
                ),
                "affected_partition_count": int(
                    reference_diagnostics[
                        "affected_partition_count"
                    ]
                ),
                "invalidated_partition_count": int(
                    reference_diagnostics[
                        "invalidated_partition_count"
                    ]
                ),
                "recomputed_partition_rows": int(
                    reference_diagnostics[
                        "recomputed_partition_rows"
                    ]
                ),
                "duplicate_extra_rows": int(
                    reference_diagnostics[
                        "duplicate_extra_rows"
                    ]
                ),
                "exact_match_count": 13,
                "rule_count": 13,
                "logical_portfolio_bytes": int(
                    args.logical_portfolio_bytes
                ),
                "physical_state_bytes": int(
                    state_metrics["size_bytes"]
                ),
                "status": "PASS",
            }
        )

        if is_timing_condition:
            print(
                f"  compact={compact_median:.3f}s "
                f"({compact_speedup:.3f}x vs same-rule full) | "
                f"exact13={exact_median:.3f}s "
                f"({all_rule_speedup_vs_full:.3f}x vs full13) | "
                f"agreement=13/13"
            )
        else:
            print("  exact agreement=13/13")

    trial_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("trial_number", T.IntegerType(), False),
        T.StructField("run_role", T.StringType(), False),
        T.StructField("is_timed", T.BooleanType(), False),
        T.StructField("raw_cdf_rows", T.LongType(), False),
        T.StructField("net_image_rows", T.LongType(), False),
        T.StructField("affected_partition_count", T.LongType(), False),
        T.StructField("invalidated_partition_count", T.LongType(), False),
        T.StructField("recomputed_partition_rows", T.LongType(), False),
        T.StructField("cdf_seconds", T.DoubleType(), False),
        T.StructField("state_update_seconds", T.DoubleType(), False),
        T.StructField("validation_ready_seconds", T.DoubleType(), False),
        T.StructField("insert_cdf_rows", T.LongType(), False),
        T.StructField("delete_cdf_rows", T.LongType(), False),
        T.StructField("update_preimage_cdf_rows", T.LongType(), False),
        T.StructField("update_postimage_cdf_rows", T.LongType(), False),
        T.StructField("uniqueness_fallback_seconds", T.DoubleType(), False),
        T.StructField("distinct_trip_keys", T.LongType(), False),
        T.StructField("duplicate_key_groups", T.LongType(), False),
        T.StructField("duplicate_extra_rows", T.LongType(), False),
        T.StructField("maximum_key_multiplicity", T.LongType(), False),
        T.StructField("compact_validation_seconds", T.DoubleType(), False),
        T.StructField("exact_total_seconds", T.DoubleType(), False),
        T.StructField("uniqueness_fraction_of_exact_total", T.DoubleType(), False),
        T.StructField("exact_match_count", T.IntegerType(), False),
        T.StructField("rule_count", T.IntegerType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    comparison_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("rule_id", T.StringType(), False),
        T.StructField("incremental_value", T.StringType(), False),
        T.StructField("ground_truth_value", T.StringType(), False),
        T.StructField("planner_mode", T.StringType(), False),
        T.StructField("exact_match", T.BooleanType(), False),
    ])

    summary_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("expected_rows", T.LongType(), False),
        T.StructField("is_timing_condition", T.BooleanType(), False),
        T.StructField("timed_repeats", T.IntegerType(), False),
        T.StructField("median_compact_seconds", T.DoubleType(), True),
        T.StructField("median_uniqueness_fallback_seconds", T.DoubleType(), True),
        T.StructField("median_exact_all_rule_seconds", T.DoubleType(), True),
        T.StructField("minimum_exact_all_rule_seconds", T.DoubleType(), True),
        T.StructField("maximum_exact_all_rule_seconds", T.DoubleType(), True),
        T.StructField("median_cdf_seconds", T.DoubleType(), True),
        T.StructField("median_state_update_seconds", T.DoubleType(), True),
        T.StructField("correctness_compact_seconds", T.DoubleType(), False),
        T.StructField("correctness_exact_all_rule_seconds", T.DoubleType(), False),
        T.StructField("full_scalar_median_seconds", T.DoubleType(), False),
        T.StructField("full_all_rule_median_seconds", T.DoubleType(), False),
        T.StructField("deequ_all_rule_median_seconds", T.DoubleType(), True),
        T.StructField("compact_speedup_vs_full_scalar", T.DoubleType(), True),
        T.StructField("exact_speedup_vs_full_all_rule", T.DoubleType(), True),
        T.StructField("exact_speedup_vs_deequ_all_rule", T.DoubleType(), True),
        T.StructField("uniqueness_fraction_of_exact_total", T.DoubleType(), True),
        T.StructField("raw_cdf_rows", T.LongType(), False),
        T.StructField("net_image_rows", T.LongType(), False),
        T.StructField("affected_partition_count", T.LongType(), False),
        T.StructField("invalidated_partition_count", T.LongType(), False),
        T.StructField("recomputed_partition_rows", T.LongType(), False),
        T.StructField("duplicate_extra_rows", T.LongType(), False),
        T.StructField("exact_match_count", T.IntegerType(), False),
        T.StructField("rule_count", T.IntegerType(), False),
        T.StructField("logical_portfolio_bytes", T.LongType(), False),
        T.StructField("physical_state_bytes", T.LongType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    trial_df = spark.createDataFrame(
        trial_records,
        schema=trial_schema,
    ).orderBy(
        "family_id",
        "target_version",
        "is_timed",
        "trial_number",
    )

    comparison_df = spark.createDataFrame(
        comparison_records,
        schema=comparison_schema,
    ).orderBy(
        "family_id",
        "target_version",
        "rule_id",
    )

    method_summary_df = spark.createDataFrame(
        summary_records,
        schema=summary_schema,
    ).orderBy("family_id", "target_version")

    comparison_count = comparison_df.count()
    mismatch_count = comparison_df.filter(
        ~F.col("exact_match")
    ).count()

    if comparison_count != 16 * 13:
        raise RuntimeError(
            f"Expected 208 rule comparisons; "
            f"found {comparison_count}."
        )
    if mismatch_count != 0:
        raise RuntimeError(
            f"Found {mismatch_count} all-rule mismatches."
        )

    timed_rows = [
        row
        for row in summary_records
        if bool(row["is_timing_condition"])
    ]
    expected_timed_condition_count = sum(
        1
        for workload in workloads
        if int(workload["cumulative_operations"])
        in timing_operation_counts
    )
    timed_condition_count = len(timed_rows)

    if timed_condition_count != expected_timed_condition_count:
        raise RuntimeError(
            "Timed-condition count mismatch: "
            f"expected={expected_timed_condition_count}, "
            f"actual={timed_condition_count}."
        )

    compact_speedups = [
        float(row["compact_speedup_vs_full_scalar"])
        for row in timed_rows
    ]
    full_speedups = [
        float(row["exact_speedup_vs_full_all_rule"])
        for row in timed_rows
    ]
    deequ_speedups = [
        float(row["exact_speedup_vs_deequ_all_rule"])
        for row in timed_rows
        if row["exact_speedup_vs_deequ_all_rule"]
        is not None
    ]
    exact_medians = [
        float(row["median_exact_all_rule_seconds"])
        for row in timed_rows
    ]
    compact_medians = [
        float(row["median_compact_seconds"])
        for row in timed_rows
    ]
    uniqueness_fractions = [
        float(row["uniqueness_fraction_of_exact_total"])
        for row in timed_rows
    ]

    overall_schema = T.StructType([
        T.StructField("status", T.StringType(), False),
        T.StructField("workload_condition_count", T.IntegerType(), False),
        T.StructField("rule_comparison_count", T.IntegerType(), False),
        T.StructField("rule_mismatch_count", T.IntegerType(), False),
        T.StructField("exact_agreement_rate", T.DoubleType(), False),
        T.StructField("timed_condition_count", T.IntegerType(), False),
        T.StructField("timed_execution_count", T.IntegerType(), False),
        T.StructField("timed_repeats", T.IntegerType(), False),
        T.StructField("median_compact_endpoint_seconds", T.DoubleType(), False),
        T.StructField("median_exact_all_rule_endpoint_seconds", T.DoubleType(), False),
        T.StructField("median_compact_speedup_vs_full_scalar", T.DoubleType(), False),
        T.StructField("minimum_compact_speedup_vs_full_scalar", T.DoubleType(), False),
        T.StructField("maximum_compact_speedup_vs_full_scalar", T.DoubleType(), False),
        T.StructField("median_exact_speedup_vs_full_all_rule", T.DoubleType(), False),
        T.StructField("minimum_exact_speedup_vs_full_all_rule", T.DoubleType(), False),
        T.StructField("maximum_exact_speedup_vs_full_all_rule", T.DoubleType(), False),
        T.StructField("median_exact_speedup_vs_deequ_all_rule", T.DoubleType(), False),
        T.StructField("median_uniqueness_fraction_of_exact_total", T.DoubleType(), False),
        T.StructField("total_invalidated_partitions", T.LongType(), False),
        T.StructField("total_recomputed_partition_rows", T.LongType(), False),
        T.StructField("logical_portfolio_bytes", T.LongType(), False),
        T.StructField("physical_state_bytes", T.LongType(), False),
        T.StructField("physical_state_num_files", T.LongType(), False),
        T.StructField("spark_version", T.StringType(), False),
        T.StructField("output_root", T.StringType(), False),
    ])

    overall_df = spark.createDataFrame(
        [
            (
                "PASS",
                16,
                comparison_count,
                mismatch_count,
                1.0,
                timed_condition_count,
                timed_condition_count * args.timed_repeats,
                args.timed_repeats,
                float(statistics.median(compact_medians)),
                float(statistics.median(exact_medians)),
                float(statistics.median(compact_speedups)),
                float(min(compact_speedups)),
                float(max(compact_speedups)),
                float(statistics.median(full_speedups)),
                float(min(full_speedups)),
                float(max(full_speedups)),
                float(statistics.median(deequ_speedups)),
                float(statistics.median(uniqueness_fractions)),
                int(
                    sum(
                        int(row["invalidated_partition_count"])
                        for row in summary_records
                    )
                ),
                int(
                    sum(
                        int(row["recomputed_partition_rows"])
                        for row in summary_records
                    )
                ),
                int(args.logical_portfolio_bytes),
                int(state_metrics["size_bytes"]),
                int(state_metrics["num_files"]),
                spark.version,
                output_root,
            )
        ],
        schema=overall_schema,
    )

    write_csv(
        trial_df,
        f"{output_root}/trial_results_csv",
    )
    write_csv(
        comparison_df,
        f"{output_root}/rule_comparison_csv",
    )
    write_csv(
        method_summary_df,
        f"{output_root}/method_summary_csv",
    )
    write_csv(
        overall_df,
        f"{output_root}/summary_csv",
    )

    print()
    print("=" * 80)
    print("ALL-RULE ENDPOINT RESULT")
    print("=" * 80)
    method_summary_df.filter(
        F.col("is_timing_condition")
    ).select(
        "workload_id",
        "median_compact_seconds",
        "compact_speedup_vs_full_scalar",
        "median_exact_all_rule_seconds",
        "exact_speedup_vs_full_all_rule",
        "exact_speedup_vs_deequ_all_rule",
        "uniqueness_fraction_of_exact_total",
        "exact_match_count",
    ).show(20, truncate=False)

    print("=" * 80)
    print("STATEGUARD_STRONG_EXPERIMENT_BEGIN")
    print("STRONG_EXPERIMENT_STATUS=PASS")
    print("COMPACT_PORTFOLIO=S01-S11")
    print("UNIQUENESS_STRATEGY=EXACT_FULL_SCAN_FALLBACK")
    print("RULE_SCOPE=R01-R13")
    print("WORKLOAD_CONDITION_COUNT=16")
    print(f"RULE_COMPARISON_COUNT={comparison_count}")
    print(f"RULE_MISMATCH_COUNT={mismatch_count}")
    print("EXACT_AGREEMENT_RATE=1.000000")
    print(
        "TIMED_CONDITION_COUNT="
        f"{timed_condition_count}"
    )
    print(
        "TIMED_EXECUTION_COUNT="
        f"{timed_condition_count * args.timed_repeats}"
    )
    print(f"TIMED_REPEATS={args.timed_repeats}")
    print(
        "MEDIAN_COMPACT_ENDPOINT_SECONDS="
        f"{statistics.median(compact_medians):.6f}"
    )
    print(
        "MEDIAN_EXACT_ALL_RULE_ENDPOINT_SECONDS="
        f"{statistics.median(exact_medians):.6f}"
    )
    print(
        "MEDIAN_COMPACT_SPEEDUP_VS_FULL_SCALAR="
        f"{statistics.median(compact_speedups):.6f}"
    )
    print(
        "MINIMUM_COMPACT_SPEEDUP_VS_FULL_SCALAR="
        f"{min(compact_speedups):.6f}"
    )
    print(
        "MAXIMUM_COMPACT_SPEEDUP_VS_FULL_SCALAR="
        f"{max(compact_speedups):.6f}"
    )
    print(
        "MEDIAN_EXACT_SPEEDUP_VS_FULL_ALL_RULE="
        f"{statistics.median(full_speedups):.6f}"
    )
    print(
        "MINIMUM_EXACT_SPEEDUP_VS_FULL_ALL_RULE="
        f"{min(full_speedups):.6f}"
    )
    print(
        "MAXIMUM_EXACT_SPEEDUP_VS_FULL_ALL_RULE="
        f"{max(full_speedups):.6f}"
    )
    print(
        "MEDIAN_EXACT_SPEEDUP_VS_DEEQU_ALL_RULE="
        f"{statistics.median(deequ_speedups):.6f}"
    )
    print(
        "MEDIAN_UNIQUENESS_FRACTION_OF_EXACT_TOTAL="
        f"{statistics.median(uniqueness_fractions):.6f}"
    )
    print(
        "LOGICAL_PORTFOLIO_SIZE_BYTES="
        f"{args.logical_portfolio_bytes}"
    )
    print(
        "PHYSICAL_STATE_SIZE_BYTES="
        f"{state_metrics['size_bytes']}"
    )
    print(
        "TRIAL_RESULTS_PATH="
        f"{output_root}/trial_results_csv"
    )
    print(
        "RULE_COMPARISON_PATH="
        f"{output_root}/rule_comparison_csv"
    )
    print(
        "METHOD_SUMMARY_PATH="
        f"{output_root}/method_summary_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_STRONG_EXPERIMENT_END")
    print("=" * 80)

    baseline_state.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
