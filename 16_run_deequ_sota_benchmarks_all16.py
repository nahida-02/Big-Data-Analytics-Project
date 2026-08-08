import argparse
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

# PyDeequ uses this value to select the compatible JVM API.
os.environ.setdefault("SPARK_VERSION", "3.5")

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

import pydeequ
from pydeequ.analyzers import (
    AnalysisRunner,
    AnalyzerContext,
    Completeness,
    Compliance,
    Distinctness,
    Maximum,
    Minimum,
    Size,
    Uniqueness,
)


RULE_DEFINITIONS: List[Tuple[str, str, str]] = [
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official Deequ implementation with a 16-condition correctness sweep and "
            "repeated timing on all 16 workload conditions; compare every result against "
            "the independently saved full-validation ground truth."
        )
    )
    parser.add_argument("--research-matrix-root", required=True)
    parser.add_argument("--performance-workload-result", required=True)
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument(
        "--timing-operation-counts",
        default="1000,5000,20000,100000",
        help=(
            "Comma-separated mutation counts selected for repeated timing. "
            "For the final all-16 benchmark, use 1000,5000,20000,100000."
        ),
    )
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
    return int(
        DeltaTable.forPath(spark, path)
        .history(1)
        .collect()[0]["version"]
    )


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

    expected_rows = 16 * 13
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} ground-truth rule rows; "
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
        if len(metrics) != 13:
            raise RuntimeError(
                f"{workload_id} has {len(metrics)} ground-truth rules."
            )

    return result


def metric_lookup(
    metric_rows: List[Any],
    name: str,
    instance: str,
) -> float:
    exact = [
        float(row["value"])
        for row in metric_rows
        if str(row["name"]) == name
        and str(row["instance"]) == instance
    ]

    if len(exact) == 1:
        return exact[0]

    # PyDeequ may render a single-column analyzer instance as either the
    # plain column name or a one-element sequence-like string.
    fallback = [
        float(row["value"])
        for row in metric_rows
        if str(row["name"]) == name
        and instance in str(row["instance"])
    ]

    if len(fallback) != 1:
        available = sorted(
            {
                (str(row["name"]), str(row["instance"]))
                for row in metric_rows
            }
        )
        raise RuntimeError(
            f"Could not uniquely find metric name={name}, "
            f"instance={instance}. Available={available}"
        )

    return fallback[0]


def integer_from_fraction(
    fraction: float,
    total_rows: int,
    label: str,
) -> int:
    estimate = fraction * total_rows
    rounded = int(round(estimate))

    # Deequ exposes ratio metrics as doubles. Counts under this experiment
    # are far below the exact-integer limit of IEEE-754 doubles, but a small
    # tolerance is retained for division round-off.
    if abs(estimate - rounded) > 1e-4:
        raise RuntimeError(
            f"{label} did not reconstruct an integer count: "
            f"fraction={fraction}, rows={total_rows}, "
            f"estimate={estimate}"
        )

    return rounded


def run_deequ_once(
    dataframe: DataFrame,
    min_valid_pickup: str,
    max_valid_pickup: str,
    max_passengers: int,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Any]]:
    selected = dataframe.select(
        "trip_key",
        "passenger_count",
        "fare_amount",
        "trip_distance",
        "tpep_pickup_datetime",
    )

    valid_fare = "fare_amount IS NULL OR fare_amount >= 0"
    valid_distance = (
        "trip_distance IS NULL OR trip_distance >= 0"
    )
    valid_passenger = (
        "passenger_count IS NULL OR "
        f"(passenger_count >= 1 AND passenger_count <= {max_passengers})"
    )
    valid_pickup = (
        "tpep_pickup_datetime IS NOT NULL AND "
        f"tpep_pickup_datetime >= '{min_valid_pickup}' AND "
        f"tpep_pickup_datetime <= '{max_valid_pickup}'"
    )

    start = time.perf_counter()

    context = (
        AnalysisRunner(dataframe.sparkSession)
        .onData(selected)
        .addAnalyzer(Size())
        .addAnalyzer(Completeness("passenger_count"))
        .addAnalyzer(Completeness("fare_amount"))
        .addAnalyzer(Compliance("valid_fare", valid_fare))
        .addAnalyzer(
            Compliance("valid_distance", valid_distance)
        )
        .addAnalyzer(
            Compliance("valid_passenger", valid_passenger)
        )
        .addAnalyzer(
            Compliance("valid_pickup_time", valid_pickup)
        )
        .addAnalyzer(Minimum("fare_amount"))
        .addAnalyzer(Maximum("fare_amount"))
        .addAnalyzer(Minimum("trip_distance"))
        .addAnalyzer(Maximum("trip_distance"))
        .addAnalyzer(Distinctness(["trip_key"]))
        .addAnalyzer(Uniqueness(["trip_key"]))
        .run()
    )

    metrics_df = AnalyzerContext.successMetricsAsDataFrame(
        dataframe.sparkSession,
        context,
    )

    metric_rows = metrics_df.collect()
    total_seconds = time.perf_counter() - start

    if len(metric_rows) != 13:
        raise RuntimeError(
            f"Expected 13 successful Deequ analyzer metrics; "
            f"found {len(metric_rows)}."
        )

    row_count = int(
        round(metric_lookup(metric_rows, "Size", "*"))
    )

    passenger_complete = metric_lookup(
        metric_rows,
        "Completeness",
        "passenger_count",
    )
    fare_complete = metric_lookup(
        metric_rows,
        "Completeness",
        "fare_amount",
    )
    valid_fare_fraction = metric_lookup(
        metric_rows,
        "Compliance",
        "valid_fare",
    )
    valid_distance_fraction = metric_lookup(
        metric_rows,
        "Compliance",
        "valid_distance",
    )
    valid_passenger_fraction = metric_lookup(
        metric_rows,
        "Compliance",
        "valid_passenger",
    )
    valid_pickup_fraction = metric_lookup(
        metric_rows,
        "Compliance",
        "valid_pickup_time",
    )
    distinctness = metric_lookup(
        metric_rows,
        "Distinctness",
        "trip_key",
    )
    uniqueness = metric_lookup(
        metric_rows,
        "Uniqueness",
        "trip_key",
    )

    non_null_passenger = integer_from_fraction(
        passenger_complete,
        row_count,
        "non-null passenger count",
    )
    non_null_fare = integer_from_fraction(
        fare_complete,
        row_count,
        "non-null fare count",
    )
    valid_fare_count = integer_from_fraction(
        valid_fare_fraction,
        row_count,
        "valid fare count",
    )
    valid_distance_count = integer_from_fraction(
        valid_distance_fraction,
        row_count,
        "valid distance count",
    )
    valid_passenger_count = integer_from_fraction(
        valid_passenger_fraction,
        row_count,
        "valid passenger count",
    )
    valid_pickup_count = integer_from_fraction(
        valid_pickup_fraction,
        row_count,
        "valid pickup-time count",
    )
    distinct_key_count = integer_from_fraction(
        distinctness,
        row_count,
        "distinct trip-key count",
    )

    duplicate_extra_rows = row_count - distinct_key_count

    metrics: Dict[str, Any] = {
        "R01": row_count,
        "R02": row_count - non_null_passenger,
        "R03": row_count - non_null_fare,
        "R04": row_count - valid_fare_count,
        "R05": row_count - valid_distance_count,
        "R06": row_count - valid_passenger_count,
        "R07": row_count - valid_pickup_count,
        "R08": metric_lookup(
            metric_rows,
            "Minimum",
            "fare_amount",
        ),
        "R09": metric_lookup(
            metric_rows,
            "Maximum",
            "fare_amount",
        ),
        "R10": metric_lookup(
            metric_rows,
            "Minimum",
            "trip_distance",
        ),
        "R11": metric_lookup(
            metric_rows,
            "Maximum",
            "trip_distance",
        ),
        "R12": duplicate_extra_rows,
        "R13": duplicate_extra_rows == 0,
    }

    diagnostics = {
        "total_seconds": float(total_seconds),
        "deequ_distinctness": float(distinctness),
        "deequ_uniqueness": float(uniqueness),
        "distinct_key_count": float(distinct_key_count),
        "successful_analyzer_count": float(len(metric_rows)),
    }

    return metrics, diagnostics, metric_rows


def values_equal(
    left: Any,
    right: Any,
    rule_id: str,
) -> bool:
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


def main() -> None:
    args = parse_arguments()

    if args.timed_repeats < 1:
        raise ValueError("--timed-repeats must be at least 1.")

    timing_operation_counts = {
        int(value.strip())
        for value in args.timing_operation_counts.split(",")
        if value.strip()
    }
    if not timing_operation_counts:
        raise ValueError(
            "--timing-operation-counts must contain at least one value."
        )

    spark = (
        SparkSession.builder
        .appName("StateGuardOfficialDeequAll16Benchmark")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")

    research_matrix_root = args.research_matrix_root.rstrip("/")
    performance_root = args.performance_workload_result.rstrip("/")
    ground_truth_root = args.ground_truth_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    workloads = read_workloads(
        spark,
        research_matrix_root,
        performance_root,
    )

    selected_timing_condition_count = sum(
        1
        for workload in workloads
        if int(workload["cumulative_operations"]) in timing_operation_counts
    )
    if selected_timing_condition_count != len(workloads):
        raise RuntimeError(
            "This final Deequ benchmark requires repeated timing for every "
            "workload condition: "
            f"selected={selected_timing_condition_count}, "
            f"total={len(workloads)}. "
            "Use --timing-operation-counts=1000,5000,20000,100000."
        )
    ground_truth = load_ground_truth(
        spark,
        ground_truth_root,
    )

    trial_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("OFFICIAL DEEQU IMPLEMENTATION-BASED SOTA BENCHMARK")
    print("=" * 80)
    print(f"Workload conditions: {len(workloads)}")
    print("Timing scope: ALL 16 workload conditions")
    print("Correctness sweep: one official Deequ run on all 16 conditions")
    print(
        "Repeated timing mutation counts: "
        + ",".join(str(value) for value in sorted(timing_operation_counts))
    )
    print(f"Timed repetitions on selected conditions: {args.timed_repeats}")
    print(
        "The correctness run is also the warm-up for each timed condition."
    )
    print("Rules compared against independent ground truth: R01-R13")
    print("=" * 80)

    for workload_index, workload in enumerate(workloads, start=1):
        workload_id = str(workload["workload_id"])
        family_id = str(workload["family_id"])
        target_version = int(workload["target_version"])
        expected_rows = int(workload["expected_rows_after"])
        working_path = str(workload["working_path"])
        final_version = int(workload["final_version"])

        if not DeltaTable.isDeltaTable(spark, working_path):
            raise RuntimeError(
                f"Missing Delta workload table: {working_path}"
            )
        if latest_version(working_path, spark) != final_version:
            raise RuntimeError(
                f"{workload_id} manifest final version is stale."
            )
        if target_version > final_version:
            raise RuntimeError(
                f"{workload_id} requests unavailable version "
                f"{target_version}."
            )
        if workload_id not in ground_truth:
            raise RuntimeError(
                f"Missing ground truth for {workload_id}."
            )

        dataframe = (
            spark.read.format("delta")
            .option("versionAsOf", target_version)
            .load(working_path)
        )

        print()
        print(
            f"[{workload_index:02d}/16] {workload_id} "
            f"expected_rows={expected_rows}"
        )

        timed_seconds: List[float] = []
        reference_metrics: Dict[str, Any] = {}
        exact_match_count = 0
        diagnostic_reference: Dict[str, float] = {}

        cumulative_operations = int(
            workload["cumulative_operations"]
        )
        is_timing_condition = (
            cumulative_operations in timing_operation_counts
        )

        # One exact run on every workload. For the selected timing conditions,
        # this run also warms the table/JVM before the measured repetitions.
        metrics, diagnostics, _ = run_deequ_once(
            dataframe,
            args.min_valid_pickup,
            args.max_valid_pickup,
            args.max_passengers,
        )

        if int(metrics["R01"]) != expected_rows:
            raise RuntimeError(
                f"{workload_id} Deequ Size mismatch: "
                f"expected={expected_rows}, "
                f"actual={metrics['R01']}"
            )

        exact_match_count = sum(
            1
            for rule_id, _, _ in RULE_DEFINITIONS
            if values_equal(
                metrics[rule_id],
                ground_truth[workload_id][rule_id],
                rule_id,
            )
        )

        if exact_match_count != 13:
            mismatches = [
                rule_id
                for rule_id, _, _ in RULE_DEFINITIONS
                if not values_equal(
                    metrics[rule_id],
                    ground_truth[workload_id][rule_id],
                    rule_id,
                )
            ]
            raise RuntimeError(
                f"{workload_id} official Deequ matched "
                f"{exact_match_count}/13 rules. "
                f"Mismatches={mismatches}"
            )

        reference_metrics = metrics
        diagnostic_reference = diagnostics

        trial_records.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": target_version,
                "cumulative_operations": cumulative_operations,
                "trial_number": 0,
                "run_role": (
                    "CORRECTNESS_AND_WARMUP"
                    if is_timing_condition
                    else "CORRECTNESS_ONLY"
                ),
                "is_timed": False,
                "total_seconds": float(
                    diagnostics["total_seconds"]
                ),
                "exact_match_count": exact_match_count,
                "rule_count": 13,
                "status": "PASS",
            }
        )

        if is_timing_condition:
            for trial_number in range(1, args.timed_repeats + 1):
                timed_metrics, timed_diagnostics, _ = run_deequ_once(
                    dataframe,
                    args.min_valid_pickup,
                    args.max_valid_pickup,
                    args.max_passengers,
                )

                for rule_id, _, _ in RULE_DEFINITIONS:
                    if not values_equal(
                        reference_metrics[rule_id],
                        timed_metrics[rule_id],
                        rule_id,
                    ):
                        raise RuntimeError(
                            f"{workload_id} Deequ trials disagree "
                            f"on {rule_id}."
                        )

                timed_matches = sum(
                    1
                    for rule_id, _, _ in RULE_DEFINITIONS
                    if values_equal(
                        timed_metrics[rule_id],
                        ground_truth[workload_id][rule_id],
                        rule_id,
                    )
                )
                if timed_matches != 13:
                    raise RuntimeError(
                        f"{workload_id} timed Deequ trial "
                        f"{trial_number} matched {timed_matches}/13."
                    )

                trial_seconds = float(
                    timed_diagnostics["total_seconds"]
                )
                timed_seconds.append(trial_seconds)

                trial_records.append(
                    {
                        "workload_id": workload_id,
                        "family_id": family_id,
                        "target_version": target_version,
                        "cumulative_operations": cumulative_operations,
                        "trial_number": trial_number,
                        "run_role": "TIMED",
                        "is_timed": True,
                        "total_seconds": trial_seconds,
                        "exact_match_count": timed_matches,
                        "rule_count": 13,
                        "status": "PASS",
                    }
                )

        for rule_id, rule_name, _ in RULE_DEFINITIONS:
            deequ_value = reference_metrics[rule_id]
            truth_value = ground_truth[workload_id][rule_id]
            comparison_records.append(
                {
                    "workload_id": workload_id,
                    "family_id": family_id,
                    "target_version": target_version,
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "ground_truth_value": metric_display(
                        truth_value,
                        rule_id,
                    ),
                    "deequ_value": metric_display(
                        deequ_value,
                        rule_id,
                    ),
                    "exact_match": values_equal(
                        deequ_value,
                        truth_value,
                        rule_id,
                    ),
                }
            )

        median_seconds = (
            float(statistics.median(timed_seconds))
            if timed_seconds
            else None
        )
        minimum_seconds = (
            float(min(timed_seconds))
            if timed_seconds
            else None
        )
        maximum_seconds = (
            float(max(timed_seconds))
            if timed_seconds
            else None
        )

        summary_records.append(
            {
                "workload_id": workload_id,
                "family_id": family_id,
                "target_version": target_version,
                "cumulative_operations": cumulative_operations,
                "expected_rows": expected_rows,
                "is_timing_condition": is_timing_condition,
                "timed_repeats": (
                    args.timed_repeats
                    if is_timing_condition
                    else 0
                ),
                "median_total_seconds": median_seconds,
                "minimum_total_seconds": minimum_seconds,
                "maximum_total_seconds": maximum_seconds,
                "correctness_run_seconds": float(
                    diagnostic_reference["total_seconds"]
                ),
                "exact_match_count": exact_match_count,
                "rule_count": 13,
                "deequ_distinctness": float(
                    diagnostic_reference["deequ_distinctness"]
                ),
                "deequ_uniqueness": float(
                    diagnostic_reference["deequ_uniqueness"]
                ),
                "status": "PASS",
            }
        )

        if is_timing_condition:
            print(
                f"  DEEQU median={median_seconds:.3f}s "
                f"range=[{minimum_seconds:.3f}, "
                f"{maximum_seconds:.3f}] "
                f"agreement={exact_match_count}/13"
            )
        else:
            print(
                "  DEEQU correctness=13/13 "
                "(not selected for repeated timing)"
            )

    trial_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("trial_number", T.IntegerType(), False),
        T.StructField("run_role", T.StringType(), False),
        T.StructField("is_timed", T.BooleanType(), False),
        T.StructField("total_seconds", T.DoubleType(), False),
        T.StructField("exact_match_count", T.IntegerType(), False),
        T.StructField("rule_count", T.IntegerType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    comparison_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("rule_id", T.StringType(), False),
        T.StructField("rule_name", T.StringType(), False),
        T.StructField("ground_truth_value", T.StringType(), False),
        T.StructField("deequ_value", T.StringType(), False),
        T.StructField("exact_match", T.BooleanType(), False),
    ])

    method_summary_schema = T.StructType([
        T.StructField("workload_id", T.StringType(), False),
        T.StructField("family_id", T.StringType(), False),
        T.StructField("target_version", T.IntegerType(), False),
        T.StructField("cumulative_operations", T.LongType(), False),
        T.StructField("expected_rows", T.LongType(), False),
        T.StructField("is_timing_condition", T.BooleanType(), False),
        T.StructField("timed_repeats", T.IntegerType(), False),
        T.StructField("median_total_seconds", T.DoubleType(), True),
        T.StructField("minimum_total_seconds", T.DoubleType(), True),
        T.StructField("maximum_total_seconds", T.DoubleType(), True),
        T.StructField("correctness_run_seconds", T.DoubleType(), False),
        T.StructField("exact_match_count", T.IntegerType(), False),
        T.StructField("rule_count", T.IntegerType(), False),
        T.StructField("deequ_distinctness", T.DoubleType(), False),
        T.StructField("deequ_uniqueness", T.DoubleType(), False),
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
        schema=method_summary_schema,
    ).orderBy(
        "family_id",
        "target_version",
    )

    total_rule_comparisons = len(comparison_records)
    total_mismatches = sum(
        1
        for row in comparison_records
        if not bool(row["exact_match"])
    )
    timed_summary_records = [
        row
        for row in summary_records
        if bool(row["is_timing_condition"])
    ]
    median_of_workload_medians = float(
        statistics.median(
            float(row["median_total_seconds"])
            for row in timed_summary_records
        )
    )
    timed_condition_count = len(timed_summary_records)
    timed_execution_count = (
        timed_condition_count * args.timed_repeats
    )

    overall_schema = T.StructType([
        T.StructField("status", T.StringType(), False),
        T.StructField("workload_condition_count", T.IntegerType(), False),
        T.StructField("rule_comparison_count", T.IntegerType(), False),
        T.StructField("rule_mismatch_count", T.IntegerType(), False),
        T.StructField("exact_agreement_rate", T.DoubleType(), False),
        T.StructField("correctness_run_count", T.IntegerType(), False),
        T.StructField("timed_condition_count", T.IntegerType(), False),
        T.StructField("timed_execution_count", T.IntegerType(), False),
        T.StructField("timed_repeats", T.IntegerType(), False),
        T.StructField(
            "median_of_workload_medians_seconds",
            T.DoubleType(),
            False,
        ),
        T.StructField("spark_version", T.StringType(), False),
        T.StructField("pydeequ_package", T.StringType(), False),
        T.StructField("deequ_maven_coordinate", T.StringType(), False),
    ])

    overall_df = spark.createDataFrame(
        [
            (
                "PASS",
                len(workloads),
                total_rule_comparisons,
                total_mismatches,
                (
                    1.0
                    if total_rule_comparisons == 0
                    else (
                        total_rule_comparisons - total_mismatches
                    )
                    / total_rule_comparisons
                ),
                len(workloads),
                timed_condition_count,
                timed_execution_count,
                args.timed_repeats,
                median_of_workload_medians,
                spark.version,
                "pydeequ==1.6.0",
                "com.amazon.deequ:deequ:2.0.16-spark-3.5",
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
    print("OFFICIAL DEEQU METHOD SUMMARY")
    print("=" * 80)
    method_summary_df.show(20, truncate=False)

    print("=" * 80)
    print("STATEGUARD_DEEQU_SOTA_BENCHMARK_BEGIN")
    print("DEEQU_SOTA_BENCHMARK_STATUS=PASS")
    print(f"WORKLOAD_CONDITION_COUNT={len(workloads)}")
    print(f"RULE_COMPARISON_COUNT={total_rule_comparisons}")
    print(f"RULE_MISMATCH_COUNT={total_mismatches}")
    print(
        "EXACT_AGREEMENT_RATE="
        f"{(total_rule_comparisons - total_mismatches) / total_rule_comparisons:.6f}"
    )
    print(f"CORRECTNESS_RUN_COUNT={len(workloads)}")
    print(f"TIMED_CONDITION_COUNT={timed_condition_count}")
    print(f"TIMED_EXECUTION_COUNT={timed_execution_count}")
    print(f"TIMED_REPEATS={args.timed_repeats}")
    print(
        "MEDIAN_OF_WORKLOAD_MEDIANS_SECONDS="
        f"{median_of_workload_medians:.6f}"
    )
    print("SPARK_VERSION=" + spark.version)
    print("PYDEEQU_PACKAGE=pydeequ==1.6.0")
    print(
        "DEEQU_MAVEN_COORDINATE="
        "com.amazon.deequ:deequ:2.0.16-spark-3.5"
    )
    print(
        f"TRIAL_RESULTS_PATH="
        f"{output_root}/trial_results_csv"
    )
    print(
        f"RULE_COMPARISON_PATH="
        f"{output_root}/rule_comparison_csv"
    )
    print(
        f"METHOD_SUMMARY_PATH="
        f"{output_root}/method_summary_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_DEEQU_SOTA_BENCHMARK_END")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
