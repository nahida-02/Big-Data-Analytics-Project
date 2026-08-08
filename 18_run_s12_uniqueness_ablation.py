import argparse
import statistics
import time
from typing import Any, Dict, List, Sequence, Set, Tuple

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_WORKLOADS = (
    "03_DELETE_ONLY_N001000,"
    "01_INSERT_ONLY_N001000,"
    "04_MIXED_40I_30U_30D_N100000,"
    "02_UPDATE_ONLY_N100000"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a targeted StateGuard S12 uniqueness ablation. "
            "For representative mutable-Delta workloads, compare exact "
            "full-snapshot uniqueness fallback against exact incremental "
            "S12 key-frequency maintenance. R12 and R13 must match the "
            "independent full-validation ground truth for every run."
        )
    )
    parser.add_argument("--research-matrix-root", required=True)
    parser.add_argument("--performance-workload-result", required=True)
    parser.add_argument("--key-frequency-state", required=True)
    parser.add_argument("--initial-state-summary", required=True)
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--selected-workloads",
        default=DEFAULT_WORKLOADS,
        help=(
            "Comma-separated workload IDs. Default: one representative "
            "small/large condition across DELETE, INSERT, MIXED and UPDATE."
        ),
    )
    parser.add_argument("--timed-repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--key-buckets", type=int, default=256)
    parser.add_argument(
        "--expected-s12-bytes",
        type=int,
        default=2359231751,
        help="Expected logical/physical Delta size previously measured for S12.",
    )
    parser.add_argument(
        "--compact-portfolio-bytes",
        type=int,
        default=318331,
        help="Logical size of the compact S01-S11 portfolio.",
    )
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def parse_csv_set(text: str, name: str) -> List[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values.")
    return values


def delta_metrics(spark: SparkSession, path: str) -> Dict[str, int]:
    detail = DeltaTable.forPath(spark, path).detail().collect()[0]
    return {
        "num_files": int(detail["numFiles"]),
        "size_bytes": int(detail["sizeInBytes"]),
    }


def read_workloads(
    spark: SparkSession,
    research_matrix_root: str,
    performance_workload_result: str,
    selected_ids: Sequence[str],
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
        )
    )

    rows = (
        matrix.join(manifest, on="family_id", how="inner")
        .filter(F.col("workload_id").isin(list(selected_ids)))
        .collect()
    )

    by_id = {
        str(row["workload_id"]): row.asDict(recursive=True)
        for row in rows
    }

    missing = [workload_id for workload_id in selected_ids if workload_id not in by_id]
    if missing:
        raise RuntimeError(
            "Selected workloads were not found in the performance matrix: "
            + ", ".join(missing)
        )

    return [by_id[workload_id] for workload_id in selected_ids]


def load_initial_state_summary(
    spark: SparkSession,
    path: str,
    expected_key_buckets: int,
) -> Dict[str, int]:
    rows = (
        spark.read.option("header", "true")
        .csv(path)
        .select(
            F.col("key_bucket_count").cast("int").alias("key_bucket_count"),
            F.col("duplicate_extra_rows")
            .cast("long")
            .alias("duplicate_extra_rows"),
            F.col("key_frequency_size_bytes")
            .cast("long")
            .alias("key_frequency_size_bytes"),
        )
        .collect()
    )

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one initial-state summary row; found {len(rows)}."
        )

    row = rows[0]
    key_bucket_count = int(row["key_bucket_count"])
    if key_bucket_count != expected_key_buckets:
        raise RuntimeError(
            "S12 key-bucket count mismatch: "
            f"expected={expected_key_buckets}, actual={key_bucket_count}."
        )

    return {
        "key_bucket_count": key_bucket_count,
        "duplicate_extra_rows": int(row["duplicate_extra_rows"]),
        "key_frequency_size_bytes": int(row["key_frequency_size_bytes"]),
    }


def load_ground_truth(
    spark: SparkSession,
    ground_truth_root: str,
    selected_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{ground_truth_root}/ground_truth_rules_csv")
        .filter(F.col("workload_id").isin(list(selected_ids)))
        .filter(F.col("rule_id").isin(["R12", "R13"]))
        .select(
            "workload_id",
            "rule_id",
            "metric_type",
            F.col("long_value").cast("long").alias("long_value"),
            F.col("boolean_value").cast("boolean").alias("boolean_value"),
        )
        .collect()
    )

    expected_rows = len(selected_ids) * 2
    if len(rows) != expected_rows:
        raise RuntimeError(
            "Ground-truth uniqueness row count mismatch: "
            f"expected={expected_rows}, actual={len(rows)}."
        )

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        workload_id = str(row["workload_id"])
        rule_id = str(row["rule_id"])
        metric_type = str(row["metric_type"])

        if metric_type == "LONG":
            value: Any = int(row["long_value"])
        elif metric_type == "BOOLEAN":
            value = bool(row["boolean_value"])
        else:
            raise RuntimeError(
                f"Unexpected metric type for {workload_id}/{rule_id}: "
                f"{metric_type}."
            )

        result.setdefault(workload_id, {})[rule_id] = value

    for workload_id in selected_ids:
        metrics = result.get(workload_id, {})
        if set(metrics) != {"R12", "R13"}:
            raise RuntimeError(
                f"{workload_id} does not contain both R12 and R13 ground truth."
            )

    return result


def compact_net_cdf(
    spark: SparkSession,
    working_path: str,
    target_version: int,
    expected_cdf_rows: int,
) -> Tuple[DataFrame, Dict[str, int]]:
    start = time.perf_counter()

    raw = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", 1)
        .option("endingVersion", target_version)
        .load(working_path)
        .select(
            "row_id",
            "trip_key",
            "_change_type",
            "_commit_version",
        )
        .withColumn(
            "_type_order",
            F.when(
                F.col("_change_type").isin("delete", "update_preimage"),
                F.lit(0),
            ).otherwise(F.lit(1)),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    raw_count = raw.count()
    if raw_count != expected_cdf_rows:
        raw.unpersist()
        raise RuntimeError(
            "CDF row count mismatch: "
            f"expected={expected_cdf_rows}, actual={raw_count}."
        )

    from pyspark.sql import Window

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
        .filter(F.col("_change_type").isin("delete", "update_preimage"))
        .select("row_id", "trip_key")
        .withColumn("_sign", F.lit(-1).cast("long"))
    )

    final_images = (
        raw.withColumn("_rn", F.row_number().over(descending))
        .filter(F.col("_rn") == 1)
        .filter(F.col("_change_type").isin("insert", "update_postimage"))
        .select("row_id", "trip_key")
        .withColumn("_sign", F.lit(1).cast("long"))
    )

    net_images = (
        first_images.unionByName(final_images)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    net_image_count = net_images.count()
    raw.unpersist()

    return net_images, {
        "raw_cdf_rows": raw_count,
        "net_image_count": net_image_count,
        "cdf_prepare_seconds": float(time.perf_counter() - start),
    }


def exact_full_snapshot_uniqueness(
    spark: SparkSession,
    working_path: str,
    target_version: int,
) -> Tuple[Dict[str, Any], float]:
    start = time.perf_counter()

    row = (
        spark.read.format("delta")
        .option("versionAsOf", target_version)
        .load(working_path)
        .select("trip_key")
        .groupBy("trip_key")
        .agg(F.count(F.lit(1)).cast("long").alias("frequency"))
        .agg(
            F.sum(
                F.greatest(
                    F.col("frequency") - F.lit(1),
                    F.lit(0),
                )
            )
            .cast("long")
            .alias("duplicate_extra_rows"),
            F.sum(
                F.when(F.col("frequency") > 1, 1).otherwise(0)
            )
            .cast("long")
            .alias("duplicate_key_groups"),
        )
        .collect()[0]
    )

    seconds = time.perf_counter() - start
    duplicate_extra_rows = int(row["duplicate_extra_rows"] or 0)

    return {
        "R12": duplicate_extra_rows,
        "R13": duplicate_extra_rows == 0,
        "duplicate_key_groups": int(row["duplicate_key_groups"] or 0),
        "affected_key_count": None,
        "affected_key_bucket_count": None,
        "affected_key_bucket_fraction": None,
    }, float(seconds)


def exact_s12_incremental_uniqueness(
    spark: SparkSession,
    net_images: DataFrame,
    key_frequency_state: str,
    baseline_duplicate_extra: int,
    key_buckets: int,
) -> Tuple[Dict[str, Any], float]:
    start = time.perf_counter()

    affected_key_deltas = (
        net_images.groupBy("trip_key")
        .agg(F.sum("_sign").cast("long").alias("delta_frequency"))
        .filter(F.col("delta_frequency") != 0)
        .withColumn(
            "key_bucket",
            F.pmod(
                F.xxhash64("trip_key"),
                F.lit(key_buckets),
            ).cast("int"),
        )
        .select("key_bucket", "trip_key", "delta_frequency")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    affected_key_count = affected_key_deltas.count()
    affected_key_buckets = sorted(
        int(row["key_bucket"])
        for row in (
            affected_key_deltas.select("key_bucket")
            .distinct()
            .collect()
        )
    )

    base_state = spark.read.format("delta").load(key_frequency_state)

    if affected_key_buckets:
        base_subset = base_state.filter(
            F.col("key_bucket").isin(affected_key_buckets)
        )
    else:
        base_subset = base_state.limit(0)

    overlay = (
        affected_key_deltas.join(
            base_subset.select(
                "key_bucket",
                "trip_key",
                F.col("frequency").cast("long").alias("base_frequency"),
            ),
            on=["key_bucket", "trip_key"],
            how="left",
        )
        .fillna({"base_frequency": 0})
        .withColumn(
            "final_frequency",
            F.col("base_frequency") + F.col("delta_frequency"),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    invalid_count = (
        overlay.filter(F.col("final_frequency") < 0)
        .limit(1)
        .count()
    )
    if invalid_count:
        overlay.unpersist()
        affected_key_deltas.unpersist()
        raise RuntimeError("S12 overlay produced a negative key frequency.")

    adjustment = (
        overlay.agg(
            F.sum(
                F.greatest(
                    F.col("final_frequency") - F.lit(1),
                    F.lit(0),
                )
                - F.greatest(
                    F.col("base_frequency") - F.lit(1),
                    F.lit(0),
                )
            )
            .cast("long")
            .alias("duplicate_extra_delta"),
            F.sum(
                (
                    F.when(F.col("final_frequency") > 1, 1).otherwise(0)
                    - F.when(F.col("base_frequency") > 1, 1).otherwise(0)
                )
            )
            .cast("long")
            .alias("duplicate_group_delta"),
        )
        .collect()[0]
    )

    duplicate_extra_rows = (
        baseline_duplicate_extra
        + int(adjustment["duplicate_extra_delta"] or 0)
    )

    duplicate_group_delta = int(
        adjustment["duplicate_group_delta"] or 0
    )

    seconds = time.perf_counter() - start

    overlay.unpersist()
    affected_key_deltas.unpersist()

    return {
        "R12": duplicate_extra_rows,
        "R13": duplicate_extra_rows == 0,
        "duplicate_key_groups_delta": duplicate_group_delta,
        "affected_key_count": affected_key_count,
        "affected_key_bucket_count": len(affected_key_buckets),
        "affected_key_bucket_fraction": (
            len(affected_key_buckets) / float(key_buckets)
        ),
    }, float(seconds)


def assert_exact(
    workload_id: str,
    method: str,
    observed: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    if int(observed["R12"]) != int(expected["R12"]):
        raise RuntimeError(
            f"{workload_id}/{method} R12 mismatch: "
            f"observed={observed['R12']}, expected={expected['R12']}."
        )
    if bool(observed["R13"]) != bool(expected["R13"]):
        raise RuntimeError(
            f"{workload_id}/{method} R13 mismatch: "
            f"observed={observed['R13']}, expected={expected['R13']}."
        )


def main() -> None:
    args = parse_arguments()

    if args.timed_repeats <= 0:
        raise ValueError("--timed-repeats must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.key_buckets <= 0:
        raise ValueError("--key-buckets must be positive.")

    selected_ids = parse_csv_set(
        args.selected_workloads,
        "--selected-workloads",
    )

    research_root = args.research_matrix_root.rstrip("/")
    performance_root = args.performance_workload_result.rstrip("/")
    key_state_path = args.key_frequency_state.rstrip("/")
    initial_summary_path = args.initial_state_summary.rstrip("/")
    ground_truth_root = args.ground_truth_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    for name, value in [
        ("research_matrix_root", research_root),
        ("performance_workload_result", performance_root),
        ("key_frequency_state", key_state_path),
        ("initial_state_summary", initial_summary_path),
        ("ground_truth_root", ground_truth_root),
        ("output_root", output_root),
    ]:
        if not value:
            raise ValueError(f"{name} cannot be empty.")

    spark = (
        SparkSession.builder.appName("StateGuardS12UniquenessAblation")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "256")

    if not DeltaTable.isDeltaTable(spark, key_state_path):
        raise RuntimeError(
            f"Missing S12 exact key-frequency Delta state: {key_state_path}"
        )

    state_metrics = delta_metrics(spark, key_state_path)
    state_summary = load_initial_state_summary(
        spark,
        initial_summary_path,
        args.key_buckets,
    )

    measured_s12_bytes = int(state_summary["key_frequency_size_bytes"])
    if state_metrics["size_bytes"] != measured_s12_bytes:
        raise RuntimeError(
            "S12 size mismatch between Delta detail and initial-state summary: "
            f"detail={state_metrics['size_bytes']}, "
            f"summary={measured_s12_bytes}."
        )

    if (
        args.expected_s12_bytes > 0
        and measured_s12_bytes != args.expected_s12_bytes
    ):
        raise RuntimeError(
            "S12 size changed from the previously measured value: "
            f"expected={args.expected_s12_bytes}, actual={measured_s12_bytes}."
        )

    workloads = read_workloads(
        spark,
        research_root,
        performance_root,
        selected_ids,
    )
    ground_truth = load_ground_truth(
        spark,
        ground_truth_root,
        selected_ids,
    )

    baseline_duplicate_extra = int(
        state_summary["duplicate_extra_rows"]
    )

    print("=" * 80)
    print("STATEGUARD S12 UNIQUENESS ABLATION")
    print("=" * 80)
    print(
        "Question: does spending ~2.2 GiB on S12 reduce the exact "
        "R12/R13 uniqueness cost versus full-snapshot fallback?"
    )
    print(f"Selected workloads: {','.join(selected_ids)}")
    print(f"Timed repeats per method: {args.timed_repeats}")
    print(f"Warm-up runs per method: {args.warmup_runs}")
    print(f"S12 key buckets: {args.key_buckets}")
    print(f"S12 state bytes: {measured_s12_bytes}")
    print(f"Compact S01-S11 bytes: {args.compact_portfolio_bytes}")
    print("=" * 80)

    trial_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []
    mismatch_count = 0

    for index, workload in enumerate(workloads, start=1):
        workload_id = str(workload["workload_id"])
        working_path = str(workload["working_path"])
        target_version = int(workload["target_version"])
        operation_count = int(workload["cumulative_operations"])
        expected_cdf_rows = int(
            workload["expected_cumulative_cdf_rows"]
        )
        truth = ground_truth[workload_id]

        print(
            f"[{index:02d}/{len(workloads):02d}] "
            f"{workload_id} operations={operation_count}"
        )

        net_images, cdf_diag = compact_net_cdf(
            spark,
            working_path,
            target_version,
            expected_cdf_rows,
        )

        def execute_method(
            method: str,
        ) -> Tuple[Dict[str, Any], float]:
            if method == "S12_INCREMENTAL":
                return exact_s12_incremental_uniqueness(
                    spark,
                    net_images,
                    key_state_path,
                    baseline_duplicate_extra,
                    args.key_buckets,
                )
            if method == "FULL_SNAPSHOT_FALLBACK":
                return exact_full_snapshot_uniqueness(
                    spark,
                    working_path,
                    target_version,
                )
            raise ValueError(f"Unknown method: {method}")

        # Warm both paths before measured trials.
        for warmup_index in range(1, args.warmup_runs + 1):
            for method in [
                "S12_INCREMENTAL",
                "FULL_SNAPSHOT_FALLBACK",
            ]:
                observed, seconds = execute_method(method)
                assert_exact(workload_id, method, observed, truth)

                trial_records.append(
                    {
                        "workload_id": workload_id,
                        "family_id": str(workload["family_id"]),
                        "operation_count": operation_count,
                        "target_version": target_version,
                        "method": method,
                        "trial_index": warmup_index,
                        "is_warmup": True,
                        "seconds": seconds,
                        "R12_duplicate_extra_rows": int(observed["R12"]),
                        "R13_uniqueness_pass": bool(observed["R13"]),
                        "exact_match": True,
                        "raw_cdf_rows": int(cdf_diag["raw_cdf_rows"]),
                        "net_image_count": int(cdf_diag["net_image_count"]),
                        "cdf_prepare_seconds": float(
                            cdf_diag["cdf_prepare_seconds"]
                        ),
                        "affected_key_count": (
                            int(observed["affected_key_count"])
                            if observed.get("affected_key_count") is not None
                            else None
                        ),
                        "affected_key_bucket_count": (
                            int(observed["affected_key_bucket_count"])
                            if observed.get("affected_key_bucket_count")
                            is not None
                            else None
                        ),
                        "affected_key_bucket_fraction": (
                            float(observed["affected_key_bucket_fraction"])
                            if observed.get("affected_key_bucket_fraction")
                            is not None
                            else None
                        ),
                    }
                )

        method_times: Dict[str, List[float]] = {
            "S12_INCREMENTAL": [],
            "FULL_SNAPSHOT_FALLBACK": [],
        }
        representative_s12: Dict[str, Any] = {}

        # Alternate method order to reduce systematic order bias.
        for trial_index in range(1, args.timed_repeats + 1):
            if trial_index % 2 == 1:
                order = [
                    "S12_INCREMENTAL",
                    "FULL_SNAPSHOT_FALLBACK",
                ]
            else:
                order = [
                    "FULL_SNAPSHOT_FALLBACK",
                    "S12_INCREMENTAL",
                ]

            for method in order:
                observed, seconds = execute_method(method)
                try:
                    assert_exact(workload_id, method, observed, truth)
                except RuntimeError:
                    mismatch_count += 1
                    raise

                method_times[method].append(seconds)
                if method == "S12_INCREMENTAL":
                    representative_s12 = observed

                trial_records.append(
                    {
                        "workload_id": workload_id,
                        "family_id": str(workload["family_id"]),
                        "operation_count": operation_count,
                        "target_version": target_version,
                        "method": method,
                        "trial_index": trial_index,
                        "is_warmup": False,
                        "seconds": seconds,
                        "R12_duplicate_extra_rows": int(observed["R12"]),
                        "R13_uniqueness_pass": bool(observed["R13"]),
                        "exact_match": True,
                        "raw_cdf_rows": int(cdf_diag["raw_cdf_rows"]),
                        "net_image_count": int(cdf_diag["net_image_count"]),
                        "cdf_prepare_seconds": float(
                            cdf_diag["cdf_prepare_seconds"]
                        ),
                        "affected_key_count": (
                            int(observed["affected_key_count"])
                            if observed.get("affected_key_count") is not None
                            else None
                        ),
                        "affected_key_bucket_count": (
                            int(observed["affected_key_bucket_count"])
                            if observed.get("affected_key_bucket_count")
                            is not None
                            else None
                        ),
                        "affected_key_bucket_fraction": (
                            float(observed["affected_key_bucket_fraction"])
                            if observed.get("affected_key_bucket_fraction")
                            is not None
                            else None
                        ),
                    }
                )

        s12_median = float(
            statistics.median(method_times["S12_INCREMENTAL"])
        )
        fallback_median = float(
            statistics.median(
                method_times["FULL_SNAPSHOT_FALLBACK"]
            )
        )
        speedup = (
            fallback_median / s12_median
            if s12_median > 0
            else float("inf")
        )

        summary_records.append(
            {
                "workload_id": workload_id,
                "family_id": str(workload["family_id"]),
                "operation_count": operation_count,
                "target_version": target_version,
                "median_s12_incremental_seconds": s12_median,
                "median_full_snapshot_fallback_seconds": fallback_median,
                "s12_speedup_vs_fallback": speedup,
                "affected_key_count": int(
                    representative_s12.get("affected_key_count") or 0
                ),
                "affected_key_bucket_count": int(
                    representative_s12.get(
                        "affected_key_bucket_count"
                    )
                    or 0
                ),
                "affected_key_bucket_fraction": float(
                    representative_s12.get(
                        "affected_key_bucket_fraction"
                    )
                    or 0.0
                ),
                "R12_duplicate_extra_rows": int(truth["R12"]),
                "R13_uniqueness_pass": bool(truth["R13"]),
                "exact_match_count": 2,
            }
        )

        print(
            f"  S12={s12_median:.3f}s | "
            f"fallback={fallback_median:.3f}s | "
            f"S12 speedup={speedup:.3f}x | "
            f"buckets={summary_records[-1]['affected_key_bucket_count']}"
            f"/{args.key_buckets} | exact=2/2"
        )

        net_images.unpersist()

    if mismatch_count != 0:
        raise RuntimeError(
            f"S12 ablation found {mismatch_count} correctness mismatches."
        )

    trial_schema = T.StructType(
        [
            T.StructField("workload_id", T.StringType(), False),
            T.StructField("family_id", T.StringType(), False),
            T.StructField("operation_count", T.LongType(), False),
            T.StructField("target_version", T.IntegerType(), False),
            T.StructField("method", T.StringType(), False),
            T.StructField("trial_index", T.IntegerType(), False),
            T.StructField("is_warmup", T.BooleanType(), False),
            T.StructField("seconds", T.DoubleType(), False),
            T.StructField("R12_duplicate_extra_rows", T.LongType(), False),
            T.StructField("R13_uniqueness_pass", T.BooleanType(), False),
            T.StructField("exact_match", T.BooleanType(), False),
            T.StructField("raw_cdf_rows", T.LongType(), False),
            T.StructField("net_image_count", T.LongType(), False),
            T.StructField("cdf_prepare_seconds", T.DoubleType(), False),
            T.StructField("affected_key_count", T.LongType(), True),
            T.StructField("affected_key_bucket_count", T.LongType(), True),
            T.StructField(
                "affected_key_bucket_fraction",
                T.DoubleType(),
                True,
            ),
        ]
    )

    method_summary_schema = T.StructType(
        [
            T.StructField("workload_id", T.StringType(), False),
            T.StructField("family_id", T.StringType(), False),
            T.StructField("operation_count", T.LongType(), False),
            T.StructField("target_version", T.IntegerType(), False),
            T.StructField(
                "median_s12_incremental_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "median_full_snapshot_fallback_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "s12_speedup_vs_fallback",
                T.DoubleType(),
                False,
            ),
            T.StructField("affected_key_count", T.LongType(), False),
            T.StructField(
                "affected_key_bucket_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "affected_key_bucket_fraction",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "R12_duplicate_extra_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "R13_uniqueness_pass",
                T.BooleanType(),
                False,
            ),
            T.StructField("exact_match_count", T.IntegerType(), False),
        ]
    )

    trial_df = spark.createDataFrame(
        [
            (
                row["workload_id"],
                row["family_id"],
                row["operation_count"],
                row["target_version"],
                row["method"],
                row["trial_index"],
                row["is_warmup"],
                row["seconds"],
                row["R12_duplicate_extra_rows"],
                row["R13_uniqueness_pass"],
                row["exact_match"],
                row["raw_cdf_rows"],
                row["net_image_count"],
                row["cdf_prepare_seconds"],
                row["affected_key_count"],
                row["affected_key_bucket_count"],
                row["affected_key_bucket_fraction"],
            )
            for row in trial_records
        ],
        schema=trial_schema,
    )

    method_summary_df = spark.createDataFrame(
        [
            (
                row["workload_id"],
                row["family_id"],
                row["operation_count"],
                row["target_version"],
                row["median_s12_incremental_seconds"],
                row["median_full_snapshot_fallback_seconds"],
                row["s12_speedup_vs_fallback"],
                row["affected_key_count"],
                row["affected_key_bucket_count"],
                row["affected_key_bucket_fraction"],
                row["R12_duplicate_extra_rows"],
                row["R13_uniqueness_pass"],
                row["exact_match_count"],
            )
            for row in summary_records
        ],
        schema=method_summary_schema,
    )

    speedups = [
        float(row["s12_speedup_vs_fallback"])
        for row in summary_records
    ]
    bucket_fractions = [
        float(row["affected_key_bucket_fraction"])
        for row in summary_records
    ]

    storage_multiplier = (
        measured_s12_bytes / float(args.compact_portfolio_bytes)
        if args.compact_portfolio_bytes > 0
        else float("inf")
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("workload_condition_count", T.IntegerType(), False),
            T.StructField("method_count", T.IntegerType(), False),
            T.StructField("warmup_runs_per_method", T.IntegerType(), False),
            T.StructField("timed_repeats_per_method", T.IntegerType(), False),
            T.StructField("timed_execution_count", T.IntegerType(), False),
            T.StructField(
                "timed_rule_comparison_count",
                T.IntegerType(),
                False,
            ),
            T.StructField("rule_mismatch_count", T.IntegerType(), False),
            T.StructField(
                "median_s12_speedup_vs_fallback",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "minimum_s12_speedup_vs_fallback",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "maximum_s12_speedup_vs_fallback",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "median_affected_key_bucket_fraction",
                T.DoubleType(),
                False,
            ),
            T.StructField("s12_state_size_bytes", T.LongType(), False),
            T.StructField(
                "compact_portfolio_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "s12_to_compact_storage_multiplier",
                T.DoubleType(),
                False,
            ),
        ]
    )

    condition_count = len(workloads)
    timed_execution_count = (
        condition_count * 2 * args.timed_repeats
    )
    timed_rule_comparison_count = timed_execution_count * 2

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                condition_count,
                2,
                args.warmup_runs,
                args.timed_repeats,
                timed_execution_count,
                timed_rule_comparison_count,
                0,
                float(statistics.median(speedups)),
                float(min(speedups)),
                float(max(speedups)),
                float(statistics.median(bucket_fractions)),
                measured_s12_bytes,
                int(args.compact_portfolio_bytes),
                float(storage_multiplier),
            )
        ],
        schema=summary_schema,
    )

    write_csv(trial_df, f"{output_root}/trial_results_csv")
    write_csv(
        method_summary_df,
        f"{output_root}/method_summary_csv",
    )
    write_csv(summary_df, f"{output_root}/summary_csv")

    print()
    print("=" * 80)
    print("S12_ABLATION_BEGIN")
    print("S12_ABLATION_STATUS=PASS")
    print(f"WORKLOAD_CONDITION_COUNT={condition_count}")
    print("METHOD_COUNT=2")
    print(f"WARMUP_RUNS_PER_METHOD={args.warmup_runs}")
    print(f"TIMED_REPEATS_PER_METHOD={args.timed_repeats}")
    print(f"TIMED_EXECUTION_COUNT={timed_execution_count}")
    print(
        "TIMED_RULE_COMPARISON_COUNT="
        f"{timed_rule_comparison_count}"
    )
    print("RULE_MISMATCH_COUNT=0")
    print(
        "MEDIAN_S12_SPEEDUP_VS_FALLBACK="
        f"{statistics.median(speedups):.6f}"
    )
    print(
        "MINIMUM_S12_SPEEDUP_VS_FALLBACK="
        f"{min(speedups):.6f}"
    )
    print(
        "MAXIMUM_S12_SPEEDUP_VS_FALLBACK="
        f"{max(speedups):.6f}"
    )
    print(
        "MEDIAN_AFFECTED_KEY_BUCKET_FRACTION="
        f"{statistics.median(bucket_fractions):.6f}"
    )
    print(f"S12_STATE_SIZE_BYTES={measured_s12_bytes}")
    print(
        "COMPACT_PORTFOLIO_SIZE_BYTES="
        f"{args.compact_portfolio_bytes}"
    )
    print(
        "S12_TO_COMPACT_STORAGE_MULTIPLIER="
        f"{storage_multiplier:.6f}"
    )
    print(
        "TRIAL_RESULTS_PATH="
        f"{output_root}/trial_results_csv"
    )
    print(
        "METHOD_SUMMARY_PATH="
        f"{output_root}/method_summary_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("S12_ABLATION_END")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
