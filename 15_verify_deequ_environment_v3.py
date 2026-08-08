import argparse
import importlib
import os
import subprocess
import sys
import time
from typing import Dict, List

# PyDeequ selects its Deequ coordinate from this environment variable.
os.environ.setdefault("SPARK_VERSION", "3.5")


def ensure_pydeequ() -> None:
    target = "/tmp/stateguard_pydeequ_1_6_0"

    if target not in sys.path:
        sys.path.insert(0, target)

    try:
        import pydeequ  # noqa: F401
        return
    except ModuleNotFoundError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                target,
                "pydeequ==1.6.0",
            ]
        )
        importlib.invalidate_caches()
        import pydeequ  # noqa: F401


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly verify the official PyDeequ/Deequ implementation "
            "on the existing Spark 3.5 Delta Lake environment."
        )
    )
    parser.add_argument("--canonical-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-rows", type=int, default=100000)
    parser.add_argument("--expected-canonical-version", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.sample_rows < 1000:
        raise ValueError("--sample-rows must be at least 1000.")

    ensure_pydeequ()

    import pydeequ
    from delta.tables import DeltaTable
    from pydeequ.analyzers import (
        AnalysisRunner,
        AnalyzerContext,
        Completeness,
        Compliance,
        Maximum,
        Minimum,
        Size,
        Uniqueness,
    )
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder
        .appName("StateGuardVerifyOfficialDeequ")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")

    canonical_path = args.canonical_path.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, canonical_path):
        raise RuntimeError(f"Not a Delta table: {canonical_path}")

    canonical_version = int(
        DeltaTable.forPath(spark, canonical_path)
        .history(1)
        .collect()[0]["version"]
    )

    if canonical_version != args.expected_canonical_version:
        raise RuntimeError(
            "Canonical version mismatch: "
            f"expected={args.expected_canonical_version}, "
            f"actual={canonical_version}"
        )

    # The analyzer execution below is the definitive compatibility test.
    # Dataproc may localize --jars without exposing the URI through
    # SparkConf, so no separate classpath pre-check is used here.

    sample_start = time.perf_counter()
    sample = (
        spark.read.format("delta")
        .option("versionAsOf", canonical_version)
        .load(canonical_path)
        .select(
            "trip_key",
            "passenger_count",
            "fare_amount",
            "trip_distance",
        )
        .limit(args.sample_rows)
        .cache()
    )
    spark_sample_count = sample.count()
    sample_seconds = time.perf_counter() - sample_start

    if spark_sample_count != args.sample_rows:
        raise RuntimeError(
            f"Expected {args.sample_rows} sample rows; "
            f"found {spark_sample_count}."
        )

    analysis_start = time.perf_counter()

    result = (
        AnalysisRunner(spark)
        .onData(sample)
        .addAnalyzer(Size())
        .addAnalyzer(Completeness("passenger_count"))
        .addAnalyzer(Completeness("fare_amount"))
        .addAnalyzer(
            Compliance(
                "non_negative_fare",
                "fare_amount IS NULL OR fare_amount >= 0",
            )
        )
        .addAnalyzer(
            Compliance(
                "non_negative_distance",
                "trip_distance IS NULL OR trip_distance >= 0",
            )
        )
        .addAnalyzer(Minimum("fare_amount"))
        .addAnalyzer(Maximum("fare_amount"))
        .addAnalyzer(Uniqueness(["trip_key"]))
        .run()
    )

    metrics_df = AnalyzerContext.successMetricsAsDataFrame(
        spark,
        result,
    )

    analysis_seconds = time.perf_counter() - analysis_start

    metrics = metrics_df.collect()

    if not metrics:
        raise RuntimeError("Deequ returned no successful metrics.")

    size_rows = [
        row
        for row in metrics
        if str(row["name"]).lower() == "size"
    ]

    if len(size_rows) != 1:
        raise RuntimeError(
            f"Expected one Deequ Size metric; found {len(size_rows)}."
        )

    deequ_size = int(round(float(size_rows[0]["value"])))

    if deequ_size != spark_sample_count:
        raise RuntimeError(
            "Deequ Size does not match Spark count: "
            f"deequ={deequ_size}, spark={spark_sample_count}"
        )

    output_metrics = (
        metrics_df.withColumn(
            "canonical_version",
            F.lit(canonical_version).cast("long"),
        )
        .withColumn(
            "sample_rows",
            F.lit(spark_sample_count).cast("long"),
        )
        .withColumn(
            "spark_version",
            F.lit(spark.version),
        )
        .withColumn(
            "pydeequ_version",
            F.lit(
                getattr(pydeequ, "__version__", "package-1.6.0")
            ),
        )
        .withColumn(
            "deequ_maven_coordinate",
            F.lit(
                "com.amazon.deequ:deequ:2.0.16-spark-3.5"
            ),
        )
    )

    summary_schema = (
        "status string, canonical_version long, sample_rows long, "
        "successful_metric_count long, spark_count long, deequ_size long, "
        "sample_load_seconds double, deequ_analysis_seconds double, "
        "spark_version string, pydeequ_package string, "
        "deequ_maven_coordinate string, canonical_path string"
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                canonical_version,
                spark_sample_count,
                len(metrics),
                spark_sample_count,
                deequ_size,
                float(sample_seconds),
                float(analysis_seconds),
                spark.version,
                "pydeequ==1.6.0",
                "com.amazon.deequ:deequ:2.0.16-spark-3.5",
                canonical_path,
            )
        ],
        schema=summary_schema,
    )

    (
        output_metrics.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{output_root}/metrics_csv")
    )
    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{output_root}/summary_csv")
    )

    print()
    print("=" * 78)
    print("OFFICIAL DEEQU SMOKE-TEST METRICS")
    print("=" * 78)
    metrics_df.orderBy("entity", "instance", "name").show(
        30,
        truncate=False,
    )

    print("=" * 78)
    print("STATEGUARD_DEEQU_ENVIRONMENT_BEGIN")
    print("DEEQU_ENVIRONMENT_STATUS=PASS")
    print(f"CANONICAL_VERSION={canonical_version}")
    print(f"SAMPLE_ROWS={spark_sample_count}")
    print(f"SUCCESSFUL_METRIC_COUNT={len(metrics)}")
    print(f"SPARK_COUNT={spark_sample_count}")
    print(f"DEEQU_SIZE={deequ_size}")
    print(f"SPARK_VERSION={spark.version}")
    print("PYDEEQU_PACKAGE=pydeequ==1.6.0")
    print(
        "DEEQU_MAVEN_COORDINATE="
        "com.amazon.deequ:deequ:2.0.16-spark-3.5"
    )
    print(f"SAMPLE_LOAD_SECONDS={sample_seconds:.3f}")
    print(f"DEEQU_ANALYSIS_SECONDS={analysis_seconds:.3f}")
    print(f"METRICS_PATH={output_root}/metrics_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_DEEQU_ENVIRONMENT_END")
    print("=" * 78)

    sample.unpersist()
    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


if __name__ == "__main__":
    main()
