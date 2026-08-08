import argparse
import time
from typing import List, Tuple

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


RAW_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]

EXPECTED_SCHEMA = T.StructType(
    [
        T.StructField("VendorID", T.IntegerType(), True),
        T.StructField("tpep_pickup_datetime", T.TimestampNTZType(), True),
        T.StructField("tpep_dropoff_datetime", T.TimestampNTZType(), True),
        T.StructField("passenger_count", T.LongType(), True),
        T.StructField("trip_distance", T.DoubleType(), True),
        T.StructField("RatecodeID", T.LongType(), True),
        T.StructField("store_and_fwd_flag", T.StringType(), True),
        T.StructField("PULocationID", T.IntegerType(), True),
        T.StructField("DOLocationID", T.IntegerType(), True),
        T.StructField("payment_type", T.LongType(), True),
        T.StructField("fare_amount", T.DoubleType(), True),
        T.StructField("extra", T.DoubleType(), True),
        T.StructField("mta_tax", T.DoubleType(), True),
        T.StructField("tip_amount", T.DoubleType(), True),
        T.StructField("tolls_amount", T.DoubleType(), True),
        T.StructField("improvement_surcharge", T.DoubleType(), True),
        T.StructField("total_amount", T.DoubleType(), True),
        T.StructField("congestion_surcharge", T.DoubleType(), True),
        T.StructField("Airport_fee", T.DoubleType(), True),
        T.StructField("cbd_congestion_fee", T.DoubleType(), True),
    ]
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable 17-month StateGuard canonical Delta table."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="GCS root containing the 2025 and 2026 raw folders.",
    )
    parser.add_argument(
        "--profile-summary",
        required=True,
        help="GCS path to the successful profiler summary_csv directory.",
    )
    parser.add_argument(
        "--delta-path",
        required=True,
        help="GCS destination for the canonical Delta table.",
    )
    parser.add_argument(
        "--result-root",
        required=True,
        help="GCS destination for canonical-build metadata.",
    )
    parser.add_argument(
        "--state-partitions",
        type=int,
        default=64,
        help="Number of logical StateGuard partitions. Default: 64.",
    )
    return parser.parse_args()


def expected_paths(raw_root: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []

    for month in range(1, 13):
        year_month = f"2025-{month:02d}"
        files.append(
            (
                year_month,
                f"{raw_root}/2025/yellow_tripdata_{year_month}.parquet",
            )
        )

    for month in range(1, 6):
        year_month = f"2026-{month:02d}"
        files.append(
            (
                year_month,
                f"{raw_root}/2026/yellow_tripdata_{year_month}.parquet",
            )
        )

    return files


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def main() -> None:
    args = parse_arguments()

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardCanonicalDeltaBuilder")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.state_partitions))

    extensions = spark.conf.get("spark.sql.extensions", "")
    catalog = spark.conf.get("spark.sql.catalog.spark_catalog", "")

    if "DeltaSparkSessionExtension" not in extensions:
        raise RuntimeError(
            "DeltaSparkSessionExtension is not enabled on this Spark cluster."
        )

    if "DeltaCatalog" not in catalog:
        raise RuntimeError(
            "DeltaCatalog is not enabled on this Spark cluster."
        )

    raw_root = args.raw_root.rstrip("/")
    delta_path = args.delta_path.rstrip("/")
    result_root = args.result_root.rstrip("/")

    profile_row = (
        spark.read.option("header", "true")
        .csv(args.profile_summary)
        .select(
            F.col("file_count").cast("long").alias("file_count"),
            F.col("total_rows").cast("long").alias("total_rows"),
            F.col("schema_mismatch_count")
            .cast("long")
            .alias("schema_mismatch_count"),
        )
        .collect()
    )

    if len(profile_row) != 1:
        raise RuntimeError(
            "Expected exactly one row in the profiler summary."
        )

    profile = profile_row[0]
    expected_file_count = int(profile["file_count"])
    expected_row_count = int(profile["total_rows"])
    schema_mismatch_count = int(profile["schema_mismatch_count"])

    if expected_file_count != 17:
        raise RuntimeError(
            f"Profiler reports {expected_file_count} files, expected 17."
        )

    if schema_mismatch_count != 0:
        raise RuntimeError(
            "Profiler summary contains schema mismatches; canonical build stopped."
        )

    files = expected_paths(raw_root)
    source_paths = [path for _, path in files]

    print("=" * 78)
    print("STATEGUARD CANONICAL DELTA BUILD")
    print("=" * 78)
    print(f"Raw root: {raw_root}")
    print(f"Canonical path: {delta_path}")
    print(f"Expected files: {expected_file_count}")
    print(f"Expected rows: {expected_row_count}")
    print(f"Logical StateGuard partitions: {args.state_partitions}")
    print(f"Spark version: {spark.version}")
    print(f"Delta extensions: {extensions}")
    print(f"Delta catalog: {catalog}")
    print("=" * 78)

    raw_df = (
        spark.read
        .schema(EXPECTED_SCHEMA)
        .parquet(*source_paths)
        .withColumn(
            "source_file",
            F.regexp_extract(F.input_file_name(), r"([^/]+)$", 1),
        )
        .withColumn(
            "source_year_month",
            F.regexp_extract(
                F.col("source_file"),
                r"yellow_tripdata_(\d{4}-\d{2})\.parquet",
                1,
            ),
        )
        .withColumn(
            "source_year",
            F.substring("source_year_month", 1, 4).cast("int"),
        )
        .withColumn(
            "source_month",
            F.substring("source_year_month", 6, 2).cast("int"),
        )
    )

    # Exact-record key used by duplicate-count and uniqueness rules.
    record_json = F.to_json(
        F.struct(*[F.col(name) for name in RAW_COLUMNS]),
        options={"ignoreNullFields": "false"},
    )

    enriched_df = (
        raw_df
        .withColumn("trip_key", F.sha2(record_json, 256))
        .withColumn("_canonical_nonce", F.monotonically_increasing_id())
        .withColumn(
            "row_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.col("source_file"),
                    F.col("trip_key"),
                    F.col("_canonical_nonce").cast("string"),
                ),
                256,
            ),
        )
        .withColumn(
            "state_partition_id",
            F.pmod(
                F.xxhash64("trip_key"),
                F.lit(args.state_partitions),
            ).cast("int"),
        )
        .drop("_canonical_nonce")
        .select(
            "row_id",
            "trip_key",
            "state_partition_id",
            "source_file",
            "source_year_month",
            "source_year",
            "source_month",
            *RAW_COLUMNS,
        )
    )

    build_start = time.perf_counter()

    (
        enriched_df.repartition(17, "source_year_month")
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .partitionBy("source_year_month")
        .save(delta_path)
    )

    build_seconds = time.perf_counter() - build_start

    canonical_df = spark.read.format("delta").load(delta_path)

    verification = (
        canonical_df.agg(
            F.count(F.lit(1)).cast("long").alias("actual_rows"),
            F.countDistinct("source_year_month")
            .cast("long")
            .alias("month_count"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("state_partition_count"),
            F.sum(
                F.when(F.col("row_id").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_row_ids"),
            F.sum(
                F.when(F.col("trip_key").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_trip_keys"),
            F.sum(
                F.when(
                    F.length(F.col("source_year_month")) != 7,
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_source_month_rows"),
            F.min("source_year_month").alias("first_source_month"),
            F.max("source_year_month").alias("last_source_month"),
        )
        .collect()[0]
    )

    delta_table = DeltaTable.forPath(spark, delta_path)
    detail = delta_table.detail().collect()[0]
    history = delta_table.history(1).collect()[0]

    properties = detail["properties"] or {}
    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    actual_rows = int(verification["actual_rows"])
    month_count = int(verification["month_count"])
    state_partition_count = int(verification["state_partition_count"])
    null_row_ids = int(verification["null_row_ids"])
    null_trip_keys = int(verification["null_trip_keys"])
    invalid_source_month_rows = int(
        verification["invalid_source_month_rows"]
    )

    errors = []

    if actual_rows != expected_row_count:
        errors.append(
            f"row count mismatch: expected={expected_row_count}, "
            f"actual={actual_rows}"
        )

    if month_count != 17:
        errors.append(f"expected 17 source months, found {month_count}")

    if state_partition_count != args.state_partitions:
        errors.append(
            f"expected {args.state_partitions} logical partitions, "
            f"found {state_partition_count}"
        )

    if null_row_ids != 0:
        errors.append(f"found {null_row_ids} null row IDs")

    if null_trip_keys != 0:
        errors.append(f"found {null_trip_keys} null trip keys")

    if invalid_source_month_rows != 0:
        errors.append(
            f"found {invalid_source_month_rows} invalid source-month rows"
        )

    if not cdf_enabled:
        errors.append("Delta Change Data Feed is not enabled")

    if errors:
        print("CANONICAL BUILD VERIFICATION FAILED")
        for error in errors:
            print(f"- {error}")
        spark.stop()
        raise RuntimeError("; ".join(errors))

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("expected_rows", T.LongType(), False),
            T.StructField("actual_rows", T.LongType(), False),
            T.StructField("month_count", T.LongType(), False),
            T.StructField("state_partition_count", T.LongType(), False),
            T.StructField("null_row_ids", T.LongType(), False),
            T.StructField("null_trip_keys", T.LongType(), False),
            T.StructField(
                "invalid_source_month_rows",
                T.LongType(),
                False,
            ),
            T.StructField("first_source_month", T.StringType(), False),
            T.StructField("last_source_month", T.StringType(), False),
            T.StructField("delta_version", T.LongType(), False),
            T.StructField("cdf_enabled", T.BooleanType(), False),
            T.StructField("num_data_files", T.LongType(), False),
            T.StructField("delta_size_bytes", T.LongType(), False),
            T.StructField("build_seconds", T.DoubleType(), False),
            T.StructField("spark_version", T.StringType(), False),
            T.StructField("delta_path", T.StringType(), False),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                expected_row_count,
                actual_rows,
                month_count,
                state_partition_count,
                null_row_ids,
                null_trip_keys,
                invalid_source_month_rows,
                verification["first_source_month"],
                verification["last_source_month"],
                int(history["version"]),
                cdf_enabled,
                int(detail["numFiles"]),
                int(detail["sizeInBytes"]),
                float(build_seconds),
                spark.version,
                delta_path,
            )
        ],
        schema=summary_schema,
    )

    write_csv(summary_df, f"{result_root}/summary_csv")

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("STATEGUARD_CANONICAL_BUILD_BEGIN")
    print("BUILD_STATUS=PASS")
    print(f"EXPECTED_ROWS={expected_row_count}")
    print(f"ACTUAL_ROWS={actual_rows}")
    print(f"MONTH_COUNT={month_count}")
    print(f"FIRST_SOURCE_MONTH={verification['first_source_month']}")
    print(f"LAST_SOURCE_MONTH={verification['last_source_month']}")
    print(f"STATE_PARTITION_COUNT={state_partition_count}")
    print(f"NULL_ROW_IDS={null_row_ids}")
    print(f"NULL_TRIP_KEYS={null_trip_keys}")
    print(f"DELTA_VERSION={int(history['version'])}")
    print(f"CDF_ENABLED={str(cdf_enabled).lower()}")
    print(f"NUM_DATA_FILES={int(detail['numFiles'])}")
    print(f"DELTA_SIZE_BYTES={int(detail['sizeInBytes'])}")
    print(f"BUILD_SECONDS={build_seconds:.3f}")
    print(f"SPARK_VERSION={spark.version}")
    print(f"CANONICAL_DELTA_PATH={delta_path}")
    print(f"BUILD_SUMMARY_PATH={result_root}/summary_csv")
    print("STATEGUARD_CANONICAL_BUILD_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
