import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


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
        description="Profile and validate all 17 StateGuard raw Parquet files."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Cloud Storage root containing the 2025 and 2026 folders.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Cloud Storage root where profiling outputs will be written.",
    )
    return parser.parse_args()


def schema_signature(schema: T.StructType) -> List[Tuple[str, str, bool]]:
    return [
        (field.name, field.dataType.simpleString(), field.nullable)
        for field in schema.fields
    ]


def schema_fingerprint(schema: T.StructType) -> str:
    canonical = json.dumps(
        schema_signature(schema),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_paths(raw_root: str) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []

    for month in range(1, 13):
        year_month = f"2025-{month:02d}"
        path = f"{raw_root}/2025/yellow_tripdata_{year_month}.parquet"
        paths.append((year_month, path))

    for month in range(1, 6):
        year_month = f"2026-{month:02d}"
        path = f"{raw_root}/2026/yellow_tripdata_{year_month}.parquet"
        paths.append((year_month, path))

    return paths


def get_file_size_bytes(spark: SparkSession, path: str) -> int:
    hadoop_configuration = spark.sparkContext._jsc.hadoopConfiguration()
    java_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)
    filesystem = java_path.getFileSystem(hadoop_configuration)
    status = filesystem.getFileStatus(java_path)
    return int(status.getLen())


def write_csv(dataframe: DataFrame, output_path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(output_path)
    )


def main() -> None:
    args = parse_arguments()

    spark = SparkSession.builder.appName("StateGuardRaw17Profiler").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", "64")

    raw_root = args.raw_root.rstrip("/")
    output_root = args.output_root.rstrip("/")
    files = expected_paths(raw_root)

    expected_signature = schema_signature(EXPECTED_SCHEMA)
    expected_fingerprint = schema_fingerprint(EXPECTED_SCHEMA)

    schema_rows: List[Dict[str, object]] = []
    size_rows: List[Tuple[str, str, str, int]] = []

    print("=" * 78)
    print("STATEGUARD RAW 17-FILE PROFILER")
    print("=" * 78)
    print(f"Raw root: {raw_root}")
    print(f"Output root: {output_root}")
    print(f"Expected file count: {len(files)}")
    print(f"Expected column count: {len(EXPECTED_SCHEMA.fields)}")
    print(f"Expected schema fingerprint: {expected_fingerprint}")
    print("=" * 78)

    for year_month, path in files:
        print(f"Inspecting schema and size: {year_month}")

        actual_schema = spark.read.parquet(path).schema
        actual_signature = schema_signature(actual_schema)
        actual_fingerprint = schema_fingerprint(actual_schema)

        exact_match = actual_signature == expected_signature
        file_size = get_file_size_bytes(spark, path)
        filename = os.path.basename(path)

        schema_rows.append(
            {
                "source_year_month": year_month,
                "source_file": filename,
                "source_path": path,
                "column_count": len(actual_schema.fields),
                "schema_match": exact_match,
                "expected_schema_fingerprint": expected_fingerprint,
                "actual_schema_fingerprint": actual_fingerprint,
                "actual_schema_json": actual_schema.json(),
            }
        )
        size_rows.append((year_month, filename, path, file_size))

        print(
            f"  schema_match={exact_match}, "
            f"columns={len(actual_schema.fields)}, "
            f"size_bytes={file_size}"
        )

    schema_result_schema = T.StructType(
        [
            T.StructField("source_year_month", T.StringType(), False),
            T.StructField("source_file", T.StringType(), False),
            T.StructField("source_path", T.StringType(), False),
            T.StructField("column_count", T.IntegerType(), False),
            T.StructField("schema_match", T.BooleanType(), False),
            T.StructField("expected_schema_fingerprint", T.StringType(), False),
            T.StructField("actual_schema_fingerprint", T.StringType(), False),
            T.StructField("actual_schema_json", T.StringType(), False),
        ]
    )

    schema_df = spark.createDataFrame(
        schema_rows,
        schema=schema_result_schema,
    ).orderBy("source_year_month")

    write_csv(schema_df, f"{output_root}/schema_validation_csv")

    mismatch_count = schema_df.filter(~F.col("schema_match")).count()

    if mismatch_count != 0:
        print("SCHEMA VALIDATION FAILED")
        schema_df.filter(~F.col("schema_match")).show(100, truncate=False)
        print(f"SCHEMA_MISMATCH_COUNT={mismatch_count}")
        spark.stop()
        sys.exit(2)

    print("All 17 schemas exactly match the locked canonical schema.")

    paths_only = [path for _, path in files]

    raw_df = (
        spark.read
        .schema(EXPECTED_SCHEMA)
        .parquet(*paths_only)
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
    )

    aggregate_expressions = [
        F.count(F.lit(1)).cast("long").alias("row_count"),
        F.min("tpep_pickup_datetime").alias("min_pickup_datetime"),
        F.max("tpep_pickup_datetime").alias("max_pickup_datetime"),
        F.min("tpep_dropoff_datetime").alias("min_dropoff_datetime"),
        F.max("tpep_dropoff_datetime").alias("max_dropoff_datetime"),
        F.min("fare_amount").alias("min_fare_amount"),
        F.max("fare_amount").alias("max_fare_amount"),
        F.min("trip_distance").alias("min_trip_distance"),
        F.max("trip_distance").alias("max_trip_distance"),
        F.sum(F.when(F.col("fare_amount") < 0, 1).otherwise(0))
        .cast("long")
        .alias("negative_fare_rows"),
        F.sum(F.when(F.col("trip_distance") < 0, 1).otherwise(0))
        .cast("long")
        .alias("negative_distance_rows"),
        F.sum(F.when(F.col("passenger_count") < 0, 1).otherwise(0))
        .cast("long")
        .alias("negative_passenger_rows"),
    ]

    for column_name in EXPECTED_SCHEMA.fieldNames():
        aggregate_expressions.append(
            F.sum(
                F.when(F.col(column_name).isNull(), 1).otherwise(0)
            ).cast("long").alias(f"null__{column_name}")
        )

    monthly_profile = (
        raw_df.groupBy("source_year_month", "source_file")
        .agg(*aggregate_expressions)
    )

    file_size_schema = T.StructType(
        [
            T.StructField("source_year_month", T.StringType(), False),
            T.StructField("source_file", T.StringType(), False),
            T.StructField("source_path", T.StringType(), False),
            T.StructField("size_bytes", T.LongType(), False),
        ]
    )

    file_size_df = spark.createDataFrame(size_rows, schema=file_size_schema)

    monthly_profile = (
        monthly_profile.join(
            file_size_df,
            on=["source_year_month", "source_file"],
            how="inner",
        )
        .select(
            "source_year_month",
            "source_file",
            "source_path",
            "size_bytes",
            "row_count",
            "min_pickup_datetime",
            "max_pickup_datetime",
            "min_dropoff_datetime",
            "max_dropoff_datetime",
            "min_fare_amount",
            "max_fare_amount",
            "min_trip_distance",
            "max_trip_distance",
            "negative_fare_rows",
            "negative_distance_rows",
            "negative_passenger_rows",
            *[f"null__{column_name}" for column_name in EXPECTED_SCHEMA.fieldNames()],
        )
        .orderBy("source_year_month")
    )

    profile_count = monthly_profile.count()

    if profile_count != 17:
        print(f"ERROR: Expected 17 monthly profile rows, but found {profile_count}.")
        spark.stop()
        sys.exit(3)

    empty_file_count = monthly_profile.filter(F.col("row_count") <= 0).count()

    if empty_file_count != 0:
        print("ERROR: One or more Parquet files contain no rows.")
        monthly_profile.filter(F.col("row_count") <= 0).show(truncate=False)
        spark.stop()
        sys.exit(4)

    summary_df = (
        monthly_profile.agg(
            F.count(F.lit(1)).cast("long").alias("file_count"),
            F.sum("size_bytes").cast("long").alias("total_size_bytes"),
            F.sum("row_count").cast("long").alias("total_rows"),
            F.min("min_pickup_datetime").alias("overall_min_pickup_datetime"),
            F.max("max_pickup_datetime").alias("overall_max_pickup_datetime"),
            F.sum("negative_fare_rows")
            .cast("long")
            .alias("total_negative_fare_rows"),
            F.sum("negative_distance_rows")
            .cast("long")
            .alias("total_negative_distance_rows"),
            F.sum("negative_passenger_rows")
            .cast("long")
            .alias("total_negative_passenger_rows"),
        )
        .withColumn("schema_mismatch_count", F.lit(mismatch_count).cast("long"))
        .withColumn("locked_column_count", F.lit(len(EXPECTED_SCHEMA.fields)).cast("long"))
        .withColumn("locked_schema_fingerprint", F.lit(expected_fingerprint))
        .withColumn("spark_version", F.lit(spark.version))
        .withColumn(
            "spark_session_timezone",
            F.lit(spark.conf.get("spark.sql.session.timeZone")),
        )
    )

    write_csv(monthly_profile, f"{output_root}/monthly_profile_csv")
    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("MONTHLY PROFILE")
    print("=" * 78)

    monthly_profile.select(
        "source_year_month",
        "source_file",
        "size_bytes",
        "row_count",
        "min_pickup_datetime",
        "max_pickup_datetime",
        "negative_fare_rows",
        "negative_distance_rows",
    ).show(30, truncate=False)

    summary = summary_df.collect()[0]

    print("=" * 78)
    print("STATEGUARD_RAW17_PROFILE_BEGIN")
    print("PROFILE_STATUS=PASS")
    print(f"FILE_COUNT={summary['file_count']}")
    print(f"TOTAL_SIZE_BYTES={summary['total_size_bytes']}")
    print(f"TOTAL_ROWS={summary['total_rows']}")
    print(f"OVERALL_MIN_PICKUP_DATETIME={summary['overall_min_pickup_datetime']}")
    print(f"OVERALL_MAX_PICKUP_DATETIME={summary['overall_max_pickup_datetime']}")
    print(f"TOTAL_NEGATIVE_FARE_ROWS={summary['total_negative_fare_rows']}")
    print(f"TOTAL_NEGATIVE_DISTANCE_ROWS={summary['total_negative_distance_rows']}")
    print(f"SCHEMA_MISMATCH_COUNT={mismatch_count}")
    print(f"SPARK_VERSION={spark.version}")
    print(
        "SPARK_SESSION_TIMEZONE="
        f"{spark.conf.get('spark.sql.session.timeZone')}"
    )
    print(f"MONTHLY_PROFILE_OUTPUT={output_root}/monthly_profile_csv")
    print(f"SCHEMA_OUTPUT={output_root}/schema_validation_csv")
    print(f"SUMMARY_OUTPUT={output_root}/summary_csv")
    print("STATEGUARD_RAW17_PROFILE_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
