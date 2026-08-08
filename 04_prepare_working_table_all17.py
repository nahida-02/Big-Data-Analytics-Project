import argparse
import time

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one mutable CDF-enabled working Delta table from the "
            "complete merged 17-file StateGuard canonical table."
        )
    )
    parser.add_argument("--canonical-path", required=True)
    parser.add_argument("--working-path", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--expected-months", type=int, default=17)
    parser.add_argument("--expected-rows", type=int, default=67721884)
    return parser.parse_args()


def write_single_csv(dataframe, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def main() -> None:
    args = parse_args()

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardPrepareWorkingTableAll17")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.state_partitions))

    canonical_path = args.canonical_path.rstrip("/")
    working_path = args.working_path.rstrip("/")
    result_root = args.result_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, canonical_path):
        raise RuntimeError(f"Not a Delta table: {canonical_path}")

    canonical_table = DeltaTable.forPath(spark, canonical_path)
    canonical_version = int(
        canonical_table.history(1).collect()[0]["version"]
    )

    canonical_df = spark.read.format("delta").load(canonical_path)

    required_columns = {
        "row_id",
        "trip_key",
        "state_partition_id",
        "source_year_month",
    }
    missing = sorted(required_columns.difference(canonical_df.columns))
    if missing:
        raise RuntimeError(
            "Canonical table is missing required columns: "
            + ", ".join(missing)
        )

    source = (
        canonical_df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.countDistinct("source_year_month")
            .cast("long")
            .alias("month_count"),
            F.min("source_year_month").alias("first_source_month"),
            F.max("source_year_month").alias("last_source_month"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("state_partition_count"),
            F.sum(
                F.when(F.col("row_id").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_row_ids"),
            F.sum(
                F.when(F.col("trip_key").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_trip_keys"),
        )
        .collect()[0]
    )

    source_rows = int(source["row_count"])
    source_months = int(source["month_count"])
    source_partitions = int(source["state_partition_count"])
    first_month = str(source["first_source_month"])
    last_month = str(source["last_source_month"])
    source_null_row_ids = int(source["null_row_ids"])
    source_null_trip_keys = int(source["null_trip_keys"])

    if source_rows != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} rows, found {source_rows}."
        )

    if source_months != args.expected_months:
        raise RuntimeError(
            f"Expected {args.expected_months} source months, "
            f"found {source_months}."
        )

    if source_partitions != args.state_partitions:
        raise RuntimeError(
            f"Expected {args.state_partitions} logical partitions, "
            f"found {source_partitions}."
        )

    if source_null_row_ids != 0:
        raise RuntimeError(
            f"Canonical table contains {source_null_row_ids} null row IDs."
        )

    if source_null_trip_keys != 0:
        raise RuntimeError(
            f"Canonical table contains {source_null_trip_keys} null trip keys."
        )

    print("=" * 78)
    print("STATEGUARD COMPLETE 17-FILE WORKING-TABLE PREPARATION")
    print("=" * 78)
    print(f"Canonical Delta version: {canonical_version}")
    print(f"Rows treated together: {source_rows}")
    print(f"Source-file/month markers present: {source_months}")
    print(f"Source range: {first_month} through {last_month}")
    print(f"Logical StateGuard partitions: {source_partitions}")
    print(f"Working path: {working_path}")
    print("=" * 78)

    build_start = time.perf_counter()

    (
        canonical_df.repartition(
            args.state_partitions,
            "state_partition_id",
        )
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .partitionBy("state_partition_id")
        .save(working_path)
    )

    build_seconds = time.perf_counter() - build_start

    if not DeltaTable.isDeltaTable(spark, working_path):
        raise RuntimeError(
            "The destination was not created as a Delta table."
        )

    working_table = DeltaTable.forPath(spark, working_path)
    working_history = working_table.history(1).collect()[0]
    working_detail = working_table.detail().collect()[0]

    properties = working_detail["properties"] or {}
    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    check = (
        spark.read.format("delta")
        .load(working_path)
        .agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.countDistinct("source_year_month")
            .cast("long")
            .alias("month_count"),
            F.min("source_year_month").alias("first_source_month"),
            F.max("source_year_month").alias("last_source_month"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("state_partition_count"),
            F.sum(
                F.when(F.col("row_id").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_row_ids"),
            F.sum(
                F.when(F.col("trip_key").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_trip_keys"),
        )
        .collect()[0]
    )

    actual_rows = int(check["row_count"])
    actual_months = int(check["month_count"])
    actual_partitions = int(check["state_partition_count"])
    actual_first_month = str(check["first_source_month"])
    actual_last_month = str(check["last_source_month"])
    null_row_ids = int(check["null_row_ids"])
    null_trip_keys = int(check["null_trip_keys"])

    verification_errors = []

    if actual_rows != source_rows:
        verification_errors.append(
            f"row mismatch: source={source_rows}, working={actual_rows}"
        )

    if actual_months != source_months:
        verification_errors.append(
            f"source-marker mismatch: source={source_months}, "
            f"working={actual_months}"
        )

    if actual_first_month != first_month:
        verification_errors.append("first source marker changed")

    if actual_last_month != last_month:
        verification_errors.append("last source marker changed")

    if actual_partitions != args.state_partitions:
        verification_errors.append(
            f"expected {args.state_partitions} logical partitions, "
            f"found {actual_partitions}"
        )

    if null_row_ids != 0:
        verification_errors.append(
            f"found {null_row_ids} null row IDs"
        )

    if null_trip_keys != 0:
        verification_errors.append(
            f"found {null_trip_keys} null trip keys"
        )

    if not cdf_enabled:
        verification_errors.append("Change Data Feed is not enabled")

    if verification_errors:
        raise RuntimeError("; ".join(verification_errors))

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("canonical_version", T.LongType(), False),
            T.StructField("working_version", T.LongType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("source_marker_count", T.LongType(), False),
            T.StructField("first_source_marker", T.StringType(), False),
            T.StructField("last_source_marker", T.StringType(), False),
            T.StructField(
                "state_partition_count",
                T.LongType(),
                False,
            ),
            T.StructField("null_row_ids", T.LongType(), False),
            T.StructField("null_trip_keys", T.LongType(), False),
            T.StructField("cdf_enabled", T.BooleanType(), False),
            T.StructField("num_data_files", T.LongType(), False),
            T.StructField("size_bytes", T.LongType(), False),
            T.StructField("build_seconds", T.DoubleType(), False),
            T.StructField("spark_version", T.StringType(), False),
            T.StructField("canonical_path", T.StringType(), False),
            T.StructField("working_path", T.StringType(), False),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                canonical_version,
                int(working_history["version"]),
                actual_rows,
                actual_months,
                actual_first_month,
                actual_last_month,
                actual_partitions,
                null_row_ids,
                null_trip_keys,
                cdf_enabled,
                int(working_detail["numFiles"]),
                int(working_detail["sizeInBytes"]),
                float(build_seconds),
                spark.version,
                canonical_path,
                working_path,
            )
        ],
        schema=summary_schema,
    )

    write_single_csv(summary_df, f"{result_root}/summary_csv")

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{result_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("STATEGUARD_WORKING_TABLE_BEGIN")
    print("WORKING_TABLE_STATUS=PASS")
    print(f"CANONICAL_VERSION={canonical_version}")
    print(f"WORKING_VERSION={int(working_history['version'])}")
    print(f"ROW_COUNT={actual_rows}")
    print(f"SOURCE_MARKER_COUNT={actual_months}")
    print(
        f"SOURCE_MARKER_RANGE="
        f"{actual_first_month}_TO_{actual_last_month}"
    )
    print(f"STATE_PARTITION_COUNT={actual_partitions}")
    print(f"NULL_ROW_IDS={null_row_ids}")
    print(f"NULL_TRIP_KEYS={null_trip_keys}")
    print(f"CDF_ENABLED={str(cdf_enabled).lower()}")
    print(f"NUM_DATA_FILES={int(working_detail['numFiles'])}")
    print(f"SIZE_BYTES={int(working_detail['sizeInBytes'])}")
    print(f"BUILD_SECONDS={build_seconds:.3f}")
    print(f"WORKING_PATH={working_path}")
    print(f"SUMMARY_PATH={result_root}/summary_csv")
    print("STATEGUARD_WORKING_TABLE_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
