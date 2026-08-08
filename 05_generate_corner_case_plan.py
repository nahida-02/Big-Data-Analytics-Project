import argparse
import time
from typing import Dict, List, Optional, Sequence, Tuple

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


METADATA_COLUMNS = {
    "row_id",
    "trip_key",
    "state_partition_id",
    "source_year_month",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic, non-destructive corner-case mutation "
            "plan for the complete merged StateGuard working Delta table."
        )
    )
    parser.add_argument("--working-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument(
        "--min-valid-pickup",
        default="2024-12-31 00:00:00",
    )
    parser.add_argument(
        "--max-valid-pickup",
        default="2026-06-01 23:59:59",
    )
    parser.add_argument("--max-passengers", type=int, default=8)
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def get_unique_extreme_row_id(
    dataframe: DataFrame,
    column_name: str,
    ascending: bool,
    excluded_row_ids: Sequence[int],
) -> Tuple[int, float]:
    grouped = (
        dataframe.filter(F.col(column_name).isNotNull())
        .groupBy(column_name)
        .agg(
            F.count(F.lit(1)).alias("value_count"),
            F.min("row_id").cast("long").alias("row_id"),
        )
        .filter(F.col("value_count") == 1)
    )

    if excluded_row_ids:
        grouped = grouped.filter(
            ~F.col("row_id").isin([int(value) for value in excluded_row_ids])
        )

    order_column = (
        F.col(column_name).asc()
        if ascending
        else F.col(column_name).desc()
    )

    rows = (
        grouped.orderBy(order_column, F.col("row_id").asc())
        .select("row_id", column_name)
        .limit(1)
        .collect()
    )

    if not rows:
        direction = "minimum" if ascending else "maximum"
        raise RuntimeError(
            f"Could not find a unique {direction} candidate for {column_name}."
        )

    return int(rows[0]["row_id"]), float(rows[0][column_name])


def main() -> None:
    args = parse_arguments()

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")

    spark = (
        SparkSession.builder
        .appName("StateGuardGenerateCornerCasePlan")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", "64")

    working_path = args.working_path.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, working_path):
        raise RuntimeError(f"Not a Delta table: {working_path}")

    delta_table = DeltaTable.forPath(spark, working_path)
    detail = delta_table.detail().collect()[0]
    current_version = int(delta_table.history(1).collect()[0]["version"])

    properties = detail["properties"] or {}
    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    if not cdf_enabled:
        raise RuntimeError("Change Data Feed is not enabled.")

    df = spark.read.format("delta").load(working_path)

    required_columns = {
        "row_id",
        "trip_key",
        "state_partition_id",
        "source_year_month",
        "passenger_count",
        "fare_amount",
        "trip_distance",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "store_and_fwd_flag",
    }

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise RuntimeError(
            "Working table is missing required columns: "
            + ", ".join(missing)
        )

    record_columns = [
        column_name
        for column_name in df.columns
        if column_name not in METADATA_COLUMNS
    ]

    table_stats = (
        df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.max("row_id").cast("long").alias("max_row_id"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("partition_count"),
        )
        .collect()[0]
    )

    row_count = int(table_stats["row_count"])
    max_row_id = int(table_stats["max_row_id"])
    partition_count = int(table_stats["partition_count"])

    if partition_count != args.state_partitions:
        raise RuntimeError(
            f"Expected {args.state_partitions} logical partitions, "
            f"found {partition_count}."
        )

    duplicate_key = (
        df.groupBy("trip_key")
        .agg(F.count(F.lit(1)).cast("long").alias("frequency"))
        .filter(F.col("frequency") > 1)
        .orderBy(F.hex("trip_key").asc())
        .limit(1)
    )

    duplicate_rows = (
        df.join(
            duplicate_key.select("trip_key"),
            on="trip_key",
            how="inner",
        )
        .orderBy(F.col("row_id").asc())
        .select("row_id", "trip_key")
        .limit(2)
        .collect()
    )

    if len(duplicate_rows) < 2:
        raise RuntimeError(
            "The expected existing duplicate group was not found."
        )

    duplicate_keep_row_id = int(duplicate_rows[0]["row_id"])
    duplicate_update_row_id = int(duplicate_rows[1]["row_id"])
    duplicate_trip_key = bytes(duplicate_rows[0]["trip_key"])

    excluded_ids: List[int] = [
        duplicate_keep_row_id,
        duplicate_update_row_id,
    ]

    min_fare_row_id, min_fare_value = get_unique_extreme_row_id(
        df,
        "fare_amount",
        ascending=True,
        excluded_row_ids=excluded_ids,
    )
    excluded_ids.append(min_fare_row_id)

    max_distance_row_id, max_distance_value = get_unique_extreme_row_id(
        df,
        "trip_distance",
        ascending=False,
        excluded_row_ids=excluded_ids,
    )
    excluded_ids.append(max_distance_row_id)

    min_pickup = F.lit(args.min_valid_pickup).cast("timestamp_ntz")
    max_pickup = F.lit(args.max_valid_pickup).cast("timestamp_ntz")

    valid_template_candidates = df.filter(
        F.col("passenger_count").isNotNull()
        & (F.col("passenger_count") >= 1)
        & (F.col("passenger_count") <= args.max_passengers)
        & F.col("fare_amount").isNotNull()
        & (F.col("fare_amount") > 0)
        & (F.col("fare_amount") < 1000)
        & F.col("trip_distance").isNotNull()
        & (F.col("trip_distance") > 0)
        & (F.col("trip_distance") < 100)
        & F.col("tpep_pickup_datetime").isNotNull()
        & (F.col("tpep_pickup_datetime") >= min_pickup)
        & (F.col("tpep_pickup_datetime") <= max_pickup)
        & (~F.col("row_id").isin(excluded_ids))
        & (F.col("trip_key") != F.lit(bytearray(duplicate_trip_key)))
    )

    template_rows = (
        valid_template_candidates.orderBy(F.col("row_id").asc())
        .limit(1)
        .collect()
    )

    if not template_rows:
        raise RuntimeError("Could not find a deterministic valid template row.")

    template_row_id = int(template_rows[0]["row_id"])

    selected_roles = [
        ("VALID_TEMPLATE", template_row_id),
        ("MIN_FARE_DELETE_TARGET", min_fare_row_id),
        ("MAX_DISTANCE_DELETE_TARGET", max_distance_row_id),
        ("DUPLICATE_KEEP_ROW", duplicate_keep_row_id),
        ("DUPLICATE_UPDATE_TARGET", duplicate_update_row_id),
    ]

    selected_frames = []

    for role, row_id in selected_roles:
        selected_frames.append(
            df.filter(F.col("row_id") == F.lit(row_id))
            .withColumn("selection_role", F.lit(role))
        )

    selected_rows_df = selected_frames[0]

    for selected_frame in selected_frames[1:]:
        selected_rows_df = selected_rows_df.unionByName(selected_frame)

    (
        selected_rows_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(f"{output_root}/selected_rows_delta")
    )

    selected_display_df = selected_rows_df.select(
        "selection_role",
        "row_id",
        F.hex("trip_key").alias("trip_key_hex"),
        "state_partition_id",
        "source_year_month",
        "passenger_count",
        "fare_amount",
        "trip_distance",
        "tpep_pickup_datetime",
        "store_and_fwd_flag",
    )

    write_csv(
        selected_display_df,
        f"{output_root}/selected_rows_csv",
    )

    next_row_id = max_row_id + 1

    scenario_specs: List[Dict[str, object]] = [
        {
            "scenario_id": "C01",
            "operation_type": "DELETE",
            "source_role": "MIN_FARE_DELETE_TARGET",
            "target_row_id": min_fare_row_id,
            "new_row_id": None,
            "planned_change": (
                f"Delete the unique low-fare outlier ({min_fare_value})."
            ),
            "expected_rules": "R01,R04,R08,R12,R13",
            "corner_case": "Delete-sensitive minimum recomputation",
        },
        {
            "scenario_id": "C02",
            "operation_type": "DELETE",
            "source_role": "MAX_DISTANCE_DELETE_TARGET",
            "target_row_id": max_distance_row_id,
            "new_row_id": None,
            "planned_change": (
                "Delete the unique maximum-distance holder "
                f"({max_distance_value})."
            ),
            "expected_rules": "R01,R11,R12,R13",
            "corner_case": "Delete-sensitive maximum recomputation",
        },
        {
            "scenario_id": "C03",
            "operation_type": "UPDATE",
            "source_role": "DUPLICATE_UPDATE_TARGET",
            "target_row_id": duplicate_update_row_id,
            "new_row_id": None,
            "planned_change": (
                "Set store_and_fwd_flag='SG_FIXED' and recompute trip_key."
            ),
            "expected_rules": "R12,R13",
            "corner_case": "Remove an existing exact duplicate",
        },
        {
            "scenario_id": "C04",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id,
            "planned_change": (
                "Clone template; set store_and_fwd_flag='SG_VALID'."
            ),
            "expected_rules": "R01,R12,R13",
            "corner_case": "Ordinary valid insertion",
        },
        {
            "scenario_id": "C05",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 1,
            "planned_change": (
                "Clone template; set passenger_count=NULL and unique marker."
            ),
            "expected_rules": "R01,R02,R12,R13",
            "corner_case": "Null passenger insertion",
        },
        {
            "scenario_id": "C06",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 2,
            "planned_change": (
                "Clone template; set fare_amount=NULL and unique marker."
            ),
            "expected_rules": "R01,R03,R08,R09,R12,R13",
            "corner_case": "Null fare insertion",
        },
        {
            "scenario_id": "C07",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 3,
            "planned_change": (
                "Clone template; set fare_amount=-9999.99."
            ),
            "expected_rules": "R01,R04,R08,R12,R13",
            "corner_case": "New negative-fare outlier",
        },
        {
            "scenario_id": "C08",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 4,
            "planned_change": (
                "Clone template; set trip_distance=-7.5."
            ),
            "expected_rules": "R01,R05,R10,R12,R13",
            "corner_case": "Negative-distance violation",
        },
        {
            "scenario_id": "C09",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 5,
            "planned_change": (
                "Clone template; set passenger_count=0."
            ),
            "expected_rules": "R01,R06,R12,R13",
            "corner_case": "Invalid lower passenger boundary",
        },
        {
            "scenario_id": "C10",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 6,
            "planned_change": (
                "Clone template; set pickup/dropoff in year 2000."
            ),
            "expected_rules": "R01,R07,R12,R13",
            "corner_case": "Historical timestamp outlier",
        },
        {
            "scenario_id": "C11",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 7,
            "planned_change": (
                "Clone template; set pickup/dropoff in year 2035."
            ),
            "expected_rules": "R01,R07,R12,R13",
            "corner_case": "Future timestamp outlier",
        },
        {
            "scenario_id": "C12",
            "operation_type": "INSERT",
            "source_role": "VALID_TEMPLATE",
            "target_row_id": None,
            "new_row_id": next_row_id + 8,
            "planned_change": (
                "Clone every original record field without modification."
            ),
            "expected_rules": "R01,R12,R13",
            "corner_case": "Create a new exact duplicate",
        },
    ]

    plan_schema = T.StructType(
        [
            T.StructField("scenario_id", T.StringType(), False),
            T.StructField("operation_type", T.StringType(), False),
            T.StructField("source_role", T.StringType(), False),
            T.StructField("target_row_id", T.LongType(), True),
            T.StructField("new_row_id", T.LongType(), True),
            T.StructField("planned_change", T.StringType(), False),
            T.StructField("expected_rules", T.StringType(), False),
            T.StructField("corner_case", T.StringType(), False),
        ]
    )

    plan_df = (
        spark.createDataFrame(scenario_specs, schema=plan_schema)
        .orderBy("scenario_id")
    )

    write_csv(plan_df, f"{output_root}/plan_csv")

    (
        plan_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/plan_json")
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("working_version", T.LongType(), False),
            T.StructField("working_row_count", T.LongType(), False),
            T.StructField("record_column_count", T.LongType(), False),
            T.StructField("scenario_count", T.LongType(), False),
            T.StructField("insert_count", T.LongType(), False),
            T.StructField("update_count", T.LongType(), False),
            T.StructField("delete_count", T.LongType(), False),
            T.StructField("template_row_id", T.LongType(), False),
            T.StructField("min_fare_target_row_id", T.LongType(), False),
            T.StructField("min_fare_value", T.DoubleType(), False),
            T.StructField(
                "max_distance_target_row_id",
                T.LongType(),
                False,
            ),
            T.StructField("max_distance_value", T.DoubleType(), False),
            T.StructField(
                "duplicate_update_row_id",
                T.LongType(),
                False,
            ),
            T.StructField("max_existing_row_id", T.LongType(), False),
            T.StructField("first_new_row_id", T.LongType(), False),
            T.StructField("last_new_row_id", T.LongType(), False),
            T.StructField("cdf_enabled", T.BooleanType(), False),
            T.StructField("working_path", T.StringType(), False),
            T.StructField("selected_rows_path", T.StringType(), False),
            T.StructField("plan_path", T.StringType(), False),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                current_version,
                row_count,
                len(record_columns),
                len(scenario_specs),
                9,
                1,
                2,
                template_row_id,
                min_fare_row_id,
                min_fare_value,
                max_distance_row_id,
                max_distance_value,
                duplicate_update_row_id,
                max_row_id,
                next_row_id,
                next_row_id + 8,
                cdf_enabled,
                working_path,
                f"{output_root}/selected_rows_delta",
                f"{output_root}/plan_csv",
            )
        ],
        schema=summary_schema,
    )

    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print("=" * 78)
    print("STATEGUARD DETERMINISTIC CORNER-CASE MUTATION PLAN")
    print("=" * 78)

    plan_df.select(
        "scenario_id",
        "operation_type",
        "corner_case",
        "expected_rules",
    ).show(20, truncate=False)

    print("=" * 78)
    print("STATEGUARD_MUTATION_PLAN_BEGIN")
    print("MUTATION_PLAN_STATUS=PASS")
    print(f"WORKING_VERSION={current_version}")
    print(f"WORKING_ROW_COUNT={row_count}")
    print(f"SCENARIO_COUNT={len(scenario_specs)}")
    print("INSERT_COUNT=9")
    print("UPDATE_COUNT=1")
    print("DELETE_COUNT=2")
    print(f"TEMPLATE_ROW_ID={template_row_id}")
    print(f"MIN_FARE_TARGET_ROW_ID={min_fare_row_id}")
    print(f"MIN_FARE_VALUE={min_fare_value}")
    print(f"MAX_DISTANCE_TARGET_ROW_ID={max_distance_row_id}")
    print(f"MAX_DISTANCE_VALUE={max_distance_value}")
    print(f"DUPLICATE_UPDATE_ROW_ID={duplicate_update_row_id}")
    print(f"FIRST_NEW_ROW_ID={next_row_id}")
    print(f"LAST_NEW_ROW_ID={next_row_id + 8}")
    print(f"SELECTED_ROWS_PATH={output_root}/selected_rows_delta")
    print(f"PLAN_PATH={output_root}/plan_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_MUTATION_PLAN_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
