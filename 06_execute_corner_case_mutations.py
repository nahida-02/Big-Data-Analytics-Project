import argparse
import time
from typing import Dict, List, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Row, SparkSession
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the deterministic 12-scenario StateGuard corner-case "
            "mutation workload, one Delta commit per scenario."
        )
    )
    parser.add_argument("--working-path", required=True)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-partitions", type=int, default=64)
    parser.add_argument("--expected-start-version", type=int, default=0)
    parser.add_argument("--expected-start-rows", type=int, default=67721884)
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def latest_version(delta_table: DeltaTable) -> int:
    return int(delta_table.history(1).collect()[0]["version"])


def trip_key_expression(
    schema: T.StructType,
    overrides: Optional[Dict[str, F.Column]] = None,
) -> F.Column:
    replacements = overrides or {}

    record_fields = []

    for column_name in RAW_COLUMNS:
        expression = replacements.get(column_name, F.col(column_name))
        record_fields.append(expression.alias(column_name))

    record_json = F.to_json(
        F.struct(*record_fields),
        options={"ignoreNullFields": "false"},
    )

    digest_hex = F.sha2(record_json, 256)
    trip_key_type = schema["trip_key"].dataType

    if isinstance(trip_key_type, T.BinaryType):
        return F.unhex(digest_hex)

    if isinstance(trip_key_type, T.StringType):
        return digest_hex

    raise RuntimeError(
        f"Unsupported trip_key type: {trip_key_type.simpleString()}"
    )


def value_as_long(row: Row, column_name: str) -> Optional[int]:
    value = row[column_name]
    return None if value is None or value == "" else int(value)


def read_single_summary(
    spark: SparkSession,
    path: str,
) -> Row:
    rows = spark.read.option("header", "true").csv(path).collect()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one summary row at {path}, found {len(rows)}."
        )

    return rows[0]


def selected_role_row(
    selected_rows: DataFrame,
    role: str,
) -> Row:
    rows = (
        selected_rows.filter(F.col("selection_role") == role)
        .limit(2)
        .collect()
    )

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one selected row for role {role}, found {len(rows)}."
        )

    return rows[0]


def cdf_counts(
    spark: SparkSession,
    working_path: str,
    version: int,
) -> Dict[str, int]:
    rows = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", version)
        .option("endingVersion", version)
        .load(working_path)
        .groupBy("_change_type")
        .count()
        .collect()
    )

    result = {
        "insert": 0,
        "delete": 0,
        "update_preimage": 0,
        "update_postimage": 0,
    }

    for row in rows:
        result[str(row["_change_type"])] = int(row["count"])

    return result


def assert_cdf(
    scenario_id: str,
    operation_type: str,
    counts: Dict[str, int],
) -> None:
    expected = {
        "INSERT": {
            "insert": 1,
            "delete": 0,
            "update_preimage": 0,
            "update_postimage": 0,
        },
        "DELETE": {
            "insert": 0,
            "delete": 1,
            "update_preimage": 0,
            "update_postimage": 0,
        },
        "UPDATE": {
            "insert": 0,
            "delete": 0,
            "update_preimage": 1,
            "update_postimage": 1,
        },
    }[operation_type]

    if counts != expected:
        raise RuntimeError(
            f"{scenario_id} CDF mismatch: expected={expected}, actual={counts}"
        )


def scenario_modifications(
    scenario_id: str,
    schema: T.StructType,
) -> Dict[str, F.Column]:
    passenger_type = schema["passenger_count"].dataType
    fare_type = schema["fare_amount"].dataType
    distance_type = schema["trip_distance"].dataType
    flag_type = schema["store_and_fwd_flag"].dataType
    pickup_type = schema["tpep_pickup_datetime"].dataType
    dropoff_type = schema["tpep_dropoff_datetime"].dataType

    modifications: Dict[str, F.Column] = {}

    if scenario_id == "C04":
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_VALID").cast(flag_type)
        )
    elif scenario_id == "C05":
        modifications["passenger_count"] = (
            F.lit(None).cast(passenger_type)
        )
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_NULL_P").cast(flag_type)
        )
    elif scenario_id == "C06":
        modifications["fare_amount"] = F.lit(None).cast(fare_type)
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_NULL_F").cast(flag_type)
        )
    elif scenario_id == "C07":
        modifications["fare_amount"] = F.lit(-9999.99).cast(fare_type)
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_NEG_F").cast(flag_type)
        )
    elif scenario_id == "C08":
        modifications["trip_distance"] = (
            F.lit(-7.5).cast(distance_type)
        )
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_NEG_D").cast(flag_type)
        )
    elif scenario_id == "C09":
        modifications["passenger_count"] = (
            F.lit(0).cast(passenger_type)
        )
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_PAX0").cast(flag_type)
        )
    elif scenario_id == "C10":
        modifications["tpep_pickup_datetime"] = (
            F.lit("2000-01-01 00:00:00").cast(pickup_type)
        )
        modifications["tpep_dropoff_datetime"] = (
            F.lit("2000-01-01 00:10:00").cast(dropoff_type)
        )
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_OLD").cast(flag_type)
        )
    elif scenario_id == "C11":
        modifications["tpep_pickup_datetime"] = (
            F.lit("2035-01-01 00:00:00").cast(pickup_type)
        )
        modifications["tpep_dropoff_datetime"] = (
            F.lit("2035-01-01 00:10:00").cast(dropoff_type)
        )
        modifications["store_and_fwd_flag"] = (
            F.lit("SG_FUTURE").cast(flag_type)
        )
    elif scenario_id == "C12":
        # Keep all original 20 record fields unchanged so this creates
        # an exact duplicate under the canonical trip_key definition.
        pass
    else:
        raise RuntimeError(
            f"No INSERT transformation is defined for {scenario_id}."
        )

    return modifications


def build_insert_row(
    template_df: DataFrame,
    working_schema: T.StructType,
    working_columns: List[str],
    scenario_id: str,
    new_row_id: int,
    state_partitions: int,
) -> DataFrame:
    modifications = scenario_modifications(
        scenario_id,
        working_schema,
    )

    result = template_df

    for column_name, expression in modifications.items():
        result = result.withColumn(column_name, expression)

    row_id_type = working_schema["row_id"].dataType

    result = result.withColumn(
        "row_id",
        F.lit(new_row_id).cast(row_id_type),
    )

    new_key = trip_key_expression(working_schema)

    result = (
        result.withColumn("trip_key", new_key)
        .withColumn(
            "state_partition_id",
            F.pmod(
                F.xxhash64(F.col("trip_key")),
                F.lit(state_partitions),
            ).cast(working_schema["state_partition_id"].dataType),
        )
    )

    return result.select(*working_columns)


def main() -> None:
    args = parse_arguments()

    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardExecuteCornerCaseMutations")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", "64")

    working_path = args.working_path.rstrip("/")
    plan_root = args.plan_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, working_path):
        raise RuntimeError(f"Not a Delta table: {working_path}")

    delta_table = DeltaTable.forPath(spark, working_path)
    detail = delta_table.detail().collect()[0]
    properties = detail["properties"] or {}

    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    if not cdf_enabled:
        raise RuntimeError("Change Data Feed is not enabled.")

    start_version = latest_version(delta_table)

    if start_version != args.expected_start_version:
        raise RuntimeError(
            f"Expected working-table version "
            f"{args.expected_start_version}, found {start_version}. "
            "The mutation workload may already have been executed."
        )

    working_df = spark.read.format("delta").load(working_path)
    working_schema = working_df.schema
    working_columns = working_df.columns

    missing_raw = [
        column_name
        for column_name in RAW_COLUMNS
        if column_name not in working_columns
    ]

    if missing_raw:
        raise RuntimeError(
            "Working table is missing canonical record columns: "
            + ", ".join(missing_raw)
        )

    required_metadata = {
        "row_id",
        "trip_key",
        "state_partition_id",
    }

    missing_metadata = sorted(
        required_metadata.difference(working_columns)
    )

    if missing_metadata:
        raise RuntimeError(
            "Working table is missing metadata columns: "
            + ", ".join(missing_metadata)
        )

    start_stats = (
        working_df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("partition_count"),
            F.sum(
                F.when(F.col("row_id").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_row_ids"),
            F.sum(
                F.when(F.col("trip_key").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_trip_keys"),
        )
        .collect()[0]
    )

    start_rows = int(start_stats["row_count"])

    if start_rows != args.expected_start_rows:
        raise RuntimeError(
            f"Expected {args.expected_start_rows} starting rows, "
            f"found {start_rows}."
        )

    if int(start_stats["partition_count"]) != args.state_partitions:
        raise RuntimeError(
            f"Expected {args.state_partitions} logical partitions."
        )

    if int(start_stats["null_row_ids"]) != 0:
        raise RuntimeError("Working table contains null row IDs.")

    if int(start_stats["null_trip_keys"]) != 0:
        raise RuntimeError("Working table contains null trip keys.")

    plan_df = (
        spark.read.option("header", "true")
        .csv(f"{plan_root}/plan_csv")
        .select(
            "scenario_id",
            "operation_type",
            "source_role",
            F.col("target_row_id")
            .cast("long")
            .alias("target_row_id"),
            F.col("new_row_id").cast("long").alias("new_row_id"),
            "corner_case",
        )
        .orderBy("scenario_id")
    )

    plan_rows = plan_df.collect()

    expected_ids = [f"C{index:02d}" for index in range(1, 13)]
    actual_ids = [str(row["scenario_id"]) for row in plan_rows]

    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Unexpected scenario order: {actual_ids}"
        )

    selected_rows = spark.read.format("delta").load(
        f"{plan_root}/selected_rows_delta"
    )

    template_df = (
        selected_rows.filter(
            F.col("selection_role") == "VALID_TEMPLATE"
        )
        .drop("selection_role")
    )

    if template_df.count() != 1:
        raise RuntimeError(
            "Expected exactly one valid template row."
        )

    selected_lookup = {
        role: selected_role_row(selected_rows, role)
        for role in [
            "MIN_FARE_DELETE_TARGET",
            "MAX_DISTANCE_DELETE_TARGET",
            "DUPLICATE_KEEP_ROW",
            "DUPLICATE_UPDATE_TARGET",
        ]
    }

    corrected_rule_effects = {
        "C01": "R01,R04,R08; key-frequency state changes",
        "C02": "R01,R02,R11; key-frequency state changes",
        "C03": "R12,R13",
        "C04": "R01; key-frequency state changes",
        "C05": "R01,R02; key-frequency state changes",
        "C06": "R01,R03; key-frequency state changes",
        "C07": "R01,R04,R08; key-frequency state changes",
        "C08": "R01,R05,R10; key-frequency state changes",
        "C09": "R01,R06; key-frequency state changes",
        "C10": "R01,R07; key-frequency state changes",
        "C11": "R01,R07; key-frequency state changes",
        "C12": "R01,R12,R13",
    }

    print("=" * 78)
    print("STATEGUARD CORNER-CASE MUTATION EXECUTION")
    print("=" * 78)
    print(f"Working path: {working_path}")
    print(f"Starting Delta version: {start_version}")
    print(f"Starting rows: {start_rows}")
    print("Execution policy: one Delta commit per scenario")
    print("=" * 78)

    audit_records: List[Dict[str, object]] = []

    for plan_row in plan_rows:
        scenario_id = str(plan_row["scenario_id"])
        operation_type = str(plan_row["operation_type"])
        source_role = str(plan_row["source_role"])
        target_row_id = value_as_long(plan_row, "target_row_id")
        new_row_id = value_as_long(plan_row, "new_row_id")
        corner_case = str(plan_row["corner_case"])

        before_version = latest_version(delta_table)
        operation_start = time.perf_counter()
        target_partition_id: Optional[int] = None
        inserted_partition_id: Optional[int] = None

        if operation_type == "DELETE":
            selected = selected_lookup[source_role]
            expected_target = int(selected["row_id"])
            target_partition_id = int(selected["state_partition_id"])

            if target_row_id != expected_target:
                raise RuntimeError(
                    f"{scenario_id} target mismatch: "
                    f"plan={target_row_id}, selected={expected_target}"
                )

            delta_table.delete(
                condition=(
                    (F.col("row_id") == F.lit(target_row_id))
                    & (
                        F.col("state_partition_id")
                        == F.lit(target_partition_id)
                    )
                )
            )

        elif operation_type == "UPDATE":
            selected = selected_lookup[source_role]
            expected_target = int(selected["row_id"])
            target_partition_id = int(selected["state_partition_id"])

            if target_row_id != expected_target:
                raise RuntimeError(
                    f"{scenario_id} target mismatch: "
                    f"plan={target_row_id}, selected={expected_target}"
                )

            flag_type = working_schema[
                "store_and_fwd_flag"
            ].dataType

            fixed_flag = F.lit("SG_FIXED").cast(flag_type)

            new_key = trip_key_expression(
                working_schema,
                overrides={"store_and_fwd_flag": fixed_flag},
            )

            delta_table.update(
                condition=(
                    (F.col("row_id") == F.lit(target_row_id))
                    & (
                        F.col("state_partition_id")
                        == F.lit(target_partition_id)
                    )
                ),
                set={
                    "store_and_fwd_flag": fixed_flag,
                    "trip_key": new_key,
                    "state_partition_id": F.pmod(
                        F.xxhash64(new_key),
                        F.lit(args.state_partitions),
                    ).cast(
                        working_schema[
                            "state_partition_id"
                        ].dataType
                    ),
                },
            )

        elif operation_type == "INSERT":
            if new_row_id is None:
                raise RuntimeError(
                    f"{scenario_id} is missing new_row_id."
                )

            insert_df = build_insert_row(
                template_df=template_df,
                working_schema=working_schema,
                working_columns=working_columns,
                scenario_id=scenario_id,
                new_row_id=new_row_id,
                state_partitions=args.state_partitions,
            )

            insert_preview = (
                insert_df.select(
                    "row_id",
                    "trip_key",
                    "state_partition_id",
                )
                .collect()[0]
            )

            inserted_partition_id = int(
                insert_preview["state_partition_id"]
            )

            (
                insert_df.write.format("delta")
                .mode("append")
                .save(working_path)
            )

        else:
            raise RuntimeError(
                f"Unsupported operation type: {operation_type}"
            )

        elapsed_seconds = time.perf_counter() - operation_start

        after_version = latest_version(delta_table)

        if after_version != before_version + 1:
            raise RuntimeError(
                f"{scenario_id} expected one new Delta version: "
                f"before={before_version}, after={after_version}"
            )

        counts = cdf_counts(
            spark,
            working_path,
            after_version,
        )

        assert_cdf(
            scenario_id,
            operation_type,
            counts,
        )

        if operation_type == "DELETE":
            remaining = (
                spark.read.format("delta")
                .load(working_path)
                .filter(
                    (F.col("row_id") == F.lit(target_row_id))
                    & (
                        F.col("state_partition_id")
                        == F.lit(target_partition_id)
                    )
                )
                .limit(1)
                .count()
            )

            if remaining != 0:
                raise RuntimeError(
                    f"{scenario_id} target row still exists."
                )

        elif operation_type == "UPDATE":
            updated_rows = (
                spark.read.format("delta")
                .load(working_path)
                .filter(F.col("row_id") == F.lit(target_row_id))
                .select(
                    "store_and_fwd_flag",
                    "trip_key",
                    "state_partition_id",
                )
                .collect()
            )

            if len(updated_rows) != 1:
                raise RuntimeError(
                    f"{scenario_id} did not preserve exactly one row."
                )

            if str(
                updated_rows[0]["store_and_fwd_flag"]
            ) != "SG_FIXED":
                raise RuntimeError(
                    f"{scenario_id} update marker was not written."
                )

        elif operation_type == "INSERT":
            inserted_rows = (
                spark.read.format("delta")
                .load(working_path)
                .filter(
                    (F.col("row_id") == F.lit(new_row_id))
                    & (
                        F.col("state_partition_id")
                        == F.lit(inserted_partition_id)
                    )
                )
                .limit(2)
                .count()
            )

            if inserted_rows != 1:
                raise RuntimeError(
                    f"{scenario_id} inserted-row verification failed."
                )

        audit_records.append(
            {
                "scenario_id": scenario_id,
                "operation_type": operation_type,
                "corner_case": corner_case,
                "corrected_rule_effects": corrected_rule_effects[
                    scenario_id
                ],
                "target_row_id": target_row_id,
                "new_row_id": new_row_id,
                "before_version": before_version,
                "after_version": after_version,
                "insert_events": counts["insert"],
                "delete_events": counts["delete"],
                "update_preimage_events": counts[
                    "update_preimage"
                ],
                "update_postimage_events": counts[
                    "update_postimage"
                ],
                "elapsed_seconds": float(elapsed_seconds),
                "status": "PASS",
            }
        )

        print(
            f"{scenario_id} {operation_type:<6} "
            f"version {before_version}->{after_version} "
            f"CDF={counts} PASS"
        )

    final_version = latest_version(delta_table)
    expected_final_version = start_version + len(plan_rows)

    if final_version != expected_final_version:
        raise RuntimeError(
            f"Expected final version {expected_final_version}, "
            f"found {final_version}."
        )

    final_df = spark.read.format("delta").load(working_path)

    final_stats = (
        final_df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.countDistinct("state_partition_id")
            .cast("long")
            .alias("partition_count"),
            F.sum(
                F.when(F.col("row_id").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_row_ids"),
            F.sum(
                F.when(F.col("trip_key").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_trip_keys"),
        )
        .collect()[0]
    )

    final_rows = int(final_stats["row_count"])
    expected_final_rows = start_rows + 9 - 2

    if final_rows != expected_final_rows:
        raise RuntimeError(
            f"Expected {expected_final_rows} final rows, "
            f"found {final_rows}."
        )

    if int(final_stats["partition_count"]) != args.state_partitions:
        raise RuntimeError(
            "Final logical partition count changed unexpectedly."
        )

    if int(final_stats["null_row_ids"]) != 0:
        raise RuntimeError("Final table contains null row IDs.")

    if int(final_stats["null_trip_keys"]) != 0:
        raise RuntimeError("Final table contains null trip keys.")

    total_cdf = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", start_version + 1)
        .option("endingVersion", final_version)
        .load(working_path)
        .groupBy("_change_type")
        .count()
        .collect()
    )

    total_cdf_counts = {
        "insert": 0,
        "delete": 0,
        "update_preimage": 0,
        "update_postimage": 0,
    }

    for row in total_cdf:
        total_cdf_counts[str(row["_change_type"])] = int(row["count"])

    expected_total_cdf = {
        "insert": 9,
        "delete": 2,
        "update_preimage": 1,
        "update_postimage": 1,
    }

    if total_cdf_counts != expected_total_cdf:
        raise RuntimeError(
            "Whole-workload CDF mismatch: "
            f"expected={expected_total_cdf}, "
            f"actual={total_cdf_counts}"
        )

    audit_schema = T.StructType(
        [
            T.StructField("scenario_id", T.StringType(), False),
            T.StructField("operation_type", T.StringType(), False),
            T.StructField("corner_case", T.StringType(), False),
            T.StructField(
                "corrected_rule_effects",
                T.StringType(),
                False,
            ),
            T.StructField("target_row_id", T.LongType(), True),
            T.StructField("new_row_id", T.LongType(), True),
            T.StructField("before_version", T.LongType(), False),
            T.StructField("after_version", T.LongType(), False),
            T.StructField("insert_events", T.LongType(), False),
            T.StructField("delete_events", T.LongType(), False),
            T.StructField(
                "update_preimage_events",
                T.LongType(),
                False,
            ),
            T.StructField(
                "update_postimage_events",
                T.LongType(),
                False,
            ),
            T.StructField(
                "elapsed_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField("status", T.StringType(), False),
        ]
    )

    audit_df = (
        spark.createDataFrame(
            audit_records,
            schema=audit_schema,
        )
        .orderBy("scenario_id")
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("start_version", T.LongType(), False),
            T.StructField("final_version", T.LongType(), False),
            T.StructField("start_rows", T.LongType(), False),
            T.StructField("final_rows", T.LongType(), False),
            T.StructField("net_row_change", T.LongType(), False),
            T.StructField("scenario_count", T.LongType(), False),
            T.StructField("insert_operations", T.LongType(), False),
            T.StructField("update_operations", T.LongType(), False),
            T.StructField("delete_operations", T.LongType(), False),
            T.StructField("cdf_insert_rows", T.LongType(), False),
            T.StructField("cdf_delete_rows", T.LongType(), False),
            T.StructField(
                "cdf_update_preimage_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "cdf_update_postimage_rows",
                T.LongType(),
                False,
            ),
            T.StructField("cdf_total_rows", T.LongType(), False),
            T.StructField(
                "state_partition_count",
                T.LongType(),
                False,
            ),
            T.StructField("null_row_ids", T.LongType(), False),
            T.StructField("null_trip_keys", T.LongType(), False),
            T.StructField("working_path", T.StringType(), False),
        ]
    )

    cdf_total_rows = sum(total_cdf_counts.values())

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                start_version,
                final_version,
                start_rows,
                final_rows,
                final_rows - start_rows,
                len(plan_rows),
                9,
                1,
                2,
                total_cdf_counts["insert"],
                total_cdf_counts["delete"],
                total_cdf_counts["update_preimage"],
                total_cdf_counts["update_postimage"],
                cdf_total_rows,
                int(final_stats["partition_count"]),
                int(final_stats["null_row_ids"]),
                int(final_stats["null_trip_keys"]),
                working_path,
            )
        ],
        schema=summary_schema,
    )

    write_csv(audit_df, f"{output_root}/scenario_audit_csv")
    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        audit_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/scenario_audit_json")
    )

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("STATEGUARD_MUTATION_EXECUTION_BEGIN")
    print("MUTATION_EXECUTION_STATUS=PASS")
    print(f"START_VERSION={start_version}")
    print(f"FINAL_VERSION={final_version}")
    print(f"START_ROWS={start_rows}")
    print(f"FINAL_ROWS={final_rows}")
    print(f"NET_ROW_CHANGE={final_rows - start_rows}")
    print(f"SCENARIO_COUNT={len(plan_rows)}")
    print("INSERT_OPERATIONS=9")
    print("UPDATE_OPERATIONS=1")
    print("DELETE_OPERATIONS=2")
    print(f"CDF_INSERT_ROWS={total_cdf_counts['insert']}")
    print(f"CDF_DELETE_ROWS={total_cdf_counts['delete']}")
    print(
        "CDF_UPDATE_PREIMAGE_ROWS="
        f"{total_cdf_counts['update_preimage']}"
    )
    print(
        "CDF_UPDATE_POSTIMAGE_ROWS="
        f"{total_cdf_counts['update_postimage']}"
    )
    print(f"CDF_TOTAL_ROWS={cdf_total_rows}")
    print(
        "SCENARIO_AUDIT_PATH="
        f"{output_root}/scenario_audit_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_MUTATION_EXECUTION_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
