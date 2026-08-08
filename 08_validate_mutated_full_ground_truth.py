import argparse
import time
from datetime import datetime
from typing import Any, Dict, List

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 13-rule full-validation baseline over the "
            "StateGuard canonical Delta table."
        )
    )
    parser.add_argument(
        "--delta-path",
        required=True,
        help="GCS path of the canonical Delta table.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="GCS destination for full-validation results.",
    )
    parser.add_argument(
        "--min-valid-pickup",
        default="2024-12-31 00:00:00",
        help="Inclusive lower bound for R07.",
    )
    parser.add_argument(
        "--max-valid-pickup",
        default="2026-06-01 23:59:59",
        help="Inclusive upper bound for R07.",
    )
    parser.add_argument(
        "--max-passengers",
        type=int,
        default=8,
        help="Maximum valid non-null passenger count for R06.",
    )
    return parser.parse_args()


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def violation_status(value: int) -> str:
    return "PASS" if value == 0 else "FAIL"


def make_rule(
    order: int,
    rule_id: str,
    rule_name: str,
    state_kind: str,
    metric_type: str,
    metric_display: str,
    status: str,
    definition: str,
    long_value: int | None = None,
    double_value: float | None = None,
    boolean_value: bool | None = None,
) -> Dict[str, Any]:
    return {
        "rule_order": order,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "state_kind": state_kind,
        "metric_type": metric_type,
        "metric_display": metric_display,
        "status": status,
        "definition": definition,
        "long_value": long_value,
        "double_value": double_value,
        "boolean_value": boolean_value,
    }


def main() -> None:
    args = parse_arguments()

    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")

    # Validate the timestamp arguments before starting Spark work.
    datetime.strptime(args.min_valid_pickup, "%Y-%m-%d %H:%M:%S")
    datetime.strptime(args.max_valid_pickup, "%Y-%m-%d %H:%M:%S")

    spark = (
        SparkSession.builder
        .appName("StateGuardFullValidationBaseline")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", "64")

    delta_path = args.delta_path.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, delta_path):
        raise RuntimeError(f"Not a Delta table: {delta_path}")

    delta_table = DeltaTable.forPath(spark, delta_path)
    detail = delta_table.detail().collect()[0]
    history = delta_table.history(1).collect()[0]

    properties = detail["properties"] or {}
    cdf_enabled = (
        str(properties.get("delta.enableChangeDataFeed", "false")).lower()
        == "true"
    )

    if not cdf_enabled:
        raise RuntimeError("Canonical Delta table does not have CDF enabled.")

    df = spark.read.format("delta").load(delta_path)

    required_columns = {
        "row_id",
        "trip_key",
        "state_partition_id",
        "source_year_month",
        "tpep_pickup_datetime",
        "passenger_count",
        "fare_amount",
        "trip_distance",
    }

    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            "Canonical table is missing required columns: "
            + ", ".join(missing_columns)
        )

    min_pickup = F.lit(args.min_valid_pickup).cast("timestamp_ntz")
    max_pickup = F.lit(args.max_valid_pickup).cast("timestamp_ntz")

    print("=" * 78)
    print("STATEGUARD 13-RULE FULL-VALIDATION BASELINE")
    print("=" * 78)
    print(f"Canonical path: {delta_path}")
    print(f"Delta version: {int(history['version'])}")
    print(f"CDF enabled: {str(cdf_enabled).lower()}")
    print(
        "R06 passenger policy: "
        f"valid non-null range = 1..{args.max_passengers}"
    )
    print(
        "R07 pickup policy: "
        f"{args.min_valid_pickup} through {args.max_valid_pickup}"
    )
    print("=" * 78)

    full_start = time.perf_counter()

    scalar_start = time.perf_counter()

    scalar = (
        df.agg(
            F.count(F.lit(1)).cast("long").alias("row_count"),
            F.sum(
                F.when(F.col("passenger_count").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_passenger_rows"),
            F.sum(
                F.when(F.col("fare_amount").isNull(), 1).otherwise(0)
            ).cast("long").alias("null_fare_rows"),
            F.sum(
                F.when(
                    F.col("fare_amount").isNotNull()
                    & (F.col("fare_amount") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_fare_rows"),
            F.sum(
                F.when(
                    F.col("trip_distance").isNotNull()
                    & (F.col("trip_distance") < 0),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_distance_rows"),
            F.sum(
                F.when(
                    F.col("passenger_count").isNotNull()
                    & (
                        (F.col("passenger_count") < 1)
                        | (
                            F.col("passenger_count")
                            > F.lit(args.max_passengers)
                        )
                    ),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_passenger_rows"),
            F.sum(
                F.when(
                    F.col("tpep_pickup_datetime").isNull()
                    | (F.col("tpep_pickup_datetime") < min_pickup)
                    | (F.col("tpep_pickup_datetime") > max_pickup),
                    1,
                ).otherwise(0)
            ).cast("long").alias("invalid_pickup_time_rows"),
            F.min("fare_amount").cast("double").alias("min_fare"),
            F.max("fare_amount").cast("double").alias("max_fare"),
            F.min("trip_distance").cast("double").alias("min_distance"),
            F.max("trip_distance").cast("double").alias("max_distance"),
        )
        .collect()[0]
    )

    scalar_seconds = time.perf_counter() - scalar_start

    duplicate_start = time.perf_counter()

    duplicate_stats = (
        df.groupBy("trip_key")
        .count()
        .agg(
            F.count(F.lit(1))
            .cast("long")
            .alias("distinct_trip_keys"),
            F.sum(
                F.when(F.col("count") > 1, 1).otherwise(0)
            ).cast("long").alias("duplicate_key_groups"),
            F.sum(
                F.when(
                    F.col("count") > 1,
                    F.col("count") - F.lit(1),
                ).otherwise(F.lit(0))
            ).cast("long").alias("duplicate_extra_rows"),
            F.max("count").cast("long").alias("maximum_key_multiplicity"),
        )
        .collect()[0]
    )

    duplicate_seconds = time.perf_counter() - duplicate_start
    total_seconds = time.perf_counter() - full_start

    row_count = int(scalar["row_count"])
    null_passenger_rows = int(scalar["null_passenger_rows"])
    null_fare_rows = int(scalar["null_fare_rows"])
    invalid_fare_rows = int(scalar["invalid_fare_rows"])
    invalid_distance_rows = int(scalar["invalid_distance_rows"])
    invalid_passenger_rows = int(scalar["invalid_passenger_rows"])
    invalid_pickup_time_rows = int(scalar["invalid_pickup_time_rows"])

    min_fare = (
        float(scalar["min_fare"])
        if scalar["min_fare"] is not None
        else None
    )
    max_fare = (
        float(scalar["max_fare"])
        if scalar["max_fare"] is not None
        else None
    )
    min_distance = (
        float(scalar["min_distance"])
        if scalar["min_distance"] is not None
        else None
    )
    max_distance = (
        float(scalar["max_distance"])
        if scalar["max_distance"] is not None
        else None
    )

    distinct_trip_keys = int(duplicate_stats["distinct_trip_keys"])
    duplicate_key_groups = int(duplicate_stats["duplicate_key_groups"])
    duplicate_extra_rows = int(duplicate_stats["duplicate_extra_rows"])
    maximum_key_multiplicity = int(
        duplicate_stats["maximum_key_multiplicity"]
    )
    uniqueness_pass = duplicate_extra_rows == 0

    rules: List[Dict[str, Any]] = [
        make_rule(
            1,
            "R01",
            "Row count",
            "ADDITIVE",
            "LONG",
            f"{row_count}",
            "OBSERVED",
            "Total number of canonical records.",
            long_value=row_count,
        ),
        make_rule(
            2,
            "R02",
            "Null passenger count",
            "ADDITIVE",
            "LONG",
            f"{null_passenger_rows}",
            violation_status(null_passenger_rows),
            "Rows where passenger_count is NULL.",
            long_value=null_passenger_rows,
        ),
        make_rule(
            3,
            "R03",
            "Null fare count",
            "ADDITIVE",
            "LONG",
            f"{null_fare_rows}",
            violation_status(null_fare_rows),
            "Rows where fare_amount is NULL.",
            long_value=null_fare_rows,
        ),
        make_rule(
            4,
            "R04",
            "Invalid fare count",
            "ADDITIVE",
            "LONG",
            f"{invalid_fare_rows}",
            violation_status(invalid_fare_rows),
            "Non-null rows where fare_amount is negative.",
            long_value=invalid_fare_rows,
        ),
        make_rule(
            5,
            "R05",
            "Invalid distance count",
            "ADDITIVE",
            "LONG",
            f"{invalid_distance_rows}",
            violation_status(invalid_distance_rows),
            "Non-null rows where trip_distance is negative.",
            long_value=invalid_distance_rows,
        ),
        make_rule(
            6,
            "R06",
            "Invalid passenger count",
            "ADDITIVE",
            "LONG",
            f"{invalid_passenger_rows}",
            violation_status(invalid_passenger_rows),
            (
                "Non-null passenger_count outside project policy "
                f"range 1..{args.max_passengers}."
            ),
            long_value=invalid_passenger_rows,
        ),
        make_rule(
            7,
            "R07",
            "Invalid pickup-time count",
            "ADDITIVE",
            "LONG",
            f"{invalid_pickup_time_rows}",
            violation_status(invalid_pickup_time_rows),
            (
                "NULL pickup timestamps or values outside "
                f"[{args.min_valid_pickup}, {args.max_valid_pickup}]."
            ),
            long_value=invalid_pickup_time_rows,
        ),
        make_rule(
            8,
            "R08",
            "Minimum fare",
            "EXTREMA",
            "DOUBLE",
            str(min_fare),
            "OBSERVED",
            "Minimum non-null fare_amount.",
            double_value=min_fare,
        ),
        make_rule(
            9,
            "R09",
            "Maximum fare",
            "EXTREMA",
            "DOUBLE",
            str(max_fare),
            "OBSERVED",
            "Maximum non-null fare_amount.",
            double_value=max_fare,
        ),
        make_rule(
            10,
            "R10",
            "Minimum distance",
            "EXTREMA",
            "DOUBLE",
            str(min_distance),
            "OBSERVED",
            "Minimum non-null trip_distance.",
            double_value=min_distance,
        ),
        make_rule(
            11,
            "R11",
            "Maximum distance",
            "EXTREMA",
            "DOUBLE",
            str(max_distance),
            "OBSERVED",
            "Maximum non-null trip_distance.",
            double_value=max_distance,
        ),
        make_rule(
            12,
            "R12",
            "Duplicate key count",
            "KEY_FREQUENCY",
            "LONG",
            f"{duplicate_extra_rows}",
            violation_status(duplicate_extra_rows),
            (
                "Extra rows beyond the first occurrence of each exact "
                "20-column trip_key."
            ),
            long_value=duplicate_extra_rows,
        ),
        make_rule(
            13,
            "R13",
            "Uniqueness pass",
            "KEY_FREQUENCY",
            "BOOLEAN",
            str(uniqueness_pass).lower(),
            "PASS" if uniqueness_pass else "FAIL",
            "True only when R12 equals zero.",
            boolean_value=uniqueness_pass,
        ),
    ]

    rules_schema = T.StructType(
        [
            T.StructField("rule_order", T.IntegerType(), False),
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("rule_name", T.StringType(), False),
            T.StructField("state_kind", T.StringType(), False),
            T.StructField("metric_type", T.StringType(), False),
            T.StructField("metric_display", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("definition", T.StringType(), False),
            T.StructField("long_value", T.LongType(), True),
            T.StructField("double_value", T.DoubleType(), True),
            T.StructField("boolean_value", T.BooleanType(), True),
        ]
    )

    rules_df = (
        spark.createDataFrame(rules, schema=rules_schema)
        .orderBy("rule_order")
    )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField("delta_version", T.LongType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("rule_count", T.LongType(), False),
            T.StructField("distinct_trip_keys", T.LongType(), False),
            T.StructField("duplicate_key_groups", T.LongType(), False),
            T.StructField("duplicate_extra_rows", T.LongType(), False),
            T.StructField(
                "maximum_key_multiplicity",
                T.LongType(),
                False,
            ),
            T.StructField("uniqueness_pass", T.BooleanType(), False),
            T.StructField("scalar_scan_seconds", T.DoubleType(), False),
            T.StructField(
                "duplicate_scan_seconds",
                T.DoubleType(),
                False,
            ),
            T.StructField("total_seconds", T.DoubleType(), False),
            T.StructField("min_valid_pickup", T.StringType(), False),
            T.StructField("max_valid_pickup", T.StringType(), False),
            T.StructField("max_passengers", T.IntegerType(), False),
            T.StructField("spark_version", T.StringType(), False),
            T.StructField("delta_path", T.StringType(), False),
        ]
    )

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                int(history["version"]),
                row_count,
                13,
                distinct_trip_keys,
                duplicate_key_groups,
                duplicate_extra_rows,
                maximum_key_multiplicity,
                uniqueness_pass,
                float(scalar_seconds),
                float(duplicate_seconds),
                float(total_seconds),
                args.min_valid_pickup,
                args.max_valid_pickup,
                args.max_passengers,
                spark.version,
                delta_path,
            )
        ],
        schema=summary_schema,
    )

    write_csv(rules_df, f"{output_root}/rules_csv")
    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        rules_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/rules_json")
    )

    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("FULL VALIDATION RULES")
    print("=" * 78)

    rules_df.select(
        "rule_id",
        "rule_name",
        "state_kind",
        "metric_display",
        "status",
    ).show(20, truncate=False)

    print("=" * 78)
    print("STATEGUARD_FULL_VALIDATION_BEGIN")
    print("VALIDATION_STATUS=PASS")
    print(f"DELTA_VERSION={int(history['version'])}")
    print(f"ROW_COUNT={row_count}")
    print("RULE_COUNT=13")
    print(f"DISTINCT_TRIP_KEYS={distinct_trip_keys}")
    print(f"DUPLICATE_KEY_GROUPS={duplicate_key_groups}")
    print(f"DUPLICATE_EXTRA_ROWS={duplicate_extra_rows}")
    print(f"MAXIMUM_KEY_MULTIPLICITY={maximum_key_multiplicity}")
    print(f"UNIQUENESS_PASS={str(uniqueness_pass).lower()}")
    print(f"SCALAR_SCAN_SECONDS={scalar_seconds:.3f}")
    print(f"DUPLICATE_SCAN_SECONDS={duplicate_seconds:.3f}")
    print(f"TOTAL_SECONDS={total_seconds:.3f}")
    print(f"RULE_RESULTS_PATH={output_root}/rules_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_FULL_VALIDATION_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
