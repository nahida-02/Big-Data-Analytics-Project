import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Row, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

RAW_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID",
    "store_and_fwd_flag", "PULocationID", "DOLocationID",
    "payment_type", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount", "congestion_surcharge", "Airport_fee",
    "cbd_congestion_fee",
]

FAMILY_CODES = {
    "INSERT_ONLY": "I",
    "UPDATE_ONLY": "U",
    "DELETE_ONLY": "D",
    "MIXED_40I_30U_30D": "M",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create four complete CDF-enabled performance tables and apply "
            "the nested StateGuard mutation matrix with one Delta MERGE "
            "commit per workload level."
        )
    )
    p.add_argument("--canonical-path", required=True)
    p.add_argument("--research-matrix-root", required=True)
    p.add_argument("--consolidated-partition-state", required=True)
    p.add_argument("--key-frequency-state", required=True)
    p.add_argument("--working-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--state-partitions", type=int, default=64)
    p.add_argument("--expected-canonical-version", type=int, default=0)
    p.add_argument("--expected-canonical-rows", type=int, default=67721884)
    p.add_argument("--base-seed", type=int, default=20260806)
    p.add_argument("--candidate-hash-modulus", type=int, default=1000000)
    p.add_argument("--candidate-hash-threshold", type=int, default=20000)
    p.add_argument("--min-valid-pickup", default="2024-12-31 00:00:00")
    p.add_argument("--max-valid-pickup", default="2026-06-01 23:59:59")
    p.add_argument("--max-passengers", type=int, default=8)
    return p.parse_args()


def latest_version(table: DeltaTable) -> int:
    return int(table.history(1).collect()[0]["version"])


def write_csv(df: DataFrame, path: str) -> None:
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)


def save_records(
    spark: SparkSession,
    records: List[Dict[str, object]],
    csv_path: str,
    json_path: str,
    order_columns: Optional[List[str]] = None,
) -> DataFrame:
    if not records:
        raise RuntimeError(f"No records available for {csv_path}.")
    df = spark.createDataFrame(records)
    if order_columns:
        df = df.orderBy(*order_columns)
    write_csv(df, csv_path)
    df.coalesce(1).write.mode("overwrite").json(json_path)
    return df


def trip_key_expr(schema: T.StructType, overrides=None) -> Column:
    replacements = overrides or {}
    fields = [
        replacements.get(name, F.col(name)).alias(name)
        for name in RAW_COLUMNS
    ]
    payload = F.to_json(
        F.struct(*fields), options={"ignoreNullFields": "false"}
    )
    digest = F.sha2(payload, 256)
    key_type = schema["trip_key"].dataType
    if isinstance(key_type, T.BinaryType):
        return F.unhex(digest)
    if isinstance(key_type, T.StringType):
        return digest
    raise RuntimeError(f"Unsupported trip_key type: {key_type.simpleString()}")


def new_row_id_expr(
    row_id_type: T.DataType,
    max_numeric_row_id: Optional[int],
    family_code: str,
    ordinal: Column,
) -> Column:
    numeric_types = (T.ByteType, T.ShortType, T.IntegerType, T.LongType)
    if isinstance(row_id_type, numeric_types):
        if max_numeric_row_id is None:
            raise RuntimeError("Missing maximum numeric row_id.")
        return (F.lit(max_numeric_row_id) + ordinal.cast("long")).cast(row_id_type)
    if isinstance(row_id_type, T.StringType):
        return F.sha2(
            F.concat_ws(
                "||", F.lit("STATEGUARD_PERFORMANCE"),
                F.lit(family_code), ordinal.cast("string")
            ),
            256,
        )
    raise RuntimeError(f"Unsupported row_id type: {row_id_type.simpleString()}")


def cdf_counts(spark: SparkSession, path: str, version: int) -> Dict[str, int]:
    counts = {
        "insert": 0,
        "delete": 0,
        "update_preimage": 0,
        "update_postimage": 0,
    }
    rows = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", version)
        .option("endingVersion", version)
        .load(path)
        .groupBy("_change_type")
        .count()
        .collect()
    )
    for row in rows:
        counts[str(row["_change_type"])] = int(row["count"])
    return counts


def expected_cdf(ins: int, upd: int, dele: int) -> Dict[str, int]:
    return {
        "insert": ins,
        "delete": dele,
        "update_preimage": upd,
        "update_postimage": upd,
    }


def rank_slice(pool: DataFrame, start_rank: int, end_rank: int) -> DataFrame:
    return pool.filter(
        (F.col("candidate_rank") >= start_rank)
        & (F.col("candidate_rank") <= end_rank)
    )


def marker(prefix: str, family_code: str, ordinal: Column, data_type) -> Column:
    return F.concat(
        F.lit(f"{prefix}_{family_code}_"),
        F.lpad(ordinal.cast("string"), 8, "0"),
    ).cast(data_type)


def update_source(
    pool: DataFrame,
    start_rank: int,
    end_rank: int,
    family_code: str,
    schema: T.StructType,
    columns: List[str],
    partitions: int,
) -> DataFrame:
    df = rank_slice(pool, start_rank, end_rank)
    old_partition = F.col("state_partition_id")
    new_flag = marker(
        "SGU", family_code, F.col("candidate_rank"),
        schema["store_and_fwd_flag"].dataType,
    )
    df = (
        df.withColumn("_sg_match_row_id", F.col("row_id"))
        .withColumn("_sg_match_partition_id", old_partition)
        .withColumn("store_and_fwd_flag", new_flag)
    )
    new_key = trip_key_expr(schema, {"store_and_fwd_flag": new_flag})
    df = (
        df.withColumn("trip_key", new_key)
        .withColumn(
            "state_partition_id",
            F.pmod(F.xxhash64("trip_key"), F.lit(partitions)).cast(
                schema["state_partition_id"].dataType
            ),
        )
        .withColumn("_sg_operation", F.lit("UPDATE"))
    )
    return df.select(
        *columns, "_sg_operation", "_sg_match_row_id",
        "_sg_match_partition_id"
    )


def delete_source(
    pool: DataFrame,
    start_rank: int,
    end_rank: int,
    columns: List[str],
) -> DataFrame:
    return (
        rank_slice(pool, start_rank, end_rank)
        .withColumn("_sg_match_row_id", F.col("row_id"))
        .withColumn("_sg_match_partition_id", F.col("state_partition_id"))
        .withColumn("_sg_operation", F.lit("DELETE"))
        .select(
            *columns, "_sg_operation", "_sg_match_row_id",
            "_sg_match_partition_id"
        )
    )


def insert_source(
    pool: DataFrame,
    template_offset: int,
    start_ordinal: int,
    end_ordinal: int,
    family_code: str,
    schema: T.StructType,
    columns: List[str],
    partitions: int,
    max_numeric_row_id: Optional[int],
) -> DataFrame:
    df = rank_slice(
        pool,
        template_offset + start_ordinal,
        template_offset + end_ordinal,
    ).withColumn(
        "_sg_insert_ordinal",
        F.col("candidate_rank") - F.lit(template_offset),
    )
    new_flag = marker(
        "SGI", family_code, F.col("_sg_insert_ordinal"),
        schema["store_and_fwd_flag"].dataType,
    )
    df = (
        df.withColumn("store_and_fwd_flag", new_flag)
        .withColumn(
            "row_id",
            new_row_id_expr(
                schema["row_id"].dataType,
                max_numeric_row_id,
                family_code,
                F.col("_sg_insert_ordinal"),
            ),
        )
    )
    new_key = trip_key_expr(schema, {"store_and_fwd_flag": new_flag})
    df = (
        df.withColumn("trip_key", new_key)
        .withColumn(
            "state_partition_id",
            F.pmod(F.xxhash64("trip_key"), F.lit(partitions)).cast(
                schema["state_partition_id"].dataType
            ),
        )
        .withColumn("_sg_operation", F.lit("INSERT"))
        .withColumn(
            "_sg_match_row_id",
            F.lit(None).cast(schema["row_id"].dataType),
        )
        .withColumn(
            "_sg_match_partition_id",
            F.lit(None).cast(schema["state_partition_id"].dataType),
        )
    )
    return df.select(
        *columns, "_sg_operation", "_sg_match_row_id",
        "_sg_match_partition_id"
    )


def union_all(frames: List[DataFrame]) -> DataFrame:
    if not frames:
        raise RuntimeError("Mutation source list is empty.")
    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result


def load_matrix(spark: SparkSession, root: str) -> Dict[str, List[Row]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/mutation_matrix_csv")
        .filter(F.col("workload_class") == "PERFORMANCE_MATRIX")
        .select(
            "workload_id", "family_id",
            F.col("target_version").cast("int").alias("target_version"),
            F.col("cumulative_operations").cast("long").alias("cumulative_operations"),
            F.col("added_operations_this_version").cast("long").alias("added_operations"),
            F.col("cumulative_insert_count").cast("long").alias("cum_insert"),
            F.col("cumulative_update_count").cast("long").alias("cum_update"),
            F.col("cumulative_delete_count").cast("long").alias("cum_delete"),
            F.col("added_insert_count").cast("long").alias("add_insert"),
            F.col("added_update_count").cast("long").alias("add_update"),
            F.col("added_delete_count").cast("long").alias("add_delete"),
            F.col("expected_rows_after").cast("long").alias("expected_rows_after"),
        )
        .orderBy("family_id", "target_version")
        .collect()
    )
    grouped: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    if set(grouped) != set(FAMILY_CODES):
        raise RuntimeError(f"Unexpected families: {sorted(grouped)}")
    for family, family_rows in grouped.items():
        versions = [int(r["target_version"]) for r in family_rows]
        if versions != [1, 2, 3, 4]:
            raise RuntimeError(f"{family} versions are invalid: {versions}")
    return grouped


def verify_table(
    spark: SparkSession,
    path: str,
    expected_version: int,
    expected_rows: int,
    partitions: int,
) -> Dict[str, int]:
    table = DeltaTable.forPath(spark, path)
    version = latest_version(table)
    if version != expected_version:
        raise RuntimeError(
            f"{path}: expected version {expected_version}, found {version}."
        )
    detail = table.detail().collect()[0]
    props = detail["properties"] or {}
    if str(props.get("delta.enableChangeDataFeed", "false")).lower() != "true":
        raise RuntimeError(f"CDF is not enabled for {path}.")
    stats = (
        spark.read.format("delta").load(path)
        .agg(
            F.count(F.lit(1)).cast("long").alias("rows"),
            F.countDistinct("state_partition_id").cast("long").alias("parts"),
            F.sum(F.when(F.col("row_id").isNull(), 1).otherwise(0)).cast("long").alias("null_ids"),
            F.sum(F.when(F.col("trip_key").isNull(), 1).otherwise(0)).cast("long").alias("null_keys"),
        )
        .collect()[0]
    )
    if int(stats["rows"]) != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows} rows, found {stats['rows']}."
        )
    if int(stats["parts"]) != partitions:
        raise RuntimeError(f"{path}: logical partition count changed.")
    if int(stats["null_ids"]) or int(stats["null_keys"]):
        raise RuntimeError(f"{path}: null row IDs or trip keys found.")
    return {
        "version": version,
        "rows": int(stats["rows"]),
        "parts": int(stats["parts"]),
        "num_files": int(detail["numFiles"]),
        "size_bytes": int(detail["sizeInBytes"]),
    }


def main() -> None:
    args = parse_args()
    if args.state_partitions <= 0:
        raise ValueError("--state-partitions must be positive.")
    if args.max_passengers < 1:
        raise ValueError("--max-passengers must be at least 1.")
    if not 0 < args.candidate_hash_threshold < args.candidate_hash_modulus:
        raise ValueError("Invalid candidate hash threshold/modulus.")

    spark = SparkSession.builder.appName(
        "StateGuardBuildPerformanceWorkloads"
    ).getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.state_partitions))
    spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

    canonical_path = args.canonical_path.rstrip("/")
    matrix_root = args.research_matrix_root.rstrip("/")
    partition_state_path = args.consolidated_partition_state.rstrip("/")
    key_state_path = args.key_frequency_state.rstrip("/")
    working_root = args.working_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, canonical_path):
        raise RuntimeError(f"Not a Delta table: {canonical_path}")
    canonical_table = DeltaTable.forPath(spark, canonical_path)
    canonical_version = latest_version(canonical_table)
    if canonical_version != args.expected_canonical_version:
        raise RuntimeError(
            f"Canonical version mismatch: {canonical_version}."
        )
    canonical = spark.read.format("delta").load(canonical_path)
    schema = canonical.schema
    columns = canonical.columns
    missing = sorted(
        {"row_id", "trip_key", "state_partition_id", *RAW_COLUMNS}
        .difference(columns)
    )
    if missing:
        raise RuntimeError("Canonical columns missing: " + ", ".join(missing))
    canonical_stats = (
        canonical.agg(
            F.count(F.lit(1)).cast("long").alias("rows"),
            F.countDistinct("state_partition_id").cast("long").alias("parts"),
        ).collect()[0]
    )
    canonical_rows = int(canonical_stats["rows"])
    if canonical_rows != args.expected_canonical_rows:
        raise RuntimeError(
            f"Canonical rows: expected={args.expected_canonical_rows}, actual={canonical_rows}."
        )
    if int(canonical_stats["parts"]) != args.state_partitions:
        raise RuntimeError("Canonical partition count mismatch.")
    if not DeltaTable.isDeltaTable(spark, partition_state_path):
        raise RuntimeError("Consolidated partition state is missing.")
    if not DeltaTable.isDeltaTable(spark, key_state_path):
        raise RuntimeError("Exact key-frequency state is missing.")

    matrix = load_matrix(spark, matrix_root)
    max_insert = max(int(rows[-1]["cum_insert"]) for rows in matrix.values())
    max_update = max(int(rows[-1]["cum_update"]) for rows in matrix.values())
    max_delete = max(int(rows[-1]["cum_delete"]) for rows in matrix.values())
    required_pool = max_update + max_delete + max_insert
    update_offset = 0
    delete_offset = max_update
    insert_offset = max_update + max_delete

    print("=" * 78)
    print("STATEGUARD PERFORMANCE WORKLOAD CONSTRUCTION")
    print("=" * 78)
    print(f"Canonical version: {canonical_version}")
    print(f"Canonical rows: {canonical_rows}")
    print(f"Required clean candidate pool: {required_pool}")
    print("One Delta MERGE commit per cumulative workload level")
    print("=" * 78)

    selection_start = time.perf_counter()
    partition_state = (
        spark.read.format("delta").load(partition_state_path)
        .select(
            "state_partition_id", "minimum_fare", "maximum_fare",
            "minimum_distance", "maximum_distance"
        )
    )
    if partition_state.count() != args.state_partitions:
        raise RuntimeError("Consolidated state must contain 64 rows.")
    duplicate_keys = (
        spark.read.format("delta").load(key_state_path)
        .filter(F.col("frequency") > 1)
        .select("trip_key").distinct()
    )
    duplicate_key_count = duplicate_keys.count()
    min_pickup = F.lit(args.min_valid_pickup).cast("timestamp_ntz")
    max_pickup = F.lit(args.max_valid_pickup).cast("timestamp_ntz")
    candidates = (
        canonical.join(F.broadcast(partition_state), "state_partition_id")
        .join(F.broadcast(duplicate_keys), "trip_key", "left_anti")
        .filter(
            F.col("passenger_count").between(1, args.max_passengers)
            & F.col("fare_amount").isNotNull()
            & (F.col("fare_amount") >= 0)
            & F.col("trip_distance").isNotNull()
            & (F.col("trip_distance") >= 0)
            & F.col("tpep_pickup_datetime").isNotNull()
            & (F.col("tpep_pickup_datetime") >= min_pickup)
            & (F.col("tpep_pickup_datetime") <= max_pickup)
            & (F.col("fare_amount") > F.col("minimum_fare"))
            & (F.col("fare_amount") < F.col("maximum_fare"))
            & (F.col("trip_distance") > F.col("minimum_distance"))
            & (F.col("trip_distance") < F.col("maximum_distance"))
        )
        .withColumn(
            "_sg_hash",
            F.pmod(
                F.xxhash64("row_id", F.lit(args.base_seed)),
                F.lit(args.candidate_hash_modulus),
            ).cast("long"),
        )
        .filter(F.col("_sg_hash") < args.candidate_hash_threshold)
    )
    window = Window.orderBy(F.col("_sg_hash"), F.col("row_id").cast("string"))
    pool = (
        candidates.withColumn(
            "candidate_rank", F.row_number().over(window).cast("long")
        )
        .filter(F.col("candidate_rank") <= required_pool)
        .select(*columns, "candidate_rank")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    pool_count = pool.count()
    if pool_count != required_pool:
        raise RuntimeError(
            f"Clean candidate pool too small: required={required_pool}, actual={pool_count}."
        )
    target_manifest = f"{output_root}/candidate_target_manifest_delta"
    (
        pool.select(
            "candidate_rank", "row_id", "trip_key", "state_partition_id",
            "source_year_month", "passenger_count", "fare_amount",
            "trip_distance", "tpep_pickup_datetime"
        )
        .repartition(8)
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").save(target_manifest)
    )
    selection_seconds = time.perf_counter() - selection_start

    row_id_type = schema["row_id"].dataType
    max_numeric_row_id: Optional[int] = None
    if isinstance(row_id_type, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
        max_numeric_row_id = int(
            canonical.agg(F.max("row_id").alias("m")).collect()[0]["m"]
        )

    version_records: List[Dict[str, object]] = []
    table_records: List[Dict[str, object]] = []

    family_order = list(FAMILY_CODES)
    for family_id in family_order:
        family_code = FAMILY_CODES[family_id]
        rows = matrix[family_id]
        path = f"{working_root}/{family_id.lower()}/working_table"
        if DeltaTable.isDeltaTable(spark, path):
            raise RuntimeError(
                f"Working table already exists: {path}. Do not rerun this construction job."
            )
        print("-" * 78)
        print(f"BUILDING {family_id}")
        print(f"PATH={path}")
        build_start = time.perf_counter()
        (
            canonical.repartition(args.state_partitions, "state_partition_id")
            .write.format("delta").mode("errorifexists")
            .option("overwriteSchema", "true")
            .option("delta.enableChangeDataFeed", "true")
            .partitionBy("state_partition_id").save(path)
        )
        build_seconds = time.perf_counter() - build_start
        verify_table(
            spark, path, expected_version=0,
            expected_rows=canonical_rows, partitions=args.state_partitions
        )
        table = DeltaTable.forPath(spark, path)
        working_schema = spark.read.format("delta").load(path).schema
        working_columns = [field.name for field in working_schema.fields]
        previous_insert = previous_update = previous_delete = 0
        total_mutation_seconds = 0.0

        for row in rows:
            workload_id = str(row["workload_id"])
            version = int(row["target_version"])
            cum_insert = int(row["cum_insert"])
            cum_update = int(row["cum_update"])
            cum_delete = int(row["cum_delete"])
            add_insert = int(row["add_insert"])
            add_update = int(row["add_update"])
            add_delete = int(row["add_delete"])
            frames: List[DataFrame] = []
            if add_update:
                frames.append(
                    update_source(
                        pool,
                        update_offset + previous_update + 1,
                        update_offset + cum_update,
                        family_code, working_schema, working_columns,
                        args.state_partitions,
                    )
                )
            if add_delete:
                frames.append(
                    delete_source(
                        pool,
                        delete_offset + previous_delete + 1,
                        delete_offset + cum_delete,
                        working_columns,
                    )
                )
            if add_insert:
                frames.append(
                    insert_source(
                        pool, insert_offset,
                        previous_insert + 1, cum_insert,
                        family_code, working_schema, working_columns,
                        args.state_partitions, max_numeric_row_id,
                    )
                )
            source = union_all(frames).persist(StorageLevel.MEMORY_AND_DISK)
            source_count = source.count()
            expected_source_count = add_insert + add_update + add_delete
            if source_count != expected_source_count:
                raise RuntimeError(
                    f"{workload_id}: expected source={expected_source_count}, actual={source_count}."
                )
            duplicate_targets = (
                source.filter(F.col("_sg_operation").isin("UPDATE", "DELETE"))
                .groupBy("_sg_match_row_id").count()
                .filter(F.col("count") != 1).limit(1).count()
            )
            duplicate_new_keys = (
                source.filter(F.col("_sg_operation").isin("INSERT", "UPDATE"))
                .groupBy("trip_key").count()
                .filter(F.col("count") != 1).limit(1).count()
            )
            if duplicate_targets or duplicate_new_keys:
                raise RuntimeError(f"{workload_id}: duplicate target or generated key.")
            before = latest_version(table)
            if before != version - 1:
                raise RuntimeError(
                    f"{workload_id}: expected before version {version - 1}, found {before}."
                )
            mutation_start = time.perf_counter()
            (
                table.alias("t")
                .merge(
                    source.alias("s"),
                    (F.col("t.row_id") == F.col("s._sg_match_row_id"))
                    & (
                        F.col("t.state_partition_id")
                        == F.col("s._sg_match_partition_id")
                    ),
                )
                .whenMatchedDelete(F.col("s._sg_operation") == "DELETE")
                .whenMatchedUpdateAll(F.col("s._sg_operation") == "UPDATE")
                .whenNotMatchedInsertAll(F.col("s._sg_operation") == "INSERT")
                .execute()
            )
            mutation_seconds = time.perf_counter() - mutation_start
            total_mutation_seconds += mutation_seconds
            after = latest_version(table)
            if after != version:
                raise RuntimeError(
                    f"{workload_id}: expected version {version}, found {after}."
                )
            actual = cdf_counts(spark, path, version)
            expected = expected_cdf(add_insert, add_update, add_delete)
            if actual != expected:
                raise RuntimeError(
                    f"{workload_id}: CDF mismatch expected={expected}, actual={actual}."
                )
            history = table.history(1).collect()[0]
            version_records.append({
                "workload_id": workload_id,
                "family_id": family_id,
                "working_path": path,
                "before_version": before,
                "after_version": after,
                "cumulative_operations": int(row["cumulative_operations"]),
                "added_operations": int(row["added_operations"]),
                "cumulative_insert_count": cum_insert,
                "cumulative_update_count": cum_update,
                "cumulative_delete_count": cum_delete,
                "added_insert_count": add_insert,
                "added_update_count": add_update,
                "added_delete_count": add_delete,
                "expected_rows_after": int(row["expected_rows_after"]),
                "actual_cdf_insert": actual["insert"],
                "actual_cdf_delete": actual["delete"],
                "actual_cdf_update_preimage": actual["update_preimage"],
                "actual_cdf_update_postimage": actual["update_postimage"],
                "source_row_count": source_count,
                "mutation_seconds": float(mutation_seconds),
                "delta_operation": str(history["operation"]),
                "operation_metrics_json": json.dumps(
                    dict(history["operationMetrics"] or {}), sort_keys=True
                ),
                "status": "PASS",
            })
            print(
                f"{workload_id} version {before}->{after} "
                f"source={source_count} CDF={actual} "
                f"seconds={mutation_seconds:.3f} PASS"
            )
            previous_insert = cum_insert
            previous_update = cum_update
            previous_delete = cum_delete
            source.unpersist()

        final_expected_rows = int(rows[-1]["expected_rows_after"])
        final_info = verify_table(
            spark, path, expected_version=4,
            expected_rows=final_expected_rows,
            partitions=args.state_partitions,
        )
        table_records.append({
            "family_id": family_id,
            "family_code": family_code,
            "working_path": path,
            "start_version": 0,
            "final_version": final_info["version"],
            "start_rows": canonical_rows,
            "final_rows": final_info["rows"],
            "net_row_change": final_info["rows"] - canonical_rows,
            "state_partition_count": final_info["parts"],
            "num_files": final_info["num_files"],
            "size_bytes": final_info["size_bytes"],
            "build_seconds": float(build_seconds),
            "mutation_seconds": float(total_mutation_seconds),
            "status": "PASS",
        })

    role_records = [
        {"candidate_role": "UPDATE_TARGET", "start_rank": 1,
         "end_rank": max_update, "row_count": max_update},
        {"candidate_role": "DELETE_TARGET", "start_rank": delete_offset + 1,
         "end_rank": delete_offset + max_delete, "row_count": max_delete},
        {"candidate_role": "INSERT_TEMPLATE", "start_rank": insert_offset + 1,
         "end_rank": insert_offset + max_insert, "row_count": max_insert},
    ]
    summary_records = [{
        "status": "PASS",
        "canonical_delta_version": canonical_version,
        "canonical_row_count": canonical_rows,
        "performance_family_count": len(matrix),
        "workload_version_count": len(version_records),
        "candidate_pool_size": pool_count,
        "duplicate_key_count_excluded": duplicate_key_count,
        "max_insert_count": max_insert,
        "max_update_count": max_update,
        "max_delete_count": max_delete,
        "target_selection_seconds": float(selection_seconds),
        "all_cdf_checks_passed": True,
        "target_manifest_path": target_manifest,
        "working_root": working_root,
        "output_root": output_root,
    }]
    pool.unpersist()

    role_df = save_records(
        spark, role_records,
        f"{output_root}/candidate_role_summary_csv",
        f"{output_root}/candidate_role_summary_json",
        ["start_rank"],
    )
    version_df = save_records(
        spark, version_records,
        f"{output_root}/version_audit_csv",
        f"{output_root}/version_audit_json",
        ["family_id", "after_version"],
    )
    table_df = save_records(
        spark, table_records,
        f"{output_root}/table_manifest_csv",
        f"{output_root}/table_manifest_json",
        ["family_id"],
    )
    save_records(
        spark, summary_records,
        f"{output_root}/summary_csv",
        f"{output_root}/summary_json",
    )

    print()
    print("=" * 78)
    print("PERFORMANCE WORKING TABLE MANIFEST")
    print("=" * 78)
    table_df.select(
        "family_id", "final_version", "final_rows", "net_row_change",
        "num_files", "size_bytes", "build_seconds", "mutation_seconds",
        "status"
    ).show(10, truncate=False)
    print("=" * 78)
    print("STATEGUARD_PERFORMANCE_WORKLOADS_BEGIN")
    print("PERFORMANCE_WORKLOAD_STATUS=PASS")
    print(f"CANONICAL_DELTA_VERSION={canonical_version}")
    print(f"CANONICAL_ROW_COUNT={canonical_rows}")
    print(f"PERFORMANCE_FAMILY_COUNT={len(matrix)}")
    print(f"WORKLOAD_VERSION_COUNT={len(version_records)}")
    print(f"CANDIDATE_POOL_SIZE={pool_count}")
    print(f"DUPLICATE_KEYS_EXCLUDED={duplicate_key_count}")
    print(f"MAX_INSERT_COUNT={max_insert}")
    print(f"MAX_UPDATE_COUNT={max_update}")
    print(f"MAX_DELETE_COUNT={max_delete}")
    print(f"TARGET_SELECTION_SECONDS={selection_seconds:.3f}")
    print("ALL_CDF_CHECKS_PASSED=true")
    print(f"TARGET_MANIFEST_PATH={target_manifest}")
    print(f"VERSION_AUDIT_PATH={output_root}/version_audit_csv")
    print(f"TABLE_MANIFEST_PATH={output_root}/table_manifest_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_PERFORMANCE_WORKLOADS_END")
    print("=" * 78)
    spark.stop()


if __name__ == "__main__":
    main()
