import argparse
from typing import Any, Dict, List

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


WORKLOAD_FAMILIES = [
    {
        "family_order": 1,
        "family_id": "INSERT_ONLY",
        "insert_fraction": 1.0,
        "update_fraction": 0.0,
        "delete_fraction": 0.0,
        "description": (
            "Clean valid inserts; isolates append-oriented incremental "
            "maintenance without delete-sensitive fallback."
        ),
    },
    {
        "family_order": 2,
        "family_id": "UPDATE_ONLY",
        "insert_fraction": 0.0,
        "update_fraction": 1.0,
        "delete_fraction": 0.0,
        "description": (
            "Clean deterministic updates to existing unique rows; "
            "each update produces one preimage and one postimage."
        ),
    },
    {
        "family_order": 3,
        "family_id": "DELETE_ONLY",
        "insert_fraction": 0.0,
        "update_fraction": 0.0,
        "delete_fraction": 1.0,
        "description": (
            "Deletes of valid non-extreme unique rows; measures ordinary "
            "delete maintenance separately from adversarial extrema loss."
        ),
    },
    {
        "family_order": 4,
        "family_id": "MIXED_40I_30U_30D",
        "insert_fraction": 0.40,
        "update_fraction": 0.30,
        "delete_fraction": 0.30,
        "description": (
            "Production-style mixed workload with 40% inserts, 30% "
            "updates and 30% deletes."
        ),
    },
]

VALIDATION_PROFILES = {
    "BALANCED": {
        "description": "Every supported rule has equal validation demand.",
        "weights": {
            **{f"R{index:02d}": 1.0 for index in range(1, 14)}
        },
    },
    "ADDITIVE_HEAVY": {
        "description": (
            "Completeness and range counters dominate the recurring "
            "validation workload."
        ),
        "weights": {
            **{f"R{index:02d}": 10.0 for index in range(1, 8)},
            **{f"R{index:02d}": 2.0 for index in range(8, 12)},
            "R12": 1.0,
            "R13": 1.0,
        },
    },
    "EXTREMA_HEAVY": {
        "description": (
            "Minimum and maximum checks dominate; useful for testing "
            "partition-extrema state selection."
        ),
        "weights": {
            **{f"R{index:02d}": 2.0 for index in range(1, 8)},
            **{f"R{index:02d}": 10.0 for index in range(8, 12)},
            "R12": 1.0,
            "R13": 1.0,
        },
    },
    "UNIQUENESS_HEAVY": {
        "description": (
            "Duplicate and uniqueness validation dominate, increasing "
            "the potential value of the large exact key-frequency state."
        ),
        "weights": {
            **{f"R{index:02d}": 2.0 for index in range(1, 12)},
            "R12": 20.0,
            "R13": 20.0,
        },
    },
    "PRODUCTION_MIXED": {
        "description": (
            "Mixed demand: additive checks are frequent, extrema checks "
            "are periodic and uniqueness checks are less frequent."
        ),
        "weights": {
            "R01": 10.0,
            "R02": 8.0,
            "R03": 8.0,
            "R04": 8.0,
            "R05": 8.0,
            "R06": 8.0,
            "R07": 8.0,
            "R08": 4.0,
            "R09": 4.0,
            "R10": 4.0,
            "R11": 4.0,
            "R12": 3.0,
            "R13": 3.0,
        },
    },
}

DEFAULT_BUDGETS = [
    0,
    32768,
    65536,
    131072,
    262144,
    524288,
    268435456,
    1073741824,
    2500000000,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic StateGuard research workload, "
            "validation-demand and storage-budget matrices. The design "
            "uses four nested cumulative mutation tables rather than "
            "creating one full 67.72-million-row copy per experiment."
        )
    )
    parser.add_argument("--canonical-path", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--state-catalog", required=True)
    parser.add_argument("--state-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--sizes",
        default="1000,5000,20000,100000",
        help="Comma-separated cumulative mutation sizes.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="Comma-separated storage budgets in bytes.",
    )
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260806)
    parser.add_argument("--expected-version", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=67721884)
    parser.add_argument("--expected-state-count", type=int, default=12)
    return parser.parse_args()


def parse_positive_ints(
    text: str,
    argument_name: str,
    allow_zero: bool = False,
) -> List[int]:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} must contain only integers."
        ) from exc

    if not values:
        raise ValueError(f"{argument_name} cannot be empty.")

    minimum = 0 if allow_zero else 1

    if any(value < minimum for value in values):
        raise ValueError(
            f"{argument_name} values must be >= {minimum}."
        )

    if values != sorted(set(values)):
        raise ValueError(
            f"{argument_name} must be strictly increasing and unique."
        )

    return values


def write_csv(dataframe: DataFrame, path: str) -> None:
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def read_single_csv_row(
    spark: SparkSession,
    path: str,
) -> Row:
    rows = spark.read.option("header", "true").csv(path).collect()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row at {path}; found {len(rows)}."
        )

    return rows[0]


def required_int(row: Row, column_name: str) -> int:
    value = row[column_name]

    if value is None or value == "":
        raise RuntimeError(f"Missing integer field: {column_name}")

    return int(value)


def split_operation_counts(
    total: int,
    insert_fraction: float,
    update_fraction: float,
    delete_fraction: float,
) -> Dict[str, int]:
    fractions = [
        insert_fraction,
        update_fraction,
        delete_fraction,
    ]

    if abs(sum(fractions) - 1.0) > 1e-12:
        raise RuntimeError("Operation fractions must sum to 1.0.")

    insert_count = int(total * insert_fraction)
    update_count = int(total * update_fraction)
    delete_count = total - insert_count - update_count

    return {
        "insert_count": insert_count,
        "update_count": update_count,
        "delete_count": delete_count,
    }


def human_budget_label(value: int) -> str:
    if value == 0:
        return "0 B"

    units = [
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ]

    for suffix, divisor in units:
        if value >= divisor:
            return f"{value / divisor:.3f} {suffix}"

    return f"{value} B"


def main() -> None:
    args = parse_arguments()

    sizes = parse_positive_ints(args.sizes, "--sizes")
    budgets = parse_positive_ints(
        args.budgets,
        "--budgets",
        allow_zero=True,
    )

    if args.timing_repeats < 2:
        raise ValueError(
            "--timing-repeats must be at least 2 for stable timing."
        )

    if args.expected_version < 0:
        raise ValueError("--expected-version must be non-negative.")

    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive.")

    if args.expected_state_count <= 0:
        raise ValueError("--expected-state-count must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardResearchWorkloadMatrix")
        .getOrCreate()
    )

    canonical_path = args.canonical_path.rstrip("/")
    output_root = args.output_root.rstrip("/")

    if not DeltaTable.isDeltaTable(spark, canonical_path):
        raise RuntimeError(f"Not a Delta table: {canonical_path}")

    canonical_table = DeltaTable.forPath(spark, canonical_path)
    canonical_version = int(
        canonical_table.history(1).collect()[0]["version"]
    )

    if canonical_version != args.expected_version:
        raise RuntimeError(
            "Canonical version mismatch: "
            f"expected={args.expected_version}, "
            f"actual={canonical_version}"
        )

    baseline_summary = read_single_csv_row(
        spark,
        args.baseline_summary,
    )
    state_summary = read_single_csv_row(
        spark,
        args.state_summary,
    )

    baseline_version = required_int(
        baseline_summary,
        "delta_version",
    )
    baseline_rows = required_int(
        baseline_summary,
        "row_count",
    )
    state_count = required_int(
        state_summary,
        "state_count",
    )
    total_state_size_bytes = required_int(
        state_summary,
        "total_state_size_bytes",
    )
    key_frequency_size_bytes = required_int(
        state_summary,
        "key_frequency_size_bytes",
    )

    if baseline_version != args.expected_version:
        raise RuntimeError(
            "Baseline summary version differs from canonical version."
        )

    if baseline_rows != args.expected_rows:
        raise RuntimeError(
            "Baseline row count mismatch: "
            f"expected={args.expected_rows}, actual={baseline_rows}"
        )

    if state_count != args.expected_state_count:
        raise RuntimeError(
            "State count mismatch: "
            f"expected={args.expected_state_count}, actual={state_count}"
        )

    state_catalog = (
        spark.read.option("header", "true")
        .csv(args.state_catalog)
        .select(
            "state_id",
            "state_name",
            "state_family",
            "supported_rules",
            F.col("size_bytes").cast("long").alias("size_bytes"),
            F.col("build_seconds")
            .cast("double")
            .alias("build_seconds"),
        )
        .orderBy("state_id")
    )

    catalog_count = state_catalog.count()

    if catalog_count != args.expected_state_count:
        raise RuntimeError(
            f"Expected {args.expected_state_count} catalog rows; "
            f"found {catalog_count}."
        )

    null_catalog_values = (
        state_catalog.filter(
            F.col("state_id").isNull()
            | F.col("state_family").isNull()
            | F.col("size_bytes").isNull()
        )
        .limit(1)
        .count()
    )

    if null_catalog_values != 0:
        raise RuntimeError(
            "State catalog contains missing required values."
        )

    workload_rows: List[Dict[str, Any]] = []

    previous_size = 0

    for level_index, cumulative_size in enumerate(sizes, start=1):
        added_operations = cumulative_size - previous_size

        for family in WORKLOAD_FAMILIES:
            family_order = int(family["family_order"])
            family_id = str(family["family_id"])

            cumulative_counts = split_operation_counts(
                cumulative_size,
                float(family["insert_fraction"]),
                float(family["update_fraction"]),
                float(family["delete_fraction"]),
            )
            added_counts = split_operation_counts(
                added_operations,
                float(family["insert_fraction"]),
                float(family["update_fraction"]),
                float(family["delete_fraction"]),
            )

            expected_cdf_rows = (
                cumulative_counts["insert_count"]
                + cumulative_counts["delete_count"]
                + 2 * cumulative_counts["update_count"]
            )
            incremental_cdf_rows = (
                added_counts["insert_count"]
                + added_counts["delete_count"]
                + 2 * added_counts["update_count"]
            )
            cumulative_net_row_change = (
                cumulative_counts["insert_count"]
                - cumulative_counts["delete_count"]
            )
            expected_rows_after = (
                baseline_rows + cumulative_net_row_change
            )
            mutation_fraction_percent = (
                100.0 * cumulative_size / baseline_rows
            )
            cdf_fraction_percent = (
                100.0 * expected_cdf_rows / expected_rows_after
            )

            workload_id = (
                f"{family_order:02d}_{family_id}_"
                f"N{cumulative_size:06d}"
            )
            table_id = f"table_{family_id.lower()}"
            workload_seed = (
                args.base_seed
                + family_order * 1_000_000
                + level_index * 10_000
            )

            workload_rows.append(
                {
                    "workload_order": (
                        family_order * 100 + level_index
                    ),
                    "workload_id": workload_id,
                    "family_id": family_id,
                    "workload_class": "PERFORMANCE_MATRIX",
                    "table_id": table_id,
                    "execution_strategy": "NESTED_CUMULATIVE",
                    "target_version": level_index,
                    "cumulative_operations": cumulative_size,
                    "added_operations_this_version": added_operations,
                    "cumulative_insert_count": (
                        cumulative_counts["insert_count"]
                    ),
                    "cumulative_update_count": (
                        cumulative_counts["update_count"]
                    ),
                    "cumulative_delete_count": (
                        cumulative_counts["delete_count"]
                    ),
                    "added_insert_count": (
                        added_counts["insert_count"]
                    ),
                    "added_update_count": (
                        added_counts["update_count"]
                    ),
                    "added_delete_count": (
                        added_counts["delete_count"]
                    ),
                    "expected_cumulative_cdf_rows": expected_cdf_rows,
                    "expected_incremental_cdf_rows": incremental_cdf_rows,
                    "expected_rows_after": expected_rows_after,
                    "mutation_fraction_percent": (
                        mutation_fraction_percent
                    ),
                    "cdf_fraction_percent": cdf_fraction_percent,
                    "timing_repeats": args.timing_repeats,
                    "workload_seed": workload_seed,
                    "clean_target_policy": (
                        "VALID_UNIQUE_NON_EXTREME"
                    ),
                    "requires_fresh_table": level_index == 1,
                    "oracle_subset": (
                        cumulative_size in {sizes[0], sizes[-2]}
                    ),
                    "full_validation_required": True,
                    "exact_correctness_required": True,
                    "description": str(family["description"]),
                }
            )

        previous_size = cumulative_size

    # The already completed adversarial workload is preserved as a distinct
    # correctness-stress experiment rather than mixed into latency claims.
    workload_rows.append(
        {
            "workload_order": 999,
            "workload_id": "CORNER_STRESS_V0_TO_V12",
            "family_id": "CORNER_STRESS",
            "workload_class": "CORRECTNESS_STRESS",
            "table_id": "all17_correctness_v1",
            "execution_strategy": "EXISTING_VERIFIED_RUN",
            "target_version": 12,
            "cumulative_operations": 12,
            "added_operations_this_version": 12,
            "cumulative_insert_count": 9,
            "cumulative_update_count": 1,
            "cumulative_delete_count": 2,
            "added_insert_count": 9,
            "added_update_count": 1,
            "added_delete_count": 2,
            "expected_cumulative_cdf_rows": 13,
            "expected_incremental_cdf_rows": 13,
            "expected_rows_after": 67721891,
            "mutation_fraction_percent": (
                100.0 * 12 / baseline_rows
            ),
            "cdf_fraction_percent": (
                100.0 * 13 / 67721891
            ),
            "timing_repeats": 1,
            "workload_seed": args.base_seed,
            "clean_target_policy": (
                "REAL_OUTLIERS_PLUS_CONTROLLED_CORNER_CASES"
            ),
            "requires_fresh_table": False,
            "oracle_subset": False,
            "full_validation_required": True,
            "exact_correctness_required": True,
            "description": (
                "Existing adversarial correctness workload with nulls, "
                "negative values, old/future timestamps, duplicate "
                "transitions and delete-sensitive extrema."
            ),
        }
    )

    workload_schema = T.StructType(
        [
            T.StructField("workload_order", T.IntegerType(), False),
            T.StructField("workload_id", T.StringType(), False),
            T.StructField("family_id", T.StringType(), False),
            T.StructField("workload_class", T.StringType(), False),
            T.StructField("table_id", T.StringType(), False),
            T.StructField("execution_strategy", T.StringType(), False),
            T.StructField("target_version", T.IntegerType(), False),
            T.StructField(
                "cumulative_operations",
                T.LongType(),
                False,
            ),
            T.StructField(
                "added_operations_this_version",
                T.LongType(),
                False,
            ),
            T.StructField(
                "cumulative_insert_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "cumulative_update_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "cumulative_delete_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "added_insert_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "added_update_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "added_delete_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "expected_cumulative_cdf_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "expected_incremental_cdf_rows",
                T.LongType(),
                False,
            ),
            T.StructField(
                "expected_rows_after",
                T.LongType(),
                False,
            ),
            T.StructField(
                "mutation_fraction_percent",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "cdf_fraction_percent",
                T.DoubleType(),
                False,
            ),
            T.StructField("timing_repeats", T.IntegerType(), False),
            T.StructField("workload_seed", T.LongType(), False),
            T.StructField(
                "clean_target_policy",
                T.StringType(),
                False,
            ),
            T.StructField(
                "requires_fresh_table",
                T.BooleanType(),
                False,
            ),
            T.StructField(
                "oracle_subset",
                T.BooleanType(),
                False,
            ),
            T.StructField(
                "full_validation_required",
                T.BooleanType(),
                False,
            ),
            T.StructField(
                "exact_correctness_required",
                T.BooleanType(),
                False,
            ),
            T.StructField("description", T.StringType(), False),
        ]
    )

    workload_df = (
        spark.createDataFrame(
            workload_rows,
            schema=workload_schema,
        )
        .orderBy("workload_order")
    )

    profile_rows: List[Dict[str, Any]] = []

    for profile_order, (
        profile_id,
        profile,
    ) in enumerate(VALIDATION_PROFILES.items(), start=1):
        weights = dict(profile["weights"])
        total_weight = float(sum(weights.values()))

        if set(weights) != {
            f"R{index:02d}" for index in range(1, 14)
        }:
            raise RuntimeError(
                f"Profile {profile_id} does not cover all 13 rules."
            )

        for rule_number in range(1, 14):
            rule_id = f"R{rule_number:02d}"
            raw_weight = float(weights[rule_id])

            profile_rows.append(
                {
                    "profile_order": profile_order,
                    "profile_id": profile_id,
                    "rule_order": rule_number,
                    "rule_id": rule_id,
                    "raw_weight": raw_weight,
                    "normalized_weight": (
                        raw_weight / total_weight
                    ),
                    "description": str(profile["description"]),
                }
            )

    profile_schema = T.StructType(
        [
            T.StructField("profile_order", T.IntegerType(), False),
            T.StructField("profile_id", T.StringType(), False),
            T.StructField("rule_order", T.IntegerType(), False),
            T.StructField("rule_id", T.StringType(), False),
            T.StructField("raw_weight", T.DoubleType(), False),
            T.StructField(
                "normalized_weight",
                T.DoubleType(),
                False,
            ),
            T.StructField("description", T.StringType(), False),
        ]
    )

    profile_df = (
        spark.createDataFrame(
            profile_rows,
            schema=profile_schema,
        )
        .orderBy("profile_order", "rule_order")
    )

    budget_rows: List[Dict[str, Any]] = []

    for budget_order, budget_bytes in enumerate(budgets, start=1):
        budget_rows.append(
            {
                "budget_order": budget_order,
                "budget_id": f"B{budget_order:02d}",
                "budget_bytes": budget_bytes,
                "budget_label": human_budget_label(budget_bytes),
                "fraction_of_all_state_storage": (
                    budget_bytes / total_state_size_bytes
                    if total_state_size_bytes > 0
                    else 0.0
                ),
                "can_fit_key_frequency_state_alone": (
                    budget_bytes >= key_frequency_size_bytes
                ),
                "can_fit_all_states": (
                    budget_bytes >= total_state_size_bytes
                ),
            }
        )

    budget_schema = T.StructType(
        [
            T.StructField("budget_order", T.IntegerType(), False),
            T.StructField("budget_id", T.StringType(), False),
            T.StructField("budget_bytes", T.LongType(), False),
            T.StructField("budget_label", T.StringType(), False),
            T.StructField(
                "fraction_of_all_state_storage",
                T.DoubleType(),
                False,
            ),
            T.StructField(
                "can_fit_key_frequency_state_alone",
                T.BooleanType(),
                False,
            ),
            T.StructField(
                "can_fit_all_states",
                T.BooleanType(),
                False,
            ),
        ]
    )

    budget_df = (
        spark.createDataFrame(
            budget_rows,
            schema=budget_schema,
        )
        .orderBy("budget_order")
    )

    performance_workload_count = (
        workload_df.filter(
            F.col("workload_class") == "PERFORMANCE_MATRIX"
        ).count()
    )
    stress_workload_count = (
        workload_df.filter(
            F.col("workload_class") == "CORRECTNESS_STRESS"
        ).count()
    )

    expected_performance_count = (
        len(WORKLOAD_FAMILIES) * len(sizes)
    )

    if performance_workload_count != expected_performance_count:
        raise RuntimeError(
            "Performance workload count does not match design."
        )

    profile_weight_check = (
        profile_df.groupBy("profile_id")
        .agg(
            F.sum("normalized_weight")
            .alias("normalized_sum"),
            F.count(F.lit(1)).alias("rule_count"),
        )
        .filter(
            (F.abs(F.col("normalized_sum") - F.lit(1.0)) > 1e-12)
            | (F.col("rule_count") != 13)
        )
        .count()
    )

    if profile_weight_check != 0:
        raise RuntimeError(
            "One or more validation profiles are invalid."
        )

    summary_schema = T.StructType(
        [
            T.StructField("status", T.StringType(), False),
            T.StructField(
                "canonical_delta_version",
                T.LongType(),
                False,
            ),
            T.StructField(
                "canonical_row_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "workload_family_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "mutation_size_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "performance_workload_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "correctness_stress_workload_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "physical_working_table_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "timing_repeats",
                T.LongType(),
                False,
            ),
            T.StructField(
                "validation_profile_count",
                T.LongType(),
                False,
            ),
            T.StructField(
                "budget_count",
                T.LongType(),
                False,
            ),
            T.StructField("state_count", T.LongType(), False),
            T.StructField(
                "total_state_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "key_frequency_size_bytes",
                T.LongType(),
                False,
            ),
            T.StructField(
                "estimated_full_table_copies_avoided",
                T.LongType(),
                False,
            ),
            T.StructField("output_root", T.StringType(), False),
        ]
    )

    naive_table_count = expected_performance_count
    physical_table_count = len(WORKLOAD_FAMILIES)

    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                canonical_version,
                baseline_rows,
                len(WORKLOAD_FAMILIES),
                len(sizes),
                performance_workload_count,
                stress_workload_count,
                physical_table_count,
                args.timing_repeats,
                len(VALIDATION_PROFILES),
                len(budgets),
                state_count,
                total_state_size_bytes,
                key_frequency_size_bytes,
                naive_table_count - physical_table_count,
                output_root,
            )
        ],
        schema=summary_schema,
    )

    write_csv(workload_df, f"{output_root}/mutation_matrix_csv")
    write_csv(
        profile_df,
        f"{output_root}/validation_profiles_csv",
    )
    write_csv(budget_df, f"{output_root}/budget_matrix_csv")
    write_csv(
        state_catalog,
        f"{output_root}/measured_state_catalog_csv",
    )
    write_csv(summary_df, f"{output_root}/summary_csv")

    (
        workload_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/mutation_matrix_json")
    )
    (
        profile_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/validation_profiles_json")
    )
    (
        budget_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/budget_matrix_json")
    )
    (
        state_catalog.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/measured_state_catalog_json")
    )
    (
        summary_df.coalesce(1)
        .write.mode("overwrite")
        .json(f"{output_root}/summary_json")
    )

    print()
    print("=" * 78)
    print("STATEGUARD RESEARCH WORKLOAD MATRIX")
    print("=" * 78)

    workload_df.select(
        "workload_id",
        "family_id",
        "target_version",
        "cumulative_operations",
        "added_operations_this_version",
        "expected_cumulative_cdf_rows",
        "timing_repeats",
        "oracle_subset",
    ).show(30, truncate=False)

    print("=" * 78)
    print("STORAGE BUDGET MATRIX")
    print("=" * 78)

    budget_df.select(
        "budget_id",
        "budget_label",
        "fraction_of_all_state_storage",
        "can_fit_key_frequency_state_alone",
        "can_fit_all_states",
    ).show(20, truncate=False)

    print("=" * 78)
    print("STATEGUARD_RESEARCH_MATRIX_BEGIN")
    print("RESEARCH_MATRIX_STATUS=PASS")
    print(f"CANONICAL_DELTA_VERSION={canonical_version}")
    print(f"CANONICAL_ROW_COUNT={baseline_rows}")
    print(f"WORKLOAD_FAMILY_COUNT={len(WORKLOAD_FAMILIES)}")
    print(f"MUTATION_SIZE_COUNT={len(sizes)}")
    print(
        "PERFORMANCE_WORKLOAD_COUNT="
        f"{performance_workload_count}"
    )
    print(
        "CORRECTNESS_STRESS_WORKLOAD_COUNT="
        f"{stress_workload_count}"
    )
    print(
        f"PHYSICAL_WORKING_TABLE_COUNT={physical_table_count}"
    )
    print(
        "FULL_TABLE_COPIES_AVOIDED="
        f"{naive_table_count - physical_table_count}"
    )
    print(f"TIMING_REPEATS={args.timing_repeats}")
    print(
        f"VALIDATION_PROFILE_COUNT={len(VALIDATION_PROFILES)}"
    )
    print(f"BUDGET_COUNT={len(budgets)}")
    print(f"STATE_COUNT={state_count}")
    print(f"TOTAL_STATE_SIZE_BYTES={total_state_size_bytes}")
    print(
        "KEY_FREQUENCY_SIZE_BYTES="
        f"{key_frequency_size_bytes}"
    )
    print(
        f"MUTATION_MATRIX_PATH={output_root}/mutation_matrix_csv"
    )
    print(
        "VALIDATION_PROFILE_PATH="
        f"{output_root}/validation_profiles_csv"
    )
    print(
        f"BUDGET_MATRIX_PATH={output_root}/budget_matrix_csv"
    )
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_RESEARCH_MATRIX_END")
    print("=" * 78)

    spark.stop()


if __name__ == "__main__":
    main()
