import argparse
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


RULE_IDS = [f"R{i:02d}" for i in range(1, 14)]
SCALAR_RULES = set(RULE_IDS[:11])
DUPLICATE_RULES = {"R12", "R13"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a budget-aware greedy StateGuard state selector with an "
            "exhaustive 2^12 oracle using measured state sizes, measured full-scan "
            "costs, measured recurring S12 path cost, five validation-demand profiles, "
            "and nine storage budgets."
        )
    )
    parser.add_argument("--research-matrix-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--s12-ablation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--amortization-validations", type=int, default=1000)
    parser.add_argument("--expected-state-count", type=int, default=12)
    parser.add_argument("--expected-profile-count", type=int, default=5)
    parser.add_argument("--expected-budget-count", type=int, default=9)
    return parser.parse_args()


def write_csv(df: DataFrame, path: str) -> None:
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(path)
    )


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def parse_rules(value: Any) -> Tuple[str, ...]:
    rules = tuple(sorted({x.strip() for x in str(value).split(",") if x.strip()}))
    if not rules:
        raise RuntimeError("A state has no supported rules.")
    unknown = set(rules) - set(RULE_IDS)
    if unknown:
        raise RuntimeError(f"Unknown rules in state catalog: {sorted(unknown)}")
    return rules


def load_states(
    spark: SparkSession,
    root: str,
    expected_count: int,
) -> List[Dict[str, Any]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/measured_state_catalog_csv")
        .select(
            "state_id",
            "state_name",
            "state_family",
            "supported_rules",
            F.col("size_bytes").cast("long").alias("size_bytes"),
            F.col("build_seconds").cast("double").alias("build_seconds"),
        )
        .collect()
    )
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} states; found {len(rows)}.")

    states: List[Dict[str, Any]] = []
    for row in rows:
        state = {
            "state_id": str(row["state_id"]),
            "state_name": str(row["state_name"]),
            "state_family": str(row["state_family"]),
            "supported_rules": parse_rules(row["supported_rules"]),
            "size_bytes": int(row["size_bytes"]),
            "build_seconds": float(row["build_seconds"]),
        }
        if state["size_bytes"] <= 0:
            raise RuntimeError(f"{state['state_id']} has non-positive storage size.")
        if state["build_seconds"] < 0:
            raise RuntimeError(f"{state['state_id']} has negative build time.")
        states.append(state)

    states.sort(key=lambda x: x["state_id"])
    expected_ids = [f"S{i:02d}" for i in range(1, expected_count + 1)]
    if [x["state_id"] for x in states] != expected_ids:
        raise RuntimeError("State IDs do not match S01-S12.")
    return states


def load_profiles(
    spark: SparkSession,
    root: str,
    expected_count: int,
) -> List[Dict[str, Any]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/validation_profiles_csv")
        .select(
            F.col("profile_order").cast("int").alias("profile_order"),
            "profile_id",
            "rule_id",
            F.col("normalized_weight").cast("double").alias("normalized_weight"),
            "description",
        )
        .collect()
    )
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = str(row["profile_id"])
        item = grouped.setdefault(
            pid,
            {
                "profile_order": int(row["profile_order"]),
                "profile_id": pid,
                "description": str(row["description"]),
                "weights": {},
            },
        )
        item["weights"][str(row["rule_id"])] = float(row["normalized_weight"])

    profiles = sorted(grouped.values(), key=lambda x: x["profile_order"])
    if len(profiles) != expected_count:
        raise RuntimeError(f"Expected {expected_count} profiles; found {len(profiles)}.")
    for profile in profiles:
        if set(profile["weights"]) != set(RULE_IDS):
            raise RuntimeError(f"{profile['profile_id']} does not cover all rules.")
        if abs(sum(profile["weights"].values()) - 1.0) > 1e-9:
            raise RuntimeError(f"{profile['profile_id']} weights do not sum to 1.")
    return profiles


def load_budgets(
    spark: SparkSession,
    root: str,
    expected_count: int,
) -> List[Dict[str, Any]]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/budget_matrix_csv")
        .select(
            F.col("budget_order").cast("int").alias("budget_order"),
            "budget_id",
            F.col("budget_bytes").cast("long").alias("budget_bytes"),
            "budget_label",
            F.col("fraction_of_all_state_storage").cast("double").alias(
                "fraction_of_all_state_storage"
            ),
            "can_fit_key_frequency_state_alone",
            "can_fit_all_states",
        )
        .collect()
    )
    budgets = [
        {
            "budget_order": int(r["budget_order"]),
            "budget_id": str(r["budget_id"]),
            "budget_bytes": int(r["budget_bytes"]),
            "budget_label": str(r["budget_label"]),
            "fraction_of_all_state_storage": float(r["fraction_of_all_state_storage"]),
            "can_fit_key_frequency_state_alone": parse_bool(
                r["can_fit_key_frequency_state_alone"]
            ),
            "can_fit_all_states": parse_bool(r["can_fit_all_states"]),
        }
        for r in rows
    ]
    budgets.sort(key=lambda x: x["budget_order"])
    if len(budgets) != expected_count:
        raise RuntimeError(f"Expected {expected_count} budgets; found {len(budgets)}.")
    if any(
        budgets[i]["budget_bytes"] > budgets[i + 1]["budget_bytes"]
        for i in range(len(budgets) - 1)
    ):
        raise RuntimeError("Budgets are not sorted by size.")
    return budgets


def load_baseline_costs(spark: SparkSession, root: str) -> Dict[str, float]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/trial_results_csv")
        .filter(
            (F.col("method") == "FULL_VALIDATION")
            & (F.lower(F.col("is_warmup")) == F.lit("false"))
        )
        .select(
            F.col("scalar_seconds").cast("double").alias("scalar_seconds"),
            F.col("duplicate_seconds").cast("double").alias("duplicate_seconds"),
            F.col("total_seconds").cast("double").alias("total_seconds"),
        )
        .collect()
    )
    if len(rows) != 48:
        raise RuntimeError(
            f"Expected 48 measured full-validation trials; found {len(rows)}."
        )
    scalar = [float(r["scalar_seconds"]) for r in rows]
    duplicate = [float(r["duplicate_seconds"]) for r in rows]
    total = [float(r["total_seconds"]) for r in rows]
    return {
        "median_scalar_seconds": float(statistics.median(scalar)),
        "median_duplicate_seconds": float(statistics.median(duplicate)),
        "median_total_seconds": float(statistics.median(total)),
        "trial_count": float(len(rows)),
    }


def load_s12_recurring_costs(
    spark: SparkSession,
    root: str,
) -> Dict[str, float]:
    rows = (
        spark.read.option("header", "true")
        .csv(f"{root}/method_summary_csv")
        .select(
            F.col("median_s12_incremental_seconds").cast("double").alias(
                "median_s12_incremental_seconds"
            ),
            F.col("median_full_snapshot_fallback_seconds").cast("double").alias(
                "median_full_snapshot_fallback_seconds"
            ),
            F.col("s12_speedup_vs_fallback").cast("double").alias(
                "s12_speedup_vs_fallback"
            ),
            F.col("exact_match_count").cast("int").alias("exact_match_count"),
        )
        .collect()
    )
    if len(rows) != 4:
        raise RuntimeError(
            f"Expected 4 measured S12 ablation workloads; found {len(rows)}."
        )

    s12_seconds = [float(r["median_s12_incremental_seconds"]) for r in rows]
    fallback_seconds = [
        float(r["median_full_snapshot_fallback_seconds"]) for r in rows
    ]
    speedups = [float(r["s12_speedup_vs_fallback"]) for r in rows]

    if any(int(r["exact_match_count"]) != 2 for r in rows):
        raise RuntimeError("S12 ablation contains a non-exact R12/R13 result.")
    if any(x <= 0.0 for x in s12_seconds + fallback_seconds):
        raise RuntimeError("S12 ablation contains a non-positive runtime.")

    return {
        "median_s12_incremental_seconds": float(statistics.median(s12_seconds)),
        "median_full_snapshot_fallback_seconds": float(
            statistics.median(fallback_seconds)
        ),
        "median_s12_speedup_vs_fallback": float(statistics.median(speedups)),
        "workload_count": float(len(rows)),
    }


def rule_cost(rule_id: str, costs: Dict[str, float]) -> float:
    if rule_id in SCALAR_RULES:
        return float(costs["median_scalar_seconds"])
    if rule_id in DUPLICATE_RULES:
        return float(costs["median_duplicate_seconds"])
    raise RuntimeError(f"Unknown rule {rule_id}.")


def evaluate(
    indices: Sequence[int],
    states: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    costs: Dict[str, float],
    amortization_validations: int,
    s12_recurring_seconds: float,
) -> Dict[str, Any]:
    chosen = [states[i] for i in indices]
    covered: Set[str] = set()
    for state in chosen:
        covered.update(state["supported_rules"])

    storage = sum(int(s["size_bytes"]) for s in chosen)
    build_seconds = sum(float(s["build_seconds"]) for s in chosen)
    gross = sum(float(weights[r]) * rule_cost(r, costs) for r in covered)
    amortized = build_seconds / float(amortization_validations)

    # S01-S11 retain the original compact-state model. S12 is calibrated with
    # the measured recurring exact S12 R12/R13 path from the completed ablation.
    # The validation-demand weights make this penalty comparable to the
    # weighted avoided full-snapshot uniqueness cost in `gross`.
    s12_selected = any(s["state_id"] == "S12" for s in chosen)
    s12_rule_weight = (
        sum(float(weights[r]) for r in DUPLICATE_RULES)
        if s12_selected
        else 0.0
    )
    s12_recurring_weighted = s12_rule_weight * float(s12_recurring_seconds)

    objective = gross - s12_recurring_weighted - amortized
    baseline_expected = sum(float(weights[r]) * rule_cost(r, costs) for r in RULE_IDS)
    remaining = (
        max(0.0, baseline_expected - gross)
        + s12_recurring_weighted
        + amortized
    )

    return {
        "state_ids": tuple(s["state_id"] for s in chosen),
        "covered_rules": tuple(sorted(covered)),
        "storage_bytes": storage,
        "build_seconds": build_seconds,
        "gross_avoided_seconds": gross,
        "amortized_build_seconds": amortized,
        "s12_recurring_weighted_seconds": s12_recurring_weighted,
        "net_benefit_seconds": objective,
        "weighted_coverage": sum(float(weights[r]) for r in covered),
        "predicted_remaining_seconds": remaining,
    }


def is_better(candidate: Dict[str, Any], incumbent: Optional[Dict[str, Any]]) -> bool:
    if incumbent is None:
        return True
    c = float(candidate["net_benefit_seconds"])
    i = float(incumbent["net_benefit_seconds"])
    if c > i + 1e-12:
        return True
    if i > c + 1e-12:
        return False
    if int(candidate["storage_bytes"]) < int(incumbent["storage_bytes"]):
        return True
    if int(candidate["storage_bytes"]) > int(incumbent["storage_bytes"]):
        return False
    if len(candidate["state_ids"]) < len(incumbent["state_ids"]):
        return True
    if len(candidate["state_ids"]) > len(incumbent["state_ids"]):
        return False
    return tuple(candidate["state_ids"]) < tuple(incumbent["state_ids"])


def oracle_select(
    states: Sequence[Dict[str, Any]],
    budget: int,
    weights: Dict[str, float],
    costs: Dict[str, float],
    amortization_validations: int,
    s12_recurring_seconds: float,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    best: Optional[Dict[str, Any]] = None
    n = len(states)
    for mask in range(1 << n):
        indices = [i for i in range(n) if mask & (1 << i)]
        if sum(int(states[i]["size_bytes"]) for i in indices) > budget:
            continue
        candidate = evaluate(
            indices,
            states,
            weights,
            costs,
            amortization_validations,
            s12_recurring_seconds,
        )
        if is_better(candidate, best):
            best = candidate
    if best is None:
        raise RuntimeError("Oracle found no feasible portfolio.")
    return best, float(time.perf_counter() - started)


def greedy_select(
    states: Sequence[Dict[str, Any]],
    budget: int,
    weights: Dict[str, float],
    costs: Dict[str, float],
    amortization_validations: int,
    s12_recurring_seconds: float,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    selected: List[int] = []
    remaining = set(range(len(states)))
    current = evaluate(
        selected,
        states,
        weights,
        costs,
        amortization_validations,
        s12_recurring_seconds,
    )

    while remaining:
        best_index: Optional[int] = None
        best_candidate: Optional[Dict[str, Any]] = None
        best_density = float("-inf")
        best_gain = float("-inf")

        for index in sorted(remaining):
            candidate = evaluate(
                sorted(selected + [index]),
                states,
                weights,
                costs,
                amortization_validations,
                s12_recurring_seconds,
            )
            if int(candidate["storage_bytes"]) > budget:
                continue
            gain = float(candidate["net_benefit_seconds"]) - float(
                current["net_benefit_seconds"]
            )
            if gain <= 1e-12:
                continue
            density = gain / float(states[index]["size_bytes"])
            key = (density, gain, -int(states[index]["size_bytes"]), states[index]["state_id"])
            current_best_key = (
                best_density,
                best_gain,
                -int(states[best_index]["size_bytes"]) if best_index is not None else float("-inf"),
                states[best_index]["state_id"] if best_index is not None else "",
            )
            if best_index is None or key > current_best_key:
                best_index = index
                best_candidate = candidate
                best_density = density
                best_gain = gain

        if best_index is None or best_candidate is None:
            break
        selected.append(best_index)
        selected.sort()
        remaining.remove(best_index)
        current = best_candidate

    return current, float(time.perf_counter() - started)


def portfolio_row(
    method: str,
    profile: Dict[str, Any],
    budget: Dict[str, Any],
    portfolio: Dict[str, Any],
    selection_seconds: float,
    oracle: Dict[str, Any],
) -> Dict[str, Any]:
    oracle_value = float(oracle["net_benefit_seconds"])
    value = float(portfolio["net_benefit_seconds"])
    absolute_gap = max(0.0, oracle_value - value)
    gap_percent = 0.0 if abs(oracle_value) <= 1e-12 else 100.0 * absolute_gap / abs(oracle_value)
    budget_bytes = int(budget["budget_bytes"])
    storage = int(portfolio["storage_bytes"])
    return {
        "profile_order": int(profile["profile_order"]),
        "profile_id": str(profile["profile_id"]),
        "budget_order": int(budget["budget_order"]),
        "budget_id": str(budget["budget_id"]),
        "budget_label": str(budget["budget_label"]),
        "budget_bytes": budget_bytes,
        "method": method,
        "selected_state_count": len(portfolio["state_ids"]),
        "selected_state_ids": ",".join(portfolio["state_ids"]),
        "covered_rule_count": len(portfolio["covered_rules"]),
        "covered_rule_ids": ",".join(portfolio["covered_rules"]),
        "selected_storage_bytes": storage,
        "budget_utilization": 0.0 if budget_bytes == 0 else storage / float(budget_bytes),
        "weighted_rule_coverage": float(portfolio["weighted_coverage"]),
        "gross_avoided_scan_seconds": float(portfolio["gross_avoided_seconds"]),
        "amortized_build_seconds": float(portfolio["amortized_build_seconds"]),
        "s12_recurring_weighted_seconds": float(
            portfolio["s12_recurring_weighted_seconds"]
        ),
        "net_benefit_seconds": value,
        "predicted_remaining_scan_seconds": float(portfolio["predicted_remaining_seconds"]),
        "selection_seconds": float(selection_seconds),
        "oracle_absolute_gap_seconds": float(absolute_gap),
        "oracle_gap_percent": float(gap_percent),
        "exactness_strategy": "STATE_OR_EXACT_FULL_SCAN_FALLBACK",
        "status": "PASS",
    }


def main() -> None:
    args = parse_args()
    if args.amortization_validations <= 0:
        raise ValueError("--amortization-validations must be positive.")

    spark = (
        SparkSession.builder
        .appName("StateGuardBudgetAwareStateSelectionV2")
        .getOrCreate()
    )

    research_root = args.research_matrix_root.rstrip("/")
    baseline_root = args.baseline_root.rstrip("/")
    s12_ablation_root = args.s12_ablation_root.rstrip("/")
    output_root = args.output_root.rstrip("/")

    states = load_states(spark, research_root, args.expected_state_count)
    profiles = load_profiles(spark, research_root, args.expected_profile_count)
    budgets = load_budgets(spark, research_root, args.expected_budget_count)
    costs = load_baseline_costs(spark, baseline_root)
    s12_costs = load_s12_recurring_costs(spark, s12_ablation_root)
    s12_recurring_seconds = float(
        s12_costs["median_s12_incremental_seconds"]
    )

    print("=" * 80)
    print("STATEGUARD BUDGET-AWARE SELECTOR VS EXHAUSTIVE ORACLE")
    print("=" * 80)
    print(f"Measured states: {len(states)}")
    print(f"Profiles: {len(profiles)}")
    print(f"Budgets: {len(budgets)}")
    print(f"Oracle search space per scenario: {1 << len(states)}")
    print(f"Median scalar full scan: {costs['median_scalar_seconds']:.6f}s")
    print(f"Median uniqueness full scan: {costs['median_duplicate_seconds']:.6f}s")
    print(
        "Measured median S12 recurring exact path: "
        f"{s12_recurring_seconds:.6f}s"
    )
    print(
        "Measured median S12-ablation fallback: "
        f"{s12_costs['median_full_snapshot_fallback_seconds']:.6f}s"
    )
    print(
        "Measured median S12 speedup vs fallback: "
        f"{s12_costs['median_s12_speedup_vs_fallback']:.6f}x"
    )
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    for profile in profiles:
        print(f"PROFILE={profile['profile_id']}")
        for budget in budgets:
            oracle, oracle_seconds = oracle_select(
                states,
                int(budget["budget_bytes"]),
                profile["weights"],
                costs,
                args.amortization_validations,
                s12_recurring_seconds,
            )
            greedy, greedy_seconds = greedy_select(
                states,
                int(budget["budget_bytes"]),
                profile["weights"],
                costs,
                args.amortization_validations,
                s12_recurring_seconds,
            )
            if float(greedy["net_benefit_seconds"]) > float(oracle["net_benefit_seconds"]) + 1e-9:
                raise RuntimeError("Greedy objective exceeds the exhaustive oracle.")
            greedy_row = portfolio_row(
                "GREEDY_MARGINAL_BENEFIT_PER_BYTE",
                profile,
                budget,
                greedy,
                greedy_seconds,
                oracle,
            )
            oracle_row = portfolio_row(
                "EXHAUSTIVE_ORACLE",
                profile,
                budget,
                oracle,
                oracle_seconds,
                oracle,
            )
            results.extend([greedy_row, oracle_row])
            print(
                f"  {budget['budget_label']:>10} "
                f"states=[{greedy_row['selected_state_ids']}] "
                f"coverage={greedy_row['weighted_rule_coverage']:.4f} "
                f"gap={greedy_row['oracle_gap_percent']:.4f}%"
            )

    greedy_rows = [r for r in results if r["method"] == "GREEDY_MARGINAL_BENEFIT_PER_BYTE"]
    max_gap = max(float(r["oracle_gap_percent"]) for r in greedy_rows)
    mean_gap = float(statistics.mean(float(r["oracle_gap_percent"]) for r in greedy_rows))
    exact_count = sum(1 for r in greedy_rows if float(r["oracle_gap_percent"]) <= 1e-9)
    production_rows = [r for r in greedy_rows if r["profile_id"] == "PRODUCTION_MIXED"]
    unique_portfolios = sorted({str(r["selected_state_ids"]) for r in production_rows})

    portfolio_schema = T.StructType([
        T.StructField("profile_order", T.IntegerType(), False),
        T.StructField("profile_id", T.StringType(), False),
        T.StructField("budget_order", T.IntegerType(), False),
        T.StructField("budget_id", T.StringType(), False),
        T.StructField("budget_label", T.StringType(), False),
        T.StructField("budget_bytes", T.LongType(), False),
        T.StructField("method", T.StringType(), False),
        T.StructField("selected_state_count", T.IntegerType(), False),
        T.StructField("selected_state_ids", T.StringType(), False),
        T.StructField("covered_rule_count", T.IntegerType(), False),
        T.StructField("covered_rule_ids", T.StringType(), False),
        T.StructField("selected_storage_bytes", T.LongType(), False),
        T.StructField("budget_utilization", T.DoubleType(), False),
        T.StructField("weighted_rule_coverage", T.DoubleType(), False),
        T.StructField("gross_avoided_scan_seconds", T.DoubleType(), False),
        T.StructField("amortized_build_seconds", T.DoubleType(), False),
        T.StructField("s12_recurring_weighted_seconds", T.DoubleType(), False),
        T.StructField("net_benefit_seconds", T.DoubleType(), False),
        T.StructField("predicted_remaining_scan_seconds", T.DoubleType(), False),
        T.StructField("selection_seconds", T.DoubleType(), False),
        T.StructField("oracle_absolute_gap_seconds", T.DoubleType(), False),
        T.StructField("oracle_gap_percent", T.DoubleType(), False),
        T.StructField("exactness_strategy", T.StringType(), False),
        T.StructField("status", T.StringType(), False),
    ])

    summary_schema = T.StructType([
        T.StructField("status", T.StringType(), False),
        T.StructField("state_count", T.IntegerType(), False),
        T.StructField("profile_count", T.IntegerType(), False),
        T.StructField("budget_count", T.IntegerType(), False),
        T.StructField("scenario_count", T.IntegerType(), False),
        T.StructField("portfolio_search_space", T.IntegerType(), False),
        T.StructField("greedy_oracle_exact_match_count", T.IntegerType(), False),
        T.StructField("greedy_oracle_exact_match_rate", T.DoubleType(), False),
        T.StructField("maximum_oracle_gap_percent", T.DoubleType(), False),
        T.StructField("mean_oracle_gap_percent", T.DoubleType(), False),
        T.StructField("production_unique_portfolio_count", T.IntegerType(), False),
        T.StructField("production_unique_portfolios", T.StringType(), False),
        T.StructField("total_state_size_bytes", T.LongType(), False),
        T.StructField("key_frequency_size_bytes", T.LongType(), False),
        T.StructField("median_scalar_scan_seconds", T.DoubleType(), False),
        T.StructField("median_duplicate_scan_seconds", T.DoubleType(), False),
        T.StructField("median_full_validation_seconds", T.DoubleType(), False),
        T.StructField("median_s12_incremental_seconds", T.DoubleType(), False),
        T.StructField(
            "median_s12_ablation_fallback_seconds", T.DoubleType(), False
        ),
        T.StructField(
            "median_s12_speedup_vs_fallback", T.DoubleType(), False
        ),
        T.StructField("amortization_validations", T.IntegerType(), False),
        T.StructField("output_root", T.StringType(), False),
    ])

    portfolio_df = spark.createDataFrame(results, schema=portfolio_schema).orderBy(
        "profile_order", "budget_order", "method"
    )
    total_state_size = sum(int(s["size_bytes"]) for s in states)
    key_frequency_size = next(int(s["size_bytes"]) for s in states if s["state_id"] == "S12")
    summary_df = spark.createDataFrame(
        [
            (
                "PASS",
                len(states),
                len(profiles),
                len(budgets),
                len(greedy_rows),
                1 << len(states),
                exact_count,
                exact_count / float(len(greedy_rows)),
                max_gap,
                mean_gap,
                len(unique_portfolios),
                " | ".join(unique_portfolios),
                total_state_size,
                key_frequency_size,
                float(costs["median_scalar_seconds"]),
                float(costs["median_duplicate_seconds"]),
                float(costs["median_total_seconds"]),
                float(s12_costs["median_s12_incremental_seconds"]),
                float(s12_costs["median_full_snapshot_fallback_seconds"]),
                float(s12_costs["median_s12_speedup_vs_fallback"]),
                args.amortization_validations,
                output_root,
            )
        ],
        schema=summary_schema,
    )

    write_csv(portfolio_df, f"{output_root}/portfolio_results_csv")
    write_csv(summary_df, f"{output_root}/summary_csv")

    print()
    print("=" * 80)
    print("PRODUCTION_MIXED GREEDY BUDGET CURVE")
    print("=" * 80)
    portfolio_df.filter(
        (F.col("profile_id") == "PRODUCTION_MIXED")
        & (F.col("method") == "GREEDY_MARGINAL_BENEFIT_PER_BYTE")
    ).orderBy("budget_order").select(
        "budget_label",
        "selected_state_ids",
        "selected_storage_bytes",
        "weighted_rule_coverage",
        "net_benefit_seconds",
        "s12_recurring_weighted_seconds",
        "predicted_remaining_scan_seconds",
        "oracle_gap_percent",
    ).show(20, truncate=False)

    print("=" * 80)
    print("STATEGUARD_BUDGET_SELECTION_BEGIN")
    print("BUDGET_SELECTION_STATUS=PASS")
    print(f"STATE_COUNT={len(states)}")
    print(f"PROFILE_COUNT={len(profiles)}")
    print(f"BUDGET_COUNT={len(budgets)}")
    print(f"SCENARIO_COUNT={len(greedy_rows)}")
    print(f"PORTFOLIO_SEARCH_SPACE={1 << len(states)}")
    print(f"GREEDY_ORACLE_EXACT_MATCH_COUNT={exact_count}")
    print(f"GREEDY_ORACLE_EXACT_MATCH_RATE={exact_count / float(len(greedy_rows)):.6f}")
    print(f"MAXIMUM_ORACLE_GAP_PERCENT={max_gap:.6f}")
    print(f"MEAN_ORACLE_GAP_PERCENT={mean_gap:.6f}")
    print(f"PRODUCTION_UNIQUE_PORTFOLIO_COUNT={len(unique_portfolios)}")
    print("PRODUCTION_UNIQUE_PORTFOLIOS=" + " | ".join(unique_portfolios))
    print(f"TOTAL_STATE_SIZE_BYTES={total_state_size}")
    print(f"KEY_FREQUENCY_SIZE_BYTES={key_frequency_size}")
    print(f"MEDIAN_SCALAR_SCAN_SECONDS={costs['median_scalar_seconds']:.6f}")
    print(f"MEDIAN_DUPLICATE_SCAN_SECONDS={costs['median_duplicate_seconds']:.6f}")
    print(
        "MEDIAN_S12_INCREMENTAL_SECONDS="
        f"{s12_costs['median_s12_incremental_seconds']:.6f}"
    )
    print(
        "MEDIAN_S12_ABLATION_FALLBACK_SECONDS="
        f"{s12_costs['median_full_snapshot_fallback_seconds']:.6f}"
    )
    print(
        "MEDIAN_S12_SPEEDUP_VS_FALLBACK="
        f"{s12_costs['median_s12_speedup_vs_fallback']:.6f}"
    )
    print("COST_MODEL_VERSION=V2_S12_RECURRING_CALIBRATED")
    print(f"PORTFOLIO_RESULTS_PATH={output_root}/portfolio_results_csv")
    print(f"SUMMARY_PATH={output_root}/summary_csv")
    print("STATEGUARD_BUDGET_SELECTION_END")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
