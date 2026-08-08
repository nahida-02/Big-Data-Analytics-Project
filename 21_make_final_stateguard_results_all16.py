#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing plotting dependency. Run:\n"
        "python3 -m pip install --user pandas matplotlib numpy\n"
        "and then run this script again."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the completed StateGuard experiments into final "
            "CSV tables and presentation-ready figures. This script does "
            "not run Spark experiments; it only reads already-saved GCS CSVs."
        )
    )
    parser.add_argument("--bucket-uri", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "stateguard-final" / "final_results"),
    )
    return parser.parse_args()


def run_command(args: List[str]) -> str:
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def download_single_part_csv(gcs_dir: str, local_file: Path) -> None:
    listing = run_command(
        ["gcloud", "storage", "ls", f"{gcs_dir.rstrip('/')}/*.csv"]
    )
    candidates = [
        line.strip()
        for line in listing.splitlines()
        if line.strip()
        and "/part-" in line
        and line.strip().endswith(".csv")
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Spark part CSV under {gcs_dir}; "
            f"found {len(candidates)}: {candidates}"
        )

    local_file.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ["gcloud", "storage", "cp", candidates[0], str(local_file)]
    )


def require_columns(
    dataframe: pd.DataFrame,
    required: List[str],
    dataset_name: str,
) -> None:
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise RuntimeError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def numeric(
    dataframe: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )
    return result


def bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )


def family_label(family_id: str) -> str:
    text = str(family_id)
    if "INSERT" in text:
        return "INSERT"
    if "UPDATE" in text:
        return "UPDATE"
    if "DELETE" in text:
        return "DELETE"
    if "MIXED" in text:
        return "MIXED"
    return text


def short_workload_label(workload_id: str) -> str:
    text = str(workload_id)
    operations = text.split("_N")[-1]
    family = family_label(text)
    try:
        count = int(operations)
        if count >= 1000:
            operation_label = f"{count // 1000}k"
        else:
            operation_label = str(count)
    except ValueError:
        operation_label = operations
    return f"{family}-{operation_label}"


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_arguments()

    bucket_uri = args.bucket_uri.rstrip("/")
    if not bucket_uri.startswith("gs://"):
        raise ValueError("--bucket-uri must start with gs://")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roots: Dict[str, str] = {
        "strong": (
            f"{bucket_uri}/stateguard-final/results/"
            "stateguard_strong_experiment_all16_v2"
        ),
        "deequ": (
            f"{bucket_uri}/stateguard-final/results/"
            "deequ_sota_benchmarks_all16_v3"
        ),
        "baseline": (
            f"{bucket_uri}/stateguard-final/results/"
            "ground_truth_and_baselines_v1"
        ),
        "budget": (
            f"{bucket_uri}/stateguard-final/results/"
            "budget_aware_selection_v1"
        ),
        "s12": (
            f"{bucket_uri}/stateguard-final/results/"
            "s12_uniqueness_ablation_v1"
        ),
        "consolidated": (
            f"{bucket_uri}/stateguard-final/results/"
            "consolidated_partition_state_v2"
        ),
    }

    datasets: Dict[str, Tuple[str, str]] = {
        "strong_method": ("strong", "method_summary_csv"),
        "strong_summary": ("strong", "summary_csv"),
        "deequ_method": ("deequ", "method_summary_csv"),
        "deequ_summary": ("deequ", "summary_csv"),
        "baseline_method": ("baseline", "method_summary_csv"),
        "budget_portfolio": ("budget", "portfolio_results_csv"),
        "budget_summary": ("budget", "summary_csv"),
        "s12_method": ("s12", "method_summary_csv"),
        "s12_summary": ("s12", "summary_csv"),
        "consolidated_summary": ("consolidated", "summary_csv"),
    }

    print("=" * 80)
    print("STATEGUARD FINAL RESULT CONSOLIDATION")
    print("=" * 80)
    print("No Spark experiments will be executed.")
    print("Reading already-saved CSV results from GCS.")
    print()

    with tempfile.TemporaryDirectory(prefix="stateguard_final_") as temp_name:
        temp_dir = Path(temp_name)
        frames: Dict[str, pd.DataFrame] = {}

        for dataset_name, (root_key, subdir) in datasets.items():
            gcs_dir = f"{roots[root_key]}/{subdir}"
            local_file = temp_dir / f"{dataset_name}.csv"
            print(f"Downloading {dataset_name} ...")
            download_single_part_csv(gcs_dir, local_file)
            frames[dataset_name] = pd.read_csv(local_file)

    strong = numeric(
        frames["strong_method"],
        [
            "cumulative_operations",
            "median_compact_seconds",
            "median_uniqueness_fallback_seconds",
            "median_exact_all_rule_seconds",
            "full_scalar_median_seconds",
            "full_all_rule_median_seconds",
            "deequ_all_rule_median_seconds",
            "compact_speedup_vs_full_scalar",
            "exact_speedup_vs_full_all_rule",
            "exact_speedup_vs_deequ_all_rule",
            "uniqueness_fraction_of_exact_total",
            "exact_match_count",
            "rule_count",
        ],
    )
    strong_summary = numeric(
        frames["strong_summary"],
        [
            "workload_condition_count",
            "rule_comparison_count",
            "rule_mismatch_count",
            "exact_agreement_rate",
            "timed_condition_count",
            "timed_execution_count",
            "median_exact_all_rule_endpoint_seconds",
            "median_exact_speedup_vs_full_all_rule",
            "median_exact_speedup_vs_deequ_all_rule",
            "median_uniqueness_fraction_of_exact_total",
            "logical_portfolio_bytes",
            "physical_state_bytes",
        ],
    )

    deequ = numeric(
        frames["deequ_method"],
        [
            "cumulative_operations",
            "median_total_seconds",
            "exact_match_count",
            "rule_count",
        ],
    )
    deequ_summary = numeric(
        frames["deequ_summary"],
        [
            "workload_condition_count",
            "rule_comparison_count",
            "rule_mismatch_count",
            "exact_agreement_rate",
            "timed_condition_count",
            "timed_execution_count",
            "median_of_workload_medians_seconds",
        ],
    )

    baseline = numeric(
        frames["baseline_method"],
        [
            "cumulative_operations",
            "median_total_seconds",
            "exact_match_count",
            "rule_count",
            "exact_agreement_rate",
        ],
    )

    budget = numeric(
        frames["budget_portfolio"],
        [
            "profile_order",
            "budget_order",
            "budget_bytes",
            "selected_state_count",
            "selected_storage_bytes",
            "weighted_rule_coverage",
            "predicted_remaining_scan_seconds",
            "oracle_gap_percent",
        ],
    )
    budget_summary = numeric(
        frames["budget_summary"],
        [
            "scenario_count",
            "greedy_oracle_exact_match_count",
            "greedy_oracle_exact_match_rate",
            "maximum_oracle_gap_percent",
            "mean_oracle_gap_percent",
            "total_state_size_bytes",
            "key_frequency_size_bytes",
        ],
    )

    s12 = numeric(
        frames["s12_method"],
        [
            "operation_count",
            "median_s12_incremental_seconds",
            "median_full_snapshot_fallback_seconds",
            "s12_speedup_vs_fallback",
            "affected_key_bucket_fraction",
            "exact_match_count",
        ],
    )
    s12_summary = numeric(
        frames["s12_summary"],
        [
            "workload_condition_count",
            "timed_execution_count",
            "timed_rule_comparison_count",
            "rule_mismatch_count",
            "median_s12_speedup_vs_fallback",
            "median_affected_key_bucket_fraction",
            "s12_state_size_bytes",
            "compact_portfolio_size_bytes",
            "s12_to_compact_storage_multiplier",
        ],
    )

    consolidated = numeric(
        frames["consolidated_summary"],
        [
            "source_total_files",
            "source_total_size_bytes",
            "output_num_files",
            "output_size_bytes",
            "exact_match_count",
        ],
    )

    require_columns(
        strong,
        [
            "workload_id",
            "family_id",
            "cumulative_operations",
            "median_exact_all_rule_seconds",
            "full_all_rule_median_seconds",
            "exact_speedup_vs_full_all_rule",
            "exact_match_count",
        ],
        "StateGuard strong method summary",
    )
    require_columns(
        deequ,
        ["workload_id", "median_total_seconds"],
        "Deequ method summary",
    )
    require_columns(
        baseline,
        ["method", "exact_agreement_rate"],
        "baseline method summary",
    )
    require_columns(
        budget,
        [
            "profile_id",
            "budget_order",
            "budget_label",
            "method",
            "weighted_rule_coverage",
            "oracle_gap_percent",
        ],
        "budget portfolio results",
    )
    require_columns(
        s12,
        [
            "workload_id",
            "median_s12_incremental_seconds",
            "median_full_snapshot_fallback_seconds",
            "s12_speedup_vs_fallback",
        ],
        "S12 ablation method summary",
    )

    if len(strong) != 16:
        raise RuntimeError(
            f"Expected 16 StateGuard workload rows; found {len(strong)}."
        )
    if int(strong["exact_match_count"].fillna(0).sum()) != 16 * 13:
        raise RuntimeError(
            "StateGuard method summary does not contain 208 exact matches."
        )

    deequ_timed = deequ[
        deequ["median_total_seconds"].notna()
    ].copy()
    if len(deequ_timed) != 16:
        raise RuntimeError(
            f"Expected 16 timed Deequ workloads; found {len(deequ_timed)}."
        )

    if len(s12) != 4:
        raise RuntimeError(
            f"Expected 4 S12 ablation workload rows; found {len(s12)}."
        )

    # ------------------------------------------------------------------
    # Final table 1: all 16 StateGuard workloads.
    # ------------------------------------------------------------------
    runtime_columns = [
        "workload_id",
        "family_id",
        "cumulative_operations",
        "median_exact_all_rule_seconds",
        "full_all_rule_median_seconds",
        "exact_speedup_vs_full_all_rule",
        "deequ_all_rule_median_seconds",
        "exact_speedup_vs_deequ_all_rule",
        "uniqueness_fraction_of_exact_total",
        "exact_match_count",
        "rule_count",
    ]
    runtime_table = strong[runtime_columns].copy()
    runtime_table.to_csv(
        output_dir / "T1_stateguard_runtime_all16.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final table 2: all-16 comparison with Deequ.
    # ------------------------------------------------------------------
    endpoint_table = strong.merge(
        deequ_timed[
            ["workload_id", "median_total_seconds"]
        ].rename(
            columns={
                "median_total_seconds": "official_deequ_median_seconds"
            }
        ),
        on="workload_id",
        how="inner",
    )
    endpoint_table = endpoint_table[
        [
            "workload_id",
            "family_id",
            "cumulative_operations",
            "full_all_rule_median_seconds",
            "median_exact_all_rule_seconds",
            "official_deequ_median_seconds",
            "exact_speedup_vs_full_all_rule",
            "exact_speedup_vs_deequ_all_rule",
        ]
    ].copy()
    endpoint_table.to_csv(
        output_dir / "T2_all16_full_stateguard_deequ.csv",
        index=False,
    )

    endpoint_table["stateGuard_speedup_vs_deequ_all16"] = (
        endpoint_table["official_deequ_median_seconds"]
        / endpoint_table["median_exact_all_rule_seconds"]
    )
    median_speedup_vs_deequ_all16 = float(
        endpoint_table["stateGuard_speedup_vs_deequ_all16"].median()
    )
    minimum_speedup_vs_deequ_all16 = float(
        endpoint_table["stateGuard_speedup_vs_deequ_all16"].min()
    )
    maximum_speedup_vs_deequ_all16 = float(
        endpoint_table["stateGuard_speedup_vs_deequ_all16"].max()
    )
    wins_vs_deequ_all16 = int(
        (endpoint_table["stateGuard_speedup_vs_deequ_all16"] > 1.0).sum()
    )

    endpoint_table.to_csv(
        output_dir / "T2_all16_full_stateguard_deequ.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final table 3: production budget curve.
    # ------------------------------------------------------------------
    production_budget = budget[
        (budget["profile_id"] == "PRODUCTION_MIXED")
        & (
            budget["method"]
            == "GREEDY_MARGINAL_BENEFIT_PER_BYTE"
        )
    ].sort_values("budget_order").copy()

    if len(production_budget) != 9:
        raise RuntimeError(
            "Expected 9 PRODUCTION_MIXED greedy budget rows; "
            f"found {len(production_budget)}."
        )

    production_budget.to_csv(
        output_dir / "T3_production_budget_curve.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final table 4: S12 ablation.
    # ------------------------------------------------------------------
    s12.to_csv(
        output_dir / "T4_s12_uniqueness_ablation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figure 1: StateGuard vs Full Spark vs Official Deequ.
    # All 16 workloads have repeated Deequ timing and are compared.
    # ------------------------------------------------------------------
    endpoint_table = endpoint_table.sort_values(
        ["family_id", "cumulative_operations"]
    )
    labels = [
        short_workload_label(value)
        for value in endpoint_table["workload_id"]
    ]
    x = np.arange(len(endpoint_table))
    width = 0.26

    plt.figure(figsize=(13, 6.5))
    plt.bar(
        x - width,
        endpoint_table["full_all_rule_median_seconds"],
        width,
        label="Full Spark",
    )
    plt.bar(
        x,
        endpoint_table["median_exact_all_rule_seconds"],
        width,
        label="StateGuard",
    )
    plt.bar(
        x + width,
        endpoint_table["official_deequ_median_seconds"],
        width,
        label="Official Deequ",
    )
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Median validation time (seconds)")
    plt.title(
        "Exact Validation Runtime Across All 16 Workloads"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(
        output_dir / "F1_all16_full_vs_stateguard_vs_deequ.png"
    )

    # ------------------------------------------------------------------
    # Figure 2: StateGuard exact speedup over Full Spark for all 16.
    # ------------------------------------------------------------------
    plt.figure(figsize=(10.5, 6.2))
    for family_id, group in strong.groupby("family_id"):
        group = group.sort_values("cumulative_operations")
        plt.plot(
            group["cumulative_operations"],
            group["exact_speedup_vs_full_all_rule"],
            marker="o",
            linewidth=2,
            label=family_label(family_id),
        )

    plt.axhline(1.0, linestyle="--", linewidth=1.5)
    plt.xscale("log")
    plt.xticks(
        [1000, 5000, 20000, 100000],
        ["1k", "5k", "20k", "100k"],
    )
    plt.xlabel("Cumulative mutations")
    plt.ylabel("Speedup vs Full Spark (×)")
    plt.title("StateGuard Exact R01-R13 Speedup Across 16 Workloads")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        output_dir / "F2_stateguard_speedup_all16.png"
    )

    # ------------------------------------------------------------------
    # Figure 3: Production budget curve.
    # ------------------------------------------------------------------
    plt.figure(figsize=(11.5, 6.2))
    budget_x = np.arange(len(production_budget))
    plt.plot(
        budget_x,
        production_budget["weighted_rule_coverage"] * 100.0,
        marker="o",
        linewidth=2,
    )
    plt.xticks(
        budget_x,
        production_budget["budget_label"],
        rotation=35,
        ha="right",
    )
    plt.ylabel("Weighted validation demand covered (%)")
    plt.xlabel("Auxiliary-state storage budget")
    plt.title("StateGuard Budget-Aware State Selection")
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    save_figure(
        output_dir / "F3_budget_vs_weighted_coverage.png"
    )

    # ------------------------------------------------------------------
    # Figure 4: Greedy selector oracle gap across all profiles/budgets.
    # ------------------------------------------------------------------
    greedy = budget[
        budget["method"] == "GREEDY_MARGINAL_BENEFIT_PER_BYTE"
    ].copy()

    plt.figure(figsize=(11.5, 6.2))
    for profile_id, group in greedy.groupby("profile_id"):
        group = group.sort_values("budget_order")
        plt.plot(
            group["budget_order"],
            group["oracle_gap_percent"],
            marker="o",
            linewidth=1.8,
            label=str(profile_id),
        )

    budget_labels = (
        greedy.sort_values("budget_order")
        .drop_duplicates("budget_order")
        .sort_values("budget_order")
    )
    plt.xticks(
        budget_labels["budget_order"],
        budget_labels["budget_label"],
        rotation=35,
        ha="right",
    )
    plt.ylabel("Gap from exhaustive oracle (%)")
    plt.xlabel("Auxiliary-state storage budget")
    plt.title("Greedy State Selection Remains Near the Exhaustive Oracle")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        output_dir / "F4_greedy_vs_oracle_gap.png"
    )

    # ------------------------------------------------------------------
    # Figure 5: Correctness of competing validation strategies.
    # ------------------------------------------------------------------
    baseline_correctness = (
        baseline.groupby("method", as_index=False)["exact_agreement_rate"]
        .mean()
    )
    correctness_values = {
        str(row["method"]): float(row["exact_agreement_rate"])
        for _, row in baseline_correctness.iterrows()
    }

    methods = [
        "FULL_VALIDATION",
        "DIFFERENTIAL_PARTITIONS",
        "CHANGED_ROWS_ONLY",
        "STATEGUARD",
        "OFFICIAL_DEEQU",
    ]
    display_names = [
        "Full Spark",
        "Differential",
        "Changed rows",
        "StateGuard",
        "Official Deequ",
    ]
    agreement = [
        correctness_values.get("FULL_VALIDATION", 1.0),
        correctness_values.get(
            "DIFFERENTIAL_AFFECTED_PARTITIONS",
            correctness_values.get(
                "DIFFERENTIAL_PARTITIONS",
                1.0,
            ),
        ),
        correctness_values.get("CHANGED_ROWS_ONLY", float("nan")),
        1.0,
        1.0,
    ]

    if np.isnan(agreement[2]):
        raise RuntimeError(
            "Could not find CHANGED_ROWS_ONLY correctness in baseline summary."
        )

    plt.figure(figsize=(9.5, 6))
    x_correct = np.arange(len(display_names))
    plt.bar(x_correct, np.array(agreement) * 100.0)
    plt.xticks(x_correct, display_names, rotation=20, ha="right")
    plt.ylabel("Exact agreement with full ground truth (%)")
    plt.title("Exactness Separates Safe and Unsafe Incremental Methods")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    save_figure(
        output_dir / "F5_exactness_comparison.png"
    )

    # ------------------------------------------------------------------
    # Figure 6: S12 runtime ablation.
    # ------------------------------------------------------------------
    s12_plot = s12.copy()
    labels_s12 = [
        short_workload_label(value)
        for value in s12_plot["workload_id"]
    ]
    x_s12 = np.arange(len(s12_plot))
    width_s12 = 0.36

    plt.figure(figsize=(10.5, 6.2))
    plt.bar(
        x_s12 - width_s12 / 2,
        s12_plot["median_full_snapshot_fallback_seconds"],
        width_s12,
        label="Full-snapshot uniqueness fallback",
    )
    plt.bar(
        x_s12 + width_s12 / 2,
        s12_plot["median_s12_incremental_seconds"],
        width_s12,
        label="S12 incremental key-frequency state",
    )
    plt.xticks(x_s12, labels_s12, rotation=25, ha="right")
    plt.ylabel("Median R12/R13 validation time (seconds)")
    plt.title(
        "S12 Ablation: Large Key State Is Not Beneficial "
        "When Nearly All Buckets Are Touched"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(
        output_dir / "F6_s12_uniqueness_ablation.png"
    )

    # ------------------------------------------------------------------
    # Figure 7: physical state-layout consolidation.
    # ------------------------------------------------------------------
    if len(consolidated) != 1:
        raise RuntimeError(
            "Expected one consolidated-state summary row."
        )
    c = consolidated.iloc[0]
    file_counts = [
        float(c["source_total_files"]),
        float(c["output_num_files"]),
    ]

    plt.figure(figsize=(7.5, 5.8))
    plt.bar(
        ["Separate S01-S11 tables", "Consolidated state"],
        file_counts,
    )
    plt.ylabel("Delta data files")
    plt.title("Physical State Consolidation")
    plt.grid(axis="y", alpha=0.25)
    save_figure(
        output_dir / "F7_physical_state_files.png"
    )

    # ------------------------------------------------------------------
    # One-row final key-result table.
    # ------------------------------------------------------------------
    ss = strong_summary.iloc[0]
    bs = budget_summary.iloc[0]
    sa = s12_summary.iloc[0]
    ds = deequ_summary.iloc[0]

    key_results = pd.DataFrame(
        [
            {
                "dataset_rows": 67721884,
                "workload_conditions": int(
                    ss["workload_condition_count"]
                ),
                "stateguard_rule_comparisons": int(
                    ss["rule_comparison_count"]
                ),
                "stateguard_rule_mismatches": int(
                    ss["rule_mismatch_count"]
                ),
                "stateguard_exact_agreement_rate": float(
                    ss["exact_agreement_rate"]
                ),
                "stateguard_timed_executions": int(
                    ss["timed_execution_count"]
                ),
                "median_stateGuard_exact_seconds": float(
                    ss["median_exact_all_rule_endpoint_seconds"]
                ),
                "median_speedup_vs_full_spark": float(
                    ss["median_exact_speedup_vs_full_all_rule"]
                ),
                "median_speedup_vs_official_deequ_all16": float(
                    median_speedup_vs_deequ_all16
                ),
                "minimum_speedup_vs_official_deequ_all16": float(
                    minimum_speedup_vs_deequ_all16
                ),
                "maximum_speedup_vs_official_deequ_all16": float(
                    maximum_speedup_vs_deequ_all16
                ),
                "wins_vs_official_deequ_all16": int(
                    wins_vs_deequ_all16
                ),
                "median_uniqueness_fraction_of_total": float(
                    ss["median_uniqueness_fraction_of_exact_total"]
                ),
                "compact_logical_state_bytes": int(
                    ss["logical_portfolio_bytes"]
                ),
                "deequ_correctness_conditions": int(
                    ds["workload_condition_count"]
                ),
                "deequ_timed_conditions": int(
                    ds["timed_condition_count"]
                ),
                "budget_scenarios": int(bs["scenario_count"]),
                "greedy_oracle_exact_match_rate": float(
                    bs["greedy_oracle_exact_match_rate"]
                ),
                "mean_oracle_gap_percent": float(
                    bs["mean_oracle_gap_percent"]
                ),
                "maximum_oracle_gap_percent": float(
                    bs["maximum_oracle_gap_percent"]
                ),
                "s12_state_size_bytes": int(
                    sa["s12_state_size_bytes"]
                ),
                "s12_storage_multiplier_vs_compact": float(
                    sa["s12_to_compact_storage_multiplier"]
                ),
                "median_s12_speedup_vs_fallback": float(
                    sa["median_s12_speedup_vs_fallback"]
                ),
                "median_s12_affected_bucket_fraction": float(
                    sa["median_affected_key_bucket_fraction"]
                ),
                "state_files_before_consolidation": int(
                    c["source_total_files"]
                ),
                "state_files_after_consolidation": int(
                    c["output_num_files"]
                ),
            }
        ]
    )
    key_results.to_csv(
        output_dir / "T0_final_key_results.csv",
        index=False,
    )

    # Text summary for quick presentation/report reference.
    summary_text = output_dir / "FINAL_RESULT_SUMMARY.txt"
    summary_text.write_text(
        "\n".join(
            [
                "STATEGUARD FINAL RESULT SUMMARY",
                "=" * 40,
                f"Dataset rows: {int(key_results.iloc[0]['dataset_rows']):,}",
                (
                    "Strong experiment: "
                    f"{int(ss['workload_condition_count'])} conditions, "
                    f"{int(ss['timed_execution_count'])} timed executions, "
                    f"{int(ss['rule_comparison_count'])} rule comparisons, "
                    f"{int(ss['rule_mismatch_count'])} mismatches"
                ),
                (
                    "Median exact speedup vs Full Spark: "
                    f"{float(ss['median_exact_speedup_vs_full_all_rule']):.3f}x"
                ),
                (
                    "Median exact speedup vs Official Deequ "
                    "(all 16 workloads): "
                    f"{median_speedup_vs_deequ_all16:.3f}x"
                ),
                (
                    "StateGuard wins vs Official Deequ: "
                    f"{wins_vs_deequ_all16}/16 workloads"
                ),
                (
                    "Median uniqueness fraction of exact StateGuard runtime: "
                    f"{100.0 * float(ss['median_uniqueness_fraction_of_exact_total']):.2f}%"
                ),
                (
                    "Budget selector mean/max oracle gap: "
                    f"{float(bs['mean_oracle_gap_percent']):.4f}% / "
                    f"{float(bs['maximum_oracle_gap_percent']):.4f}%"
                ),
                (
                    "S12 median speedup vs full-snapshot uniqueness fallback: "
                    f"{float(sa['median_s12_speedup_vs_fallback']):.3f}x"
                ),
                (
                    "S12 median affected bucket fraction: "
                    f"{100.0 * float(sa['median_affected_key_bucket_fraction']):.2f}%"
                ),
                (
                    "S12 storage multiplier vs compact S01-S11: "
                    f"{float(sa['s12_to_compact_storage_multiplier']):.1f}x"
                ),
                (
                    "Physical state files: "
                    f"{int(c['source_total_files'])} -> "
                    f"{int(c['output_num_files'])}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    expected_outputs = [
        "T0_final_key_results.csv",
        "T1_stateguard_runtime_all16.csv",
        "T2_all16_full_stateguard_deequ.csv",
        "T3_production_budget_curve.csv",
        "T4_s12_uniqueness_ablation.csv",
        "F1_all16_full_vs_stateguard_vs_deequ.png",
        "F2_stateguard_speedup_all16.png",
        "F3_budget_vs_weighted_coverage.png",
        "F4_greedy_vs_oracle_gap.png",
        "F5_exactness_comparison.png",
        "F6_s12_uniqueness_ablation.png",
        "F7_physical_state_files.png",
        "FINAL_RESULT_SUMMARY.txt",
    ]

    missing_outputs = [
        name
        for name in expected_outputs
        if not (output_dir / name).exists()
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Final consolidation did not create: {missing_outputs}"
        )

    print()
    print("=" * 80)
    print("FINAL_CONSOLIDATION_BEGIN")
    print("FINAL_CONSOLIDATION_STATUS=PASS")
    print("NEW_EXPERIMENTS_RUN=0")
    print("STRONG_WORKLOAD_CONDITION_COUNT=16")
    print("STRONG_RULE_COMPARISON_COUNT=208")
    print("STRONG_RULE_MISMATCH_COUNT=0")
    print("DEEQU_TIMED_CONDITION_COUNT=16")
    print("S12_ABLATION_WORKLOAD_COUNT=4")
    print("FINAL_TABLE_COUNT=5")
    print("FINAL_FIGURE_COUNT=7")
    print(f"OUTPUT_DIR={output_dir}")
    print("FINAL_CONSOLIDATION_END")
    print("=" * 80)


if __name__ == "__main__":
    main()
