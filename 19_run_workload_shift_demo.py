#!/usr/bin/env python3
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_STATES = [f"S{i:02d}" for i in range(1, 13)]
DEFAULT_PHASES = [
    "ADDITIVE_HEAVY",
    "EXTREMA_HEAVY",
    "UNIQUENESS_HEAVY",
    "PRODUCTION_MIXED",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a workload-shift demonstration from the already-completed "
            "StateGuard selector-v2 results. No Spark experiment is executed."
        )
    )
    parser.add_argument("--selector-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-id", default="B04")
    parser.add_argument(
        "--phase-profiles",
        default=",".join(DEFAULT_PHASES),
        help="Comma-separated profile IDs in chronological phase order.",
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
    return result.stdout


def download_csv_parts(gcs_dir: str, local_dir: Path) -> List[Path]:
    listing = run_command(["gcloud", "storage", "ls", f"{gcs_dir.rstrip('/')}/*.csv"])
    uris = [line.strip() for line in listing.splitlines() if line.strip().endswith(".csv")]
    if not uris:
        raise RuntimeError(f"No CSV part files found under {gcs_dir}")

    local_paths = []
    for i, uri in enumerate(uris):
        dest = local_dir / f"part_{i:04d}.csv"
        run_command(["gcloud", "storage", "cp", uri, str(dest)])
        local_paths.append(dest)
    return local_paths


def read_gcs_csv(gcs_dir: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="stateguard_shift_") as tmp:
        local_dir = Path(tmp)
        parts = download_csv_parts(gcs_dir, local_dir)
        frames = [pd.read_csv(p) for p in parts]
        return pd.concat(frames, ignore_index=True)


def parse_states(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    states = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in states if x not in EXPECTED_STATES]
    if unknown:
        raise RuntimeError(f"Unknown state IDs encountered: {unknown}")
    return states


def join_states(states: Set[str]) -> str:
    return ",".join(sorted(states))


def main() -> None:
    args = parse_args()
    selector_root = args.selector_root.rstrip("/")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_profiles = [x.strip() for x in args.phase_profiles.split(",") if x.strip()]
    if len(phase_profiles) < 2:
        raise ValueError("At least two workload phases are required.")
    if len(set(phase_profiles)) != len(phase_profiles):
        raise ValueError("Phase profile IDs must be unique within this demonstration.")

    print("=" * 80)
    print("STATEGUARD WORKLOAD-SHIFT DEMONSTRATION")
    print("=" * 80)
    print("No Spark experiments will be executed.")
    print(f"Selector source: {selector_root}")
    print(f"Fixed budget: {args.budget_id}")
    print(f"Phase sequence: {' -> '.join(phase_profiles)}")
    print()

    portfolio = read_gcs_csv(f"{selector_root}/portfolio_results_csv")
    summary = read_gcs_csv(f"{selector_root}/summary_csv")

    required_portfolio_columns = {
        "profile_id",
        "budget_id",
        "budget_label",
        "method",
        "selected_state_count",
        "selected_state_ids",
        "selected_storage_bytes",
        "weighted_rule_coverage",
        "predicted_remaining_scan_seconds",
        "oracle_gap_percent",
        "s12_recurring_weighted_seconds",
        "status",
    }
    missing = sorted(required_portfolio_columns - set(portfolio.columns))
    if missing:
        raise RuntimeError(f"Selector-v2 portfolio output is missing columns: {missing}")

    # These columns were added to the selector-v2 summary and therefore act as
    # a guard against accidentally reading selector-v1 results.
    required_summary_columns = {
        "median_s12_incremental_seconds",
        "median_s12_ablation_fallback_seconds",
        "median_s12_speedup_vs_fallback",
    }
    missing_summary = sorted(required_summary_columns - set(summary.columns))
    if missing_summary:
        raise RuntimeError(
            "Selector source does not look like recurring-cost-calibrated v2; "
            f"missing summary columns: {missing_summary}"
        )

    method = "GREEDY_MARGINAL_BENEFIT_PER_BYTE"
    selected = portfolio[
        (portfolio["method"] == method)
        & (portfolio["budget_id"].astype(str) == str(args.budget_id))
        & (portfolio["profile_id"].isin(phase_profiles))
    ].copy()

    if len(selected) != len(phase_profiles):
        found = sorted(selected["profile_id"].astype(str).unique().tolist())
        raise RuntimeError(
            f"Expected {len(phase_profiles)} phase rows at budget {args.budget_id}; "
            f"found {len(selected)}. Profiles found: {found}"
        )

    selected_by_profile = {
        str(row["profile_id"]): row for _, row in selected.iterrows()
    }

    records = []
    previous_states: Set[str] = set()
    unique_portfolios = set()
    transition_changes = 0
    s12_selected_phase_count = 0

    for phase_order, profile_id in enumerate(phase_profiles, start=1):
        row = selected_by_profile[profile_id]
        states = set(parse_states(row["selected_state_ids"]))

        added = states - previous_states
        removed = previous_states - states
        retained = previous_states & states
        changed = phase_order > 1 and states != previous_states

        if changed:
            transition_changes += 1
        if "S12" in states:
            s12_selected_phase_count += 1

        portfolio_key = tuple(sorted(states))
        unique_portfolios.add(portfolio_key)

        records.append(
            {
                "phase_order": phase_order,
                "phase_id": f"P{phase_order}",
                "profile_id": profile_id,
                "budget_id": str(row["budget_id"]),
                "budget_label": str(row["budget_label"]),
                "selected_state_count": int(row["selected_state_count"]),
                "selected_state_ids": join_states(states),
                "selected_storage_bytes": int(row["selected_storage_bytes"]),
                "weighted_rule_coverage": float(row["weighted_rule_coverage"]),
                "predicted_remaining_scan_seconds": float(
                    row["predicted_remaining_scan_seconds"]
                ),
                "oracle_gap_percent": float(row["oracle_gap_percent"]),
                "s12_recurring_weighted_seconds": float(
                    row["s12_recurring_weighted_seconds"]
                ),
                "added_state_ids": join_states(added),
                "removed_state_ids": join_states(removed),
                "retained_state_ids": join_states(retained),
                "portfolio_changed_from_previous": bool(changed),
                "s12_selected": bool("S12" in states),
                "status": str(row["status"]),
            }
        )

        previous_states = states

    out = pd.DataFrame(records)
    if not (out["status"] == "PASS").all():
        raise RuntimeError("One or more selected phase rows are not PASS.")

    table_path = output_dir / "T5_workload_shift_portfolios.csv"
    out.to_csv(table_path, index=False)

    # Build one state-by-phase matrix figure.
    matrix = np.zeros((len(EXPECTED_STATES), len(phase_profiles)), dtype=int)
    for col, record in enumerate(records):
        active = set(parse_states(record["selected_state_ids"]))
        for row_index, state_id in enumerate(EXPECTED_STATES):
            matrix[row_index, col] = 1 if state_id in active else 0

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(phase_profiles)))
    ax.set_xticklabels(
        [p.replace("_", "\n") for p in phase_profiles],
        fontsize=10,
    )
    ax.set_yticks(range(len(EXPECTED_STATES)))
    ax.set_yticklabels(EXPECTED_STATES)
    ax.set_xlabel("Workload phase at fixed storage budget")
    ax.set_ylabel("Auxiliary state")
    ax.set_title(
        f"Workload-Responsive State Portfolio at Fixed {records[0]['budget_label']} Budget"
    )

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(
                c,
                r,
                "✓" if matrix[r, c] else "",
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.tight_layout()
    figure_path = output_dir / "F8_workload_shift_adaptation.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_row = summary.iloc[0]
    summary_path = output_dir / "WORKLOAD_SHIFT_SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("STATEGUARD WORKLOAD-SHIFT SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write("WORKLOAD_SHIFT_STATUS=PASS\n")
        f.write("SOURCE_COST_MODEL=V2_S12_RECURRING_CALIBRATED\n")
        f.write(f"FIXED_BUDGET_ID={args.budget_id}\n")
        f.write(f"FIXED_BUDGET_LABEL={records[0]['budget_label']}\n")
        f.write(f"PHASE_COUNT={len(records)}\n")
        f.write(f"TRANSITION_COUNT={max(0, len(records) - 1)}\n")
        f.write(f"PORTFOLIO_CHANGE_COUNT={transition_changes}\n")
        f.write(f"UNIQUE_PORTFOLIO_COUNT={len(unique_portfolios)}\n")
        f.write(f"S12_SELECTED_PHASE_COUNT={s12_selected_phase_count}\n")
        f.write(
            "MEDIAN_S12_INCREMENTAL_SECONDS="
            f"{float(summary_row['median_s12_incremental_seconds']):.6f}\n"
        )
        f.write(
            "MEDIAN_S12_ABLATION_FALLBACK_SECONDS="
            f"{float(summary_row['median_s12_ablation_fallback_seconds']):.6f}\n"
        )
        f.write(
            "MEDIAN_S12_SPEEDUP_VS_FALLBACK="
            f"{float(summary_row['median_s12_speedup_vs_fallback']):.6f}\n"
        )
        for record in records:
            f.write(
                f"{record['phase_id']}={record['profile_id']} | "
                f"states={record['selected_state_ids']} | "
                f"added={record['added_state_ids']} | "
                f"removed={record['removed_state_ids']}\n"
            )

    print("WORKLOAD_SHIFT_BEGIN")
    print("WORKLOAD_SHIFT_STATUS=PASS")
    print("SOURCE_COST_MODEL=V2_S12_RECURRING_CALIBRATED")
    print(f"FIXED_BUDGET_ID={args.budget_id}")
    print(f"FIXED_BUDGET_LABEL={records[0]['budget_label']}")
    print(f"PHASE_COUNT={len(records)}")
    print(f"TRANSITION_COUNT={max(0, len(records) - 1)}")
    print(f"PORTFOLIO_CHANGE_COUNT={transition_changes}")
    print(f"UNIQUE_PORTFOLIO_COUNT={len(unique_portfolios)}")
    print(f"S12_SELECTED_PHASE_COUNT={s12_selected_phase_count}")
    for record in records:
        print(
            f"{record['phase_id']} {record['profile_id']}: "
            f"states=[{record['selected_state_ids']}] "
            f"added=[{record['added_state_ids']}] "
            f"removed=[{record['removed_state_ids']}]"
        )
    print(f"TABLE_PATH={table_path}")
    print(f"FIGURE_PATH={figure_path}")
    print(f"SUMMARY_PATH={summary_path}")
    print("WORKLOAD_SHIFT_END")


if __name__ == "__main__":
    main()
