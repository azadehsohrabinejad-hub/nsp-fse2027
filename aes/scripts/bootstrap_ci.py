#!/usr/bin/env python3
"""
bootstrap_ci.py
===============
Compute 95% bootstrap confidence intervals for the 2×2 capability-difficulty matrix.

Metrics per quadrant:
  - pass_rate              : mean(final_pass) over all task×strategy rows
  - strategy_spread_pp     : (max_strategy_pass_rate - min_strategy_pass_rate) × 100
  - routing_relevant_count : number of tasks where 0 < sum(strategy_passes) < 8
  - oracle_headroom_pp     : (oracle_pass_rate - best_fixed_strategy_pass_rate) × 100

Bootstrap unit: tasks (sample tasks with replacement; keep all strategy rows for each
sampled task).

Usage
-----
  python scripts/bootstrap_ci.py \\
    --t2-labeled  <csv> \\
    --t2-unlabeled <csv> \\
    --t1-labeled  <csv> \\
    --t1-unlabeled <csv> \\
    [--n-bootstrap 2000] \\
    [--seed 42] \\
    [--out reports/ase2026_paper/bootstrap_ci_results.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

STRATEGIES = [
    "direct_baseline",
    "contract_first",
    "failure_mode_first",
    "pattern_skeleton",
    "multi_view",
    "semantic_diff",
    "role_decomposed",
    "simulation_trace",
]

OUTCOME_COL = "final_pass"
TASK_COL    = "task_id"
STRATEGY_COL = "strategy"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_quadrant(csv_path: str) -> pd.DataFrame:
    """Load a CSV, keep only the 8 fixed strategies, cast outcome to int."""
    df = pd.read_csv(csv_path)
    # drop oracle_teacher
    df = df[df[STRATEGY_COL] != "oracle_teacher"].copy()
    # keep only the canonical 8 strategies
    df = df[df[STRATEGY_COL].isin(STRATEGIES)].copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(float).round().astype(int)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute the four scalar metrics from a (possibly resampled) dataframe.

    Parameters
    ----------
    df : rows of (task_id, strategy, final_pass) — all 8 strategies must be
         representable per task for spread / routing_relevant to be meaningful.

    Returns
    -------
    dict with keys: pass_rate, strategy_spread_pp, routing_relevant_count,
                    oracle_headroom_pp
    """
    tasks = df[TASK_COL].unique()
    n_tasks = len(tasks)

    # ── pass_rate ──────────────────────────────────────────────────────────
    pass_rate = df[OUTCOME_COL].mean()

    # ── per-task pivot (tasks × strategies) ───────────────────────────────
    # Only include strategy columns that actually exist in the sample
    pivot = (
        df.groupby([TASK_COL, STRATEGY_COL])[OUTCOME_COL]
        .mean()           # in case a task appears multiple times via resampling
        .unstack(STRATEGY_COL)
        .reindex(columns=STRATEGIES)
    )

    # ── strategy_spread_pp ─────────────────────────────────────────────────
    # Strategy pass rate = mean over tasks (rows)
    strategy_means = pivot.mean(axis=0)          # shape (8,)
    strategy_spread_pp = float((strategy_means.max() - strategy_means.min()) * 100)

    # ── routing_relevant_count ────────────────────────────────────────────
    # A task is routing-relevant if 0 < number_of_strategies_that_passed < 8
    task_pass_counts = pivot.sum(axis=1)          # sum across 8 strategies per task
    routing_relevant_count = int(
        ((task_pass_counts > 0) & (task_pass_counts < len(STRATEGIES))).sum()
    )

    # ── oracle_headroom_pp ────────────────────────────────────────────────
    # oracle_pass_rate  = mean over tasks of max(strategy_passes for that task)
    # best_fixed_pass   = max over strategies of mean(strategy_passes over tasks)
    oracle_pass_rate = float(pivot.max(axis=1).mean())
    best_fixed_pass  = float(strategy_means.max())
    oracle_headroom_pp = float((oracle_pass_rate - best_fixed_pass) * 100)

    return {
        "pass_rate":              float(pass_rate),
        "strategy_spread_pp":     float(strategy_spread_pp),
        "routing_relevant_count": int(routing_relevant_count),
        "oracle_headroom_pp":     float(oracle_headroom_pp),
    }


def bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Bootstrap 95% CIs by resampling tasks with replacement.

    Returns a dict:
      {
        "point": {...},          # point estimates on original data
        "ci_lo": {...},          # 2.5th percentile
        "ci_hi": {...},          # 97.5th percentile
        "n_tasks": int,
        "n_runs":  int,
      }
    """
    if rng is None:
        rng = np.random.default_rng(42)

    tasks = df[TASK_COL].unique()
    n_tasks = len(tasks)

    point = compute_metrics(df)

    boot_records = {k: [] for k in point}

    for _ in range(n_bootstrap):
        sampled_tasks = rng.choice(tasks, size=n_tasks, replace=True)
        # Build resampled dataframe; assign unique IDs to duplicated tasks
        # so that pivot doesn't collapse duplicates into a single row.
        parts = []
        for i, tid in enumerate(sampled_tasks):
            chunk = df[df[TASK_COL] == tid].copy()
            chunk[TASK_COL] = f"__bs_{i}__{tid}"
            parts.append(chunk)
        boot_df = pd.concat(parts, ignore_index=True)

        m = compute_metrics(boot_df)
        for k, v in m.items():
            boot_records[k].append(v)

    ci_lo = {k: float(np.percentile(boot_records[k], 2.5))  for k in boot_records}
    ci_hi = {k: float(np.percentile(boot_records[k], 97.5)) for k in boot_records}

    return {
        "point":   point,
        "ci_lo":   ci_lo,
        "ci_hi":   ci_hi,
        "n_tasks": int(n_tasks),
        "n_runs":  int(len(df)),
    }


# ── formatting ───────────────────────────────────────────────────────────────

def fmt_pct(point: float, lo: float, hi: float, decimals: int = 1) -> str:
    """Format a percentage value with CI, e.g. '91.7% [87.5, 95.8]'."""
    p  = round(point * 100, decimals)
    lo = round(lo   * 100, decimals)
    hi = round(hi   * 100, decimals)
    return f"{p}% [{lo}, {hi}]"


def fmt_pp(point: float, lo: float, hi: float, decimals: int = 1) -> str:
    """Format a pp value with CI, e.g. '8.3pp [4.2, 12.5]'."""
    p  = round(point, decimals)
    lo = round(lo,    decimals)
    hi = round(hi,    decimals)
    return f"{p}pp [{lo}, {hi}]"


def fmt_count(point: float, lo: float, hi: float) -> str:
    """Format an integer count with CI."""
    return f"{int(round(point))} [{int(round(lo))}, {int(round(hi))}]"


def print_markdown_table(results: dict) -> None:
    """Print the summary markdown table to stdout."""

    header = (
        "| Quadrant | Pass Rate (95% CI) | Spread pp (95% CI) "
        "| Routing-Relevant (95% CI) | Oracle Headroom pp (95% CI) |"
    )
    sep = (
        "|---|---|---|---|---|"
    )

    print()
    print("## 2×2 Capability-Difficulty Matrix — Bootstrap 95% CIs (N=2000)")
    print()
    print(header)
    print(sep)

    order = ["T2+Labeled", "T2+Unlabeled", "T1+Labeled", "T1+Unlabeled"]
    for name in order:
        r = results[name]
        pt, lo, hi = r["point"], r["ci_lo"], r["ci_hi"]

        pass_str    = fmt_pct(pt["pass_rate"],              lo["pass_rate"],              hi["pass_rate"])
        spread_str  = fmt_pp( pt["strategy_spread_pp"],     lo["strategy_spread_pp"],     hi["strategy_spread_pp"])
        rr_str      = fmt_count(pt["routing_relevant_count"], lo["routing_relevant_count"], hi["routing_relevant_count"])
        oracle_str  = fmt_pp( pt["oracle_headroom_pp"],     lo["oracle_headroom_pp"],     hi["oracle_headroom_pp"])

        print(f"| {name} | {pass_str} | {spread_str} | {rr_str} | {oracle_str} |")

    print()
    print("*Bootstrap unit: tasks resampled with replacement.*")
    print("*oracle_teacher excluded from all quadrants.*")
    print()

    # Also print per-strategy means for reference
    print("### Point Estimates Detail")
    print()
    hdr2 = "| Quadrant | N_tasks | N_runs | Pass | Spread | RR_tasks | Oracle_Δ |"
    print(hdr2)
    print("|---|---|---|---|---|---|---|")
    for name in order:
        r = results[name]
        pt = r["point"]
        print(
            f"| {name} "
            f"| {r['n_tasks']} "
            f"| {r['n_runs']} "
            f"| {pt['pass_rate']:.3f} "
            f"| {pt['strategy_spread_pp']:.1f}pp "
            f"| {pt['routing_relevant_count']} "
            f"| {pt['oracle_headroom_pp']:.1f}pp |"
        )
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for the 2×2 capability-difficulty matrix."
    )
    parser.add_argument("--t2-labeled",   required=True, help="CSV for T2+Labeled quadrant")
    parser.add_argument("--t2-unlabeled", required=True, help="CSV for T2+Unlabeled quadrant")
    parser.add_argument("--t1-labeled",   required=True, help="CSV for T1+Labeled quadrant")
    parser.add_argument("--t1-unlabeled", required=True, help="CSV for T1+Unlabeled quadrant")
    parser.add_argument("--n-bootstrap",  type=int, default=2000,
                        help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        default="reports/ase2026_paper/bootstrap_ci_results.json",
        help="Output JSON path (default: reports/ase2026_paper/bootstrap_ci_results.json)",
    )
    args = parser.parse_args()

    quadrant_specs = [
        ("T2+Labeled",   args.t2_labeled),
        ("T2+Unlabeled", args.t2_unlabeled),
        ("T1+Labeled",   args.t1_labeled),
        ("T1+Unlabeled", args.t1_unlabeled),
    ]

    rng = np.random.default_rng(args.seed)

    results = {}
    for name, csv_path in quadrant_specs:
        print(f"[bootstrap_ci] Processing {name} from {csv_path} ...", flush=True)
        df = load_quadrant(csv_path)
        print(
            f"  -> {df[TASK_COL].nunique()} tasks, "
            f"{df[STRATEGY_COL].nunique()} strategies, "
            f"{len(df)} rows after filtering",
            flush=True,
        )
        ci = bootstrap_ci(df, n_bootstrap=args.n_bootstrap, rng=rng)
        results[name] = ci

    # ── print markdown table ──────────────────────────────────────────────
    print_markdown_table(results)

    # ── save JSON ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "n_bootstrap":    args.n_bootstrap,
                    "seed":           args.seed,
                    "confidence":     0.95,
                    "bootstrap_unit": "tasks",
                    "outcome_column": OUTCOME_COL,
                    "strategies":     STRATEGIES,
                    "excluded":       ["oracle_teacher"],
                },
                "quadrants": results,
            },
            f,
            indent=2,
        )
    print(f"[bootstrap_ci] Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
