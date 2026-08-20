#!/usr/bin/env python3
"""Compute family-level Capability-Clash analytics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "reports" / "capability_clash" / "capability_clash_results.csv"
DEFAULT_MANIFEST = ROOT / "data" / "capability_clash" / "manifest_matrix.jsonl"
DEFAULT_CSV = ROOT / "reports" / "capability_clash" / "family_level_results.csv"
DEFAULT_MD = ROOT / "reports" / "capability_clash" / "FAMILY_LEVEL_RESULTS.md"


def load_manifest(manifest_path: Path) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            metadata = payload.get("metadata") or {}
            mutation_family = metadata.get("mutation_family") or payload.get("mutation_family") or "unknown"
            repo_slug = payload.get("dataset_source") or payload.get("repo_slug") or "unknown"
            mapping[payload["task_id"]] = (mutation_family, repo_slug)
    return mapping


def format_table(df: pd.DataFrame, headers: Tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def compute_family_results(results_path: Path, manifest_path: Path, csv_path: Path, md_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    df = pd.read_csv(results_path)
    df = df[df["task_id"].isin(manifest.keys())].copy()
    df[["mutation_family", "repo_slug"]] = df["task_id"].apply(lambda tid: pd.Series(manifest[tid]))
    df["pass_flag"] = (df["final_pass"] >= 1.0).astype(float)

    agg = (
        df.groupby(["mutation_family", "repo_slug", "model", "strategy"])
        .agg(runs=("task_id", "count"), passes=("pass_flag", "sum"))
        .reset_index()
    )
    agg["pass_rate"] = agg["passes"] / agg["runs"].where(agg["runs"] > 0, 1)
    agg.to_csv(csv_path, index=False)

    family_model = (
        df.groupby(["mutation_family", "model"])
        .agg(pass_rate=("pass_flag", "mean"))
        .reset_index()
        .sort_values(["mutation_family", "model"])
    )
    family_strategy = (
        df.groupby(["mutation_family", "strategy"])
        .agg(pass_rate=("pass_flag", "mean"))
        .reset_index()
    )
    strategy_spread = []
    for family, subset in family_strategy.groupby("mutation_family"):
        best_row = subset.loc[subset["pass_rate"].idxmax()]
        worst_row = subset.loc[subset["pass_rate"].idxmin()]
        spread = best_row["pass_rate"] - worst_row["pass_rate"]
        strategy_spread.append(
            {
                "mutation_family": family,
                "best": f"{best_row['strategy']} ({best_row['pass_rate']:.2%})",
                "worst": f"{worst_row['strategy']} ({worst_row['pass_rate']:.2%})",
                "spread": spread,
            }
        )
    strategy_spread_df = pd.DataFrame(strategy_spread).sort_values("mutation_family")

    hardest = (
        family_model.groupby("mutation_family")
        .agg(avg_pass_rate=("pass_rate", "mean"))
        .reset_index()
    )
    model_pivot = family_model.pivot(index="mutation_family", columns="model", values="pass_rate").fillna(0.0)
    hardest = hardest.join(model_pivot, on="mutation_family")
    hardest["model_gap"] = (hardest.get("gpt-4.1-mini", 0.0) - hardest.get("gpt-4.1-nano", 0.0)).abs()
    hardest = hardest.sort_values("avg_pass_rate")
    hardest["rank"] = range(1, len(hardest) + 1)

    repo_family = (
        df.groupby(["mutation_family", "repo_slug"])
        .agg(pass_rate=("pass_flag", "mean"))
        .reset_index()
        .sort_values(["mutation_family", "repo_slug"])
    )

    strategy_overall = (
        df.groupby("strategy")
        .agg(pass_rate=("pass_flag", "mean"))
        .reset_index()
        .sort_values("pass_rate", ascending=False)
    )
    model_overall = (
        df.groupby("model")
        .agg(pass_rate=("pass_flag", "mean"))
        .reset_index()
        .sort_values("pass_rate", ascending=False)
    )

    lines = [
        "# Family-Level Capability-Clash Results",
        "",
        "## Pass Rate by Mutation Family × Model Tier",
        format_table(
            family_model.assign(pass_rate=lambda d: d["pass_rate"].map(lambda x: f"{x:.2%}")),
            ("Mutation Family", "Model", "Avg Pass Rate"),
        ),
        "",
        "## Strategy Spread per Mutation Family (averaged across models)",
        format_table(
            strategy_spread_df.assign(spread=lambda d: d["spread"].map(lambda x: f"{x:.2%}")),
            ("Mutation Family", "Best Strategy", "Worst Strategy", "Spread"),
        ),
        "",
        "## Hardest Family Ranking (overall average pass rate)",
        format_table(
            hardest.assign(
                avg_pass_rate=lambda d: d["avg_pass_rate"].map(lambda x: f"{x:.2%}"),
                model_gap=lambda d: d["model_gap"].map(lambda x: f"{x:.2%}"),
            )[["rank", "mutation_family", "avg_pass_rate", "model_gap"]],
            ("Rank", "Mutation Family", "Avg Pass Rate", "Model Gap"),
        ),
        "",
        f"Hardest family = {hardest.iloc[0]['mutation_family']} ({hardest.iloc[0]['avg_pass_rate']}). "
        f"Easiest non-ceiling family = {hardest.iloc[-1]['mutation_family']} ({hardest.iloc[-1]['avg_pass_rate']}).",
        "",
        "## Repo-by-Family Performance (avg pass rate across both tiers)",
        format_table(
            repo_family.assign(pass_rate=lambda d: d["pass_rate"].map(lambda x: f"{x:.2%}")),
            ("Mutation Family", "Repo", "Avg Pass Rate"),
        ),
        "",
        "## Overall Strategy + Model Metrics",
        format_table(
            strategy_overall.assign(pass_rate=lambda d: d["pass_rate"].map(lambda x: f"{x:.2%}")),
            ("Strategy", "Avg Pass Rate"),
        ),
        "",
        format_table(
            model_overall.assign(pass_rate=lambda d: d["pass_rate"].map(lambda x: f"{x:.2%}")),
            ("Model", "Avg Pass Rate"),
        ),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute family-level Capability-Clash analytics.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    compute_family_results(args.results, args.manifest, args.csv, args.markdown)


if __name__ == "__main__":
    main()
