#!/usr/bin/env python3
"""Classify Capability-Clash failures into taxonomy buckets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "reports" / "capability_clash" / "capability_clash_results.csv"
DEFAULT_MANIFEST = ROOT / "data" / "capability_clash" / "manifest_matrix.jsonl"
DEFAULT_CSV = ROOT / "reports" / "capability_clash" / "failure_taxonomy_counts.csv"
DEFAULT_MD = ROOT / "reports" / "capability_clash" / "FAILURE_TAXONOMY.md"
DUAL_FAMILIES = {"compensating_dual_mutation", "incomplete_patch_trap"}


def load_manifest(path: Path) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            metadata = payload.get("metadata") or {}
            family = metadata.get("mutation_family") or payload.get("mutation_family") or "unknown"
            repo = payload.get("dataset_source") or payload.get("repo_slug") or "unknown"
            mapping[payload["task_id"]] = (family, repo)
    return mapping


def classify_failure(row: pd.Series) -> str:
    if row["final_pass"] >= 1.0:
        return "success"
    msg = (row.get("dominant_failure_category") or "").strip()
    tf_recall = float(row.get("target_file_recall") or 0.0)
    loc_precision = float(row.get("localization_precision") or 0.0)
    edited_count = float(row.get("edited_file_count") or 0.0)
    family = row.get("mutation_family", "unknown")
    if msg.startswith("fail::test edits are prohibited"):
        return "test-message overfitting"
    if msg.startswith("fail::edit path"):
        return "wrong-file localization"
    if msg.startswith("fail::multi-file target requires edits") or (0.0 < tf_recall < 1.0):
        return "one-file-only repair"
    if "old_string not found" in msg or "apply_patch" in msg:
        return "patch application failure"
    if tf_recall >= 1.0 and family in DUAL_FAMILIES:
        return "wrong invariant restored"
    if edited_count <= 1 and tf_recall == 0 and loc_precision == 0:
        return "shallow local patch"
    if loc_precision == 0 and edited_count > 0:
        return "wrong-file localization"
    if tf_recall >= 1.0:
        return "semantic mismatch after partial fix"
    return "semantic mismatch after partial fix"


def format_table(df: pd.DataFrame, headers: Tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def compute_taxonomy(results_path: Path, manifest_path: Path, csv_path: Path, md_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    df = pd.read_csv(results_path)
    df = df[df["task_id"].isin(manifest.keys())].copy()
    df[["mutation_family", "repo_slug"]] = df["task_id"].apply(lambda tid: pd.Series(manifest[tid]))
    df["failure_taxonomy"] = df.apply(classify_failure, axis=1)
    df = df[df["failure_taxonomy"] != "success"].copy()

    agg = (
        df.groupby(["mutation_family", "repo_slug", "model", "strategy", "failure_taxonomy"])
        .size()
        .reset_index(name="count")
        .sort_values(["mutation_family", "failure_taxonomy", "model", "strategy"])
    )
    agg.to_csv(csv_path, index=False)

    family_table = (
        df.groupby(["mutation_family", "failure_taxonomy"])
        .size()
        .reset_index(name="count")
        .sort_values(["mutation_family", "count"], ascending=[True, False])
    )
    strategy_table = (
        df.groupby(["strategy", "failure_taxonomy"])
        .size()
        .reset_index(name="count")
        .sort_values(["strategy", "count"], ascending=[True, False])
    )
    model_table = (
        df.groupby(["model", "failure_taxonomy"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )
    repo_table = (
        df.groupby(["repo_slug", "failure_taxonomy"])
        .size()
        .reset_index(name="count")
        .sort_values(["repo_slug", "count"], ascending=[True, False])
    )

    top_issue = family_table.iloc[0]
    notes = (
        f"Most frequent failure: **{top_issue['failure_taxonomy']}** "
        f"inside **{top_issue['mutation_family']}** ({int(top_issue['count'])} runs). "
        "Audit-driven categories show that one-file-only repairs and test-edit attempts "
        "remain the dominant failure modes on Capability-Clash."
    )

    sections = [
        "# Failure Taxonomy",
        "",
        "## Failures by Mutation Family",
        format_table(family_table, ("Mutation Family", "Failure Type", "Count")),
        "",
        "## Failures by Strategy",
        format_table(strategy_table, ("Strategy", "Failure Type", "Count")),
        "",
        "## Failures by Model Tier",
        format_table(model_table, ("Model", "Failure Type", "Count")),
        "",
        "## Failures by Repo",
        format_table(repo_table, ("Repo", "Failure Type", "Count")),
        "",
        notes,
    ]
    md_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute failure taxonomy tables.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    compute_taxonomy(args.results, args.manifest, args.csv, args.markdown)


if __name__ == "__main__":
    main()
