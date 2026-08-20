#!/usr/bin/env python3
"""Merge Capability-Clash snapshot CSVs into canonical result artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "capability_clash" / "manifest_matrix.jsonl"
DEFAULT_SNAPSHOT_DIR = ROOT / "reports" / "decomposition" / "real_world" / "real_repo" / "model_snapshots"
DEFAULT_RESULTS = ROOT / "reports" / "capability_clash" / "capability_clash_results.csv"
DEFAULT_MATRIX = ROOT / "reports" / "capability_clash" / "capability_clash_task_matrix.csv"
DEFAULT_COVERAGE = ROOT / "reports" / "capability_clash" / "run_coverage_report.md"


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            entries[payload["task_id"]] = payload
    return entries


def collect_snapshots(snapshot_dir: Path) -> List[Path]:
    files = sorted(snapshot_dir.glob("manifest_chunk_*_latest.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshot CSVs found under {snapshot_dir}")
    return files


def merge_results(manifest: Dict[str, Dict[str, object]], snapshots: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in snapshots:
        df = pd.read_csv(path)
        df["snapshot_source"] = path.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["task_id", "model", "strategy", "snapshot_source"])
    merged = merged.drop_duplicates(subset=["task_id", "model", "strategy"], keep="last")
    merged = merged[merged["task_id"].isin(manifest.keys())]
    merged = merged.reset_index(drop=True)
    return merged


def build_matrix(results: pd.DataFrame, manifest: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    df = results.copy()
    df["pass_flag"] = (df["final_pass"] >= 1.0).astype(float)
    matrix = (
        df.pivot_table(
            index="task_id",
            columns=["model", "strategy"],
            values="pass_flag",
            aggfunc="max",
            fill_value=0.0,
        )
        .reset_index()
        .sort_values("task_id")
    )
    matrix.columns = ["task_id"] + [f"{lvl0}__{lvl1}" for lvl0, lvl1 in matrix.columns.tolist()[1:]]
    dataset_sources = {task_id: entry.get("dataset_source", entry.get("repo_slug", "")) for task_id, entry in manifest.items()}
    matrix.insert(1, "dataset_source", matrix["task_id"].map(dataset_sources))
    return matrix


def write_coverage(results: pd.DataFrame, manifest: Dict[str, Dict[str, object]], coverage_path: Path) -> None:
    now = datetime.now(timezone.utc).astimezone()
    manifest_tasks = list(manifest.keys())
    model_summary = (
        results.assign(pass_flag=results["final_pass"] >= 1.0)
        .groupby("model")
        .agg(runs=("task_id", "count"), passes=("pass_flag", "sum"))
        .reset_index()
    )
    model_summary["pass_rate"] = model_summary["passes"] / model_summary["runs"]

    per_task = (
        results.assign(pass_flag=results["final_pass"] >= 1.0)
        .groupby(["task_id", "model"])
        .agg(passes=("pass_flag", "sum"), attempts=("strategy", "count"))
        .reset_index()
    )
    dataset_sources = {task_id: entry.get("dataset_source", entry.get("repo_slug", "")) for task_id, entry in manifest.items()}
    per_task["dataset_source"] = per_task["task_id"].map(dataset_sources)
    per_task = per_task.sort_values(["dataset_source", "task_id", "model"])

    lines: List[str] = [
        "# Capability-Clash Run Coverage",
        "",
        f"- Run tag: capability_clash_full_matrix_{now.strftime('%Y%m%d_%H%M%S')}",
        f"- Generated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Tasks: {len(manifest_tasks)}",
        "- Strategies: 8",
        "- Models: " + ", ".join(sorted(results['model'].unique())),
        "",
        "| Model | Runs | Passes | Pass Rate |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in model_summary.iterrows():
        lines.append(f"| {row['model']} | {int(row['runs'])} | {int(row['passes'])} | {row['pass_rate']:.2%} |")
    lines.append("")
    lines.append("## Task Coverage")
    lines.append("| Task | Dataset | Model | Passes | Attempts |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in per_task.iterrows():
        lines.append(
            f"| {row['task_id']} | {row['dataset_source']} | {row['model']} | "
            f"{int(row['passes'])} | {int(row['attempts'])} |"
        )
    coverage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Capability-Clash snapshot CSVs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    snapshots = collect_snapshots(args.snapshot_dir)
    merged = merge_results(manifest, snapshots)
    merged.to_csv(args.results, index=False)
    matrix = build_matrix(merged, manifest)
    matrix.to_csv(args.matrix, index=False)
    write_coverage(merged, manifest, args.coverage)
    print(f"Wrote {len(merged)} rows to {args.results}")


if __name__ == "__main__":
    main()
