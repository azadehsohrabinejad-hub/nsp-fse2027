#!/usr/bin/env python3
"""Compute Capability-Clash semantic audit metrics from benchmark runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

DEFAULT_RESULTS = Path('reports/capability_clash/capability_clash_results.csv')
DEFAULT_MANIFEST = Path('data/capability_clash/manifest.jsonl')
DEFAULT_OUTPUT = Path('reports/capability_clash/semantic_audit_results.csv')
DEFAULT_SUMMARY = Path('reports/capability_clash/SEMANTIC_AUDIT_SUMMARY.md')


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    with manifest_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            task_dir = payload.get('task_dir')
            if task_dir:
                task_dir_path = Path(task_dir)
                task_json_path = task_dir_path / 'task.json'
                if task_json_path.exists():
                    try:
                        payload['_task_json'] = json.loads(task_json_path.read_text(encoding='utf-8'))
                    except json.JSONDecodeError:
                        payload['_task_json'] = {}
                else:
                    payload['_task_json'] = {}
            mapping[payload['task_id']] = payload
    return mapping


def _parse_files(value: object) -> Set[str]:
    if isinstance(value, str):
        if not value:
            return set()
        parts = [entry.strip() for entry in value.split(';')]
        return {entry for entry in parts if entry}
    if isinstance(value, list):
        return {str(entry).strip() for entry in value if str(entry).strip()}
    return set()


def compute_metrics(results: pd.DataFrame, manifest: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in results.iterrows():
        task_id = row['task_id']
        manifest_entry = manifest.get(task_id, {})
        mutation_family = manifest_entry.get('mutation_family', 'unknown')
        task_json = manifest_entry.get('_task_json') or {}
        task_metadata = task_json.get('metadata') or {}
        manifest_targets = manifest_entry.get('target_files') or []
        target_files = set(task_json.get('target_files') or task_metadata.get('target_files') or manifest_targets or [])
        trap_files = set(task_metadata.get('trap_files') or manifest_entry.get('trap_files') or [])
        edited_files = _parse_files(row.get('edited_files'))
        touched_targets = edited_files & target_files if target_files else set()
        missing_targets = target_files - touched_targets if target_files else set()
        touched_traps = edited_files & trap_files if trap_files else set()
        dual_gap = max(len(missing_targets), 0)
        touched_target_count = len(touched_targets)
        total_target_count = len(target_files)
        coverage_ratio = (touched_target_count / total_target_count) if total_target_count else 0.0

        localization_tag = 'none'
        localization_detail = 'none'
        if touched_traps and not touched_targets:
            localization_tag = 'trap_only'
            localization_detail = 'trap_only'
        elif total_target_count and touched_target_count == total_target_count and not touched_traps:
            localization_tag = 'dual_only'
            localization_detail = 'full_targets_clean'
        elif touched_targets or touched_traps or edited_files:
            localization_tag = 'partial_edit'
            if touched_targets and touched_traps:
                localization_detail = 'targets_plus_traps'
            elif touched_targets:
                localization_detail = 'targets_partial'
            elif touched_traps:
                localization_detail = 'trap_only_partial'
            else:
                localization_detail = 'non_target_edit'

        missing_edit_kind = 'none'
        if total_target_count == 0:
            if not edited_files:
                missing_edit_kind = 'not_applicable'
            elif touched_traps:
                missing_edit_kind = 'no_targets_defined_trap_edit'
            else:
                missing_edit_kind = 'no_targets_defined'
        elif touched_target_count == 0:
            missing_edit_kind = 'all_targets_missing'
        elif dual_gap == 0:
            missing_edit_kind = 'none'
        elif dual_gap == 1:
            missing_edit_kind = 'single_edit_missing'
        else:
            missing_edit_kind = 'multi_edits_missing'

        def safe_float(value: object) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        metrics = {
            'patch_completeness': safe_float(row.get('ground_truth_recall')),  # coverage of GT patches
            'invariant_restoration': safe_float(row.get('target_file_recall')),  # both mutation files restored
                'localization_quality': safe_float(row.get('localization_precision')),  # trap avoidance proxy
                'semantic_partial_fix': safe_float(row.get('repair_gain')),  # how much failing tests improved
                'counterfactual_robustness': safe_float(row.get('candidate_overlap_rate')),  # edits overlap with traps?
            }
        final_pass = safe_float(row.get('final_pass')) >= 1.0
        audit_success = metrics['patch_completeness'] >= 0.99 and metrics['invariant_restoration'] >= 0.99
        audit_disagrees = audit_success != final_pass
        incomplete_fix = final_pass and not audit_success
        rows.append(
            {
                'task_id': task_id,
                'mutation_family': mutation_family,
                'model': row['model'],
                'strategy': row['strategy'],
                'final_pass': safe_float(row.get('final_pass')),
                **metrics,
                'localization_tag': localization_tag,
                'localization_detail': localization_detail,
                'missing_edit_kind': missing_edit_kind,
                'missing_target_count': dual_gap,
                'touched_target_count': touched_target_count,
                'total_target_count': total_target_count,
                'target_coverage_ratio': coverage_ratio,
                'edited_target_files': ";".join(sorted(touched_targets)),
                'missing_target_files': ";".join(sorted(missing_targets)),
                'edited_trap_files': ";".join(sorted(touched_traps)),
                'touched_trap_files': bool(touched_traps),
                'audit_disagrees_tests': audit_disagrees,
                'incomplete_fix_detected': incomplete_fix,
            }
        )
    return pd.DataFrame(rows)


def write_summary(metrics: pd.DataFrame, summary_path: Path) -> None:
    lines = ["# Semantic Audit Summary", ""]
    grouped = metrics.groupby('mutation_family')[['patch_completeness','invariant_restoration','localization_quality','semantic_partial_fix','counterfactual_robustness']].mean().reset_index()
    lines.append("| Mutation Family | Patch Completeness | Invariant Restoration | Localization Quality | Semantic Partial Fix | Counterfactual Robustness |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in grouped.iterrows():
        lines.append(
            "| {family} | {pc:.2f} | {ir:.2f} | {lq:.2f} | {spf:.2f} | {cr:.2f} |".format(
                family=row['mutation_family'], pc=row['patch_completeness'], ir=row['invariant_restoration'], lq=row['localization_quality'], spf=row['semantic_partial_fix'], cr=row['counterfactual_robustness']
            )
        )
    lines.append("")
    model_group = metrics.groupby('model')[['patch_completeness','invariant_restoration','localization_quality','semantic_partial_fix','counterfactual_robustness']].mean().reset_index()
    lines.append("## Model-level Averages")
    lines.append("| Model | Patch Completeness | Invariant Restoration | Localization Quality | Semantic Partial Fix | Counterfactual Robustness |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in model_group.iterrows():
        lines.append(
            "| {model} | {pc:.2f} | {ir:.2f} | {lq:.2f} | {spf:.2f} | {cr:.2f} |".format(
                model=row['model'], pc=row['patch_completeness'], ir=row['invariant_restoration'], lq=row['localization_quality'], spf=row['semantic_partial_fix'], cr=row['counterfactual_robustness']
            )
        )
    lines.append("")
    disagree_group = metrics.groupby('mutation_family')[['audit_disagrees_tests','incomplete_fix_detected']].mean().reset_index()
    lines.append("## Semantic Audit vs Pass/Fail")
    lines.append("| Mutation Family | Disagreement Rate | Incomplete-Fix Rate |")
    lines.append("| --- | --- | --- |")
    for _, row in disagree_group.iterrows():
        lines.append("| {family} | {dis:.2%} | {inc:.2%} |".format(
            family=row['mutation_family'],
            dis=row['audit_disagrees_tests'],
            inc=row['incomplete_fix_detected'],
        ))
    lines.append("")
    tag_counts = metrics['localization_tag'].value_counts(normalize=True)
    lines.append("## Localization Tag Distribution")
    lines.append("| Tag | Share |")
    lines.append("| --- | --- |")
    for tag, share in tag_counts.items():
        lines.append(f"| {tag} | {share:.2%} |")

    def _format_share_table(frame: pd.DataFrame, title: str) -> None:
        lines.append("")
        lines.append(title)
        if frame.empty:
            lines.append("(no data)")
            return
        headers = ['Mutation Family'] + list(frame.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(['---'] * len(headers)) + " |")
        for family, row in frame.iterrows():
            cells = [family]
            for col in frame.columns:
                cells.append(f"{row[col]:.0%}")
            lines.append("| " + " | ".join(cells) + " |")

    tag_share = (
        metrics.groupby(['mutation_family', 'localization_tag']).size()
        .unstack(fill_value=0)
    )
    tag_share = tag_share.div(tag_share.sum(axis=1), axis=0)
    tag_share = tag_share.reindex(columns=['trap_only', 'dual_only', 'partial_edit', 'none'], fill_value=0.0)
    _format_share_table(tag_share, "## Localization Tag Mix by Family")

    detail_share = (
        metrics.groupby(['mutation_family', 'localization_detail']).size()
        .unstack(fill_value=0)
    )
    detail_share = detail_share.div(detail_share.sum(axis=1), axis=0)
    _format_share_table(detail_share, "## Localization Detail Mix by Family")

    missing_share = (
        metrics.groupby(['mutation_family', 'missing_edit_kind']).size()
        .unstack(fill_value=0)
    )
    missing_share = missing_share.div(missing_share.sum(axis=1), axis=0)
    missing_cols = ['all_targets_missing', 'single_edit_missing', 'multi_edits_missing', 'none']
    for col in missing_cols:
        if col not in missing_share.columns:
            missing_share[col] = 0.0
    missing_share = missing_share[missing_cols]
    _format_share_table(missing_share, "## Missing Edit Breakdown by Family")

    summary_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute semantic audit metrics from Capability-Clash runs.")
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    results_df = pd.read_csv(args.results)
    manifest = load_manifest(args.manifest)
    metrics_df = compute_metrics(results_df, manifest)
    metrics_df.to_csv(args.output, index=False)
    write_summary(metrics_df, args.summary)


if __name__ == '__main__':
    main()
