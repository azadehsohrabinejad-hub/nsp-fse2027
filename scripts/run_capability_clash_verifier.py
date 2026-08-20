#!/usr/bin/env python3
"""Train and evaluate lightweight verifiers on Capability-Clash semantic audit outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "reports" / "capability_clash" / "semantic_audit_results.csv"
DEFAULT_RESULTS = ROOT / "reports" / "capability_clash" / "verifier_results.csv"
DEFAULT_MD = ROOT / "reports" / "capability_clash" / "VERIFIER_RESULTS.md"


def evaluate(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1


def run_verifiers(audit_path: Path, results_csv: Path, markdown_path: Path) -> None:
    df = pd.read_csv(audit_path)
    df = df[df["final_pass"] >= 1.0].copy()
    if df.empty:
        raise SystemExit("No passing runs found for verifier training.")
    groups = df["task_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["incomplete_fix_detected"], groups))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    feature_cols = [
        "patch_completeness",
        "invariant_restoration",
        "localization_quality",
        "semantic_partial_fix",
        "counterfactual_robustness",
        "missing_target_count",
        "touched_target_count",
        "total_target_count",
    ]
    cat_cols = ["localization_tag", "mutation_family", "model", "strategy"]
    feature_frame = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, drop_first=False)
    feature_frame = feature_frame.fillna(0.0)
    X_train = feature_frame.iloc[train_idx]
    X_test = feature_frame.iloc[test_idx]
    y_train = train_df["incomplete_fix_detected"].astype(int)
    y_test = test_df["incomplete_fix_detected"].astype(int)

    baseline_pred = (test_df["invariant_restoration"] < 0.999).astype(int)
    baseline_precision, baseline_recall, baseline_f1 = evaluate(y_test, baseline_pred)

    lr_precision = lr_recall = lr_f1 = 0.0
    lr_note = "Skipped logistic regression because only one class was present."
    if y_train.nunique() >= 2 and y_test.nunique() >= 2:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        lr_pred = clf.predict(X_test)
        lr_precision, lr_recall, lr_f1 = evaluate(y_test, lr_pred)
        lr_note = "Logistic regression trained on audit-derived features."

    results = pd.DataFrame(
        [
            {"verifier": "target_file_recall < 1.0", "precision": baseline_precision, "recall": baseline_recall, "f1": baseline_f1},
            {"verifier": "logistic_regression", "precision": lr_precision, "recall": lr_recall, "f1": lr_f1},
        ]
    )
    results.to_csv(results_csv, index=False)

    summary = [
        "# Verifier Prototype Results",
        "",
        f"- Training samples: {len(train_df)} passing runs; test samples: {len(test_df)} passing runs.",
        f"- Baseline rule (`target_file_recall < 1.0`): precision {baseline_precision:.2f}, recall {baseline_recall:.2f}, F1 {baseline_f1:.2f}.",
        f"- Logistic regression (localization + repair features): precision {lr_precision:.2f}, recall {lr_recall:.2f}, F1 {lr_f1:.2f}.",
        f"- Note: {lr_note}",
        "",
    ]
    if lr_f1 > baseline_f1 + 1e-6:
        summary.append("**Outcome:** The learned verifier beats the simple threshold; include it as supporting evidence.")
    else:
        summary.append("**Outcome:** The learned verifier does not beat the simple threshold; record as a negative result.")
    markdown_path.write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Capability-Clash verifier baselines.")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    run_verifiers(args.audit, args.results, args.markdown)


if __name__ == "__main__":
    main()
