#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'reports/capability_clash/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = ROOT / 'reports/capability_clash/capability_clash_results.csv'
AUDIT_CSV = ROOT / 'reports/capability_clash/semantic_audit_results.csv'
FAMILY_CSV = ROOT / 'reports/capability_clash/family_level_results.csv'
VERIFIER_CSV = ROOT / 'reports/capability_clash/verifier_results.csv'

sns.set_style('whitegrid')


def save_fig(fig, name: str) -> None:
    png_path = FIG_DIR / f'{name}.png'
    pdf_path = FIG_DIR / f'{name}.pdf'
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def fig_strategy_by_family():
    family = pd.read_csv(FAMILY_CSV)
    family_strategy = family.groupby(['mutation_family','strategy'])['pass_rate'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=family_strategy, x='mutation_family', y='pass_rate', hue='strategy', ax=ax)
    ax.set_ylabel('Pass rate')
    ax.set_xlabel('Mutation family')
    ax.set_title('Strategy spread by mutation family (avg across models)')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, 'fig_strategy_spread_by_family')


def fig_semantic_disagreement():
    results = pd.read_csv(RESULTS_CSV)
    audit = pd.read_csv(AUDIT_CSV)
    manifest = pd.read_json(ROOT / 'data/capability_clash/manifest.jsonl', lines=True)
    merged = results.merge(audit[['task_id','model','strategy','patch_completeness','invariant_restoration']], on=['task_id','model','strategy'], how='left')
    merged = merged.merge(manifest[['task_id','mutation_family']], on='task_id', how='left')
    merged['tests_pass'] = merged['final_pass'] == 1.0
    merged['audit_flag'] = (merged['patch_completeness'] < 1.0) | (merged['invariant_restoration'] < 1.0)
    merged['category'] = 'Tests fail'
    merged.loc[merged['tests_pass'] & merged['audit_flag'], 'category'] = 'Audit disagrees'
    merged.loc[merged['tests_pass'] & (~merged['audit_flag']), 'category'] = 'Clean pass'
    shares = merged.groupby(['mutation_family','category']).size().reset_index(name='count')
    shares['share'] = shares['count'] / shares.groupby('mutation_family')['count'].transform('sum')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=shares, x='mutation_family', y='share', hue='category', ax=ax)
    ax.set_ylabel('Share of runs')
    ax.set_title('Semantic audit vs pass/fail disagreement by family')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, 'fig_semantic_audit_disagreement')


def fig_difficulty_ladder():
    df = pd.read_csv(RESULTS_CSV)
    manifest = pd.read_json(ROOT / 'data/capability_clash/manifest.jsonl', lines=True)
    family_overall = df.merge(manifest[['task_id','mutation_family']], on='task_id', how='left')
    rung_easy = df['initial_pass'].mean()
    rung_cc = df['final_pass'].mean()
    hardest = family_overall.groupby('mutation_family')['final_pass'].mean().min()
    ladder = pd.DataFrame({
        'tier': ['Legacy easy tier', 'Capability-Clash overall', 'Hardest family'],
        'pass_rate': [rung_easy, rung_cc, hardest],
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=ladder, x='tier', y='pass_rate', ax=ax, palette='viridis')
    ax.set_ylabel('Pass rate')
    ax.set_title('Difficulty ladder (pass rate)')
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    save_fig(fig, 'fig_difficulty_ladder')


def fig_model_gap():
    family = pd.read_csv(FAMILY_CSV)
    family_model = family.groupby(['mutation_family','model'])['pass_rate'].mean().reset_index()
    pivot = family_model.pivot(index='mutation_family', columns='model', values='pass_rate').fillna(0)
    pivot['gap'] = (pivot.max(axis=1) - pivot.min(axis=1))
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot['gap'].sort_values().plot(kind='bar', ax=ax, color='tomato')
    ax.set_ylabel('Pass-rate gap')
    ax.set_title('Model-tier gap by mutation family')
    ax.set_ylim(0, 1)
    save_fig(fig, 'fig_model_gap_by_family')


def fig_verifier():
    verifier = pd.read_csv(VERIFIER_CSV)
    metrics = verifier.melt(id_vars='verifier', value_vars=['precision','recall','f1'], var_name='metric', value_name='score')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=metrics, x='metric', y='score', hue='verifier', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Verifier baseline vs logistic regression')
    save_fig(fig, 'fig_verifier_comparison')


def main():
    fig_strategy_by_family()
    fig_semantic_disagreement()
    fig_difficulty_ladder()
    fig_model_gap()
    fig_verifier()

if __name__ == '__main__':
    main()
