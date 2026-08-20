# CCBench Artifact — ASE 2026

**Paper:** Capability-Clash: Hard Semantic Floors, Precision-over-Scale Inversion, and Reward Corruption in LLM Code Repair

---

## Directory Structure

```
ccbench/
├── paper/
│   ├── submission_paper.tex     # Full paper source (ACM sigconf, double-blind)
│   └── refs.bib                 # Bibliography
├── data/
│   (task manifests — see below)
├── results/
│   ├── capability_clash_results_complete_1184.csv   # Full 1,184-run matrix (37 tasks × 8 strategies × 4 models)
│   ├── capability_clash_results_4model_37task.csv   # Per-task pass/fail for all 4 models
│   ├── capability_clash_task_matrix.csv             # Task × model difficulty matrix
│   ├── ollama_e0_results.csv                        # E0 ablation: qwen2.5-coder:3b, 103 runs, 0/103 pass
│   ├── semantic_audit_results.csv                   # Audit of 72 test-passing repairs (100% disagreement)
│   ├── verifier_results.csv                         # Verifier study: 5 judges × F1 scores
│   ├── t3_results.csv                               # T3 (Opus) full run results
│   ├── failure_taxonomy_counts.csv                  # Per-family failure breakdown
│   ├── family_level_results_37task.csv              # Pass rates per mutation family × model
│   └── table_capability_clash_main.csv              # Main paper table (Table 3)
├── figures/
│   ├── fig_difficulty_ladder.{png,pdf}              # Figure 1: task difficulty ladder
│   ├── fig_model_gap_by_family.{png,pdf}            # Figure 2: per-family model gap
│   ├── fig_semantic_audit_disagreement.{png,pdf}    # Figure 3: audit disagreement rates
│   ├── fig_strategy_spread_by_family.{png,pdf}      # Figure 4: strategy spread
│   └── fig_verifier_comparison.{png,pdf}            # Figure 5: verifier F1 comparison
└── scripts/
    ├── generate_capability_clash_tasks.py           # Task generation + mutation injection
    ├── validate_capability_clash_tasks.py           # Four-step verification protocol
    ├── run_capability_clash_matrix.py               # Main benchmark runner (single model)
    ├── run_ccbench_multimodel.py                    # Multi-model runner (4 models parallel)
    ├── run_capability_clash_full_matrix.sh          # Shell wrapper: all 4 models × 8 strategies
    ├── run_capability_clash_model.sh                # Shell wrapper: single model run
    ├── merge_capability_clash_results.py            # Merge per-model CSVs into full matrix
    ├── compute_semantic_audit_metrics.py            # Fleiss kappa, audit disagreement stats
    ├── llm_judge_verifier.py                        # Reference-free LLM judge (F1=0.615)
    ├── llm_judge_gt_verifier.py                     # GT-guided LLM judge (F1=0.993)
    ├── llm_judge_t3_verifier.py                     # T3 judge (gpt-4.1, F1=0.917)
    ├── run_capability_clash_verifier.py             # Verifier study runner
    ├── compute_failure_taxonomy.py                  # Failure mode taxonomy counts
    ├── compute_family_level_results.py              # Per-family pass rate aggregation
    ├── generate_capability_clash_figures.py         # Reproduce all 5 paper figures
    └── bootstrap_ci.py                              # Bootstrap 95% CIs for pass rates
```

---

## Benchmark Overview

- **37 tasks** × **8 strategies** × **4 models** = **1,184 canonical runs**
- **4 models**: gpt-4.1-mini (T2), gpt-4.1-nano (T1), claude-haiku-4-5 (T2), claude-opus-4-6 (T3)
- **5 mutation families**: compensating_dual, temporal_state, incomplete_patch, false_localization, semantic_shadow
- **5 repositories**: braindecode (Python), simtradedata (Python), tmuxp (Python), tc-template-node-postgres (Node.js), appwrite (PHP)
- **E0 ablation**: qwen2.5-coder:3b (Ollama, 3B open-weight) — 103 runs, 0/103 pass rate
- **Total paper runs**: 1,562 (including ablations E0–E5, temperature study, 47-task replication)

---

## Reproducing Results

### Prerequisites
```bash
cd /path/to/DataCollector
source venv/bin/activate
export OPENAI_API_KEY=<your key>
export ANTHROPIC_API_KEY=<your key>
```

### Run the full 4-model benchmark
```bash
python scripts/run_ccbench_multimodel.py \
  --manifest data/capability_clash_manifest_37task.jsonl \
  --models gpt-4.1-mini gpt-4.1-nano claude-haiku-4-5 claude-opus-4-6 \
  --strategies all \
  --out results/
```

### Reproduce figures
```bash
python scripts/generate_capability_clash_figures.py \
  --results results/capability_clash_results_complete_1184.csv \
  --out figures/
```

### Reproduce semantic audit stats
```bash
python scripts/compute_semantic_audit_metrics.py \
  --audit results/semantic_audit_results.csv
```

---

## Key Numbers (paper-ready)

| Metric | Value |
|--------|-------|
| Hard floor (all 4 models fail) | 21/37 tasks (56.8%) |
| Haiku pass rate | 19.3% [14.9%, 24.0%] |
| Opus pass rate | 8.8% [5.4%, 12.2%] |
| Precision-over-scale inversion | z=3.67, p<0.001 |
| Reward corruption rate | 72/72 = 100% |
| GT-guided verifier F1 | 0.993 |
| Reference-free verifier F1 | 0.615 |
| E0 open-weight pass rate | 0/103 = 0.0% |
| 47-task replication floor | 41/47 = 87.2% |

---

## Vendored Repositories (required to re-run harness)

The five repositories used in the benchmark are not bundled due to size. Clone them before running any benchmark commands:

```bash
mkdir -p experiments/real_repos && cd experiments/real_repos

# Python repos
git clone https://github.com/braindecode/braindecode.git
git clone https://github.com/tony/tmuxp.git
git clone https://github.com/Mnemox-AI/idea-reality-mcp.git   # simtradedata proxy
git clone https://github.com/simtrade/simtradedata.git

# Node.js repo
git clone https://github.com/topcoder-platform/tc-template-node-postgres.git
cd tc-template-node-postgres && npm install && cd ../..

# PHP repo (optional — 4 tasks excluded from core analyses due to Composer env mismatch)
git clone https://github.com/appwrite/appwrite.git
```

After cloning, each task's `repo_path` in the manifest must point to the local clone. The harness copies `repo_path` to a temp workspace and applies/reverts mutations there — the source clone is never modified.

---

## Notes

- Task mutation source code is in `src/decomposition/real_repo/` (harness) and `experiments/real_repo_tasks/` (per-task workspaces).
- Mutations are PRE-APPLIED in each task's `workspace/` subdirectory. The harness uses `repo_path` from the manifest — this should point to the pre-mutated workspace, not the clean clone.
- PHP/Appwrite tasks (4 of 37) fail due to Composer environment mismatch in the harness, not repair failure. Excluded from family-conditioned and routing analyses.
