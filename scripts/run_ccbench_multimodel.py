#!/usr/bin/env python3
"""
Run CCBench (37 tasks × 8 strategies) with all 6 remaining models.

Models already run: gpt-4.1-mini, gpt-4.1-nano
Remaining models:
  Anthropic: claude-haiku-4-5, claude-opus-4-6
  Groq:      llama-4-scout-17b-16e-instruct, llama-3.3-70b-versatile,
             moonshot-kimi-k2-instruct, qwen/qwen3-32b

Usage:
    source venv/bin/activate
    export $(grep -E "^(OPENAI|ANTHROPIC|GROQ)_API_KEY=" .env | xargs)
    python scripts/run_ccbench_multimodel.py [--model MODEL] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Load .env
for line in (PROJECT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MANIFEST = PROJECT / "data" / "capability_clash" / "manifest_expanded.jsonl"
RUNS_OUT = PROJECT / "reports" / "capability_clash" / "runs"
SNAPSHOT_DIR = PROJECT / "reports" / "decomposition" / "real_world" / "real_repo" / "model_snapshots"
LOG_DIR = PROJECT / "reports" / "capability_clash" / "logs"

# (provider, model_name, short_label)
ALL_REMAINING_MODELS = [
    ("anthropic", "claude-haiku-4-5",                "haiku"),
    ("anthropic", "claude-opus-4-6",                 "opus"),
    ("groq",      "llama-4-scout-17b-16e-instruct",  "scout"),
    ("groq",      "llama-3.3-70b-versatile",         "l70b"),
    ("groq",      "moonshotai/moonshot-kimi-k2-0711","kimi"),
    ("groq",      "qwen/qwen3-32b",                  "qwen"),
]

STRATEGIES_ALPHA = "direct_baseline,contract_first,multi_view,failure_mode_first"
STRATEGIES_BETA  = "pattern_skeleton,simulation_trace,semantic_diff,role_decomposed"

PYTHON = sys.executable


def run_benchmark(provider: str, model: str, label: str, strategies: str,
                  manifest: Path, dry_run: bool = False) -> bool:
    """Run benchmark for one model × strategy-block combination."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{label}_{ts}.log"

    cmd = [
        PYTHON, "-m", "src.decomposition.runners.run_real_repo_benchmark",
        "--tasks-file", str(manifest),
        "--strategies", strategies,
        "--mode", "real_world_research",
        "--skip-oracle",
        "--provider", provider,
        "--model", model,
    ]

    if dry_run:
        print(f"DRY-RUN: {' '.join(cmd)}")
        return True

    print(f"\n[{ts}] Running {model} ({provider}) | {strategies.split(',')[0]}…")
    print(f"  Log: {log_file}")

    env = {**os.environ, "LLM_PROVIDER": provider, "LLM_MODEL": model}
    with log_file.open("w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(PROJECT), env=env,
        )
    if result.returncode != 0:
        print(f"  ✗ FAILED (rc={result.returncode}). See {log_file}")
        return False

    # Snapshot the strategy_comparison.csv with a model-specific name
    strategy_csv = PROJECT / "reports" / "decomposition" / "real_world" / "real_repo" / "strategy_comparison.csv"
    if strategy_csv.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        strat_tag = strategies.split(",")[0]
        dest = SNAPSHOT_DIR / f"ccbench_{label}_{strat_tag}_{ts}.csv"
        shutil.copy2(strategy_csv, dest)
        latest = SNAPSHOT_DIR / f"ccbench_{label}_{strat_tag}_latest.csv"
        shutil.copy2(dest, latest)
        print(f"  ✓ Snapshot: {dest.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Run only this model (by short label or full name)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_to_run = ALL_REMAINING_MODELS
    if args.model:
        models_to_run = [m for m in ALL_REMAINING_MODELS
                         if args.model in (m[1], m[2])]
        if not models_to_run:
            print(f"Model '{args.model}' not found. Options: {[m[2] for m in ALL_REMAINING_MODELS]}")
            sys.exit(1)

    print(f"CCBench multi-model runner")
    print(f"Manifest: {MANIFEST} ({sum(1 for _ in open(MANIFEST))} tasks)")
    print(f"Models to run: {[m[2] for m in models_to_run]}")

    failed = []
    for provider, model, label in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model} (provider={provider}, label={label})")
        print(f"{'='*60}")

        for strat_block, strats in [("alpha", STRATEGIES_ALPHA), ("beta", STRATEGIES_BETA)]:
            ok = run_benchmark(provider, model, f"{label}_{strat_block}", strats, MANIFEST, args.dry_run)
            if not ok:
                failed.append(f"{model}:{strat_block}")
            if not args.dry_run:
                time.sleep(2)  # Brief pause between blocks

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED blocks: {failed}")
    else:
        print("All blocks completed successfully.")
    print(f"\nNext step: merge results with scripts/merge_capability_clash_results.py")


if __name__ == "__main__":
    main()
