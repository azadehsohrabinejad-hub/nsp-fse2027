#!/usr/bin/env python3
"""Run the Capability-Clash matrix in manageable chunks."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / 'data' / 'capability_clash' / 'chunks'
LOG_DIR = ROOT / 'reports' / 'capability_clash' / 'runs'
SNAPSHOT_DIR = ROOT / 'reports' / 'decomposition' / 'real_world' / 'real_repo' / 'model_snapshots'

STRATEGY_BLOCKS = [
    ('alpha', 'direct_baseline,contract_first,multi_view,failure_mode_first'),
    ('beta', 'pattern_skeleton,simulation_trace,semantic_diff,role_decomposed'),
]
MODELS = ['gpt-4.1-nano', 'gpt-4.1-mini']
PYTHON_BIN = Path(sys.executable)
STRATEGY_CSV = ROOT / 'reports' / 'decomposition' / 'real_world' / 'real_repo' / 'strategy_comparison.csv'


def snapshot_results(chunk: Path, model: str, label: str, timestamp: str) -> None:
    if not STRATEGY_CSV.exists():
        return
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = SNAPSHOT_DIR / f"{chunk.stem}_{model}_{label}_{timestamp}.csv"
    shutil.copy2(STRATEGY_CSV, dest)
    latest = SNAPSHOT_DIR / f"{chunk.stem}_{model}_{label}_latest.csv"
    shutil.copy2(dest, latest)
    print(f"[capability-clash] Snapshot saved to {dest}")


def run_chunk(chunk: Path, model: str, label: str, strategies: str, dry_run: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f"{chunk.stem}_{model}_{label}_{timestamp}.log"
    cmd = [
        str(PYTHON_BIN),
        '-m',
        'src.decomposition.runners.run_real_repo_benchmark',
        '--tasks-file',
        str(chunk),
        '--strategies',
        strategies,
        '--mode',
        'real_world_research',
        '--skip-oracle',
        '--provider',
        'openai',
        '--model',
        model,
    ]
    if dry_run:
        print('DRY-RUN', ' '.join(cmd))
        return
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('CMD: ' + ' '.join(cmd) + '\n')
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Run failed for {chunk.name} {model} {label}. See {log_path}")
    snapshot_results(chunk, model, label, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full Capability-Clash matrix via chunked manifests.')
    parser.add_argument('--dry-run', action='store_true', help='Print the commands without executing.')
    args = parser.parse_args()

    chunks = sorted(CHUNKS_DIR.glob('manifest_chunk_*.jsonl'))
    if not chunks:
        raise SystemExit(f'No chunk manifests found under {CHUNKS_DIR}')

    for model in MODELS:
        for chunk in chunks:
            for label, strategies in STRATEGY_BLOCKS:
                run_chunk(chunk, model, label, strategies, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
