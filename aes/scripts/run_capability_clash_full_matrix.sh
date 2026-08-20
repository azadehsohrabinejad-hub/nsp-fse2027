#!/bin/zsh
set -euo pipefail
cd /Users/karanallagh/Desktop/DataCollector
if [ -f .env ]; then
  set -a
  source .env >/dev/null 2>&1
  set +a
fi
source venv/bin/activate
SNAPSHOT_DIR="/Users/karanallagh/Desktop/DataCollector/reports/decomposition/real_world/real_repo/model_snapshots"
TASKS_FILE="${CAPABILITY_CLASH_TASKS_FILE:-data/capability_clash/manifest_matrix.jsonl}"
mkdir -p "$SNAPSHOT_DIR"
run_model() {
  MODEL="$1"
  echo "[capability-clash] Running model $MODEL on $TASKS_FILE" >&2
  python -m src.decomposition.runners.run_real_repo_benchmark \
    --tasks-file "$TASKS_FILE" \
    --strategies direct_baseline,contract_first,multi_view,failure_mode_first,pattern_skeleton,simulation_trace,semantic_diff,role_decomposed \
    --model "$MODEL" \
    --mode real_world_research \
    --skip-oracle
  TS=$(date +%Y%m%d_%H%M%S)
  SNAP_FILE="$SNAPSHOT_DIR/strategy_comparison_${MODEL}_${TS}.csv"
  cp reports/decomposition/real_world/real_repo/strategy_comparison.csv "$SNAP_FILE"
  ln -sf "$SNAP_FILE" "$SNAPSHOT_DIR/strategy_comparison_${MODEL}_latest.csv"
  echo "[capability-clash] Copied strategy comparison to $SNAP_FILE" >&2
}
for MODEL in "$@"; do
  run_model "$MODEL"
  echo "[capability-clash] Completed model $MODEL" >&2
  # small pause between tiers to reduce contention
  sleep 5
done
