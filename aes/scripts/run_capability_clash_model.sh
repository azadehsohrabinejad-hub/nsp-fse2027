#!/bin/zsh
set -euo pipefail
MODEL="$1"
cd /Users/karanallagh/Desktop/DataCollector
if [ -f .env ]; then
  set -a
  source .env >/dev/null 2>&1
  set +a
fi
source venv/bin/activate
TASKS_FILE="${CAPABILITY_CLASH_TASKS_FILE:-data/capability_clash/manifest_matrix.jsonl}"
python -m src.decomposition.runners.run_real_repo_benchmark \
  --tasks-file "$TASKS_FILE" \
  --strategies direct_baseline,contract_first,multi_view,failure_mode_first,pattern_skeleton,simulation_trace,semantic_diff,role_decomposed \
  --model "$MODEL" \
  --mode real_world_research \
  --skip-oracle
