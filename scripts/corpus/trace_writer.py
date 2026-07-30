import pandas as pd
import os
import json

# 1. Define paths
manifest_path = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0.csv"
base_trace_dir = r"D:\raz\razieh\traces"

print("Reading manifest...")
df = pd.read_csv(manifest_path)

# 2. Helper function to create a minimal valid round
def create_minimal_round(round_index):
    return {
        "round_index": round_index,
        "context": {"files_retrieved": [], "context_tokens": 0},
        "prompt": {"input_tokens": 0, "rendered_prompt_path": None},
        "response": {"output_tokens": 0, "parse_status": "unknown"},
        "edits": {"statistics": {"applied_edit_count": 0, "files_touched": 0}},
        "execution": {"tests": {"pass_rate": 0.0, "failing_tests": []}},
        "observation": {
            "test_features": {"pass_rate": 0.0, "delta_pass_rate": 0.0},
            "behavior_features": {"invalid_patch": False, "test_regression": False}
        }
    }

# 3. Generate a trace for each run
print(f"Generating {len(df)} traces...")
for index, row in df.iterrows():
    run_id = str(row["run_id"])
    
    # Parse run_id to extract components (task_id__model__strategy)
    parts = run_id.split("__")
    task_id = parts[0] if len(parts) > 0 else "unknown"
    model_name = parts[1] if len(parts) > 1 else "unknown"
    strategy_id = parts[2] if len(parts) > 2 else "unknown"

    # Create directory structure: traces/model/strategy/task/
    run_dir = os.path.join(base_trace_dir, model_name, strategy_id, task_id)
    os.makedirs(run_dir, exist_ok=True)

    # Build the NSP Trace object
    trace = {
        "schema": {"name": "NSP Trace", "version": "1.0.0"},
        "run": {
            "run_id": run_id,
            "seed": 1, # Assuming default seed 1 for pilot
            "status": "completed" if row["execution_status"] == "Completed" else "failed",
            "actual_rounds": int(row["num_rounds"]) if pd.notna(row["num_rounds"]) else 0
        },
        "task": {
            "task_id": task_id,
            "benchmark": "CCBench"
        },
        "model": {
            "model_name": model_name
        },
        "strategy": {
            "strategy_id": strategy_id
        },
        "environment": {
            "repository_state_hash": "unknown_local_metadata_only"
        },
        "initial_state": {
            "tests": {"status": "unknown", "pass_rate": 0.0}
        },
        "rounds": [create_minimal_round(i+1) for i in range(int(row["num_rounds"]) if pd.notna(row["num_rounds"]) else 0)],
        "final_state": {
            "all_tests_passed": bool(row["execution_status"] == "Completed"),
            "final_pass_rate": 1.0 if row["execution_status"] == "Completed" else 0.0
        },
        "semantic_evaluation": {
            "available": bool(row["completeness_available"]),
            "semantic_score": float(row["semantic_completeness"]) if pd.notna(row["semantic_completeness"]) else None,
            "drift_detected": False # Will be computed later by NSP model
        },
        "integrity": {
            "validation_status": "valid_metadata_only",
            "validation_errors": []
        }
    }

    # Save to JSON file
    file_path = os.path.join(run_dir, f"{run_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2)

print("Done! All NSP Trace JSON files have been generated in the traces/ directory.")