import os
import json

# 1. Define paths
trace_dir = r"D:\raz\razieh\traces"
seq_dir = r"D:\raz\razieh\data\nsp_sequences"

# Create output directory if it doesn't exist
os.makedirs(seq_dir, exist_ok=True)

print("Starting feature extraction...")

# 2. Define the 21 features based on Section 29 of your standard
def extract_features(round_data, final_pass_rate):
    obs = round_data.get("observation", {})
    test_feats = obs.get("test_features", {})
    edit_feats = obs.get("edit_features", {})
    context_feats = obs.get("context_features", {})
    usage_feats = obs.get("usage_features", {})
    behavior_feats = obs.get("behavior_features", {})

    # Extracting values, defaulting to 0.0 if missing
    y_t = [
        test_feats.get("pass_rate", final_pass_rate),  # 1. pass_rate
        test_feats.get("delta_pass_rate", 0.0),        # 2. delta_pass_rate
        test_feats.get("failing_test_count", 0),       # 3. failing_test_count
        test_feats.get("new_failure_count", 0),        # 4. new_failure_count
        test_feats.get("resolved_failure_count", 0),   # 5. resolved_failure_count
        edit_feats.get("edit_count", 0),               # 6. edit_count
        edit_feats.get("files_touched", 0),            # 7. files_touched
        edit_feats.get("lines_added", 0),              # 8. lines_added
        edit_feats.get("lines_removed", 0),            # 9. lines_removed
        edit_feats.get("repeated_file_edit_ratio", 0.0), # 10. repeated_file_edit_ratio
        edit_feats.get("reverted_edit_ratio", 0.0),    # 11. reverted_edit_ratio
        context_feats.get("retrieved_file_count", 0),  # 12. retrieved_file_count
        context_feats.get("context_tokens", 0),        # 13. context_tokens
        int(context_feats.get("context_truncated", False) == True), # 14. context_truncated
        usage_feats.get("input_tokens", 0),            # 15. input_tokens
        usage_feats.get("output_tokens", 0),           # 16. output_tokens
        usage_feats.get("latency_seconds", 0.0),       # 17. latency_seconds
        int(behavior_feats.get("invalid_patch", False) == True),    # 18. invalid_patch
        int(behavior_feats.get("build_broken", False) == True),     # 19. build_broken
        int(behavior_feats.get("test_regression", False) == True),  # 20. test_regression
        int(behavior_feats.get("scope_expansion", False) == True)   # 21. scope_expansion
    ]
    return y_t

# 3. Process all trace files
processed_count = 0
for root, dirs, files in os.walk(trace_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                trace = json.load(f)
            
            run_id = trace.get("run", {}).get("run_id", file.replace(".json", ""))
            rounds = trace.get("rounds", [])
            final_pass_rate = trace.get("final_state", {}).get("final_pass_rate", 0.0)
            semantic_score = trace.get("semantic_evaluation", {}).get("semantic_score")
            
            # Build sequence Y = [y1, y2, ..., yT]
            sequence_Y = []
            for r in rounds:
                y_t = extract_features(r, final_pass_rate)
                sequence_Y.append(y_t)
            
            # If no rounds were recorded, create a dummy single-step sequence
            if not sequence_Y:
                sequence_Y = [[final_pass_rate] + [0]*20]

            # 4. Save the sequence to a JSON file
            output_data = {
                "run_id": run_id,
                "sequence_length": len(sequence_Y),
                "feature_vector_length": len(sequence_Y[0]),
                "Y": sequence_Y,
                "semantic_label": semantic_score
            }
            
            out_file_path = os.path.join(seq_dir, f"{run_id}.json")
            with open(out_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            processed_count += 1

print(f"\n--- Extraction Summary ---")
print(f"Total sequences generated: {processed_count}")
print(f"Saved to: {seq_dir}")
print("Done! Data is now ready for the NSP Model (Algorithm 2 & 3).")