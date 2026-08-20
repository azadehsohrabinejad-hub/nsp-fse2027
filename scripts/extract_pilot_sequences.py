import os
import json

# 1. Define paths
trace_dir = r"D:\raz\razieh\traces\topcoder_pilot"
seq_dir = r"D:\raz\razieh\data\nsp_pilot_sequences"

# Create output directory
os.makedirs(seq_dir, exist_ok=True)

print("Extracting features from real Pilot traces...")
processed_count = 0

# 2. Walk through all generated JSON files
for root, dirs, files in os.walk(trace_dir):
    for file in files:
        if file == "trace.json":
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                trace = json.load(f)
            
            run_id = trace.get("run", {}).get("run_id", "unknown")
            rounds = trace.get("rounds", [])
            
            # Build sequence Y = [y1, y2, ..., yT]
            sequence_Y = []
            for r in rounds:
                # Default vector of 21 zeros
                y_t = [0.0] * 21
                
                # 1. Extract pass_rate (feature 1)
                pass_rate = r.get("execution", {}).get("tests", {}).get("pass_rate", 0.0)
                y_t[0] = pass_rate
                
                # 2. Extract edit_count (feature 6)
                edit_count = r.get("edits", {}).get("statistics", {}).get("applied_edit_count", 0)
                y_t[5] = float(edit_count)
                
                # 3. Extract invalid_patch (feature 18)
                invalid_patch = r.get("observation", {}).get("behavior_features", {}).get("invalid_patch", False)
                y_t[17] = 1.0 if invalid_patch else 0.0
                
                sequence_Y.append(y_t)
            # If no rounds were recorded, create a dummy single-step sequence
            if not sequence_Y:
                sequence_Y = [[0.0] * 21]
                
            # 3. Save the sequence to a JSON file
            output_data = {
                "run_id": run_id,
                "sequence_length": len(sequence_Y),
                "feature_vector_length": 21,
                "Y": sequence_Y,
                "semantic_label": None
            }
            
            out_file_path = os.path.join(seq_dir, f"{run_id}.json")
            with open(out_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            processed_count += 1

print(f"\n--- Extraction Summary ---")
print(f"Total sequences generated: {processed_count}")
print(f"Saved to: {seq_dir}")