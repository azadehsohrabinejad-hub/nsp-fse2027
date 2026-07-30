import pandas as pd
import os

# 1. Define paths
input_csv = r"D:\raz\razieh\results\capability_clash_results_complete_1184.csv"
output_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"

print("Reading the main results file...")
df = pd.read_csv(input_csv)

# 2. Create run_id if not exists
df["run_id"] = df["task_id"].astype(str) + "__" + df["model"].astype(str) + "__" + df["strategy"].astype(str)

# 3. Map existing columns to the new standard schema
# We rename your columns to match the required schema names
rename_mapping = {
    "repair_rounds": "num_rounds",
    "round_trace_path": "trace_path",
    "final_status": "execution_status",
    "initial_pass": "initial_pass_count",
    "final_pass": "final_pass_count"
}
df.rename(columns=rename_mapping, inplace=True)

# 4. Add default corpus membership columns (Step 1.7)
df["canonical_status"] = "Pending_Review"
df["exclusion_reason"] = ""
df["review_status"] = "Not_Reviewed"
df["manifest_schema_version"] = "1.0"

# 5. Check if trace file actually exists (Step 1.8 - Trace validation preparation)
print("Checking if trace files exist...")
BASE_DIR = r"D:\raz\razieh"

def check_trace_exists(rel_path):
    if pd.isna(rel_path) or rel_path == "":
        return False
    # Combine base directory with the relative path from CSV
    abs_path = os.path.join(BASE_DIR, str(rel_path))
    return os.path.exists(abs_path)

df["trace_exists"] = df["trace_path"].apply(check_trace_exists)

# 6. Select and order the final columns for the working manifest
final_columns = [
    "run_id", "task_id", "model", "strategy", 
    "canonical_status", "exclusion_reason", "review_status",
    "setup_status", "execution_status", "num_rounds", 
    "trace_path", "trace_exists", 
    "initial_pass_count", "final_pass_count",
    "manifest_schema_version"
]

# Keep only the columns we need (ignore the rest for now)
df_working = df[final_columns]

# 7. Save the working manifest
print(f"Saving working manifest to {output_csv}...")
df_working.to_csv(output_csv, index=False)

print("\n--- Summary ---")
print(f"Total rows: {len(df_working)}")
print(f"Traces found: {df_working['trace_exists'].sum()}")
print(f"Traces missing: {(~df_working['trace_exists']).sum()}")
print("Done! Step 1.6, 1.7, and 1.8 are complete.")