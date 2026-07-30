import pandas as pd
import os

# 1. Define paths
input_csv = r"D:\raz\razieh\results\capability_clash_results_complete_1184.csv"
output_csv = r"D:\raz\razieh\data\manifests\all_discovered_runs_raw.csv"
duplicates_csv = r"D:\raz\razieh\data\validation\duplicate_run_ids.csv"

print("Reading the main results file...")
df = pd.read_csv(input_csv)

# 2. Create standard run_id (Step 1.5)
# Format: task_id__model__strategy
print("Generating unique run_id...")
df["run_id"] = df["task_id"].astype(str) + "__" + df["model"].astype(str) + "__" + df["strategy"].astype(str)

# 3. Add required columns for raw manifest
df["source_path"] = input_csv
df["discovered_run_id"] = df["run_id"]
df["discovery_status"] = "Discovered"

# 4. Select and reorder columns to match Step 1.4 requirements
required_columns = [
    "source_path",
    "run_id",
    "task_id",
    "model",
    "strategy",
    "setup_status",
    "final_status",
    "repair_rounds",
    "round_trace_path",
    "setup_summary_path",
    "discovered_run_id",
    "discovery_status"
]

# Keep only columns that exist in the dataframe to avoid errors
existing_columns = [col for col in required_columns if col in df.columns]
df_raw = df[existing_columns]

# 5. Save the raw manifest
print(f"Saving raw manifest to {output_csv}...")
df_raw.to_csv(output_csv, index=False)

# 6. Check for duplicates (Step 1.5)
print("Checking for duplicates...")
duplicates = df_raw[df_raw["run_id"].duplicated(keep=False)].sort_values("run_id")

if not duplicates.empty:
    duplicates.to_csv(duplicates_csv, index=False)
    print(f"WARNING: Found {len(duplicates)} duplicate rows! Saved to {duplicates_csv}")
else:
    print("No duplicates found. All run_ids are unique!")

print(f"\nTotal discovered records: {len(df_raw)}")
print("Done! Step 1.4 and 1.5 are complete.")