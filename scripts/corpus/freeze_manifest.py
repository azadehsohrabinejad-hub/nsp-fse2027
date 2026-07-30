import pandas as pd
import json
import os

# 1. Define paths
working_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"
final_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0.csv"
summary_json = r"D:\raz\razieh\reports\corpus_summary_v1.0.json"

print("Reading working manifest...")
df = pd.read_csv(working_csv)

# 2. Sanity Checks (Step 1.21)
print("Running sanity checks...")
assert df["run_id"].is_unique, "Error: run_id is not unique!"
assert not ((df["canonical_status"] == "Excluded") & (df["exclusion_reason"] == "None")).any(), "Error: Excluded runs must have a reason!"
assert not ((df["canonical_status"] == "Canonical_Core") & (df["setup_status"] == "failed")).any(), "Error: Canonical runs cannot have setup failures!"

valid_completeness = df["semantic_completeness"].dropna()
assert valid_completeness.between(0, 1).all(), "Error: Completeness must be between 0 and 1!"
print("All sanity checks passed successfully!")

# 3. Sort and Save Final Manifest (Step 1.22)
print("\nSorting and saving final manifest...")
df = df.sort_values(["task_id", "model", "strategy", "run_id"])
df.to_csv(final_csv, index=False)

# 4. Generate Machine-Readable Summary (Step 1.24)
print("Generating summary JSON...")
summary = {
    "manifest_version": "1.0",
    "manifest_schema_version": "1.0",
    "total_discovered_runs": len(df),
    "canonical_runs": int(len(df[df["canonical_status"] == "Canonical_Core"])),
    "excluded_runs": int(len(df[df["canonical_status"] == "Excluded"])),
    "ablation_runs": 0,
    "completeness_available": int(df["completeness_available"].sum()),
    "models": {k: int(v) for k, v in df["model"].value_counts().items()},
    "strategies": {k: int(v) for k, v in df["strategy"].value_counts().items()}
}

with open(summary_json, 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n--- Final Summary ---")
print(f"Canonical Corpus successfully frozen at: {final_csv}")
print(f"Total Canonical Runs: {summary['canonical_runs']}")
print(f"Total Excluded Runs: {summary['excluded_runs']}")
print("Done! Step 1 is complete.")