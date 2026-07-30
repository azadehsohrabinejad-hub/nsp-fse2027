import pandas as pd

# 1. Define paths
manifest_path = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"
completeness_csv = r"D:\raz\razieh\results\semantic_audit_results.csv"
output_path = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"

print("Reading manifest and completeness data...")
df_manifest = pd.read_csv(manifest_path)
df_comp = pd.read_csv(completeness_csv)

# 2. Create run_id in completeness dataframe
df_comp["run_id"] = df_comp["task_id"].astype(str) + "__" + df_comp["model"].astype(str) + "__" + df_comp["strategy"].astype(str)

# 3. Select only the columns we need from completeness file
# We will map 'patch_completeness' to 'semantic_completeness'
df_comp_subset = df_comp[["run_id", "patch_completeness"]].copy()
df_comp_subset.rename(columns={"patch_completeness": "semantic_completeness"}, inplace=True)

# 4. Merge them together
print("Merging completeness into manifest...")
df_merged = pd.merge(df_manifest, df_comp_subset, on="run_id", how="left")

# 5. Add metadata about the completeness method
df_merged["completeness_available"] = df_merged["semantic_completeness"].notna()
df_merged["completeness_method"] = "semantic_audit"
df_merged["completeness_version"] = "1.0"

# 6. Save the updated manifest
print(f"Saving updated manifest to {output_path}...")
df_merged.to_csv(output_path, index=False)

# 7. Print summary
print("\n--- Completeness Summary ---")
print(f"Total runs: {len(df_merged)}")
print(f"Runs with completeness data: {df_merged['completeness_available'].sum()}")
print(f"Runs without completeness data: {(~df_merged['completeness_available']).sum()}")