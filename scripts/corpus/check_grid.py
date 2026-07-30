import pandas as pd

# 1. Define path
input_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"

print("Reading manifest...")
df = pd.read_csv(input_csv)

print("\n--- Grid Analysis ---")
print(f"Total rows: {len(df)}")

# 2. Show counts for models, strategies, and tasks
print("\nModels found:")
print(df["model"].value_counts(dropna=False))

print("\nStrategies found:")
print(df["strategy"].value_counts(dropna=False))

print("\nTasks found:")
print(f"Total unique tasks: {df['task_id'].nunique()}")

# 3. Calculate expected grid size
num_tasks = df["task_id"].nunique()
num_models = df["model"].nunique()
num_strategies = df["strategy"].nunique()

expected_runs = num_tasks * num_models * num_strategies

print("\n--- Grid Math ---")
print(f"Expected grid: {num_tasks} tasks x {num_models} models x {num_strategies} strategies = {expected_runs} runs")
print(f"Actual rows in CSV: {len(df)}")
print(f"Difference (Missing runs): {expected_runs - len(df)}")

# 4. Check if 1184 is just Canonical or Total
canonical_count = len(df[df["canonical_status"] == "Canonical_Core"])
excluded_count = len(df[df["canonical_status"] == "Excluded"])
print(f"\nCanonical Core: {canonical_count}")
print(f"Excluded: {excluded_count}")