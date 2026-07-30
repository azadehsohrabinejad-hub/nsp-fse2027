import pandas as pd

# 1. Define paths
completeness_csv = r"D:\raz\razieh\results\semantic_audit_results.csv"

print("Reading semantic audit results...")
df_comp = pd.read_csv(completeness_csv)

# 2. Show columns
print("\n--- Columns in semantic_audit_results.csv ---")
print(list(df_comp.columns))

# 3. Show first 3 rows
print("\n--- First 3 rows ---")
print(df_comp.head(3).to_string())