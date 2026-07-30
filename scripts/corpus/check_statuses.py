import pandas as pd

# Define path
input_csv = r"D:\raz\razieh\results\capability_clash_results_complete_1184.csv"

print("Reading data...")
df = pd.read_csv(input_csv)

print("\n--- Setup Status Values ---")
# This shows how many times each setup status occurred
print(df["setup_status"].value_counts(dropna=False))

print("\n--- Final Status Values ---")
# This shows how many times each final status occurred
print(df["final_status"].value_counts(dropna=False))

print("\n--- Repair Rounds Distribution ---")
# This shows if any rounds were actually recorded
print(df["repair_rounds"].value_counts(dropna=False).sort_index())