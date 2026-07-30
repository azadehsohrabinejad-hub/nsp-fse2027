import pandas as pd
import os

# 1. Define paths
input_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"
output_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0_working.csv"

print("Reading working manifest...")
df = pd.read_csv(input_csv)

# 2. Apply Canonical Policy (Step 1.14)
print("Classifying runs...")

for index, row in df.iterrows():
    setup = str(row["setup_status"]).lower()
    # We renamed it to execution_status in the previous step, so we read that
    exec_status = str(row["execution_status"]).lower() 
    
    # Rule 1: Setup Failures -> Excluded
    if setup == "failed" or exec_status == "setup_failed":
        df.at[index, "canonical_status"] = "Excluded"
        df.at[index, "exclusion_reason"] = "Setup_Failure"
        df.at[index, "execution_status"] = "Failed_Before_Repair"
        df.at[index, "repair_attempted"] = False
        
    # Rule 2: Successful Setup -> Canonical Core
    elif setup == "success":
        df.at[index, "canonical_status"] = "Canonical_Core"
        df.at[index, "exclusion_reason"] = "None"
        df.at[index, "repair_attempted"] = True
        
        # Map original statuses to standard execution statuses
        if exec_status == "passed_initial":
            df.at[index, "execution_status"] = "Completed"
        elif exec_status == "passed_after_repair":
            df.at[index, "execution_status"] = "Completed"
        elif exec_status == "exhausted_repairs":
            df.at[index, "execution_status"] = "Repair_Failed"
        else:
            df.at[index, "execution_status"] = "Unknown"

# 3. Update review status since we reviewed them
df["review_status"] = "Reviewed"

# 4. Save the updated manifest
print(f"Saving updated manifest to {output_csv}...")
df.to_csv(output_csv, index=False)

# 5. Print final counts
print("\n--- Corpus Classification Summary ---")
print("Canonical Statuses:")
print(df["canonical_status"].value_counts())
print("\nExecution Statuses:")
print(df["execution_status"].value_counts())

print("\nDone! Runs classified successfully.")