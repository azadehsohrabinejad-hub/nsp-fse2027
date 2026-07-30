import pandas as pd
import os

# 1. Define paths
final_csv = r"D:\raz\razieh\data\manifests\corpus_manifest_v1.0.csv"
audit_report = r"D:\raz\razieh\reports\corpus_audit_v1.0.md"

print("Generating audit report...")
df = pd.read_csv(final_csv)

# 2. Calculate stats
total_runs = len(df)
canonical_runs = len(df[df["canonical_status"] == "Canonical_Core"])
excluded_runs = len(df[df["canonical_status"] == "Excluded"])
valid_traces = df["trace_exists"].sum() # Note: we know traces are missing locally, but we record the fact
completeness_available = df["completeness_available"].sum()

exclusion_reasons = df[df["canonical_status"] == "Excluded"]["exclusion_reason"].value_counts().to_dict()

# 3. Write Markdown report
with open(audit_report, 'w', encoding='utf-8') as f:
    f.write("# Canonical Corpus Audit v1.0\n\n")
    f.write("## 1. Purpose\n")
    f.write("This document provides the official audit for the repair-trajectory corpus used in NSP Paper 1.\n\n")
    
    f.write("## 5. Total discovered runs\n")
    f.write(f"**{total_runs}** runs were discovered in the raw results file.\n\n")
    
    f.write("## 6. Canonical runs\n")
    f.write(f"**{canonical_runs}** runs met all criteria and are included in the Canonical Core.\n\n")
    
    f.write("## 7. Excluded runs by reason\n")
    for reason, count in exclusion_reasons.items():
        f.write(f"- {reason}: {count}\n")
    f.write("\n")
    
    f.write("## 12. Patch availability\n")
    f.write(f"Completeness metrics are available for **{completeness_available}** runs.\n\n")

    f.write("## 16. Known limitations\n")
    f.write("- Trace files (JSONL) are currently stored on the remote execution server and are not present in the local Windows workspace. Trace validation was performed based on CSV metadata rather than raw file parsing.\n")

print(f"Audit report saved to: {audit_report}")
print("Done!")