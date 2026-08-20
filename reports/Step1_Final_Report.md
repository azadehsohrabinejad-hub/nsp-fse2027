Executive Report — End of Step 1: Building and Freezing the Canonical Corpus
1. Objective
The goal of this step was to clean, standardize, and freeze the model execution dataset (Repair Trajectories) for use in the NSP paper. Initially, the data existed as a raw, unstructured file with 1,184 rows, requiring separation of environment-related errors from actual model execution runs.

2. Statistical Summary of Results
At the end of this step, the dataset was frozen with the following status:

Metric	Value
Total Discovered Runs	1,184 runs
Canonical Core (valid runs)	1,054 runs — runs where the model made at least one repair attempt
Excluded Runs	130 runs — due to environment/setup errors occurring before the model started
Completeness Metrics Available	480 runs from the Canonical set have semantic audit results
3. Key Actions Performed (Workflow)
a) Infrastructure Setup and Backup
Installed Git on Windows OS.

Created standard project folder structure (data/manifests, scripts/corpus, reports, etc.).

Performed initial data backup and initialized Git repository with the main branch.

b) Data Discovery and Raw Manifest Creation
Identified the main results file (capability_clash_results_complete_1184.csv).

Extracted key metadata (e.g., task_id, model, strategy, setup_status).

Generated a unique standardized identifier (run_id) for each run in the format: task_id__model__strategy.

Verified and confirmed no duplicate data existed among the 1,184 rows.

c) Classification and Canonical Corpus Extraction
Precisely distinguished Setup Failures from Repair Failures based on setup_status and final_status values.

Applied the Canonical policy: any run where the environment was successfully set up (even if the model failed to fix the code) was accepted as part of the Canonical Core.

d) Resolving the Grid Puzzle (1,200 vs. 1,184)
In the initial roadmap, 1,200 runs were expected (30 tasks × 5 models × 8 strategies). Statistical analysis revealed:

Actual number of tasks: 37

Number of models evaluated: 4 (gpt-4.1-mini, gpt-4.1-nano, claude-haiku, claude-opus)

Number of strategies: 8

Result: 37 × 4 × 8 = 1,184. Therefore, the grid is fully complete with no missing runs.

e) Integrating Completeness Metrics
Merged semantic_audit_results.csv with the main manifest.

Extracted and recorded patch_completeness as the semantic_completeness metric for runs that had undergone semantic evaluation.

f) Validation, Freezing, and Git Commit
Performed numerical sanity checks (ID uniqueness, completeness value ranges, and status consistency).

Generated the final corpus_manifest_v1.0.csv file and set it to read-only mode.

Produced an audit text report and a checksum file.

Finalized changes in Git with an official commit and tag v1.0-frozen-corpus.

4. Documented Limitations and Notes
Trace Files: The raw execution trace files (JSON/JSONL format) were not available in the local Windows environment (likely remaining on the Linux execution server). Therefore, trace validation in this step was performed based on the metadata recorded in the CSV file, rather than parsing physical trace files.

5. Final Outputs and Files Generated
The following files have been committed to the project repository:

data/manifests/corpus_manifest_v1.0.csv — Official frozen manifest

reports/corpus_audit_v1.0.md — Audit report

reports/corpus_summary_v1.0.json — Statistical summary for system use

reports/corpus_v1_checksums.sha256 — SHA256 checksums for integrity verification

scripts/corpus/*.py — Reproducible scripts for this step

6. Conclusion
Step 1 has been successfully completed. The data has been transformed from a raw state into a standardized, validated, and versioned dataset. The project is now ready to proceed to Step 2.

