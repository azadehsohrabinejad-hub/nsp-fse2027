import os
import json
import csv

# 1. Define paths
trace_dir = r"D:\raz\razieh\traces"
report_path = r"D:\raz\razieh\data\validation\trace_validation_report.csv"

print("Starting NSP Trace Validation...")
validation_results = []

# 2. Check if directory exists
if not os.path.exists(trace_dir):
    print(f"Error: Trace directory not found at {trace_dir}")
    exit()

# 3. Walk through all generated JSON files
for root, dirs, files in os.walk(trace_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    trace = json.load(f)
                except Exception as e:
                    validation_results.append({
                        "file": file, "valid": False, "error": f"JSON Parse Error: {str(e)}"
                    })
                    continue

            errors = []
            
            # Rule 1: run_id exists
            run_id = trace.get("run", {}).get("run_id")
            if not run_id:
                errors.append("Missing run_id")

            # Rule 2 & 3: Rounds start at 1 and no duplicates
            rounds = trace.get("rounds", [])
            if rounds:
                indices = [r.get("round_index") for r in rounds]
                if indices[0] != 1:
                    errors.append("Rounds do not start at 1")
                if len(indices) != len(set(indices)):
                    errors.append("Duplicate round_index found")

            # Rule 5: Pass rate between 0 and 1 (if available)
            final_pass_rate = trace.get("final_state", {}).get("final_pass_rate")
            if final_pass_rate is not None:
                if not (0.0 <= final_pass_rate <= 1.0):
                    errors.append(f"Invalid pass_rate: {final_pass_rate}")

            # Check mandatory top-level fields
            required_top = ["schema", "run", "task", "model", "strategy", "rounds", "final_state"]
            for field in required_top:
                if field not in trace:
                    errors.append(f"Missing top-level field: {field}")

            # Append result
            validation_results.append({
                "file": file,
                "run_id": run_id,
                "valid": len(errors) == 0,
                "errors": "; ".join(errors) if errors else "None"
            })

# 4. Save report to CSV
print(f"Validated {len(validation_results)} trace files.")
with open(report_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["file", "run_id", "valid", "errors"])
    writer.writeheader()
    writer.writerows(validation_results)

# 5. Print Summary
valid_count = sum(1 for r in validation_results if r["valid"])
invalid_count = len(validation_results) - valid_count

print("\n--- Validation Summary ---")
print(f"Total Traces Checked: {len(validation_results)}")
print(f"Valid Traces: {valid_count}")
print(f"Invalid Traces: {invalid_count}")
print(f"Detailed report saved to: {report_path}")