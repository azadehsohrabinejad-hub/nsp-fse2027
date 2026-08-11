import pandas as pd
import re
import json
import os

def prepare_pilot_tasks(input_file, output_dir):
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file) if input_file.endswith('.xlsx') else pd.read_csv(input_file)
    
    pilot_task_ids = []
    
    print("Filtering Python tasks with specific commit hashes...")
    for index, row in df.iterrows():
        tech = str(row.get('technologies', ''))
        desc = str(row.get('description', ''))
        track = str(row.get('trackType', '')).lower()
        
        # 1. Filter for Python Development tasks
        if 'python' in tech.lower() and track == 'development':
            
            # 2. Find Commit Hash (e.g., "commit hash 47be786c" or "commit 47be786c")
            commit_match = re.search(r'(?:commit\s*hash\s*|against\s*commit\s*|commit\s*)([0-9a-f]{7,40})', desc, re.IGNORECASE)
            if not commit_match:
                continue
                
            commit_hash = commit_match.group(1)
            
            # 3. Find GitHub Repo URL
            repo_match = re.search(r'https://github\.com/[\w\-]+/[\w\-]+', desc)
            if not repo_match:
                continue
                
            repo_url = repo_match.group(0)
            
            # 4. Find Branch (if mentioned)
            branch_match = re.search(r'(?:branch\s*`?)([\w\-]+)`?', desc, re.IGNORECASE)
            branch = branch_match.group(1) if branch_match else None
            
            # Create task directory and task.json
            task_id = f"tc_{row.get('id', index)}"
            task_dir = os.path.join(output_dir, task_id)
            os.makedirs(task_dir, exist_ok=True)
            
            task_json = {
                "schema_version": "1.0",
                "task_id": task_id,
                "benchmark": "Topcoder",
                "language": "python",
                "description": desc[:500], # Truncate to 500 chars to avoid LLM context limits
                "github_repo": repo_url,
                "branch": branch,
                "commit_hash": commit_hash,
                "target_files": [], # Empty: LLM has to figure it out
                "setup_commands": ["pip install -r requirements.txt"],
                "test_commands": ["pytest"],
                "max_rounds": 3 # Allow up to 3 rounds to generate T>1 trajectories
            }
            
            with open(os.path.join(task_dir, "task.json"), 'w', encoding='utf-8') as f:
                json.dump(task_json, f, indent=4)
                
            pilot_task_ids.append(task_id)
            
    print(f"\nFound {len(pilot_task_ids)} Python tasks with specific commits.")
    print(f"First 5 tasks: {pilot_task_ids[:5]}")
    
    # Save the list of pilot tasks
    list_path = os.path.join(output_dir, "pilot_tasks_list.json")
    with open(list_path, 'w') as f:
        json.dump(pilot_task_ids, f, indent=4)
        
    print(f"Task list saved to: {list_path}")

if __name__ == "__main__":
    input_filepath = r"D:\raz\razieh\data\Challenges.xlsx"
    if not os.path.exists(input_filepath):
        input_filepath = r"D:\raz\razieh\data\Challenges.csv"
        
    output_directory = r"D:\raz\razieh\data\nsp_tasks"
    prepare_pilot_tasks(input_filepath, output_directory)