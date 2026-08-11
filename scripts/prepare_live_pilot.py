import json
import os

def prepare_live_tasks():
    live_path = r"D:\raz\razieh\data\nsp_tasks\live_python_tasks.json"
    output_dir = r"D:\raz\razieh\data\nsp_tasks"
    
    with open(live_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"Preparing task.json for first 5 live tasks...")
    task_ids = []
    
    for task in tasks[:5]:
        task_id = task['task_id']
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Use the main branch, no specific commit
        task_json = {
            "schema_version": "1.0",
            "task_id": task_id,
            "benchmark": "Topcoder",
            "language": "python",
            "description": task.get('description_excerpt', 'Fix the issue described in the challenge.'),
            "github_repo": task['github_repo'],
            "branch": None, # CHANGED: Let Git use default branch
            "commit_hash": None,
            "target_files": [],
            "setup_commands": ["pip install -r requirements.txt"],
            "test_commands": ["pytest"],
            "max_rounds": 3
        }
        
        # If main doesn't work, runner will fail, we can try master later
        with open(os.path.join(task_dir, "task.json"), 'w', encoding='utf-8') as f:
            json.dump(task_json, f, indent=4)
            
        task_ids.append(task_id)
        
    # Save the list
    list_path = os.path.join(output_dir, "pilot_tasks_list.json")
    with open(list_path, 'w') as f:
        json.dump(task_ids, f, indent=4)
        
    print(f"Prepared {len(task_ids)} tasks. IDs: {task_ids}")

if __name__ == "__main__":
    prepare_live_tasks()