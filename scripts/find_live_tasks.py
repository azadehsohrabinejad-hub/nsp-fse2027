import json
import os
import subprocess

def check_repos():
    tasks_path = r"D:\raz\razieh\data\nsp_tasks\python_tasks.json"
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"Checking {len(tasks)} Python tasks for live GitHub repositories...")
    live_tasks = []
    
    for i, task in enumerate(tasks):
        repo_url = task.get('github_repo')
        if not repo_url:
            continue
            
        # Use git ls-remote to silently check if repo exists and is accessible
        cmd = ["git", "ls-remote", repo_url]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(f"[{i+1}/{len(tasks)}] ✅ LIVE: {repo_url}")
            live_tasks.append(task)
        else:
            print(f"[{i+1}/{len(tasks)}] ❌ DEAD: {repo_url}")
            
    print(f"\nFound {len(live_tasks)} live Python tasks.")
    
    # Save live tasks
    live_path = r"D:\raz\razieh\data\nsp_tasks\live_python_tasks.json"
    with open(live_path, 'w', encoding='utf-8') as f:
        json.dump(live_tasks, f, indent=4)
        
    print(f"Live tasks saved to: {live_path}")

if __name__ == "__main__":
    check_repos()