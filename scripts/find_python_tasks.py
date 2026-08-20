import json
import os

# Load the qualified tasks
tasks_path = r"D:\raz\razieh\data\nsp_tasks\qualified_tasks.json"
with open(tasks_path, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

python_tasks = []

for task in tasks:
    # Check if Python is in the technologies list
    if 'Python' in task.get('technologies', '') or 'python' in task.get('description_excerpt', '').lower():
        python_tasks.append(task)

print(f"Found {len(python_tasks)} Python tasks.")
print("\n--- Top 5 Python Tasks ---")
for i, task in enumerate(python_tasks[:5]):
    print(f"\n[{i+1}] ID: {task['task_id']}")
    print(f"Name: {task['name']}")
    print(f"Repo: {task['github_repo']}")
    print(f"Tech: {task['technologies']}")
    print(f"Desc: {task['description_excerpt'][:150]}...")

# Save the filtered list
py_tasks_path = r"D:\raz\razieh\data\nsp_tasks\python_tasks.json"
with open(py_tasks_path, 'w', encoding='utf-8') as f:
    json.dump(python_tasks, f, indent=4)
    
print(f"\nSaved to: {py_tasks_path}")