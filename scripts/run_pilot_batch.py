import os 
import sys 
import json 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from scripts.runner.NSPRepairRunner import NSPRepairRunner, OllamaProvider 
tasks_list_path = r"D:\raz\razieh\data\nsp_tasks\pilot_tasks_list.json" 
with open(tasks_list_path, 'r') as f: task_ids = json.load(f) 
provider = OllamaProvider(model_name="qwen2.5-coder:3b") 
model_name = "qwen2.5-coder-3b" 
print(f"Starting Batch Run for {len(task_ids)} tasks...") 
for task_id in task_ids: 
    task_dir = os.path.join(r"D:\raz\razieh\data\nsp_tasks", task_id) 
    print(f"\n=====================================") 
    print(f"Starting Task: {task_id}") 
    print(f"=====================================") 
    try: 
        runner = NSPRepairRunner(task_dir, provider, model_name) 
        runner.run() 
    except Exception as e: 
        print(f"[BatchRunner] Error running task {task_id}: {e}") 
