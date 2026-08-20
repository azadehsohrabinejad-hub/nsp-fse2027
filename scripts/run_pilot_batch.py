import os
import sys
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.runner.NSPRepairRunner import NSPRepairRunner, OpenAIProvider

def main():
    # 1. Path to our simple Python Smoke Test
    task_dir = r"D:\raz\razieh\data\nsp_tasks\nsp_smoke_test_001"
    
    # 2. Put your actual OpenRouter Key here (without any spaces)
    MY_API_KEY = "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" 
    
    # 3. Setup OpenRouter Provider
    provider = OpenAIProvider(
        model_name="openai/gpt-4o-mini",
        api_key=MY_API_KEY,  # Pass the key directly
        base_url="https://openrouter.ai/api/v1"
    )
    model_name = "gpt-4o-mini"
    
    print(f"Starting Smoke Test with OpenRouter ({model_name})...")
    
    try:
        runner = NSPRepairRunner(task_dir, provider, model_name)
        runner.run()
    except Exception as e:
        print(f"[BatchRunner] Error running task: {e}")

if __name__ == "__main__":
    main()