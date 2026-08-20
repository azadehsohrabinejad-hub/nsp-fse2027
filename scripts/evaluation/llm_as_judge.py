import os
import sys
import json
import requests

# Helper function to call OpenRouter
def call_openrouter(prompt, api_key, model="openai/gpt-4o-mini"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict code reviewer. Evaluate the semantic correctness of the patch. Return ONLY a JSON object with 'score' (0.0 to 1.0) and 'reason'."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    else:
        return {"score": 0.0, "reason": f"API Error: {response.status_code}"}

def evaluate_patch(buggy_code, patched_code, task_description, api_key):
    prompt = f"""
    Task Description: {task_description}
    
    Buggy Code:
    ```
    {buggy_code}
    ```
    
    Patched Code:
    ```
    {patched_code}
    ```
    
    Question: Does the patched code semantically fix the bug described in the task without introducing new issues? 
    Rate the semantic correctness from 0.0 (completely wrong/harmful) to 1.0 (perfect fix).
    """
    
    result = call_openrouter(prompt, api_key)
    return result.get("score", 0.0)

if __name__ == "__main__":
    # Example usage: Evaluate our smoke test repair
    API_KEY = "YOUR_OPENROUTER_API_KEY_HERE" # Put your key here
    
    task_desc = "Fix the add function in calculator.py to correctly add two numbers."
    buggy_code = "def add(a, b):\n    return a * b"
    patched_code = "def add(a, b):\n    return a + b"
    
    print("Running LLM-as-a-Judge...")
    score = evaluate_patch(buggy_code, patched_code, task_desc, API_KEY)
    
    print(f"\n=== Judge Result ===")
    print(f"Semantic Score: {score}")
    
    # Save the result
    with open("reports/judge_evaluation.json", 'w') as f:
        json.dump({"score": score}, f, indent=4)
    print("Saved to reports/judge_evaluation.json")