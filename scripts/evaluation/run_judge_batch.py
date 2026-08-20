import os
import sys
import json
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- LLM Judge Configuration ---
OPENROUTER_API_KEY = "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"
JUDGE_MODEL = "openai/gpt-4o-mini"

def call_judge(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict code reviewer. Evaluate the semantic correctness of the patch. Return ONLY a JSON object with 'score' (0.0 to 1.0) and 'reason'."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        else:
            return {"score": 0.0, "reason": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"score": 0.0, "reason": f"Exception: {str(e)}"}

def evaluate_trace(trace_path, task_dir):
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    
    run_id = trace.get("run", {}).get("run_id", "unknown")
    task_id = trace.get("task", {}).get("task_id", "unknown")
    
    # Read task description
    task_json_path = os.path.join(task_dir, "task.json")
    task_desc = "No description available."
    if os.path.exists(task_json_path):
        with open(task_json_path, 'r') as f:
            task_desc = json.load(f).get("description", task_desc)
    
    # Read buggy code (from the first prompt)
    prompt_path = os.path.join(os.path.dirname(trace_path), "artifacts", "prompts", "round_001.txt")
    buggy_code = "Buggy code not found in prompt."
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            content = f.read()
            # Extract code block if exists
            if "```" in content:
                buggy_code = content.split("```")[1]
                if buggy_code.startswith("python\n"):
                    buggy_code = buggy_code[7:]
            else:
                buggy_code = content
    
    # Read patched code (from the first response)
    response_path = os.path.join(os.path.dirname(trace_path), "artifacts", "responses", "round_001.json")
    patched_code = "No patch applied."
    if os.path.exists(response_path):
        with open(response_path, 'r') as f:
            resp_data = json.load(f)
            edits = resp_data.get("edits", [])
            if edits:
                # Reconstruct patched code by applying the first edit
                old_str = edits[0].get("old_string", edits[0].get("line", ""))
                new_str = edits[0].get("new_string", edits[0].get("with", ""))
                patched_code = buggy_code.replace(old_str, new_str)
    
    # Construct Judge Prompt
    judge_prompt = f"""
    Task Description: {task_desc}
    
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
    
    result = call_judge(judge_prompt)
    
    return {
        "run_id": run_id,
        "task_id": task_id,
        "judge_score": result.get("score", 0.0),
        "judge_reason": result.get("reason", "N/A")
    }

def main():
    traces_root = r"D:\raz\razieh\traces\topcoder_pilot"
    tasks_root = r"D:\raz\razieh\data\nsp_tasks"
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("ERROR: Please put your OpenRouter API Key in the script!")
        return
        
    print("=== Running LLM-as-a-Judge on all traces ===")
    
    all_results = []
    
    # Find all trace.json files
    for root, dirs, files in os.walk(traces_root):
        for file in files:
            if file == "trace.json":
                trace_path = os.path.join(root, file)
                
                # Extract task_id from path to find task_dir
                # Path format: ...\topcoder_pilot\tc_XXXX\model\...\trace.json
                parts = root.replace(traces_root, "").split(os.sep)
                task_id = parts[1] if len(parts) > 1 else ""
                task_dir = os.path.join(tasks_root, task_id)
                
                print(f"Evaluating: {task_id}...")
                res = evaluate_trace(trace_path, task_dir)
                all_results.append(res)
                
    # Save results
    output_path = r"D:\raz\razieh\reports\llm_judge_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nDone! Evaluated {len(all_results)} traces. Results saved to {output_path}")

if __name__ == "__main__":
    main()