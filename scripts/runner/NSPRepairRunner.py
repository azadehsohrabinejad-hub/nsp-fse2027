import os
import sys
import json
import subprocess
import time
import requests
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Provider Abstraction ---
class LLMProvider:
    def generate(self, prompt: str) -> dict:
        raise NotImplementedError

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> dict:
        print("[MockLLM] Generating response...")
        return {
            "plan": "Fix the markdown rendering issue.",
            "edits": [
                {
                    "file": "src/calculator.py",
                    "operation": "replace",
                    "old_string": "return a * b",
                    "new_string": "return a + b"
                }
            ]
        }

class OllamaProvider(LLMProvider):
    def __init__(self, model_name="qwen2.5-coder:3b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> dict:
        print(f"[OllamaProvider] Sending prompt to model...")
        system_prompt = (
            "You are a code repair agent. Your output MUST be ONLY a valid JSON object.\n"
            "Do not output markdown or explanations outside the JSON.\n"
            "The JSON MUST follow this exact schema:\n"
            "{\n"
            "  \"plan\": \"Brief explanation\",\n"
            "  \"edits\": [\n"
            "    {\n"
            "      \"file\": \"relative/path/to/file.py\",\n"
            "      \"old_string\": \"the exact existing code block to replace\",\n"
            "      \"new_string\": \"the new code block\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Example:\n"
            "{\n"
            "  \"plan\": \"Change multiplication to addition\",\n"
            "  \"edits\": [\n"
            "    {\n"
            "      \"file\": \"src/calculator.py\",\n"
            "      \"old_string\": \"return a * b\",\n"
            "      \"new_string\": \"return a + b\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\n{prompt}\n\nReturn ONLY the JSON object:",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1} # Lower temperature for strict format
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            print(f"[OllamaProvider] Error: {e}")
            return {"plan": "Error", "edits": []}

# --- Repository Manager ---
class RepositoryManager:
    @staticmethod
    def clone_and_checkout(task_dir, repo_url, branch=None, commit_hash=None):
        workspace_dir = os.path.join(task_dir, "workspace")
        
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
            
        print(f"[RepoManager] Cloning {repo_url}...")
        clone_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([repo_url, workspace_dir])
        
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[RepoManager] Clone failed: {result.stderr}")
            return False
            
        if commit_hash:
            print(f"[RepoManager] Fetching commit {commit_hash}...")
            fetch_cmd = ["git", "fetch", "--depth", "1", "origin", commit_hash]
            subprocess.run(fetch_cmd, cwd=workspace_dir, capture_output=True, text=True)
            
            checkout_cmd = ["git", "checkout", commit_hash]
            res = subprocess.run(checkout_cmd, cwd=workspace_dir, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[RepoManager] Checkout failed: {res.stderr}")
                return False
                
        print("[RepoManager] Repository ready.")
        return True

# --- NSP Repair Runner ---
class NSPRepairRunner:
    def __init__(self, task_dir, provider, model_name="mock-llm"):
        self.task_dir = task_dir
        self.workspace_dir = os.path.join(task_dir, "workspace")
        self.provider = provider
        self.model_name = model_name
        
        with open(os.path.join(task_dir, "task.json"), 'r') as f:
            self.task = json.load(f)
            
        self.run_id = f"{self.task['task_id']}__{model_name}__direct_baseline__seed_001"
        self.trace_dir = os.path.join(r"D:\raz\razieh\traces", "topcoder_pilot", self.task['task_id'], model_name, "direct_baseline", "seed_001")
        self.artifacts_dir = os.path.join(self.trace_dir, "artifacts")
        
        os.makedirs(os.path.join(self.artifacts_dir, "prompts"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "responses"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "patches"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "logs"), exist_ok=True)

    def run_tests(self, stdout_path, stderr_path):
        cmd = self.task["test_commands"][0].split()
        result = subprocess.run(cmd, cwd=self.workspace_dir, capture_output=True, text=True)
        
        with open(stdout_path, 'w') as f: f.write(result.stdout)
        with open(stderr_path, 'w') as f: f.write(result.stderr)
            
        return result.returncode == 0

    def apply_patch(self, edits, target_files=None):
        applied_count = 0
        if not edits:
            return 0
            
        for edit in edits:
            # Security Check: Skip if not a dictionary
            if not isinstance(edit, dict):
                continue
                
            # 1. Find the file path
            file_str = edit.get("file") or edit.get("path") or edit.get("filename") or edit.get("file_path")
            if not file_str and target_files:
                file_str = target_files[0]
            elif not file_str:
                continue
                
            file_path = os.path.join(self.workspace_dir, str(file_str).replace('/', os.sep))
            
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 2. Find old_string and new_string
            old_str = edit.get("old_string") or edit.get("line") or edit.get("find") or edit.get("old") or edit.get("original_line") or edit.get("original")
            new_str = edit.get("new_string") or edit.get("with") or edit.get("replacement") or edit.get("new") or edit.get("new_line") or edit.get("replacement_line")
            
            # Type Check: Convert to string if they are integers or other types
            if old_str is None or new_str is None:
                continue
                
            old_str = str(old_str).strip().rstrip(';')
            new_str = str(new_str).strip().rstrip(';')
            
            # 4. Apply the replacement
            if old_str in content:
                content = content.replace(old_str, new_str)
                with open(file_path, 'w') as f:
                    f.write(content)
                applied_count += 1
                print(f"[PatchEngine] Successfully applied edit to {file_str}")
            else:
                print(f"[PatchEngine] Failed to apply edit to {file_str} (old_string not found).")
                
        return applied_count
        
    def run(self):
        print(f"=== Starting Run: {self.run_id} ===")
        
        # 1. Setup Repository
        if self.task.get("github_repo"):
            if not RepositoryManager.clone_and_checkout(
                self.task_dir, 
                self.task.get("github_repo"), 
                self.task.get("branch"), 
                self.task.get("commit_hash")
            ):
                print("Failed to setup repository. Aborting task.")
                return
        else:
            print("[Runner] No github_repo specified. Using local workspace files.")
            
        # 2. Run Setup Commands
        for cmd in self.task.get("setup_commands", []):
            print(f"Running setup: {cmd}")
            subprocess.run(cmd.split(), cwd=self.workspace_dir)
            
        # 3. Initial Test
        print("Running initial tests...")
        initial_pass = self.run_tests(
            os.path.join(self.artifacts_dir, "logs", "initial_test_stdout.txt"),
            os.path.join(self.artifacts_dir, "logs", "initial_test_stderr.txt")
        )
        print(f"Initial test passed: {initial_pass}")
        
        rounds = []
        for i in range(1, self.task["max_rounds"] + 1):
            print(f"\n--- Round {i} ---")
            round_start_time = time.time()
            
            target_file = self.task['target_files'][0] if self.task.get('target_files') else ""
            buggy_code = ""
            if target_file:
                with open(os.path.join(self.workspace_dir, target_file), 'r') as f:
                    buggy_code = f.read()
                
            prompt = f"Task: {self.task['description']}\nFix the file: {target_file}\nCurrent buggy code:\n```\n{buggy_code}\n```"
            prompt_path = os.path.join(self.artifacts_dir, "prompts", f"round_{i:03d}.txt")
            with open(prompt_path, 'w') as f: f.write(prompt)
            
            response = self.provider.generate(prompt)
            response_path = os.path.join(self.artifacts_dir, "responses", f"round_{i:03d}.json")
            with open(response_path, 'w') as f: json.dump(response, f, indent=2)
            
            # Apply patch with target_files fallback
            applied_count = self.apply_patch(response.get("edits", []), self.task.get("target_files"))
            
            passed = self.run_tests(
                os.path.join(self.artifacts_dir, "logs", f"round_{i:03d}_test_stdout.txt"),
                os.path.join(self.artifacts_dir, "logs", f"round_{i:03d}_test_stderr.txt")
            )
            print(f"Round {i} test passed: {passed}")
            
            y_t = [0.0] * 21
            y_t[0] = 1.0 if passed else 0.0
            
            rounds.append({
                "round_index": i,
                "timing": {"total_seconds": time.time() - round_start_time},
                "prompt": {"rendered_prompt_path": prompt_path},
                "response": {"raw_response_path": response_path, "parse_status": "success" if applied_count > 0 else "failed"},
                "edits": {"statistics": {"applied_edit_count": applied_count}},
                "execution": {"tests": {"status": "passed" if passed else "failed", "pass_rate": y_t[0]}},
                "observation": {
                    "test_features": {"pass_rate": y_t[0], "delta_pass_rate": y_t[0]},
                    "behavior_features": {"invalid_patch": applied_count == 0}
                }
            })
            
            if passed:
                print("Tests passed! Stopping early.")
                break
                
        trace = {
            "schema": {"name": "NSP Trace", "version": "1.0.0"},
            "run": {"run_id": self.run_id, "status": "completed", "actual_rounds": len(rounds)},
            "task": {"task_id": self.task["task_id"]},
            "model": {"model_name": self.model_name},
            "strategy": {"strategy_id": "direct_baseline"},
            "initial_state": {"tests": {"status": "failed", "pass_rate": 0.0}},
            "rounds": rounds,
            "final_state": {"all_tests_passed": rounds[-1]["execution"]["tests"]["status"] == "passed", "final_pass_rate": 1.0 if passed else 0.0},
            "integrity": {"validation_status": "valid", "validation_errors": []}
        }
        
        trace_path = os.path.join(self.trace_dir, "trace.json")
        with open(trace_path, 'w') as f: json.dump(trace, f, indent=2)
        print(f"\nTrace successfully saved to: {trace_path}")

if __name__ == "__main__":
    # Using the local smoke test that doesn't need npm
    task_dir = r"D:\raz\razieh\data\nsp_tasks\nsp_smoke_test_001"
    
    # Using OllamaProvider
    provider = OllamaProvider(model_name="qwen2.5-coder:3b")
    model_name = "qwen2.5-coder-3b"
    
    runner = NSPRepairRunner(task_dir, provider, model_name)
    runner.run()