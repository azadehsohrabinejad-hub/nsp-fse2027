import os
import sys
import json
import subprocess
import time

# --- Provider Abstraction ---
class LLMProvider:
    def generate(self, prompt: str) -> dict:
        raise NotImplementedError

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> dict:
        # Simulate an LLM response returning structured edits
        print("[MockLLM] Generating response...")
        return {
            "plan": "The add function multiplies its arguments instead of adding them.",
            "edits": [
                {
                    "file": "src/calculator.py",
                    "operation": "replace",
                    "old_string": "return a * b",
                    "new_string": "return a + b"
                }
            ]
        }

# --- NSP Repair Runner ---
class NSPRepairRunner:
    def __init__(self, task_dir, provider):
        self.task_dir = task_dir
        self.workspace_dir = os.path.join(task_dir, "workspace")
        self.provider = provider
        
        with open(os.path.join(task_dir, "task.json"), 'r') as f:
            self.task = json.load(f)
            
        self.run_id = f"{self.task['task_id']}__mock-llm__direct_baseline__seed_001"
        self.trace_dir = os.path.join(r"D:\raz\razieh\traces", "smoke", self.task['task_id'], "mock-llm", "direct_baseline", "seed_001")
        self.artifacts_dir = os.path.join(self.trace_dir, "artifacts")
        
        # Create directory structure
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

    def apply_patch(self, edits):
        for edit in edits:
            file_path = os.path.join(self.workspace_dir, edit["file"].replace('/', os.sep))
            with open(file_path, 'r') as f:
                content = f.read()
            
            content = content.replace(edit["old_string"], edit["new_string"])
            
            with open(file_path, 'w') as f:
                f.write(content)

    def run(self):
        print(f"=== Starting Run: {self.run_id} ===")
        
        # 1. Initial Test (Expected to fail)
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
            
            # 2. Generate Prompt
            prompt = f"Task: {self.task['description']}\nFix the file: {self.task['target_files'][0]}"
            prompt_path = os.path.join(self.artifacts_dir, "prompts", f"round_{i:03d}.txt")
            with open(prompt_path, 'w') as f: f.write(prompt)
            
            # 3. Call LLM Provider
            response = self.provider.generate(prompt)
            response_path = os.path.join(self.artifacts_dir, "responses", f"round_{i:03d}.json")
            with open(response_path, 'w') as f: json.dump(response, f, indent=2)
            
            # 4. Apply Patch
            self.apply_patch(response["edits"])
            patch_path = os.path.join(self.artifacts_dir, "patches", f"round_{i:03d}.diff")
            with open(patch_path, 'w') as f: f.write(str(response["edits"]))
            
            # 5. Run Tests
            passed = self.run_tests(
                os.path.join(self.artifacts_dir, "logs", f"round_{i:03d}_test_stdout.txt"),
                os.path.join(self.artifacts_dir, "logs", f"round_{i:03d}_test_stderr.txt")
            )
            print(f"Round {i} test passed: {passed}")
            
            # 6. Build Round Data
            y_t = [0.0] * 21
            y_t[0] = 1.0 if passed else 0.0 # pass_rate
            
            rounds.append({
                "round_index": i,
                "timing": {"total_seconds": time.time() - round_start_time},
                "prompt": {"rendered_prompt_path": prompt_path},
                "response": {"raw_response_path": response_path, "parse_status": "success"},
                "edits": {"statistics": {"applied_edit_count": len(response["edits"])}},
                "execution": {"tests": {"status": "passed" if passed else "failed", "pass_rate": y_t[0]}},
                "observation": {
                    "test_features": {"pass_rate": y_t[0], "delta_pass_rate": y_t[0]},
                    "behavior_features": {"invalid_patch": False}
                }
            })
            
            if passed:
                print("Tests passed! Stopping early.")
                break
                
        # 7. Build Final Trace JSON
        trace = {
            "schema": {"name": "NSP Trace", "version": "1.0.0"},
            "run": {"run_id": self.run_id, "status": "completed", "actual_rounds": len(rounds)},
            "task": {"task_id": self.task["task_id"]},
            "model": {"model_name": "mock-llm"},
            "strategy": {"strategy_id": "direct_baseline"},
            "initial_state": {"tests": {"status": "failed", "pass_rate": 0.0}},
            "rounds": rounds,
            "final_state": {"all_tests_passed": rounds[-1]["execution"]["tests"]["status"] == "passed", "final_pass_rate": 1.0},
            "integrity": {"validation_status": "valid", "validation_errors": []}
        }
        
        trace_path = os.path.join(self.trace_dir, "trace.json")
        with open(trace_path, 'w') as f: json.dump(trace, f, indent=2)
        print(f"\nTrace successfully saved to: {trace_path}")

if __name__ == "__main__":
    task_dir = r"D:\raz\razieh\data\nsp_tasks\nsp_smoke_test_001"
    provider = MockLLMProvider()
    runner = NSPRepairRunner(task_dir, provider)
    runner.run()