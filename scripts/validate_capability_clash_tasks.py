#!/usr/bin/env python3
"""Validate generated Capability-Clash tasks by running tests."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = PROJECT_ROOT / "data" / "capability_clash" / "tasks"
MANIFEST_PATH = PROJECT_ROOT / "data" / "capability_clash" / "manifest.jsonl"


def load_candidate(task_dir: Path) -> Dict[str, Any]:
    with (task_dir / "candidate.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_mutations(task_dir: Path) -> List[Dict[str, Any]]:
    with (task_dir / "mutations.json").open("r", encoding="utf-8") as fp:
        payload = json.load(fp) or {}
    return payload.get("mutations") or []


def copy_repo(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".mypy_cache", ".pytest_cache")
    shutil.copytree(src, dest, ignore=ignore, symlinks=True)


def apply_mutations(workspace: Path, mutations: List[Dict[str, Any]]) -> None:
    for mutation in mutations:
        file_path = workspace / mutation["file"]
        text = file_path.read_text(encoding="utf-8")
        search = mutation["search"]
        replace = mutation["replace"]
        if search not in text:
            raise RuntimeError(f"validation: unable to apply mutation in {mutation['file']}")
        text = text.replace(search, replace, 1)
        file_path.write_text(text, encoding="utf-8")


def revert_mutations(workspace: Path, mutations: List[Dict[str, Any]]) -> None:
    for mutation in mutations:
        file_path = workspace / mutation["file"]
        text = file_path.read_text(encoding="utf-8")
        search = mutation["replace"]
        replace = mutation["search"]
        if search not in text:
            raise RuntimeError(f"validation: unable to revert mutation in {mutation['file']}")
        text = text.replace(search, replace, 1)
        file_path.write_text(text, encoding="utf-8")


def run_commands(commands: List[str], cwd: Path, env: Dict[str, str], log_file: Path) -> Tuple[bool, str]:
    combined_output: List[str] = []
    run_env = os.environ.copy()
    run_env.update(env)
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            env=run_env,
            capture_output=True,
            text=True,
        )
        output = f"$ {cmd}\n{proc.stdout}\n{proc.stderr}\n"
        combined_output.append(output)
        if proc.returncode != 0:
            log_file.write_text("".join(combined_output), encoding="utf-8")
            return False, "".join(combined_output)
    log_file.write_text("".join(combined_output), encoding="utf-8")
    return True, "".join(combined_output)


def validation_sequence(
    candidate: Dict[str, Any],
    mutations: List[Dict[str, Any]],
    skip_setup: bool = False,
) -> Dict[str, Any]:
    repo_path = Path(candidate["repo_path"])
    env = candidate.get("env") or {}
    if skip_setup:
        setup: List[str] = []
    else:
        setup = []
        for cmd in candidate.get("setup_commands") or []:
            stripped = cmd.strip()
            if stripped.startswith("pip install -e ."):
                setup.append(cmd.replace("pip install -e .", f"pip install -e {repo_path}", 1))
            else:
                setup.append(cmd)
    tests = candidate.get("test_commands") or []
    commands = setup + tests
    if not commands:
        raise RuntimeError(f"candidate {candidate['task_id']} missing commands")
    results: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="cc_baseline_") as baseline_tmp:
        baseline_path = Path(baseline_tmp)
        copy_repo(repo_path, baseline_path)
        ok, _ = run_commands(commands, baseline_path, env, Path(candidate_dir(candidate) / "validation" / "baseline.log"))
        results["baseline_pass"] = ok

    with tempfile.TemporaryDirectory(prefix="cc_mutated_") as mutated_tmp:
        mutated_path = Path(mutated_tmp)
        copy_repo(repo_path, mutated_path)
        apply_mutations(mutated_path, mutations)
        ok, _ = run_commands(commands, mutated_path, env, Path(candidate_dir(candidate) / "validation" / "mutated.log"))
        results["mutated_pass"] = ok

        partials: List[Dict[str, Any]] = []
        for entry in candidate.get("partial_reverts") or []:
            revert_ids = entry.get("revert_mutations") or []
            expected = entry.get("expected_status", "fail")
            with tempfile.TemporaryDirectory(prefix="cc_partial_") as partial_tmp:
                partial_path = Path(partial_tmp)
                copy_repo(repo_path, partial_path)
                apply_mutations(partial_path, mutations)
                subset = [m for m in mutations if m.get("id") in revert_ids]
                revert_mutations(partial_path, subset)
                logfile = Path(candidate_dir(candidate) / "validation" / f"partial_{entry.get('id')}.log")
                ok, _ = run_commands(commands, partial_path, env, logfile)
                partials.append({"id": entry.get("id"), "observed": "pass" if ok else "fail", "expected": expected})
        results["partials"] = partials
    return results


def candidate_dir(candidate: Dict[str, Any]) -> Path:
    return TASKS_DIR / candidate["task_id"]


def ensure_validation_dirs(task_dir: Path) -> None:
    (task_dir / "validation").mkdir(parents=True, exist_ok=True)


def evaluate_results(results: Dict[str, Any]) -> bool:
    if not results.get("baseline_pass"):
        return False
    if results.get("mutated_pass"):
        return False
    for entry in results.get("partials") or []:
        if entry["observed"] != entry["expected"]:
            return False
    return True


def append_manifest(entries: List[Dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry))
            fp.write("\n")


def validate_tasks(
    limit: int | None = None,
    task_ids: List[str] | None = None,
    skip_setup: bool = False,
) -> None:
    if task_ids:
        task_dirs = []
        for tid in task_ids:
            task_dir = TASKS_DIR / tid
            if not task_dir.is_dir():
                raise RuntimeError(f"task directory missing: {task_dir}")
            task_dirs.append(task_dir)
    else:
        task_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())
    manifest_entries: List[Dict[str, Any]] = []
    for idx, task_dir in enumerate(task_dirs):
        if limit is not None and idx >= limit:
            break
        candidate = load_candidate(task_dir)
        mutations = load_mutations(task_dir)
        ensure_validation_dirs(task_dir)
        results = validation_sequence(candidate, mutations, skip_setup=skip_setup)
        success = evaluate_results(results)
        manifest_entry = {
            "task_id": candidate["task_id"],
            "repo_slug": candidate["repo_slug"],
            "mutation_family": candidate.get("mutation_family"),
            "validated": success,
            "results": results,
            "task_dir": str(task_dir),
        }
        manifest_entries.append(manifest_entry)
        status = "VALID" if success else "INVALID"
        print(f"[validate] {candidate['task_id']}: {status}")
    append_manifest(manifest_entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Capability-Clash tasks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Validate only the specified task_id (repeatable).",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip running setup_commands for each task validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_tasks(args.limit, args.task_ids, skip_setup=args.skip_setup)


if __name__ == "__main__":
    main()
