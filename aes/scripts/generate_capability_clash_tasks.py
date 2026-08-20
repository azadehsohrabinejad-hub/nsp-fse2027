#!/usr/bin/env python3
"""Generate Capability-Clash task workspaces from mined candidates."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import difflib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "capability_clash" / "candidates.jsonl"
TASKS_DIR = PROJECT_ROOT / "data" / "capability_clash" / "tasks"


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _force_rmtree(path: Path) -> None:
    """Remove a directory tree robustly, falling back to os-level rm on macOS ENOTEMPTY."""
    import subprocess
    try:
        shutil.rmtree(path)
    except OSError:
        subprocess.run(["rm", "-rf", str(path)], check=True)


def copy_repo(src: Path, dest: Path) -> None:
    if dest.exists():
        _force_rmtree(dest)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".mypy_cache", ".pytest_cache")
    shutil.copytree(src, dest, ignore=ignore, symlinks=True)


def apply_mutations(workspace: Path, mutations: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Apply string replacements and return per-file original/mutated text."""
    originals: Dict[str, str] = {}
    for mutation in mutations:
        file_rel = mutation["file"]
        search = mutation["search"]
        replace = mutation["replace"]
        file_path = workspace / file_rel
        data = file_path.read_text(encoding="utf-8")
        if file_rel not in originals:
            originals[file_rel] = data
        if search not in data:
            raise RuntimeError(f"Search string missing while applying mutation in {file_rel}")
        data = data.replace(search, replace, 1)
        file_path.write_text(data, encoding="utf-8")
    return originals


def write_patch(task_dir: Path, originals: Dict[str, str], workspace: Path) -> None:
    patch_lines: List[str] = []
    for file_rel, original_text in originals.items():
        mutated_text = (workspace / file_rel).read_text(encoding="utf-8")
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            mutated_text.splitlines(keepends=True),
            fromfile=f"a/{file_rel}",
            tofile=f"b/{file_rel}",
        )
        patch_lines.extend(diff)
    (task_dir / "ground_truth.patch").write_text("".join(patch_lines), encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_task_json(candidate: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(candidate.get("metadata") or {})
    metadata.update(
        {
            "mutation_family": candidate.get("mutation_family"),
            "repo_key": candidate.get("repo_key"),
            "partial_reverts": candidate.get("partial_reverts") or [],
            "trap_files": candidate.get("trap_files") or [],
            "repo_slug": candidate.get("repo_slug"),
        }
    )
    prompt = candidate.get("prompt") or candidate.get("description") or ""
    return {
        "task_id": candidate.get("task_id"),
        "prompt": prompt,
        "repo_path": "workspace",
        "dataset": "capability_clash",
        "dataset_source": candidate.get("repo_slug"),
        "task_type": "bugfix",
        "difficulty": "H",
        "language": candidate.get("language", "python"),
        "runtime_family": candidate.get("runtime_family", candidate.get("language", "python")),
        "test_commands": candidate.get("test_commands"),
        "setup_commands": candidate.get("setup_commands"),
        "timeout_seconds": candidate.get("timeout_seconds") or 420,
        "env": candidate.get("env") or {},
        "target_files": candidate.get("target_files") or [],
        "file_context": candidate.get("file_context") or [],
        "allowed_edit_paths": candidate.get("target_files") or [],
        "metadata": metadata,
        "reportable": True,
        "task_is_real_world": True,
        "allow_file_creation": False,
    }


def generate_tasks(candidates_path: Path, limit: int | None = None) -> None:
    candidates = load_candidates(candidates_path)
    if limit is not None:
        candidates = candidates[:limit]
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        task_id = candidate.get("task_id")
        if not task_id:
            print(f"[generate] Skipping candidate with missing task_id from {candidate.get('_template')}")
            continue
        task_dir = TASKS_DIR / task_id
        workspace_dir = task_dir / "workspace"
        if task_dir.exists():
            _force_rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        copy_repo(Path(candidate["repo_path"]), workspace_dir)
        originals = apply_mutations(workspace_dir, candidate.get("mutations") or [])
        write_patch(task_dir, originals, workspace_dir)
        write_json(task_dir / "mutations.json", {"mutations": candidate.get("mutations")})
        write_json(task_dir / "candidate.json", candidate)
        write_json(task_dir / "task.json", build_task_json(candidate))
        print(f"[generate] Created task {task_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Capability-Clash tasks.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tasks(args.candidates, args.limit)


if __name__ == "__main__":
    main()
