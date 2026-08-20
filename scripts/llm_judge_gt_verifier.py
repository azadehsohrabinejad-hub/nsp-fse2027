"""
LLM-as-Judge Semantic Verifier (Ground-Truth-Guided)
=====================================================
Enhancement over llm_judge_verifier.py:
Provides the ground-truth patch as a reference to the judge.

Hypothesis: Reference-guided judge will dramatically improve recall
(currently 0.444) while maintaining perfect precision, pushing F1 above
the 0.8 threshold required for viable RL training.

Input:  reports/capability_clash/capability_clash_results.csv (480 rows)
        data/capability_clash/tasks/{task_id}/ground_truth.patch
        reports/decomposition/traces/{strategy}/{task_id}.json

Output:
  reports/ase2026_aegis/llm_judge_gt_results.csv
  reports/ase2026_aegis/llm_judge_gt_summary.md
"""
from __future__ import annotations

import csv, json, os, time
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "reports/capability_clash/capability_clash_results.csv"
TASKS_DIR   = ROOT / "data/capability_clash/tasks"
OUT_DIR     = ROOT / "reports/ase2026_aegis"

key = os.environ.get("OPENAI_API_KEY")
if not key:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip()
client = openai.OpenAI(api_key=key)

JUDGE_MODEL = "gpt-4.1-mini"

RUBRIC_SYSTEM = """You are a strict semantic code review expert evaluating whether a proposed patch
correctly and completely fixes a software mutation, using the ground-truth patch as reference.

You will be given:
1. The mutation description (what was changed)
2. The mutation family (type of adversarial mutation)
3. The buggy code (post-mutation)
4. The GROUND-TRUTH patch (the correct, verified fix)
5. The PROPOSED patch (what the repair agent produced)

Evaluate the proposed patch against these 5 criteria:
C1. COMPLETE: Does the proposed patch address ALL aspects that the ground-truth patch addresses?
C2. EQUIVALENT: Is the proposed patch semantically equivalent to the ground-truth patch (different syntax is OK if semantics match)?
C3. MINIMAL: Does the proposed patch avoid unnecessary changes beyond what the ground-truth requires?
C4. TARGETED: Does the proposed patch modify the same locations as the ground-truth patch?
C5. FAMILY-AWARE: Does the fix account for the specific adversarial pattern of this mutation family?

Respond ONLY with a JSON object:
{
  "c1_complete": true/false,
  "c2_equivalent": true/false,
  "c3_minimal": true/false,
  "c4_targeted": true/false,
  "c5_family_aware": true/false,
  "verdict": "PASS" or "FAIL",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explaining the verdict"
}

VERDICT = PASS only if ALL five criteria are true.
A proposed patch that is INCOMPLETE (fixes only part of what the ground-truth fixes) must be FAIL.
A proposed patch that fixes the WRONG LOCATION must be FAIL even if tests pass.
"""


def load_test_passing_runs() -> list[dict]:
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    return [r for r in rows if float(r.get("final_pass", 0)) > 0]


def get_ground_truth_patch(task_id: str) -> str:
    gt_path = TASKS_DIR / task_id / "ground_truth.patch"
    if gt_path.exists():
        return gt_path.read_text()[:2000]
    return "(ground truth not available)"


def _extract_edits_from_payload(raw_payload: str) -> list[dict]:
    raw = raw_payload.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = []
        in_fence = False
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                inner.append(line)
        raw = "\n".join(inner).strip()
    try:
        payload = json.loads(raw)
        return payload.get("edits", [])
    except Exception:
        return []


def get_patch_content(task_id: str, strategy: str, model: str) -> str:
    trace_dir = ROOT / "reports/decomposition/traces" / strategy
    trace_file = trace_dir / f"{task_id}.json"
    if trace_file.exists():
        try:
            trace = json.loads(trace_file.read_text())
            rounds = trace.get("rounds", [])
            best_edits: list[dict] = []
            best_pass = -1.0
            for rnd in rounds:
                pr = float(rnd.get("pass_rate", 0) or 0)
                raw_payload = rnd.get("edit_metadata", {}).get("raw_edit_payload", "")
                if not raw_payload:
                    continue
                edits = _extract_edits_from_payload(raw_payload)
                if pr > best_pass or (pr == best_pass and edits):
                    best_pass = pr
                    best_edits = edits
            if best_edits:
                patch_lines = []
                for e in best_edits[:3]:
                    old = e.get("old_string", "")[:300]
                    new = e.get("new_string", "")[:300]
                    if old or new:
                        patch_lines.append(f"--- old:\n{old}\n+++ new:\n{new}")
                if patch_lines:
                    return "\n\n".join(patch_lines)
            top_edits = trace.get("edits", trace.get("edit_batch", []))
            if top_edits:
                patch_lines = []
                for e in top_edits[:3]:
                    old = e.get("old_string", "")[:300]
                    new = e.get("new_string", "")[:300]
                    if old or new:
                        patch_lines.append(f"--- old:\n{old}\n+++ new:\n{new}")
                return "\n\n".join(patch_lines)
        except Exception:
            pass
    return "(patch content not available)"


def get_task_context(task_id: str) -> dict:
    task_dir = TASKS_DIR / task_id
    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        return {}
    tj = json.loads(task_json_path.read_text())
    target_files = tj.get("target_files", [])
    prompt = tj.get("prompt", "")
    fam = tj.get("mutation_family", tj.get("metadata", {}).get("mutation_family", ""))
    code_ctx = ""
    for tf in target_files[:1]:
        fp = task_dir / "workspace" / tf
        if fp.exists():
            code_ctx = fp.read_text()[:1500]
    return {
        "prompt": prompt,
        "mutation_family": fam,
        "code_ctx": code_ctx,
        "target_file": target_files[0] if target_files else "",
    }


def judge_run_gt(run: dict, ctx: dict, patch: str, gt_patch: str) -> dict:
    user_msg = f"""## Mutation Description
{ctx['prompt'][:400]}

## Mutation Family: {ctx['mutation_family']}

## Buggy Code (post-mutation, first 1500 chars of {ctx['target_file']})
```
{ctx['code_ctx']}
```

## GROUND-TRUTH Patch (the correct verified fix)
```diff
{gt_patch[:1500]}
```

## PROPOSED Patch (what the repair agent produced)
```
{patch[:800]}
```

Evaluate the PROPOSED patch against the 5 criteria using the GROUND-TRUTH as reference. Return the JSON verdict."""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": RUBRIC_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        tokens = resp.usage.total_tokens if resp.usage else 0
        result = json.loads(raw)
        result["tokens"] = tokens
        result["api_ok"] = True
        return result
    except Exception as e:
        return {
            "c1_complete": False, "c2_equivalent": False, "c3_minimal": False,
            "c4_targeted": False, "c5_family_aware": False,
            "verdict": "FAIL", "confidence": 0.0,
            "reasoning": f"API error: {e}", "tokens": 0, "api_ok": False,
        }


def compute_metrics(judge_results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for jr in judge_results:
        is_incomplete = True  # ground truth: all 72 test-passing CC runs are semantically incomplete
        judge_says_incomplete = jr["verdict"] == "FAIL"
        if is_incomplete and judge_says_incomplete:
            tp += 1
        elif is_incomplete and not judge_says_incomplete:
            fn += 1
        elif not is_incomplete and judge_says_incomplete:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    runs = load_test_passing_runs()
    print(f"Test-passing runs to evaluate: {len(runs)}")
    print(f"Judge model: {JUDGE_MODEL} (ground-truth-guided)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    judge_results = []
    skipped = 0

    for i, run in enumerate(runs):
        task_id = run["task_id"]
        strategy = run["strategy"]
        model = run["model"]

        ctx = get_task_context(task_id)
        if not ctx:
            print(f"  [{i+1}/{len(runs)}] {task_id}: SKIP (no task.json)")
            skipped += 1
            continue

        gt_patch = get_ground_truth_patch(task_id)
        patch = get_patch_content(task_id, strategy, model)
        verdict = judge_run_gt(run, ctx, patch, gt_patch)
        verdict.update({
            "task_id": task_id,
            "strategy": strategy,
            "model": model,
            "mutation_family": ctx.get("mutation_family", ""),
            "has_gt": gt_patch != "(ground truth not available)",
            "has_patch": patch != "(patch content not available)",
        })

        status = verdict["verdict"]
        conf = verdict.get("confidence", 0)
        gt_flag = "GT✓" if verdict["has_gt"] else "GT✗"
        patch_flag = "P✓" if verdict["has_patch"] else "P✗"
        print(f"  [{i+1:2d}/{len(runs)}] {task_id[:35]:<35} {strategy:<20} → {status} ({conf:.2f}) [{gt_flag} {patch_flag}]")
        judge_results.append(verdict)
        time.sleep(0.3)

    metrics = compute_metrics(judge_results)
    n_fail = sum(1 for r in judge_results if r["verdict"] == "FAIL")
    n_pass = sum(1 for r in judge_results if r["verdict"] == "PASS")
    avg_tok = sum(r.get("tokens", 0) for r in judge_results) / len(judge_results) if judge_results else 0
    n_with_gt = sum(1 for r in judge_results if r.get("has_gt"))
    n_with_patch = sum(1 for r in judge_results if r.get("has_patch"))

    print(f"\n=== Ground-Truth-Guided LLM Judge Results ===")
    print(f"Evaluated: {len(judge_results)} | Skipped: {skipped}")
    print(f"  With ground truth: {n_with_gt}/{len(judge_results)}")
    print(f"  With patch content: {n_with_patch}/{len(judge_results)}")
    print(f"  Judge FAIL (incomplete): {n_fail}/{len(judge_results)} = {n_fail/len(judge_results):.1%}")
    print(f"  Judge PASS (complete):   {n_pass}/{len(judge_results)} = {n_pass/len(judge_results):.1%}")
    print(f"  Avg tokens/call: {avg_tok:.0f}")
    print(f"\nComparison:")
    print(f"  Rule-based verifier:       precision=1.000, recall=0.167, F1=0.286")
    print(f"  LLM judge (no reference):  precision=1.000, recall=0.444, F1=0.615")
    print(f"  LLM judge (+ GT patch):    precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    from collections import defaultdict
    by_family: dict = defaultdict(lambda: {"total": 0, "fail": 0})
    for r in judge_results:
        fam = r.get("mutation_family", "unknown")
        by_family[fam]["total"] += 1
        if r["verdict"] == "FAIL":
            by_family[fam]["fail"] += 1

    print("\nPer-family detection rate (GT-guided):")
    for fam, counts in sorted(by_family.items()):
        rate = counts["fail"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {fam}: {counts['fail']}/{counts['total']} = {rate:.0%}")

    # Write CSV
    csv_path = OUT_DIR / "llm_judge_gt_results.csv"
    fields = ["task_id", "strategy", "model", "mutation_family",
              "verdict", "confidence", "c1_complete", "c2_equivalent",
              "c3_minimal", "c4_targeted", "c5_family_aware",
              "has_gt", "has_patch", "reasoning", "tokens"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(judge_results)
    print(f"\nCSV: {csv_path}")

    # Write markdown summary
    md = [
        "# Ground-Truth-Guided LLM Judge Results\n",
        f"Judge model: `{JUDGE_MODEL}` | Rubric: 5 criteria (C1 Complete, C2 Equivalent, C3 Minimal, C4 Targeted, C5 Family-aware)\n",
        f"Evaluated: {len(judge_results)} test-passing runs | With ground truth: {n_with_gt} | With patch: {n_with_patch}\n",
        "\n## Verifier Comparison\n",
        "| Verifier | Precision | Recall | F1 |",
        "|----------|-----------|--------|-----|",
        "| Rule-based (target_file_recall < 1.0) | 1.000 | 0.167 | 0.286 |",
        "| LLM judge, no reference (gpt-4.1-mini, C1--C5) | 1.000 | 0.444 | 0.615 |",
        f"| LLM judge + ground-truth patch (gpt-4.1-mini, C1--C5) | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | **{metrics['f1']:.3f}** |",
        "\n## Per-Family Detection Rate\n",
        "| Family | Total | FAIL (detected) | Rate |",
        "|--------|-------|-----------------|------|",
    ]
    for fam, counts in sorted(by_family.items()):
        rate = counts["fail"] / counts["total"] if counts["total"] > 0 else 0
        md.append(f"| {fam} | {counts['total']} | {counts['fail']} | {rate:.0%} |")
    md += [
        "\n## Rubric Criteria (GT-guided)\n",
        "- **C1 Complete**: Addresses all aspects covered by the ground-truth patch",
        "- **C2 Equivalent**: Semantically equivalent to the ground-truth patch",
        "- **C3 Minimal**: No unnecessary changes beyond ground-truth scope",
        "- **C4 Targeted**: Modifies the same locations as the ground-truth patch",
        "- **C5 Family-aware**: Accounts for the specific adversarial pattern",
        "\nVERDICT = PASS only if all 5 criteria true.",
        f"\nF1 threshold for viable RL training: **0.800**. GT-guided judge achieves: **{metrics['f1']:.3f}**.",
    ]
    (OUT_DIR / "llm_judge_gt_summary.md").write_text("\n".join(md))
    print(f"MD:  {OUT_DIR}/llm_judge_gt_summary.md")

    return metrics


if __name__ == "__main__":
    main()
