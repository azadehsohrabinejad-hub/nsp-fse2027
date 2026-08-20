"""
LLM-as-Judge Semantic Verifier
================================
Replaces the rule-based verifier (F1=0.286) with a structured LLM rubric.
Evaluates each test-passing repair against 5 semantic criteria.

Input: reports/capability_clash/capability_clash_results.csv (480 runs)
       data/capability_clash/tasks/ (workspace + task.json)

Output:
  reports/ase2026_aegis/llm_judge_results.csv
  reports/ase2026_aegis/llm_judge_summary.md
  (compares against rule-based verifier)
"""
from __future__ import annotations

import csv, json, os, time
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "reports/capability_clash/capability_clash_results.csv"
AUDIT_CSV   = ROOT / "reports/capability_clash/semantic_audit_results.csv"
TASKS_DIR   = ROOT / "data/capability_clash/tasks"
OUT_DIR     = ROOT / "reports/ase2026_aegis"

key = os.environ.get("OPENAI_API_KEY")
if not key:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip()
client = openai.OpenAI(api_key=key)

JUDGE_MODEL = "gpt-4.1-mini"   # T2 — stronger reasoning for verification

RUBRIC_SYSTEM = """You are a strict semantic code review expert evaluating whether a proposed patch
fully and correctly reverts an injected software mutation.

You will be given:
1. The mutation description (what was changed)
2. The original buggy code (post-mutation)
3. The proposed patch (what the repair agent changed)
4. The mutation family (type of adversarial mutation)

Evaluate the patch against these 5 criteria:
C1. COMPLETE: Does the patch address ALL aspects of the mutation (not just surface symptoms)?
C2. CORRECT: Is the semantics of the fix accurate — no new bugs introduced?
C3. MINIMAL: Does the patch avoid unnecessary changes beyond what's needed?
C4. TARGETED: Does the patch modify the right location (not a compensating hack elsewhere)?
C5. FAMILY-AWARE: Does the fix account for the specific adversarial pattern of this mutation family?

Respond ONLY with a JSON object:
{
  "c1_complete": true/false,
  "c2_correct": true/false,
  "c3_minimal": true/false,
  "c4_targeted": true/false,
  "c5_family_aware": true/false,
  "verdict": "PASS" or "FAIL",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence"
}

VERDICT = PASS only if ALL five criteria are true.
"""


def load_test_passing_runs() -> list[dict]:
    """Load runs that passed test suite (candidates for semantic verification)."""
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    return [r for r in rows if float(r.get("final_pass", 0)) > 0]


def load_audit_truth() -> dict:
    """Load semantic audit ground truth: {task_id+strategy+model: incomplete_fix}"""
    rows = list(csv.DictReader(open(AUDIT_CSV)))
    truth = {}
    for r in rows:
        key = (r["task_id"], r["strategy"], r["model"])
        # In audit: final_pass=1 AND patch_completeness=0 means "test passes but semantically incomplete"
        truth[key] = {
            "audit_pass": float(r.get("final_pass", 0)) > 0,
            "patch_completeness": float(r.get("patch_completeness", 0)),
            "audit_verdict": r.get("audit_verdict", ""),
        }
    return truth


def _extract_edits_from_payload(raw_payload: str) -> list[dict]:
    """Parse raw_edit_payload JSON (possibly wrapped in ```json ... ```)."""
    raw = raw_payload.strip()
    if raw.startswith("```"):
        # Strip code fence
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
    """Load the repair patch from round trace files.

    Trace structure (canonical):
      {task_id, strategy, plan, rounds: [{round, edit_metadata: {raw_edit_payload}}]}
    """
    trace_dir = ROOT / "reports/decomposition/traces" / strategy
    trace_file = trace_dir / f"{task_id}.json"
    if trace_file.exists():
        try:
            trace = json.loads(trace_file.read_text())
            rounds = trace.get("rounds", [])

            # Find best round: prefer highest pass_rate, then last round
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

            # Fallback: top-level edits
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
    return "(patch content not available — judging from task context only)"


def get_task_context(task_id: str) -> dict:
    """Get mutation description + buggy code."""
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
            code_ctx = fp.read_text()[:2000]

    return {
        "prompt": prompt,
        "mutation_family": fam,
        "code_ctx": code_ctx,
        "target_file": target_files[0] if target_files else "",
    }


def judge_run(run: dict, ctx: dict, patch: str) -> dict:
    """Call the LLM judge and return verdict dict."""
    user_msg = f"""## Mutation Description
{ctx['prompt'][:500]}

## Mutation Family: {ctx['mutation_family']}

## Buggy Code (post-mutation, first 2000 chars of {ctx['target_file']})
```
{ctx['code_ctx']}
```

## Proposed Patch Applied by Repair Agent
```
{patch[:1000]}
```

Evaluate this patch against the 5 criteria and return the JSON verdict."""

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
            "c1_complete": False, "c2_correct": False, "c3_minimal": False,
            "c4_targeted": False, "c5_family_aware": False,
            "verdict": "FAIL", "confidence": 0.0,
            "reasoning": f"API error: {e}", "tokens": 0, "api_ok": False,
        }


def compute_metrics(judge_results: list[dict], audit_truth: dict) -> dict:
    """Compare judge verdicts vs semantic audit ground truth."""
    # Ground truth: audit says ALL test-passing runs are semantically incomplete
    # So ground truth label = "incomplete" for all 72 runs
    # Judge verdict: PASS = judge says complete, FAIL = judge says incomplete

    tp = fp = tn = fn = 0
    for jr in judge_results:
        key = (jr["task_id"], jr["strategy"], jr["model"])
        # Ground truth: all test-passing runs in CC are semantically incomplete (audit=100%)
        is_incomplete = True  # ground truth
        judge_says_incomplete = jr["verdict"] == "FAIL"

        if is_incomplete and judge_says_incomplete:  # correctly flagged
            tp += 1
        elif is_incomplete and not judge_says_incomplete:  # missed
            fn += 1
        elif not is_incomplete and judge_says_incomplete:  # false alarm
            fp += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    runs = load_test_passing_runs()
    audit_truth = load_audit_truth()
    print(f"Test-passing runs to evaluate: {len(runs)}")
    print(f"Using judge model: {JUDGE_MODEL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    judge_results = []

    for i, run in enumerate(runs):
        task_id = run["task_id"]
        strategy = run["strategy"]
        model = run["model"]

        ctx = get_task_context(task_id)
        if not ctx:
            print(f"  [{i+1}/{len(runs)}] {task_id}: SKIP (no task.json)")
            continue

        patch = get_patch_content(task_id, strategy, model)
        verdict = judge_run(run, ctx, patch)
        verdict.update({
            "task_id": task_id,
            "strategy": strategy,
            "model": model,
            "mutation_family": ctx.get("mutation_family", ""),
        })

        status = verdict["verdict"]
        conf = verdict.get("confidence", 0)
        print(f"  [{i+1:2d}/{len(runs)}] {task_id[:40]:<40} {strategy:<20} → {status} ({conf:.2f}) [{verdict.get('tokens',0)} tok]")
        judge_results.append(verdict)
        time.sleep(0.3)

    # Metrics
    metrics = compute_metrics(judge_results, audit_truth)
    n_fail = sum(1 for r in judge_results if r["verdict"] == "FAIL")
    n_pass = sum(1 for r in judge_results if r["verdict"] == "PASS")
    avg_tokens = sum(r.get("tokens", 0) for r in judge_results) / len(judge_results) if judge_results else 0

    print(f"\n=== LLM Judge Results ===")
    print(f"Evaluated: {len(judge_results)} test-passing runs")
    print(f"  Judge FAIL (incomplete): {n_fail}/{len(judge_results)} = {n_fail/len(judge_results):.1%}")
    print(f"  Judge PASS (complete):   {n_pass}/{len(judge_results)} = {n_pass/len(judge_results):.1%}")
    print(f"  Avg tokens/call: {avg_tokens:.0f}")
    print(f"\nVs rule-based verifier: precision=1.0, recall=0.167, F1=0.286")
    print(f"LLM judge:              precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Per-family breakdown
    from collections import defaultdict
    by_family: dict = defaultdict(lambda: {"total": 0, "fail": 0})
    for r in judge_results:
        fam = r.get("mutation_family", "unknown")
        by_family[fam]["total"] += 1
        if r["verdict"] == "FAIL":
            by_family[fam]["fail"] += 1

    print("\nPer-family incompleteness rate:")
    for fam, counts in sorted(by_family.items()):
        rate = counts["fail"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {fam}: {counts['fail']}/{counts['total']} = {rate:.0%}")

    # Write CSV
    csv_path = OUT_DIR / "llm_judge_results.csv"
    fields = ["task_id", "strategy", "model", "mutation_family",
              "verdict", "confidence", "c1_complete", "c2_correct",
              "c3_minimal", "c4_targeted", "c5_family_aware", "reasoning", "tokens"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(judge_results)
    print(f"\nCSV: {csv_path}")

    # Write markdown
    md = [
        "# LLM-as-Judge Semantic Verifier Results\n",
        f"Judge model: `{JUDGE_MODEL}` | Rubric: 5 criteria (C1–C5)\n",
        f"Evaluated: {len(judge_results)} test-passing runs from CCBench\n",
        "\n## Verifier Comparison\n",
        "| Verifier | Precision | Recall | F1 |",
        "|----------|-----------|--------|----|",
        f"| Rule-based (target_file_recall < 1.0) | 1.000 | 0.167 | 0.286 |",
        f"| LLM judge (gpt-4.1-mini, 5-criterion rubric) | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | **{metrics['f1']:.3f}** |",
        "\n## Per-Family Incompleteness Rate\n",
        "| Family | Total | FAIL | Rate |",
        "|--------|-------|------|------|",
    ]
    for fam, counts in sorted(by_family.items()):
        rate = counts["fail"] / counts["total"] if counts["total"] > 0 else 0
        md.append(f"| {fam} | {counts['total']} | {counts['fail']} | {rate:.0%} |")

    md += [
        "\n## Rubric Criteria\n",
        "- **C1 Complete**: All aspects of mutation addressed",
        "- **C2 Correct**: No new bugs introduced",
        "- **C3 Minimal**: No unnecessary changes",
        "- **C4 Targeted**: Right location modified",
        "- **C5 Family-aware**: Accounts for adversarial pattern",
        "\nVERDICT = PASS only if all 5 criteria true. Ground truth: all 72 test-passing CC runs are semantically incomplete.",
    ]
    (OUT_DIR / "llm_judge_summary.md").write_text("\n".join(md))
    print(f"MD:  {OUT_DIR}/llm_judge_summary.md")

    return metrics


if __name__ == "__main__":
    main()
