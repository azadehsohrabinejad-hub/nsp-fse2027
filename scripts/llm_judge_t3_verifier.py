"""
LLM-as-Judge with T3 model (gpt-4.1) + Ground Truth
=====================================================
Tests whether the strongest available model (gpt-4.1) as judge,
combined with ground-truth reference, pushes F1 above 0.9.

Compares:
  no-ref mini:  F1=0.615  (known)
  GT-ref mini:  F1=?      (running)
  GT-ref T3:    F1=?      (this script)
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

JUDGE_MODEL = "gpt-4.1"  # T3 — strongest available

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
C2. EQUIVALENT: Is the proposed patch semantically equivalent to the ground-truth (different syntax OK)?
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
An incomplete patch (fixes only part of what ground-truth fixes) must be FAIL.
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
        inner, in_fence = [], False
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                inner.append(line)
        raw = "\n".join(inner).strip()
    try:
        return json.loads(raw).get("edits", [])
    except Exception:
        return []


def get_patch_content(task_id: str, strategy: str, model: str) -> str:
    trace_dir = ROOT / "reports/decomposition/traces" / strategy
    trace_file = trace_dir / f"{task_id}.json"
    if trace_file.exists():
        try:
            trace = json.loads(trace_file.read_text())
            best_edits, best_pass = [], -1.0
            for rnd in trace.get("rounds", []):
                pr = float(rnd.get("pass_rate", 0) or 0)
                raw_payload = rnd.get("edit_metadata", {}).get("raw_edit_payload", "")
                if not raw_payload:
                    continue
                edits = _extract_edits_from_payload(raw_payload)
                if pr > best_pass or (pr == best_pass and edits):
                    best_pass, best_edits = pr, edits
            if best_edits:
                parts = []
                for e in best_edits[:3]:
                    old, new = e.get("old_string", "")[:300], e.get("new_string", "")[:300]
                    if old or new:
                        parts.append(f"--- old:\n{old}\n+++ new:\n{new}")
                if parts:
                    return "\n\n".join(parts)
        except Exception:
            pass
    return "(patch content not available)"


def get_task_context(task_id: str) -> dict:
    tj_path = TASKS_DIR / task_id / "task.json"
    if not tj_path.exists():
        return {}
    tj = json.loads(tj_path.read_text())
    target_files = tj.get("target_files", [])
    fam = tj.get("mutation_family", tj.get("metadata", {}).get("mutation_family", ""))
    code_ctx = ""
    for tf in target_files[:1]:
        fp = TASKS_DIR / task_id / "workspace" / tf
        if fp.exists():
            code_ctx = fp.read_text()[:1500]
    return {"prompt": tj.get("prompt", ""), "mutation_family": fam,
            "code_ctx": code_ctx, "target_file": target_files[0] if target_files else ""}


def judge_run(ctx: dict, patch: str, gt_patch: str) -> dict:
    user_msg = f"""## Mutation Description
{ctx['prompt'][:400]}

## Mutation Family: {ctx['mutation_family']}

## Buggy Code (first 1500 chars of {ctx['target_file']})
```
{ctx['code_ctx']}
```

## GROUND-TRUTH Patch (correct verified fix)
```diff
{gt_patch[:1500]}
```

## PROPOSED Patch (repair agent output)
```
{patch[:800]}
```

Evaluate the PROPOSED patch against the 5 criteria. Return JSON verdict."""

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
        result = json.loads(resp.choices[0].message.content)
        result["tokens"] = resp.usage.total_tokens if resp.usage else 0
        result["api_ok"] = True
        return result
    except Exception as e:
        return {"c1_complete": False, "c2_equivalent": False, "c3_minimal": False,
                "c4_targeted": False, "c5_family_aware": False,
                "verdict": "FAIL", "confidence": 0.0,
                "reasoning": f"API error: {e}", "tokens": 0, "api_ok": False}


def compute_metrics(results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        incomplete = True  # ground truth: all 72 are semantically incomplete
        flagged = r["verdict"] == "FAIL"
        if incomplete and flagged: tp += 1
        elif incomplete and not flagged: fn += 1
        elif not incomplete and flagged: fp += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    runs = load_test_passing_runs()
    print(f"Test-passing runs: {len(runs)} | Judge: {JUDGE_MODEL} (GT-guided)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, run in enumerate(runs):
        task_id, strategy, model = run["task_id"], run["strategy"], run["model"]
        ctx = get_task_context(task_id)
        if not ctx:
            print(f"  [{i+1}/{len(runs)}] {task_id}: SKIP")
            continue
        gt_patch = get_ground_truth_patch(task_id)
        patch = get_patch_content(task_id, strategy, model)
        verdict = judge_run(ctx, patch, gt_patch)
        verdict.update({"task_id": task_id, "strategy": strategy, "model": model,
                        "mutation_family": ctx.get("mutation_family", "")})
        status = verdict["verdict"]
        conf = verdict.get("confidence", 0)
        print(f"  [{i+1:2d}/{len(runs)}] {task_id[:35]:<35} {strategy:<20} → {status} ({conf:.2f})")
        results.append(verdict)
        time.sleep(0.5)  # T3 is more expensive, be conservative

    m = compute_metrics(results)
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    avg_tok = sum(r.get("tokens", 0) for r in results) / len(results) if results else 0

    print(f"\n=== T3 Judge (gpt-4.1 + Ground Truth) ===")
    print(f"  Judge FAIL: {n_fail}/{len(results)} = {n_fail/len(results):.1%}")
    print(f"  Avg tokens: {avg_tok:.0f}")
    print(f"\nComparison:")
    print(f"  Rule-based:            precision=1.000, recall=0.167, F1=0.286")
    print(f"  LLM mini (no ref):     precision=1.000, recall=0.444, F1=0.615")
    print(f"  LLM T3 gpt-4.1 (+ GT): precision={m['precision']:.3f}, recall={m['recall']:.3f}, F1={m['f1']:.3f}")

    from collections import defaultdict
    by_fam: dict = defaultdict(lambda: {"total": 0, "fail": 0})
    for r in results:
        fam = r.get("mutation_family", "unknown")
        by_fam[fam]["total"] += 1
        if r["verdict"] == "FAIL": by_fam[fam]["fail"] += 1
    print("\nPer-family (T3 GT-guided):")
    for fam, c in sorted(by_fam.items()):
        rate = c["fail"] / c["total"] if c["total"] > 0 else 0
        print(f"  {fam}: {c['fail']}/{c['total']} = {rate:.0%}")

    csv_path = OUT_DIR / "llm_judge_t3_results.csv"
    fields = ["task_id", "strategy", "model", "mutation_family",
              "verdict", "confidence", "c1_complete", "c2_equivalent",
              "c3_minimal", "c4_targeted", "c5_family_aware", "reasoning", "tokens"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"CSV: {csv_path}")

    md_path = OUT_DIR / "llm_judge_t3_summary.md"
    md_path.write_text(f"""# T3 LLM Judge (gpt-4.1 + Ground Truth)

Judge model: `{JUDGE_MODEL}` | Ground-truth-guided | 5-criterion rubric

| Verifier | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Rule-based | 1.000 | 0.167 | 0.286 |
| LLM mini (no reference) | 1.000 | 0.444 | 0.615 |
| LLM T3 gpt-4.1 (+ GT patch) | {m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** |

F1 threshold for viable RL training: **0.800**
""")
    print(f"MD: {md_path}")
    return m


if __name__ == "__main__":
    main()
