"""Build eval/report.md from runs/no_brain and runs/brain (each written by eval.run_sweep.arm). Safe to re-run."""
from __future__ import annotations

import json
from pathlib import Path

from eval.run_sweep import table, stale_servings


def load(name: str) -> list[dict]:
    p = Path("runs") / name / "episodes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def per_shift(recs: list[dict]) -> str:
    rows = ["| Shift | Success | Claimed done | Verifier | False done | Tool calls | Prompt tok | Decision |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        rows.append(f"| {r['episode']} | {'✅' if r['success'] else '❌'} | {'yes' if r['claimed_done'] else 'no'} | "
                    f"{'pass' if r['verifier_pass'] else 'fail'} | {'**yes**' if r['false_done'] else 'no'} | {r['tool_calls']} | "
                    f"{r['tokens']['prompt']:,} | {r['decision']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    nb, wb = load("no_brain"), load("brain")
    md = ["# Robot Agent Lab — with vs without the shared brain", "",
          f"Same seeds, same models, same prompts (MemDelta). {len(nb)} no-brain shifts, {len(wb)} brain shifts. "
          "Gotchas: sticky gripper (every shift), blue rack moves at shift 3, operator camera nudged at shift 5.", ""]
    if nb and wb:
        md += [table(nb, wb), ""]
    md += ["## No brain, shift by shift", "", per_shift(nb), "", "## Shared brain, shift by shift", "", per_shift(wb), ""]
    bf = Path("runs/brain/brain_final.json")
    if bf.exists():
        brain = json.loads(bf.read_text())
        md += [f"Stale `bin_blue` positions served after the rack moved: **{stale_servings(wb, Path('runs/brain'))}**", "",
               "## What the brain held at the end", "", "| id | tier | subject | relation | object | from shift |", "|---|---|---|---|---|---|"]
        md += [f"| {b['id']} | {b['tier']} | {b['subject']} | {b['relation']} | {str(b['object'])[:90]} | {b['valid_from']} |" for b in brain]
    Path("eval/report.md").write_text("\n".join(md))
    print("\n".join(md[:30]))
