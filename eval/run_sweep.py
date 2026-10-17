"""With-brain vs without-brain, same seeds, same models, same prompts (MemDelta). Writes eval/report.md.

    uv run python -m eval.run_sweep --episodes 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from llm.crusoe import Crusoe
from brain.ledger import Brain
from run_episode import run_episode


def arm(llm: Crusoe, name: str, episodes: int, seed0: int, with_brain: bool) -> list[dict]:
    out = Path("runs") / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    brain = Brain(out / "brain.sqlite") if with_brain else None
    recs = []
    for ep in range(1, episodes + 1):
        recs.append(run_episode(llm, ep, seed0 + ep, brain, out_dir=out, log=lambda s: print(f"[{name}] {s}"), playbook=""))  # no learned playbook: isolate the brain effect
    if brain:
        (out / "brain_final.json").write_text(json.dumps(brain.current(), indent=1))
        (out / "served.json").write_text(json.dumps(brain.served, indent=1))
    return recs


def stale_servings(recs: list[dict], out: Path, shift_episode: int = 3) -> int:
    """A served bin_blue position from before the shift, served on/after the shift episode, is a stale serving."""
    p = out / "served.json"
    if not p.exists():
        return 0
    served = json.loads(p.read_text())
    brain = json.loads((out / "brain_final.json").read_text())
    # entries by id with their valid_from
    return sum(1 for s in served if s["subject"] == "bin_blue" and s["relation"].startswith("position")
               and s["episode"] >= shift_episode and any(b["id"] == s["id"] and b["valid_from"] < shift_episode for b in brain))


def table(a: list[dict], b: list[dict]) -> str:
    def agg(rs):
        n = len(rs)
        f = lambda k: sum(1 for r in rs if r.get(k)) / n
        tok = lambda k: sum(r["tokens"][k] for r in rs) / n
        vc = [r["verifier_correct"] for r in rs if r["verifier_correct"] is not None]
        first = next((r["episode"] for r in rs if r["success"]), None)
        return {"success": f("success"), "false_done": f("false_done"), "claimed": f("claimed_done"),
                "verifier_acc": (sum(vc) / len(vc)) if vc else 0, "tool_calls": sum(r["tool_calls"] for r in rs) / n,
                "prompt_tok": tok("prompt"), "cached_tok": tok("cached"), "completion_tok": tok("completion"),
                "first_success": first, "replans": sum(r["replans"] for r in rs)}
    A, B = agg(a), agg(b)
    rows = [("Shifts", len(a), len(b)), ("Success rate", f"{A['success']:.0%}", f"{B['success']:.0%}"),
            ("First successful shift", A["first_success"], B["first_success"]),
            ("False 'done' claims", f"{A['false_done']:.0%}", f"{B['false_done']:.0%}"),
            ("Verifier accuracy vs ground truth", f"{A['verifier_acc']:.0%}", f"{B['verifier_acc']:.0%}"),
            ("Tool calls / shift", f"{A['tool_calls']:.1f}", f"{B['tool_calls']:.1f}"),
            ("Prompt tokens / shift", f"{A['prompt_tok']:,.0f}", f"{B['prompt_tok']:,.0f}"),
            ("  of which cached", f"{A['cached_tok']:,.0f}", f"{B['cached_tok']:,.0f}"),
            ("Completion tokens / shift", f"{A['completion_tok']:,.0f}", f"{B['completion_tok']:,.0f}"),
            ("Re-plans (total)", A["replans"], B["replans"])]
    s = "| Metric | No brain | Shared brain |\n|---|---|---|\n" + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=100)
    args = ap.parse_args()
    llm = Crusoe()
    nb = arm(llm, "no_brain", args.episodes, args.seed0, with_brain=False)
    wb = arm(llm, "brain", args.episodes, args.seed0, with_brain=True)
    stale = stale_servings(wb, Path("runs/brain"))
    md = ["# Night Shift — with vs without the shared brain", "",
          f"Same seeds, same models, same prompts. {args.episodes} shifts per arm. Gotchas: sticky gripper (all shifts), "
          "blue rack moves at shift 3, operator camera nudged at shift 5.", "", table(nb, wb), "",
          f"Stale bin_blue positions served after the rack moved (brain arm): **{stale}**", "",
          "Per-shift records: runs/no_brain/episodes.jsonl and runs/brain/episodes.jsonl. Brain contents: runs/brain/brain_final.json.",
          "", "Usage by role: ", "```", json.dumps(llm.usage.snapshot(), indent=1), "```"]
    Path("eval/report.md").write_text("\n".join(md))
    print("\n".join(md[:8]))
