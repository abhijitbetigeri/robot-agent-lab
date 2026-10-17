"""Version N+1: the team rewrites its own operator playbook from its shift history and keeps the change only if it
scores at least as well on a held-out shift it has not seen. (Darwin Gödel Machine / RSEA pattern, in miniature.)

    uv run python improve.py --history runs/brain --seed 500

Playbook lives in runs/playbook.json: {"version", "rules", "history": [{"version", "rules", "score_old", "score_new", "kept"}]}.
The operator's system prompt gets the current rules appended, so a kept change alters every future shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.crusoe import Crusoe
from run_episode import run_episode

PLAYBOOK = Path("runs/playbook.json")
IMPROVER_SYSTEM = """You are the IMPROVER for a lab robot's night-shift agent team. You read the records of recent shifts (what the
operator did, what the sealed verifier said, where it overclaimed or wasted calls) and the current operator PLAYBOOK.
Rewrite the playbook: at most 8 short imperative rules that would have made those shifts succeed with fewer tool calls and
no false "done" claims. Keep rules that clearly helped; drop or rewrite ones that did not. Rules must be about HOW to operate
(sequencing, retries, verification before claiming, when to look), never about specific tube positions.
Output JSON: {"rules": ["...", ...], "rationale": "<two sentences>"}"""


def load_playbook() -> dict:
    if PLAYBOOK.exists():
        return json.loads(PLAYBOOK.read_text())
    return {"version": 0, "rules": [], "history": []}


def save_playbook(pb: dict) -> None:
    PLAYBOOK.parent.mkdir(parents=True, exist_ok=True)
    PLAYBOOK.write_text(json.dumps(pb, indent=1))


def playbook_text(pb: dict) -> str:
    if not pb.get("rules"):
        return ""
    return f"\nPLAYBOOK v{pb['version']} (rules this team learned from its own shifts):\n" + "\n".join(f"- {r}" for r in pb["rules"])


def score(rec: dict) -> float:
    """Higher is better. Success dominates; a false done is punished; fewer tool calls and re-plans are better."""
    return 100.0 * rec["success"] - 50.0 * rec["false_done"] - 0.5 * rec["tool_calls"] - 5.0 * rec["replans"]


def propose(llm: Crusoe, history_dir: Path, pb: dict, last_k: int = 4) -> dict:
    recs = [json.loads(l) for l in (history_dir / "episodes.jsonl").read_text().splitlines() if l.strip()][-last_k:]
    digest = []
    for r in recs:
        digest.append({"shift": r["episode"], "success": r["success"], "claimed_done": r["claimed_done"], "false_done": r["false_done"],
                       "tool_calls": r["tool_calls"], "decision": r["decision"],
                       "verifier": [(v["id"], v["pass"], v["reason"]) for v in r["verdicts"].get("verdicts", [])],
                       "operator_notes": (r.get("claim") or {}).get("notes", "")})
    user = f"RECENT SHIFTS:\n{json.dumps(digest, indent=1)}\n\nCURRENT PLAYBOOK v{pb['version']}:\n{json.dumps(pb['rules'])}"
    out = llm.json("planner", IMPROVER_SYSTEM, user, max_tokens=2500)
    rules = [str(x) for x in out.get("rules", [])][:8]
    return {"rules": rules, "rationale": out.get("rationale", "")}


def gate(llm: Crusoe, old: dict, new_rules: list[str], seed: int, out_dir: Path) -> tuple[float, float, dict, dict]:
    """One held-out shift with the old playbook, one with the new. Same seed, no brain, so only the playbook differs."""
    new = {"version": old["version"] + 1, "rules": new_rules}
    r_old = run_episode(llm, episode=1, seed=seed, brain=None, out_dir=out_dir / "gate_old", playbook=playbook_text(old),
                        log=lambda s: print("[gate old]", s))
    r_new = run_episode(llm, episode=1, seed=seed, brain=None, out_dir=out_dir / "gate_new", playbook=playbook_text(new),
                        log=lambda s: print("[gate new]", s))
    return score(r_old), score(r_new), r_old, r_new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="runs/brain", help="dir with episodes.jsonl to learn from")
    ap.add_argument("--seed", type=int, default=500, help="held-out seed the team has never run")
    ap.add_argument("--out", default="runs/improve")
    a = ap.parse_args()
    llm = Crusoe()
    pb = load_playbook()
    prop = propose(llm, Path(a.history), pb)
    print(f"proposed v{pb['version'] + 1}: {json.dumps(prop['rules'], indent=1)}\nrationale: {prop['rationale']}")
    s_old, s_new, r_old, r_new = gate(llm, pb, prop["rules"], a.seed, Path(a.out))
    kept = s_new >= s_old
    pb["history"].append({"version": pb["version"] + 1, "rules": prop["rules"], "rationale": prop["rationale"], "seed": a.seed,
                          "score_old": s_old, "score_new": s_new, "kept": kept,
                          "old": {k: r_old[k] for k in ("success", "false_done", "tool_calls", "replans")},
                          "new": {k: r_new[k] for k in ("success", "false_done", "tool_calls", "replans")}})
    if kept:
        pb["version"] += 1
        pb["rules"] = prop["rules"]
    save_playbook(pb)
    print(f"held-out seed {a.seed}: old={s_old:.1f} new={s_new:.1f} -> {'KEPT as v' + str(pb['version']) if kept else 'REJECTED, playbook stays v' + str(pb['version'])}")
    print("usage", json.dumps(llm.usage.totals()))
