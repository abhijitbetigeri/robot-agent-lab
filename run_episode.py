"""One shift: goal -> definition of done -> operator loop -> sealed verifier -> ground truth -> curator.

    uv run python run_episode.py --episode 1 --brain runs/brain.sqlite        # with shared brain
    uv run python run_episode.py --episode 1 --no-brain                       # fresh team, no memory
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sim.env import LabEnv, Gotchas
from llm.crusoe import Crusoe
from brain.ledger import Brain
from agents import team

GOAL = "Put tonight's samples where they belong."


def run_episode(llm: Crusoe, episode: int, seed: int, brain: Brain | None, gotchas: Gotchas | None = None,
                out_dir: str | Path = "runs", log=print) -> dict:
    out_dir = Path(out_dir)
    env = LabEnv(seed=seed, episode=episode, gotchas=gotchas, frames_dir=out_dir / "frames")
    usage0 = llm.usage.totals()
    t0 = time.time()
    briefing = brain.briefing("planner", episode, GOAL) if brain else "SHARED BRAIN: none (this team has no memory)."
    log(f"=== shift {episode} seed {seed} brain={'on' if brain else 'off'}")
    log(briefing)

    dod = team.plan(llm, GOAL, briefing)
    if not dod.get("checks"):
        dod = team.plan(llm, GOAL, briefing, failure_summary="You produced no checks. The operator can see the bench with look(); "
                        "the protocol covers every cap colour. Do not ask what is present; write checks that cover whatever is there.")
    log("DoD: " + json.dumps(dod.get("checks"), indent=None)[:600])
    if dod.get("questions_for_human"):
        log("planner questions: " + json.dumps(dod["questions_for_human"]))

    monitor = team.Monitor()
    result, replans = None, 0
    while True:
        result = team.operate(llm, env, dod, briefing, GOAL, monitor, log=log)
        if result["decision"] == "replan" and replans < 2:
            replans += 1
            dod = team.plan(llm, GOAL, briefing, failure_summary=result["monitor_reason"])
            log("replanned: " + json.dumps(dod.get("checks"))[:300])
            continue
        break

    claim_check = team.claim_vs_trace(result["claim"], env.trace)
    try:
        frame = env.verifier_frame()
    except RuntimeError:
        frame = None
    verdicts = team.verify(llm, dod, frame) if frame else {"verdicts": [], "overall_pass": False, "unavailable": True}
    if verdicts.get("unavailable"):
        result["decision"] = "ping"
        (out_dir / "inbox.md").open("a").write(f"shift {episode}: verifier camera unavailable; need a human to check the bench.\n")
    truth = env.ground_truth()

    claimed = bool(result["claim"] and result["claim"].get("done"))
    rec = {
        "episode": episode, "seed": seed, "brain": brain is not None,
        "success": truth["success"], "claimed_done": claimed, "verifier_pass": verdicts.get("overall_pass", False),
        "false_done": claimed and not truth["success"],
        "verifier_correct": (verdicts.get("overall_pass", False) == truth["success"]) if frame else None,
        "claim_evidence_ok": claim_check["ok"], "decision": result["decision"], "turns": result["turns"], "replans": replans,
        "tool_calls": len(env.trace.calls), "wall_s": round(time.time() - t0, 1),
        "tokens": {k: llm.usage.totals()[k] - usage0[k] for k in ("prompt", "cached", "completion", "calls")},
        "truth": truth, "verdicts": verdicts, "dod": dod, "claim": result["claim"],
    }
    if brain:
        outcome = f"verifier overall_pass={verdicts.get('overall_pass')}; operator decision={result['decision']}"
        deltas = team.curate(llm, env.trace, verdicts, result["claim"], briefing, outcome, episode=episode)
        applied = brain.apply_deltas(deltas, source_agent=f"curator@shift{episode}", episode=episode)
        rec["brain_deltas"] = applied
        rec["brain_stats"] = brain.stats()
        log(f"brain: {len(applied)} deltas -> {brain.stats()}")
    log(f"result: success={rec['success']} claimed={claimed} verifier={rec['verifier_pass']} decision={rec['decision']} "
        f"tools={rec['tool_calls']} tokens={rec['tokens']}")
    (out_dir / "episodes.jsonl").open("a").write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--brain", default="runs/brain.sqlite")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--camera-down", action="store_true", help="on-stage: verifier camera unavailable -> ping")
    a = ap.parse_args()
    llm = Crusoe()
    brain = None if a.no_brain else Brain(a.brain)
    g = Gotchas(verifier_cam_down=a.camera_down)
    run_episode(llm, a.episode, a.seed, brain, g)
