"""The agent team: planner, operator, verifier, monitor, curator. Each is a small function over the Crusoe client."""
from __future__ import annotations

import json
import re
from collections import Counter

from llm.crusoe import Crusoe, content_of, parse_json, image_part
from agents import prompts as P


# ------------------------------------------------------------------------------------------------ planner
def plan(llm: Crusoe, goal: str, briefing: str, failure_summary: str | None = None) -> dict:
    user = f"GOAL FROM HUMAN: {goal}\n\n{briefing}"
    if failure_summary:
        user += f"\n\nPREVIOUS ATTEMPT FAILED: {failure_summary}\nWrite a revised plan; keep check ids stable where the check is unchanged."
    out = llm.json("planner", P.PLANNER_SYSTEM, user)
    out.setdefault("checks", [])
    out.setdefault("questions_for_human", [])
    return out


# ------------------------------------------------------------------------------------------------ operator
def _dod_text(dod: dict) -> str:
    return "DEFINITION OF DONE:\n" + "\n".join(f"- {c['id']}: {c['check']} [{c.get('how', 'frame')}]" for c in dod["checks"]) \
        + "\nPLAN:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(dod.get("plan", [])))


def _progress_of(text: str) -> tuple[list | None, bool]:
    try:
        j = parse_json(text[text.rfind("{\"progress"):] if "{\"progress" in text else text)
        return j.get("progress"), bool(j.get("infeasible", False))
    except Exception:  # noqa: BLE001
        return None, False


HARNESS_TOOLS = [
    {"type": "function", "function": {"name": "report_progress",
        "description": "Report your honest bounds on how complete the job is, after every action. low/high in 0..1. Set infeasible=true if the goal cannot be met.",
        "parameters": {"type": "object", "properties": {"low": {"type": "number"}, "high": {"type": "number"}, "infeasible": {"type": "boolean"}},
                       "required": ["low", "high"]}}},
    {"type": "function", "function": {"name": "claim_done",
        "description": "Declare the job finished. evidence maps each check id to the call_ids of tool results that prove it. notes: what surprised you, for the shared brain.",
        "parameters": {"type": "object", "properties": {"evidence": {"type": "object"}, "notes": {"type": "string"}}, "required": ["evidence"]}}},
]
OBSERVE_ONLY = {"park", "look", "scene_state", "report_progress"}


def operate(llm: Crusoe, env, dod: dict, briefing: str, goal: str, monitor, max_turns: int = 45, log=print) -> dict:
    """Run the operator loop. Returns {claim, turns, decision, monitor_reason}."""
    tools = env.tool_schemas() + HARNESS_TOOLS
    messages = [{"role": "system", "content": P.OPERATOR_SYSTEM},
                {"role": "user", "content": f"GOAL: {goal}\n\n{_dod_text(dod)}\n\n{briefing}\n\nBegin. Start with look()."}]
    claim, decision, reason = None, "continue", ""
    progress, infeasible = None, False
    for turn in range(1, max_turns + 1):
        progress, infeasible = None, False
        resp = llm.chat("operator", messages, tools=tools)
        msg = resp.choices[0].message
        text = msg.content or ""
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": text or None,
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if name == "report_progress":
                        progress = [float(args.get("low", 0)), float(args.get("high", 0))]
                        infeasible = bool(args.get("infeasible", False))
                        result = {"ok": True, "recorded": progress}
                        env.trace.add(name, args, result)
                    elif name == "claim_done":
                        claim = {"done": True, "evidence": args.get("evidence") or {}, "notes": args.get("notes", "")}
                        result = {"ok": True, "recorded": "claim"}
                        env.trace.add(name, args, result)
                    else:
                        result = env.call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = {"ok": False, "error": str(e)[:200]}
                    env.trace.add(name, {"raw": tc.function.arguments}, result)
                log(f"  [{turn}] {name}({tc.function.arguments}) -> {json.dumps(result)[:160]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            if claim:
                break
        else:
            messages.append({"role": "assistant", "content": text})
            log(f"  [{turn}] operator: {text[:200]}")
            if '"done"' in text:
                try:
                    claim = parse_json(text)
                    if claim.get("done"):
                        break
                except Exception:  # noqa: BLE001
                    pass
            messages.append({"role": "user", "content": "Continue. Use tools. Call claim_done when every check holds."})
            p2, i2 = _progress_of(text)
            progress, infeasible = (p2 or progress), (i2 or infeasible)
        decision, reason = monitor(turn, env.trace, progress, infeasible)
        if decision != "continue":
            log(f"  monitor -> {decision}: {reason}")
            break
    return {"claim": claim, "turns": turn, "decision": decision, "monitor_reason": reason}


# ------------------------------------------------------------------------------------------------ monitor (rules first)
class Monitor:
    """Deterministic rules (DeLoop loop rule, BAGEN interval estimate, budget). The LLM only writes the replan summary."""
    def __init__(self, max_turns: int = 45, stall_turns: int = 6):
        self.max_turns, self.stall_turns = max_turns, stall_turns
        self.progress_hist: list[list] = []
        self.last_ncalls = 0
        self.stall = 0
        self.replans = 0

    def __call__(self, turn: int, trace, progress, infeasible) -> tuple[str, str]:
        calls = trace.calls
        if infeasible:
            return "stop", "operator reports infeasible"
        acting = [c for c in calls if c["tool"] not in OBSERVE_ONLY]
        if len(acting) >= 2:
            a, b = acting[-1], acting[-2]
            if a["tool"] == b["tool"] and a["args"] == b["args"] and a["result"].get("ok") == b["result"].get("ok") \
                    and a["result"].get("holding") == b["result"].get("holding") and a["tool"] != "gripper":
                self.replans += 1
                return ("stop" if self.replans > 2 else "replan"), f"loop: {a['tool']} {a['args']} repeated with same result"
        if len(calls) >= 8 and all(c["tool"] in OBSERVE_ONLY for c in calls[-8:]):
            self.replans += 1
            return ("stop" if self.replans > 2 else "replan"), "observing without acting for 8 calls; last grasp did not hold"
        if progress:
            self.progress_hist.append(progress)
            if len(self.progress_hist) >= 4 and progress[1] < self.progress_hist[-4][0]:
                return "replan", "progress bounds collapsed"
        self.stall = self.stall + 1 if len(calls) == self.last_ncalls else 0
        self.last_ncalls = len(calls)
        if self.stall >= self.stall_turns:
            return "replan", "no tool use for 6 turns"
        if turn >= self.max_turns:
            return "stop", "turn budget exhausted"
        return "continue", ""


# ------------------------------------------------------------------------------------------------ verifier (sealed)
def verify(llm: Crusoe, dod: dict, frame_path: str | None, samples: int = 3) -> dict:
    if frame_path is None:
        return {"verdicts": [], "overall_pass": False, "unavailable": True}
    checks = "\n".join(f"- {c['id']}: {c['check']}" for c in dod["checks"] if c.get("how", "frame") == "frame")
    msgs = [{"role": "system", "content": P.VERIFIER_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": f"CHECKS:\n{checks}\n\nDecide from this photo."},
                                         image_part(frame_path, max_side=512)]}]
    votes: dict[str, list] = {}
    reasons: dict[str, str] = {}
    for _ in range(samples):
        try:
            out = parse_json(content_of(llm.chat("verifier", msgs, max_tokens=2000)))
        except Exception as e:  # noqa: BLE001
            out = {"verdicts": [], "error": str(e)[:100]}
        for v in out.get("verdicts", []):
            votes.setdefault(v["id"], []).append(bool(v.get("pass")))
            reasons[v["id"]] = v.get("reason", "")
    verdicts = [{"id": cid, "pass": Counter(vs).most_common(1)[0][0], "votes": vs, "reason": reasons.get(cid, "")}
                for cid, vs in votes.items()]
    overall = bool(verdicts) and all(v["pass"] for v in verdicts)
    return {"verdicts": verdicts, "overall_pass": overall}


def claim_vs_trace(claim: dict | None, trace) -> dict:
    """OverclaimBench-style check: every cited call id must exist and be a success; every frame check needs evidence."""
    if not claim:
        return {"ok": False, "reason": "no done claim"}
    ids = {c["call_id"]: c for c in trace.calls}
    missing, failed = [], []
    for cid, refs in (claim.get("evidence") or {}).items():
        if not isinstance(refs, list):
            refs = [refs]
        for r in refs or []:
            if isinstance(r, str):
                m = re.search(r"(\d+)", r)
                r = int(m.group(1)) if m else r
            if r not in ids:
                missing.append(r)
            elif ids[r]["result"].get("ok") is False:
                failed.append(r)
    return {"ok": not missing and not failed, "missing_call_ids": missing, "failed_call_ids": failed,
            "checks_cited": len(claim.get("evidence") or {})}


# ------------------------------------------------------------------------------------------------ curator
def auto_facts(trace, verdicts: dict, claim: dict | None, episode: int) -> list[dict]:
    """Facts the harness can derive without a model. These are the ones that must be right."""
    out = []
    calls = trace.calls
    # rack positions as observed by the camera (deterministic; a moved rack supersedes the old row)
    for c in reversed(calls):
        if c["tool"] == "look" and c["result"].get("racks"):
            for r in c["result"]["racks"]:
                out.append({"op": "add", "tier": "fact", "subject": f"bin_{r['color']}", "relation": "position_xy",
                            "object": f"[{r['x']:.3f}, {r['y']:.3f}]", "evidence_ref": f"call#{c['call_id']}", "confidence": 0.95})
            break
    # sticky gripper: a close that reported holding=null followed by a close that held
    nulls = [c for c in calls if c["tool"] == "gripper" and c["args"].get("cmd") == "close" and c["result"].get("holding") is None]
    helds = [c for c in calls if c["tool"] == "gripper" and c["args"].get("cmd") == "close" and c["result"].get("holding") is True]
    if nulls:
        obj = "first close often reports holding=null and does NOT grip; close a second time and check holding=true" if helds \
            else "close reported holding=null; the tube was not gripped. Do not lift or place until holding=true"
        out.append({"op": "add", "tier": "procedure", "subject": "gripper", "relation": "close_behaviour", "object": obj,
                    "evidence_ref": f"call#{nulls[0]['call_id']}", "confidence": 0.9})
    # overclaim: operator claimed done but the verifier failed
    if claim and verdicts.get("verdicts") and not verdicts.get("overall_pass"):
        failed = [v["id"] for v in verdicts["verdicts"] if not v["pass"]]
        out.append({"op": "add", "tier": "insight", "subject": "operator", "relation": "overclaim",
                    "object": f"shift {episode}: claimed done but verifier failed {failed}; verify holding=true before placing and park+look before claiming",
                    "evidence_ref": "verifier", "confidence": 0.85})
    return out


def curate(llm: Crusoe, trace, verdicts: dict, claim: dict | None, briefing: str, truth_free_outcome: str, episode: int = 0, max_llm_deltas: int = 4) -> list[dict]:
    deltas = auto_facts(trace, verdicts, claim, episode)
    log = "\n".join(f"call#{c['call_id']} {c['tool']}({json.dumps(c['args'])}) -> {json.dumps(c['result'])[:140]}" for c in trace.calls)
    user = (f"SHIFT LOG:\n{log}\n\nOPERATOR NOTES: {(claim or {}).get('notes', '')}\n\nVERIFIER: {json.dumps(verdicts)}\n\n"
            f"OUTCOME: {truth_free_outcome}\n\nBRIEFING THE TEAM HAD:\n{briefing}\n\n"
            f"Already recorded automatically (do not repeat): {json.dumps(deltas)}\n"
            f"Emit at most {max_llm_deltas} deltas: lessons that would have saved this shift, equipment quirks, and up/down votes on briefing entries. "
            "Never record tube positions (they change every shift) or restate what a tool does.")
    try:
        out = llm.json("curator", P.CURATOR_SYSTEM, user, max_tokens=1500)
        extra = [d for d in out.get("deltas", []) if d.get("op") in ("add", "upvote", "downvote")][:max_llm_deltas]
        extra = [d for d in extra if not (d.get("op") == "add" and "tube" in str(d.get("subject", "")).lower() and "position" in str(d.get("relation", "")).lower())]
        deltas += extra
    except Exception as e:  # noqa: BLE001
        deltas.append({"op": "add", "tier": "insight", "subject": "curator", "relation": "error", "object": str(e)[:80], "confidence": 0.1})
    return deltas
