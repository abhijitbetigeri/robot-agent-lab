"""End-to-end plumbing test with a scripted fake LLM: planner -> operator tool loop -> claim check -> verifier -> curator -> brain.
No network, no credits. Run: uv run python tests/test_pipeline_mock.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.crusoe import Usage  # noqa: E402
from brain.ledger import Brain  # noqa: E402
import run_episode  # noqa: E402


class _TC:
    def __init__(self, i, name, args):
        self.id = f"call_{i}"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}


def _resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, model_dump=lambda: {"content": content})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=50)))


class FakeCrusoe:
    """Operator script: park, look, then pick red -> red rack, blue -> blue rack using the look() detections, then claim done."""
    def __init__(self):
        self.usage = Usage()
        self.n = 0
        self.plan_calls = 0
        self.detections = None

    def chat(self, role, messages, **kw):
        self.usage.add(role, _resp().usage)
        if role == "verifier":
            return _resp(json.dumps({"verdicts": [{"id": "C1", "pass": True, "reason": "red on red patch"},
                                                  {"id": "C2", "pass": True, "reason": "blue on blue patch"},
                                                  {"id": "C3", "pass": True, "reason": "green untouched"}], "overall_pass": True}))
        if role != "operator":
            raise AssertionError(role)
        # pull the latest look() result out of the tool messages
        for m in reversed(messages):
            if m.get("role") == "tool" and '"tubes"' in m["content"]:
                self.detections = json.loads(m["content"])
                break
        self.n += 1
        prog = json.dumps({"progress": [min(1, self.n / 12), min(1, self.n / 10)], "infeasible": False})
        if self.n == 1:
            return _resp("Parking then looking. " + prog, [_TC(1, "park", {}), _TC(2, "look", {})])
        d = self.detections
        tube = {t["color"]: (t["x"], t["y"]) for t in d["tubes"]}
        rack = {r["color"]: (r["x"], r["y"]) for r in d["racks"]}
        seq = []
        for c in ("red", "blue"):
            x, y = tube[c]; rx, ry = rack[c]
            seq += [("move_above", {"x": x, "y": y}), ("lower", {}), ("gripper", {"cmd": "close"}), ("gripper", {"cmd": "close"}),
                    ("lift", {}), ("move_above", {"x": rx, "y": ry}), ("lower", {}), ("gripper", {"cmd": "open"}), ("lift", {})]
        step = self.n - 2
        if step < len(seq):
            name, args = seq[step]
            return _resp(f"{name}. " + prog, [_TC(10 + step, name, args)])
        if step == len(seq):
            return _resp("Parking for the final photo. " + prog, [_TC(99, "park", {})])
        ev = {"C1": [3, 7], "C2": [12, 16], "C3": [2]}
        return _resp(json.dumps({"done": True, "evidence": ev, "notes": "gripper needed two close commands", "progress": [1, 1], "infeasible": False}))

    def json(self, role, system, user, **kw):
        self.usage.add(role, _resp().usage)
        if role == "planner":
            self.plan_calls += 1
            self.n = 0  # new shift
            return {"checks": [{"id": "C1", "check": "red tube rests inside the red patch", "how": "frame"},
                               {"id": "C2", "check": "blue tube rests inside the blue patch", "how": "frame"},
                               {"id": "C3", "check": "green tube is on the bench outside both patches", "how": "frame"},
                               {"id": "C4", "check": "jaws open and empty at the end", "how": "state"}],
                    "plan": ["park", "look", "red -> red rack", "blue -> blue rack", "park", "claim"], "questions_for_human": []}
        if role == "curator":
            return {"deltas": [{"op": "add", "tier": "procedure", "subject": "gripper", "relation": "needs_retry_on_close", "object": "true", "evidence_ref": "call#5", "confidence": 0.8},
                               {"op": "add", "tier": "fact", "subject": "bin_blue", "relation": "position_xy", "object": "[0.22,-0.17]", "evidence_ref": "call#2", "confidence": 0.9}]}
        raise AssertionError(role)


def test_pipeline():
    llm = FakeCrusoe()
    brain = Brain(":memory:")
    rec = run_episode.run_episode(llm, episode=1, seed=5, brain=brain, out_dir="runs/mock", log=lambda s: None)
    assert rec["success"], rec["truth"]
    assert rec["claimed_done"] and rec["verifier_pass"] and rec["claim_evidence_ok"], rec
    assert rec["tokens"]["calls"] > 10 and rec["tokens"]["cached"] > 0
    assert brain.stats()["current"] >= 2, brain.stats()
    # second shift reads the briefing; a moved rack supersedes the fact
    rec2 = run_episode.run_episode(llm, episode=3, seed=6, brain=brain, out_dir="runs/mock", log=lambda s: None)
    assert rec2["success"]
    assert llm.plan_calls == 2
    print("PASS test_pipeline", {k: rec[k] for k in ("success", "tool_calls", "turns", "decision")}, brain.stats())


if __name__ == "__main__":
    test_pipeline()
