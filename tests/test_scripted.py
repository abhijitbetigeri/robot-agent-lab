"""Scripted checks that run with no LLM. If these pass, the bench is honest and repeatable."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.env import LabEnv, Gotchas  # noqa: E402


def pick_and_place(env, color, dest_xy):
    gt = env.ground_truth()
    x, y = gt["block_xy"][color]
    env.move_above(x, y)
    env.lower()
    r = env.gripper("close")
    if r["holding"] is None:  # sticky gripper: second try
        r = env.gripper("close")
    env.lift()
    env.move_above(*dest_xy)
    env.lower()
    env.gripper("open")
    env.lift()
    return r


def test_reach():
    env = LabEnv(seed=1, episode=1, gotchas=Gotchas(flaky_gripper=False))
    r = env.move_above(0.26, 0.06)
    assert r["ok"], r


def test_grasp():
    env = LabEnv(seed=1, episode=1, gotchas=Gotchas(flaky_gripper=False))
    gt = env.ground_truth()
    x, y = gt["block_xy"]["red"]
    env.move_above(x, y); env.lower()
    r = env.gripper("close")
    assert r["holding"] is True, r
    r = env.lift()
    assert r["holding"] is True, r


def test_sort():
    env = LabEnv(seed=1, episode=1, gotchas=Gotchas(flaky_gripper=False))
    bins = env.ground_truth()["bins_xy"]
    pick_and_place(env, "red", bins["red"])
    pick_and_place(env, "blue", bins["blue"])
    gt = env.ground_truth()
    assert gt["success"], gt


def test_flaky():
    env = LabEnv(seed=1, episode=1, gotchas=Gotchas(flaky_gripper=True))
    gt = env.ground_truth()
    x, y = gt["block_xy"]["red"]
    env.move_above(x, y); env.lower()
    assert env.gripper("close")["holding"] is None
    assert env.gripper("close")["holding"] is True


def test_frames(tmp_path=Path("runs/frames_test")):
    env = LabEnv(seed=1, episode=1, frames_dir=tmp_path)
    p = env.look()["image"]; v = env.verifier_frame()
    assert os.path.getsize(p) < 150_000 and os.path.getsize(v) < 150_000


def test_look():
    """Detector: all tubes and racks within 1 cm of truth; the nudged camera biases detections by ~3 cm in x."""
    import numpy as np
    env = LabEnv(seed=3, episode=1)
    gt, look = env.ground_truth(), env.look()
    assert {t["color"] for t in look["tubes"]} == {"red", "green", "blue"}, look
    assert {r["color"] for r in look["racks"]} == {"red", "blue"}, look
    for t in look["tubes"]:
        assert np.hypot(t["x"] - gt["block_xy"][t["color"]][0], t["y"] - gt["block_xy"][t["color"]][1]) < 0.01, t
    env5 = LabEnv(seed=3, episode=5)
    gt5, look5 = env5.ground_truth(), env5.look()
    bias = [t["x"] - gt5["block_xy"][t["color"]][0] for t in look5["tubes"]]
    assert all(0.02 < abs(b) < 0.04 for b in bias), bias  # camera moved +3 cm in x, so the bench appears shifted -3 cm


def test_bin_shift():
    e1 = LabEnv(seed=1, episode=1); e3 = LabEnv(seed=1, episode=3)
    assert abs(e3.ground_truth()["bins_xy"]["blue"][1] - e1.ground_truth()["bins_xy"]["blue"][1] - 0.08) < 1e-6


if __name__ == "__main__":
    for t in [test_reach, test_grasp, test_sort, test_flaky, test_frames, test_look, test_bin_shift]:
        t(); print("PASS", t.__name__)
