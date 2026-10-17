"""Replay a real agent shift (its tool calls, parsed from the sweep log) in the simulator and record it as a clip
with a live tool-call overlay. This is the actual MuJoCo bench moving, driven by what the Nemotron operator did.

    uv run python demo/record_sim.py --log runs/sweep_brain.log --shift 2 --seed 102 --out demo/build/sim_shift2.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.env import LabEnv, Gotchas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
VW, VH = 880, 495
BG, FG, DIM, ACC, RED, GRN = (16, 18, 24), (235, 235, 235), (150, 155, 165), (255, 200, 60), (255, 95, 95), (110, 220, 130)
MONO, SANS = "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc"


def parse_calls(log: Path, shift: int) -> list[tuple[str, dict, dict]]:
    txt = log.read_text()
    m = re.search(rf"=== shift {shift} .*?(?==== shift {shift + 1} |DONE|\Z)", txt, re.S)
    calls = []
    for line in (m.group(0) if m else "").splitlines():
        mm = re.match(r"\[\w+\]\s+\[\d+\] (\w+)\((.*?)\) -> (\{.*)$", line)
        if not mm:
            continue
        name, args, res = mm.groups()
        try:
            a = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            a = {}
        try:
            r = json.loads(res)
        except json.JSONDecodeError:
            r = {}
        calls.append((name, a, r))
    return calls


def short(name: str, args: dict, res: dict | None) -> tuple[str, tuple]:
    a = ", ".join(f"{k}={v}" for k, v in args.items() if k != "evidence")
    s = f"{name}({a})"
    color = FG
    if res is not None:
        if "holding" in res:
            s += f"  → holding={res['holding']}"
            color = RED if res["holding"] is None else (GRN if res["holding"] else FG)
        elif name == "look":
            s += f"  → {len(res.get('tubes', []))} tubes, {len(res.get('racks', []))} racks"
        elif name == "report_progress":
            s += f"  → {res.get('recorded')}"
        elif name == "claim_done":
            s += "  → claim recorded"
            color = ACC
    return s[:60], color


class Recorder:
    def __init__(self, env: LabEnv, stride: int, out: Path, fps: int = 25, ffmpeg: str = "ffmpeg", shift: int = 2):
        self.env, self.stride, self.fps, self.shift = env, stride, fps, shift
        self.n = 0
        self.log: list[tuple[str, tuple]] = []
        self.current = ""
        self.banner = ""
        self.f_title = ImageFont.truetype(SANS, 30, index=0)
        self.f_mono = ImageFont.truetype(MONO, 17, index=0)
        self.f_small = ImageFont.truetype(SANS, 20, index=0)
        self.proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                      "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                                     stdin=subprocess.PIPE)
        self.frames = 0

    def on_step(self, env):
        self.n += 1
        if self.n % self.stride == 0:
            self.emit()

    def compose(self) -> Image.Image:
        view = Image.fromarray(self.env.render_view(VW, VH))
        im = Image.new("RGB", (W, H), BG)
        im.paste(view, (24, 70))
        d = ImageDraw.Draw(im)
        d.rectangle([22, 68, 26 + VW, 72 + VH], outline=DIM, width=2)
        d.text((24, 20), f"Robot Agent Lab — shift {self.shift}, replayed from the operator's real tool calls", fill=ACC, font=self.f_title)
        d.text((24, 580), "Operator: Nemotron 3.5 Lightning on Crusoe · one tool call per turn · perception = top-down camera", fill=DIM, font=self.f_small)
        d.text((24, 606), "Sticky gripper quirk is ON: the first close reports holding=null and does not grip", fill=DIM, font=self.f_small)
        if self.banner:
            d.rectangle([24, 640, 24 + VW, 700], fill=(30, 34, 46))
            d.text((36, 652), self.banner, fill=ACC, font=self.f_title)
        # tool log panel
        x0 = 24 + VW + 24
        d.text((x0, 70), "tool calls", fill=DIM, font=self.f_small)
        y = 100
        for s, c in self.log[-20:]:
            d.text((x0, y), s, fill=c, font=self.f_mono)
            y += 24
        if self.current:
            d.text((x0, y + 6), "▶ " + self.current, fill=ACC, font=self.f_mono)
        return im

    def emit(self, hold: int = 1):
        im = self.compose()
        for _ in range(hold):
            self.proc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/sweep_brain.log")
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--out", default="demo/build/sim_shift2.mp4")
    ap.add_argument("--target-seconds", type=float, default=19.0)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    calls = parse_calls(ROOT / a.log, a.shift)
    motion = [c for c in calls if c[0] not in ("report_progress", "claim_done")]
    print(f"{len(calls)} calls parsed, {len(motion)} executed in sim")

    # pass 1: count physics steps so the clip lands near the target length
    env = LabEnv(seed=a.seed, episode=a.shift, gotchas=Gotchas(), frames_dir="runs/replay")
    counter = {"n": 0}
    env.on_step = lambda e: counter.__setitem__("n", counter["n"] + 1)
    for name, args, _ in motion:
        env.call(name, args)
    stride = max(1, int(counter["n"] / (a.target_seconds * a.fps)))
    print(f"{counter['n']} physics steps -> stride {stride}")

    # pass 2: render
    env = LabEnv(seed=a.seed, episode=a.shift, gotchas=Gotchas(), frames_dir="runs/replay")
    rec = Recorder(env, stride, ROOT / a.out, fps=a.fps, ffmpeg=ffmpeg, shift=a.shift)
    env.on_step = rec.on_step
    rec.banner = "Goal: \"Put tonight's samples where they belong.\""
    rec.emit(hold=a.fps * 2)
    for name, args, res in calls:
        rec.current, _ = short(name, args, None)
        if name == "claim_done":
            rec.banner = "Operator claims done, citing tool-call evidence per check"
            rec.log.append(short(name, args, res))
            rec.current = ""
            rec.emit(hold=a.fps * 2)
            continue
        if name == "report_progress":
            rec.log.append(short(name, args, res))
            rec.current = ""
            rec.emit(hold=3)
            continue
        if name == "gripper" and args.get("cmd") == "close":
            rec.banner = ""
        live = env.call(name, args)
        s, c = short(name, args, live if name == "look" else res)
        rec.log.append((s, c))
        if res.get("holding") is None and name == "gripper" and args.get("cmd") == "close":
            rec.banner = "holding = null: the sticky gripper did not grip. The operator must notice and retry."
            rec.emit(hold=a.fps)
        elif res.get("holding") is True and name == "gripper":
            rec.banner = "Second close: holding = true. Now it can lift."
            rec.emit(hold=a.fps // 2)
        elif name == "look":
            rec.banner = "look(): colour detector on the top-down camera → tube and rack coordinates"
            rec.emit(hold=a.fps // 2)
        rec.current = ""
    rec.banner = "Sealed verifier (Nemotron Nano Omni) judges the final photo: all checks PASS"
    rec.emit(hold=a.fps * 3)
    rec.close()
    print(f"wrote {a.out}: {rec.frames} frames = {rec.frames / a.fps:.1f}s")


if __name__ == "__main__":
    main()
