"""Build the narrated demo: use case -> architecture -> the real simulation running -> verification -> shared brain -> results.

    uv run python demo/record_sim.py          # the simulation clip (shift 2 replayed from the operator's real tool calls)
    uv run python demo/make_video.py          # -> demo/robot-agent-lab-demo.mp4
macOS `say` for narration; ffmpeg from imageio-ffmpeg. Every number and frame comes from runs/ (real output).
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "demo" / "build"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
BG, FG, DIM, ACC, RED, GRN, BLU = (247, 248, 250), (28, 31, 38), (110, 116, 128), (36, 81, 214), (204, 52, 52), (28, 140, 76), (36, 81, 214)
MONO, SANS = "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc"
VOICE, RATE = "Samantha", 180


def font(path, size):
    return ImageFont.truetype(path, size, index=0)


def _wrap(s: str, n: int) -> list[str]:
    out, cur = [], ""
    for w in s.split(" "):
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + ([cur] if cur else [])


def slide(title: str, lines=None, image: Path | Image.Image | None = None, mono=True, footer="", img_caption="", img_w=500):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 36), title, fill=ACC, font=font(SANS, 40))
    y = 110
    f = font(MONO if mono else SANS, (20 if image is not None else 22) if mono else 27)
    for text, color in (lines or []):
        chunks = [text[:58]] if (image is not None and mono) else _wrap(text, 44 if image is not None else 80)
        for chunk in chunks:
            d.text((48, y), chunk, fill=color, font=f)
            y += 28 if mono else 36
    if image is not None:
        img = (Image.open(image) if isinstance(image, Path) else image).convert("RGB")
        img.thumbnail((img_w, 520))
        im.paste(img, (W - img.width - 48, 110))
        d.rectangle([W - img.width - 50, 108, W - 46, 112 + img.height], outline=DIM, width=2)
        if img_caption:
            d.text((W - img.width - 48, 118 + img.height), img_caption, fill=DIM, font=font(SANS, 18))
    if footer:
        d.text((48, H - 48), footer, fill=DIM, font=font(SANS, 20))
    return im


# ------------------------------------------------------------------ use-case picture: the bench, labelled
def bench_picture() -> Image.Image:
    from sim.env import LabEnv, Gotchas
    env = LabEnv(seed=102, episode=1, gotchas=Gotchas(flaky_gripper=False), frames_dir=OUT)
    img = Image.fromarray(env.render_view(1120, 840, lookat=(0.21, 0.0, 0.02), distance=0.48, azimuth=150.0, elevation=-42.0))
    d = ImageDraw.Draw(img)
    f = font(SANS, 26)
    legend = [("red tube  -> cold rack (red patch)", RED), ("blue tube -> sequencer rack (blue patch)", BLU), ("green tube = control, stays put", GRN)]
    y = 30
    for t, c in legend:
        tw = d.textlength(t, font=f)
        d.rectangle([24, y - 4, 36 + tw, y + 32], fill=(255, 255, 255))
        d.text((30, y), t, fill=c, font=f)
        y += 40
    return img


# ------------------------------------------------------------------ architecture diagram
def architecture() -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 36), "How it works", fill=ACC, font=font(SANS, 40))
    ft, fs = font(SANS, 22), font(SANS, 17)

    def box(x, y, w, h, title, sub, color=FG, fill=(255, 255, 255)):
        d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=color, width=2)
        d.text((x + 14, y + 10), title, fill=color, font=ft)
        yy = y + 40
        for line in sub:
            d.text((x + 14, yy), line, fill=DIM, font=fs)
            yy += 22

    def arrow(x1, y1, x2, y2, color=DIM, label=""):
        d.line([x1, y1, x2, y2], fill=color, width=3)
        v = np.array([x2 - x1, y2 - y1], float); v /= (np.linalg.norm(v) + 1e-9)
        p = np.array([-v[1], v[0]])
        tip = np.array([x2, y2]); a = tip - 14 * v + 7 * p; b = tip - 14 * v - 7 * p
        d.polygon([tuple(tip), tuple(a), tuple(b)], fill=color)
        if label:
            d.text(((x1 + x2) / 2 + 8, (y1 + y2) / 2 - 10), label, fill=color, font=fs)

    box(48, 100, 250, 110, "Human", ["types one line, walks away:", "\"Put tonight's samples", " where they belong.\""], ACC)
    box(370, 100, 300, 110, "Planner  (Nemotron 3 Super)", ["writes the DEFINITION OF DONE:", "checks decidable from a photo", "or from robot state"], FG)
    box(370, 265, 300, 115, "Operator  (Nemotron 3.5 Lightning)", ["six tools: park, look, move_above,", "lower, lift, gripper", "claims done with evidence per check"], FG)
    box(760, 265, 470, 115, "Simulated bench  (MuJoCo, SO-101)", ["tubes, racks, two cameras", "quirks: sticky gripper, rack moves, camera nudge", "ground truth -> eval only"], FG)
    box(760, 435, 470, 110, "Verifier  (Nemotron Nano Omni)  SEALED", ["sees only the checks + one top-down photo,", "never the operator; 3 votes", "accuracy measured vs ground truth"], GRN)
    box(370, 435, 300, 110, "Monitor  (rules + Nano)", ["loop / stall / progress collapse /", "budget / camera down ->", "continue | replan | stop | ping"], RED)
    box(48, 595, 620, 110, "Shared Brain  (SQLite ledger)", ["(subject, relation, object) + valid_from/valid_to + provenance", "same key, new value -> old row superseded, never deleted", "read before every shift; Curator (Nano) writes deltas after"], BLU, fill=(236, 241, 252))
    box(760, 595, 470, 110, "Improver  (Version N+1)", ["rewrites the operator playbook from shift history", "kept ONLY if it scores >= on a held-out seed", "the team has never run"], ACC)
    arrow(298, 155, 370, 155, DIM); d.text((306, 128), "goal", fill=DIM, font=fs)
    arrow(520, 210, 520, 265, DIM); d.text((530, 228), "definition of done", fill=DIM, font=fs)
    arrow(670, 300, 760, 300, DIM); d.text((678, 244), "tool calls", fill=DIM, font=fs)
    arrow(760, 368, 670, 368, DIM); d.text((668, 384), "observations", fill=DIM, font=fs)
    arrow(995, 380, 995, 435, DIM); d.text((1005, 398), "one photo", fill=DIM, font=fs)
    arrow(520, 380, 520, 435, DIM); d.text((530, 398), "tool log", fill=DIM, font=fs)
    arrow(760, 490, 670, 490, GRN); d.text((690, 500), "verdicts", fill=GRN, font=fs)
    arrow(358, 595, 450, 545, BLU); d.text((190, 566), "briefing in, deltas out", fill=BLU, font=fs)
    arrow(995, 545, 995, 595, ACC); d.text((1005, 560), "shift records", fill=ACC, font=fs)
    d.text((48, 250), "Crusoe Managed Inference serves", fill=DIM, font=fs)
    d.text((48, 272), "every model. The operator runs in a", fill=DIM, font=fs)
    d.text((48, 294), "NemoClaw / OpenShell sandbox.", fill=DIM, font=fs)
    d.text((48, 345), "Four properties, one loop:", fill=FG, font=ft)
    for i, t in enumerate(["collaborate through shared memory", "verify independently", "detect failures", "improve autonomously"]):
        d.text((60, 380 + 26 * i), f"{i + 1}. {t}", fill=FG, font=fs)
    return im


def log_lines(path: Path, shift: int, wanted: tuple[str, ...], limit: int = 9):
    txt = path.read_text()
    m = re.search(rf"=== shift {shift} .*?(?==== shift {shift + 1} |DONE|\Z)", txt, re.S)
    out = []
    for line in (m.group(0) if m else "").splitlines():
        line = line.replace("[brain] ", "").rstrip()
        if line.startswith(("DoD:", "- [", "SHARED BRAIN", "brain:", "result:", "replanned")):
            continue
        if any(w in line for w in wanted):
            mm = re.match(r"\s*\[(\d+)\] (\w+)\((.*?)\) -> (\{.*\})", line)
            if not mm:
                continue
            n, tool, args, res = mm.groups()
            try:
                r = json.loads(res)
            except Exception:  # noqa: BLE001
                r = {}
            a = re.sub(r'[{}"]', "", args).replace("cmd:", "")
            held = {None: "nothing held", True: "tube held", False: "empty"}.get(r.get("holding"), "")
            txt2 = f"[{n:>2}] {tool}({a})  ->  {held}" if "holding" in r else f"[{n:>2}] {tool}({a})"
            color = RED if r.get("holding") is None and tool == "gripper" and "close" in a else (GRN if r.get("holding") else FG)
            out.append((txt2[:58], color))
        if len(out) >= limit:
            break
    return out


def say(text: str, name: str) -> tuple[Path, float]:
    wav = OUT / f"{name}.wav"
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "--data-format=LEI16@22050", "-o", str(wav), text], check=True)
    with wave.open(str(wav)) as w:
        return wav, w.getnframes() / w.getframerate()


def still_segment(ffmpeg, idx, im, wav, dur):
    png = OUT / f"s{idx}.png"; im.save(png)
    mp4 = OUT / f"s{idx}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(png), "-i", str(wav),
                    "-t", f"{dur + 0.7:.2f}", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", str(mp4)], check=True)
    return mp4


def clip_segment(ffmpeg, idx, clip: Path, wav, dur):
    """Video clip + narration. If the clip is shorter than the narration, hold the last frame; if longer, the clip plays out."""
    mp4 = OUT / f"s{idx}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(clip), "-i", str(wav),
                    "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={dur + 2:.1f}[v];[1:a]apad[a]", "-map", "[v]", "-map", "[a]",
                    "-t", f"{max(dur + 0.7, clip_seconds(ffmpeg, clip)):.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(mp4)], check=True)
    return mp4


def clip_seconds(ffmpeg, clip: Path) -> float:
    out = subprocess.run([ffmpeg, "-i", str(clip)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def main():
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    log = ROOT / "runs" / "sweep_brain.log"
    frames = ROOT / "runs" / "brain" / "frames"
    db = sqlite3.connect(ROOT / "runs" / "brain" / "brain.sqlite")
    pb = json.loads((ROOT / "eval" / "playbook_v1.json").read_text())["history"][0]
    sim_clip = OUT / "sim_shift2.mp4"
    assert sim_clip.exists(), "run demo/record_sim.py first"

    segs: list[tuple[str, object, str]] = []   # (kind, image-or-clip, narration)

    # 1. use case
    segs.append(("still", slide("Robot Agent Lab — the problem", [
        ("A small science lab. Sample tubes arrive all day with coloured caps.", FG),
        ("Before the night runs, every tube must be in the right place:", FG),
        ("   red cap  -> cold rack", RED), ("   blue cap -> sequencer rack", BLU), ("   green cap = control, must stay on the bench", GRN), ("", FG),
        ("Today a technician does this twice a night.", FG),
        ("A tube in the wrong rack ruins a run that costs more than the robot.", FG), ("", FG),
        ("Can a team of AI agents run that night shift with nobody watching?", ACC)], mono=False, image=bench_picture(), img_w=600,
        img_caption="the simulated bench (MuJoCo, SO-101 arm)", footer="Recursive Agents Hackathon · Crusoe x Cerebral Valley · Oct 17 2026"),
        "Robot Agent Lab. The problem: a small science lab. Sample tubes arrive all day with coloured caps. Before the night runs, "
        "red caps go to the cold rack, blue caps to the sequencer rack, and the green control stays on the bench. "
        "A tube in the wrong rack ruins an expensive run. Can a team of agents run the night shift with nobody watching?"))

    # 2. architecture
    segs.append(("still", architecture(),
        "How it works. A human types one line. A planner turns it into a definition of done: checks decidable from a photo or from robot state. "
        "An operator drives the arm through six tools. A sealed verifier sees only the checks and one photo, never the operator. "
        "A monitor catches loops and stalls and decides: continue, re-plan, stop, or ping a human. "
        "Underneath, a shared brain every shift reads first and a curator updates after. "
        "And an improver that rewrites the team's playbook, kept only if it scores better on a shift it has never seen. "
        "All Nemotron models, all on Crusoe."))

    # 3. the simulation, running
    segs.append(("clip", sim_clip,
        "This is the actual simulation, replaying shift two from the operator's real tool calls. "
        "It looks through the camera, moves above the red tube, lowers, closes. The sticky gripper reports no hold, "
        "so it closes again, holding is true, and it carries the tube to the cold rack. Then the blue tube to the sequencer rack. "
        "The green control is left alone. It parks for a clear photo and claims done with evidence. The sealed verifier passes it."))

    # 4. why the verifier exists: shift 1 false done
    l1 = log_lines(log, 1, ("gripper", "] lift"), limit=8)
    segs.append(("still", slide("Shift 1: the robot says it is done. It is not.", [("what the robot actually did (real log):", DIM)] + l1 +
                 [("", FG), ("robot: \"done\"", ACC), ("independent checker: FAIL", RED), ("  the red tube is not in the red rack", RED)],
                 image=frames / "ep01_ver_038.jpg", img_caption="the photo the checker judged from", footer="runs/sweep_brain.log, shift 1"),
        "Shift one, with empty memory. The gripper reports nothing held. The robot lifts nothing, places nothing, and says it is done anyway. "
        "An independent checker that only sees the photo says no. Agents overclaim. That is why the check is independent."))

    # 5. the brain learns and supersedes
    hist = db.execute("select id,object,valid_from,valid_to from facts where subject='bin_blue' and relation='position_xy' order by id").fetchall()
    rows = db.execute("select tier,subject,relation,object from facts where valid_from<=2 and (valid_to is null or valid_to>2) and tier<>'fact' order by id").fetchall()
    brief = [(f"[{t[0]}] {s} {r}: {str(o)}"[:58], ACC) for t, s, r, o in rows[:3]]
    hl = [("blue rack position      valid from -> until", DIM)]
    for i, o, vf, vt in hist[:1] + hist[-3:]:
        hl.append((f"  #{i:<3} {str(o):16s} shift {vf}  -> {('shift ' + str(vt)) if vt else 'CURRENT'}", GRN if vt is None else DIM))
    segs.append(("still", slide("The shared memory: learn once, update when the bench changes",
                 [("notes the next shift read before starting:", DIM)] + brief + [("", FG), ("the blue rack moved at shift 3. Memory of its position:", DIM)] + hl +
                 [("", FG), ("old position handed to an agent after the move: never", GRN)],
                 image=frames / "ep03_ver_035.jpg", img_caption="shift 3: rack moved, tube placed correctly", footer="brain/ledger.py · Temporal Validity (2606.26511) · Zep (2501.13956) · ACE (2510.04618)"),
        "After every shift, notes go into a shared memory. Shift two read the gripper lesson and succeeded. "
        "At shift three the rack moved, and the memory replaced the old position. Nothing is deleted; the old fact just stops being used."))

    # 6. results + improvement
    segs.append(("still", slide("Measured, and self-improving", [
        ("4 shifts with memory vs 4 without, same starting conditions", DIM),
        ("  tubes placed correctly in 3 of 4 shifts either way; every false 'done' was caught", FG),
        ("  the checker itself was right 75% of the time, measured against the simulator", FG), ("", FG),
        ("The team rewrote its own playbook, then had to prove it:", DIM),
        (f"  on a shift it had never seen:  old score {pb['score_old']}   ->   new score {pb['score_new']}   kept", GRN),
        ("  old playbook: claimed done falsely       new playbook: no false claim", FG), ("", FG),
        ("Honest caveat: memory has not yet reduced the number of moves. Details in the README.", DIM)],
        footer="eval/report.md · eval/playbook_v1.json · eval/brain_final.json"),
        "Everything is measured against simulator ground truth, including the verifier's own accuracy. "
        "One self-improvement cycle: the team rewrote its playbook, and the gate kept it because the new version made no false claim on an unseen shift. "
        "Caveats are in the README, with the numbers."))

    # 7. end
    segs.append(("still", slide("github.com/abhijitbetigeri/robot-agent-lab", [
        ("Shared memory. Independent checking. Failure detection. Self-improvement.", FG), ("", FG),
        ("Crusoe Managed Inference  ·  NVIDIA Nemotron: Super, Lightning, Nano, Nano Omni", DIM),
        ("NemoClaw / OpenShell sandbox for the operator  ·  You.com researcher", DIM), ("", FG),
        ("All code written today. No physical hardware. SO-101 model: MuJoCo Menagerie (Apache-2.0).", DIM)], mono=False),
        "Robot Agent Lab. Built today on Crusoe Managed Inference with Nemotron. Repo on GitHub."))

    parts, total = [], 0.0
    for i, (kind, obj, text) in enumerate(segs):
        wav, dur = say(text, f"n{i}")
        total += dur + 0.7
        parts.append(still_segment(ffmpeg, i, obj, wav, dur) if kind == "still" else clip_segment(ffmpeg, i, obj, wav, dur))
        print(f"segment {i}: {dur:.1f}s")
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    final = ROOT / "demo" / "robot-agent-lab-demo.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(final)], check=True)
    print(f"wrote {final} (~{total:.0f}s)")


if __name__ == "__main__":
    main()
