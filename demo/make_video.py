"""Build the 1-minute voiced demo from real artifacts: sweep logs, verifier frames, ledger rows, results table.

    uv run python demo/make_video.py            -> demo/night-shift-demo.mp4
Uses macOS `say` for narration and the ffmpeg bundled with imageio-ffmpeg. No screen recording needed.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "build"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
BG, FG, DIM, ACC, RED, GRN = (16, 18, 24), (235, 235, 235), (150, 155, 165), (255, 200, 60), (255, 95, 95), (110, 220, 130)
MONO = "/System/Library/Fonts/Menlo.ttc"
SANS = "/System/Library/Fonts/Helvetica.ttc"
VOICE = "Samantha"
RATE = 182


def font(path, size):
    return ImageFont.truetype(path, size, index=0)


def slide(title: str, lines: list[tuple[str, tuple]] | None = None, image: Path | None = None, mono=True, footer=""):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 36), title, fill=ACC, font=font(SANS, 40))
    y = 110
    f = font(MONO if mono else SANS, (20 if image is not None else 22) if mono else 28)
    for text, color in (lines or []):
        chunks = [text[:56]] if image is not None else _wrap(text, 80)
        for chunk in chunks:
            d.text((48, y), chunk, fill=color, font=f)
            y += 28 if mono else 38
    if image is not None:
        img = Image.open(image).convert("RGB")
        img.thumbnail((500, 500))
        im.paste(img, (W - img.width - 48, 110))
        d.rectangle([W - img.width - 50, 108, W - 46, 112 + img.height], outline=DIM, width=2)
        d.text((W - img.width - 48, 118 + img.height), "verifier camera (sealed)", fill=DIM, font=font(SANS, 18))
    if footer:
        d.text((48, H - 48), footer, fill=DIM, font=font(SANS, 20))
    return im


def _wrap(s: str, n: int) -> list[str]:
    out, cur = [], ""
    for w in s.split(" "):
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + ([cur] if cur else [])


def compact(line: str) -> str:
    m = re.match(r"\s*\[(\d+)\] (\w+)\((.*?)\) -> (\{.*\})", line)
    if not m:
        if line.startswith("result:"):
            return line[:56]
        return line[:56]
    n, tool, args, res = m.groups()
    try:
        r = json.loads(res)
    except Exception:  # noqa: BLE001
        r = {}
    a = re.sub(r'[{}"]', "", args).replace("cmd:", "")
    keys = [k for k in ("holding", "ok", "reached_xy_err_m", "recorded") if k in r]
    tail = ", ".join(f"{k}={r[k]}" for k in keys[:2])
    return f"[{n:>2}] {tool}({a})  ->  {tail}"[:56]


def log_lines(path: Path, shift: int, wanted: tuple[str, ...], limit: int = 11) -> list[tuple[str, tuple]]:
    txt = path.read_text()
    m = re.search(rf"=== shift {shift} .*?(?==== shift {shift + 1} |DONE|\Z)", txt, re.S)
    body = m.group(0) if m else ""
    out = []
    for line in body.splitlines():
        line = line.replace("[brain] ", "").rstrip()
        if line.startswith(("DoD:", "- [", "SHARED BRAIN", "brain:", "result:", "replanned")):
            continue
        if any(w in line for w in wanted):
            color = FG
            if "holding\": null" in line or "FAIL" in line or "verifier=False" in line:
                color = RED
            if "holding\": true" in line or "verifier=True" in line or "PASS" in line:
                color = GRN
            out.append((compact(line), color))
        if len(out) >= limit:
            break
    return out


def say(text: str, name: str) -> tuple[Path, float]:
    wav = OUT / f"{name}.wav"
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "--data-format=LEI16@22050", "-o", str(wav), text], check=True)
    with wave.open(str(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    return wav, dur


def segment(ffmpeg: str, idx: int, im: Image.Image, wav: Path, dur: float) -> Path:
    png = OUT / f"s{idx}.png"
    im.save(png)
    mp4 = OUT / f"s{idx}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(png), "-i", str(wav),
                    "-t", f"{dur + 0.6:.2f}", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(mp4)], check=True)
    return mp4


def main():
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    log = ROOT / "runs" / "sweep_brain.log"
    frames = ROOT / "runs" / "brain" / "frames"
    db = sqlite3.connect(ROOT / "runs" / "brain" / "brain.sqlite")
    pb = json.loads((ROOT / "eval" / "playbook_v1.json").read_text())["history"][0]

    segs = []
    # 1. title
    segs.append((slide("Night Shift", [
        ("A lab's overnight sample sorting, run by a team of Nemotron agents.", FG), ("", FG),
        ("The human types one line:", DIM), ("  \"Put tonight's samples where they belong.\"", ACC), ("", FG),
        ("Then walks away.", FG), ("", FG),
        ("Planner · Operator · sealed Verifier · Monitor · Curator", DIM),
        ("Crusoe Managed Inference · NVIDIA Nemotron · simulated SO-101 arm", DIM)], mono=False,
        footer="Recursive Agents Hackathon · Crusoe x Cerebral Valley · Oct 17 2026"),
        "Night Shift. A lab's overnight sample sorting, run by a team of Nemotron agents on Crusoe. "
        "A human types one line: put tonight's samples where they belong. Then walks away."))
    # 2. shift 1: false done caught
    l1 = log_lines(log, 1, ("gripper", "] lift", "claim_done"), limit=9)
    segs.append((slide("Shift 1 — no memory yet", [("operator tool log (real):", DIM)] + l1 + [("", FG), ("Verifier: C1 FAIL", RED), ("  red tube is not inside the red patch", RED)],
                       image=frames / "ep01_ver_038.jpg", footer="runs/sweep_brain.log, shift 1 (real output)"),
        "Shift one. The gripper reports a null hold. The operator lifts nothing, places nothing, and claims done anyway. "
        "The sealed verifier sees only the camera frame, never the operator, and fails it."))
    # 3. shift 2: brain briefing, success
    rows = db.execute("select id,tier,subject,relation,object from facts where valid_from<=2 and (valid_to is null or valid_to>2) order by id").fetchall()
    brief = [(f"[{t[0]}#{i}] {s} {r}: {str(o)[:60]}", ACC) for i, t, s, r, o in rows[:5]]
    l2 = log_lines(log, 2, ("gripper", "] lift", "claim_done"), limit=6)
    brief = [(t[:56], c) for t, c in brief]
    segs.append((slide("Shift 2 — reads the shared brain first", [("SHARED BRAIN briefing:", DIM)] + brief + [("", FG), ("operator tool log:", DIM)] + l2 + [("", FG), ("Verifier: all checks PASS", GRN)],
                       image=frames / "ep02_ver_036.jpg", footer="brain/ledger.py: facts, procedures, insights with provenance"),
        "The curator writes what happened into the shared brain. Shift two reads it before starting: close the gripper twice, "
        "check holding is true before lifting. It succeeds, and the verifier passes it."))
    # 4. supersession
    hist = db.execute("select id,object,valid_from,valid_to,source_agent from facts where subject='bin_blue' and relation='position_xy' order by id").fetchall()
    hl = [("bin_blue  position_xy      valid_from -> valid_to", DIM)]
    for i, o, vf, vt, src in hist:
        hl.append((f"  #{i:<3} {str(o):16s} shift {vf}  -> {('shift ' + str(vt)) if vt else 'CURRENT'}", GRN if vt is None else DIM))
    segs.append((slide("Shift 3 — the rack moves; the old fact loses", hl + [("", FG),
                       ("same subject+relation, new value ->", FG), ("  old row gets valid_to. Nothing deleted.", FG), ("", FG),
                       ("stale positions served after the move: 0", GRN)],
                       image=frames / "ep03_ver_035.jpg", footer="Temporal Validity (2606.26511) · Zep (2501.13956) · ACE (2510.04618)"),
        "At shift three the blue rack moves. The brain supersedes the old position deterministically. Nothing is deleted; "
        "the old fact just loses. Zero stale positions were served afterwards."))
    # 5. results + improvement
    segs.append((slide("Verified, measured, and self-improving", [
        ("4 shifts per arm, same seeds, same models, same prompts", DIM),
        ("  tubes racked 3/4 in both arms; false 'done' claims caught by the verifier", FG),
        ("  verifier accuracy measured against sim ground truth: 75%", FG), ("", FG),
        ("Version N+1 — the team rewrote its own operator playbook:", DIM),
        (f"  held-out seed {pb['seed']}: old score {pb['score_old']}  ->  new score {pb['score_new']}   KEPT", GRN),
        ("  old: false done = yes        new: false done = no", FG), ("", FG),
        ("Monitor: loop / stall / progress-collapse / budget / camera-down -> continue | replan | stop | ping", DIM)],
        footer="eval/report.md · eval/playbook_v1.json"),
        "Four shifts per arm, same seeds. False completions are caught, and the verifier's own accuracy is measured against ground truth. "
        "Then one self-improvement cycle: the team rewrote its own playbook, and kept it only because it scored better on a shift it had never seen."))
    # 6. end
    segs.append((slide("github.com/abhijitbetigeri/robot-agent-lab", [
        ("Shared memory · independent verification · failure detection · autonomous improvement", FG), ("", FG),
        ("Crusoe Managed Inference  ·  NVIDIA Nemotron (Super, Lightning, Nano, Nano Omni)", DIM),
        ("NemoClaw / OpenShell sandbox for the operator  ·  You.com researcher", DIM), ("", FG),
        ("All code written today. No physical hardware. SO-101 model: MuJoCo Menagerie (Apache-2.0).", DIM)], mono=False),
        "Night Shift. Built today on Crusoe Managed Inference with Nemotron. Repo on GitHub."))

    parts = []
    total = 0.0
    for i, (im, text) in enumerate(segs):
        wav, dur = say(text, f"n{i}")
        total += dur + 0.6
        parts.append(segment(ffmpeg, i, im, wav, dur))
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    final = ROOT / "demo" / "night-shift-demo.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(final)], check=True)
    print(f"wrote {final} ({total:.1f}s)")


if __name__ == "__main__":
    main()
