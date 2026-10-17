"""Crusoe Managed Inference client: one endpoint, per-role Nemotron models, backoff, usage + cache accounting.

Facts from Crusoe's hackathon skill file (Oct 13, 2026):
- base_url https://api.inference.crusoecloud.com/v1 (OpenAI chat completions; no Responses API)
- model IDs are case-sensitive; confirm against GET /v1/models (run `python -m llm.crusoe models`)
- reasoning tokens count against max_tokens; give reasoning models >= 1024
- Nemotron Super / Lightning: chat_template_kwargs {"enable_thinking": bool, "force_nonempty_content": true}
- rate limits are per model per minute; spreading roles across models multiplies headroom
- export CRUSOE_API_KEY in single quotes (keys can contain '$')
"""
from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI, RateLimitError, InternalServerError, APIStatusError

BASE_URL = "https://api.inference.crusoecloud.com/v1"


def _load_dotenv() -> None:
    """Load KEY='value' lines from .env next to the repo root (gitignored). Shell env wins."""
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


_load_dotenv()

# role -> (model id, default kwargs). Confirm IDs with `python -m llm.crusoe models` before 11:00.
ROLES: dict[str, dict] = {
    "planner":    {"model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B", "thinking": True,  "max_tokens": 3000},
    "operator":   {"model": "nvidia/Nemotron-3.5-Lightning-30B-A3B", "thinking": False, "max_tokens": 1500},
    "researcher": {"model": "nvidia/Nemotron-3.5-Lightning-30B-A3B", "thinking": False, "max_tokens": 1200},
    "verifier":   {"model": "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B", "thinking": None, "max_tokens": 1500},
    "monitor":    {"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "thinking": False, "max_tokens": 600},
    "curator":    {"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "thinking": False, "max_tokens": 1500},
}
FALLBACK_TOOL_MODEL = "openai/gpt-oss-120b"  # Crusoe: most reliable tool caller. Use only if Lightning misbehaves; log it.


class Usage:
    """Per-role token accounting; the eval table is built from this."""
    def __init__(self):
        self.by_role = defaultdict(lambda: {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "retries": 0})

    def add(self, role: str, usage) -> None:
        u = self.by_role[role]
        u["calls"] += 1
        u["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
        u["completion"] += getattr(usage, "completion_tokens", 0) or 0
        det = getattr(usage, "prompt_tokens_details", None)
        u["cached"] += (getattr(det, "cached_tokens", 0) or 0) if det else 0

    def totals(self) -> dict:
        t = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "retries": 0}
        for u in self.by_role.values():
            for k in t:
                t[k] += u[k]
        t["cache_hit_rate"] = round(t["cached"] / t["prompt"], 3) if t["prompt"] else 0.0
        return t

    def snapshot(self) -> dict:
        return {"by_role": {k: dict(v) for k, v in self.by_role.items()}, "totals": self.totals()}


class Crusoe:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("CRUSOE_API_KEY")
        if not key:
            sys.exit("CRUSOE_API_KEY not set (export it in single quotes)")
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self.usage = Usage()

    # ---------------------------------------------------------------- core call
    def chat(self, role: str, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice=None, response_format: dict | None = None, model: str | None = None,
             thinking: bool | None = "default", max_tokens: int | None = None, temperature: float = 1.0,
             top_p: float = 0.95, max_tries: int = 6):
        cfg = ROLES[role]
        model = model or cfg["model"]
        if thinking == "default":
            thinking = cfg["thinking"]
        kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens or cfg["max_tokens"],
                            temperature=temperature, top_p=top_p)
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        ctk: dict = {}
        if thinking is not None and "nvidia/" in model.lower():
            ctk["enable_thinking"] = bool(thinking)
            if tools:
                ctk["force_nonempty_content"] = True
        elif thinking is not None and "deepseek" in model.lower():
            ctk["thinking"] = bool(thinking)
        if ctk:
            kwargs["extra_body"] = {"chat_template_kwargs": ctk}

        for attempt in range(max_tries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.usage.add(role, resp.usage)
                return resp
            except (RateLimitError, InternalServerError) as e:
                self.usage.by_role[role]["retries"] += 1
                if attempt == max_tries - 1:
                    raise
                time.sleep(min(60.0, 1.0 * 2 ** attempt) + random.uniform(0, 0.5))
            except APIStatusError as e:
                # 400 on chat_template_kwargs → retry once without it (model may not accept the toggle)
                if e.status_code == 400 and "extra_body" in kwargs:
                    kwargs.pop("extra_body")
                    continue
                raise

    def text(self, role: str, system: str, user: str, **kw) -> str:
        r = self.chat(role, [{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return content_of(r)

    def json(self, role: str, system: str, user: str, **kw) -> dict:
        """Ask for JSON; tolerate models that wrap it in prose or fences."""
        r = self.chat(role, [{"role": "system", "content": system + "\nReply with a single JSON object and nothing else."},
                             {"role": "user", "content": user}], **kw)
        return parse_json(content_of(r))

    def models(self) -> list[dict]:
        return [m.model_dump() for m in self.client.models.list().data]


# -------------------------------------------------------------------- helpers
def content_of(resp) -> str:
    m = resp.choices[0].message
    c = m.content or ""
    if not c and resp.choices[0].finish_reason == "length":
        return "[EMPTY: reasoning exhausted max_tokens]"
    return c


def parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {s[:200]}")
    return json.loads(s[start:end + 1])


def image_part(path: str | Path, max_side: int = 768) -> dict:
    """Image as a base64 data URI part (no outbound network needed by the model)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": uri}}


if __name__ == "__main__":
    c = Crusoe()
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        ms = c.models()
        Path("llm/models.json").write_text(json.dumps(ms, indent=1))
        ids = {m["id"] for m in ms}
        for role, cfg in ROLES.items():
            print(("OK  " if cfg["model"] in ids else "MISSING ") + f"{role:10s} {cfg['model']}")
        print(f"\n{len(ids)} models written to llm/models.json")
    else:
        for role in ROLES:
            try:
                r = c.chat(role, [{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=1024)
                print(f"{role:10s} {ROLES[role]['model']:50s} -> {content_of(r)[:40]!r}")
            except Exception as e:  # noqa: BLE001
                print(f"{role:10s} {ROLES[role]['model']:50s} -> ERROR {e}")
        print(json.dumps(c.usage.snapshot()["totals"]))
