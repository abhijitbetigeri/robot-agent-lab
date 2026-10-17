"""You.com researcher: a tool the operator can call when it hits something outside the protocol.
Returns a few cited snippets; the curator may store the answer in the brain as a fact with a source and a date.

    POST https://api.you.com/v1/search   header X-API-Key   body {"query": ..., "count": 5}
"""
from __future__ import annotations

import os
import time

import httpx

SCHEMA = {"type": "function", "function": {
    "name": "research",
    "description": "Ask the web (You.com) a short factual question when a tool error or situation is outside the lab protocol. Returns cited snippets. Do not use for things you can see with look().",
    "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}


def research(question: str, count: int = 4) -> dict:
    key = os.environ.get("YOU_API_KEY")
    if not key:
        return {"ok": False, "error": "YOU_API_KEY not set"}
    try:
        r = httpx.post("https://api.you.com/v1/search", headers={"X-API-Key": key},
                       json={"query": question, "count": count}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}
    hits = []
    results = data.get("results") or {}
    results = results.get("web", []) if isinstance(results, dict) else results   # shape seen live: {"results": {"web": [...]}}
    for h in results[:count]:
        snippet = h.get("snippet") or h.get("description") or " ".join(h.get("snippets") or [])
        hits.append({"title": (h.get("title") or "")[:120], "url": h.get("url"), "snippet": (snippet or "")[:300]})
    return {"ok": True, "question": question, "hits": hits, "source": "you.com", "retrieved_at": time.strftime("%Y-%m-%d")}
