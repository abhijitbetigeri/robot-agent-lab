"""The shared brain: a bi-temporal triple ledger with deterministic supersession.

Design, with the papers each piece comes from (see README):
- rows are (subject, relation, object) triples with valid_from / valid_to  -> Zep/Graphiti 2501.13956, Temporal Validity 2606.26511
- a new row with the same (subject, relation) closes the old one; nothing is deleted  -> supersession, not similarity
- tiers: fact / procedure / insight  -> G-Memory 2506.07398
- provenance: source_agent, episode, evidence_ref on every row  -> Collaborative Memory 2505.18279
- retrieval = relevance x exp(-age/window) x usefulness  -> Generative Agents 2304.03442 + our decay thesis
- helpful/harmful votes prune insights  -> ExpeL 2308.10144
- writes arrive as itemized deltas from the curator, never a rewrite  -> ACE 2510.04618
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('fact','procedure','insight')),
  valid_from INTEGER NOT NULL,          -- episode number
  valid_to INTEGER,                     -- NULL = current
  source_agent TEXT NOT NULL, episode INTEGER NOT NULL, evidence_ref TEXT,
  confidence REAL DEFAULT 0.7, helpful INTEGER DEFAULT 0, harmful INTEGER DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_key ON facts(subject, relation, valid_to);
"""


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


class Brain:
    def __init__(self, path: str | Path = "runs/brain.sqlite", decay_window: float = 4.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.decay_window = decay_window
        self.served: list[dict] = []  # what retrieve() handed out, for stale-serving audits

    # ---------------------------------------------------------------- writes (deltas only)
    def add(self, subject: str, relation: str, obj, tier: str, source_agent: str, episode: int,
            evidence_ref: str | None = None, confidence: float = 0.7) -> dict:
        """Assert a triple. Same (subject, relation) with a different object supersedes the current row."""
        obj_s = obj if isinstance(obj, str) else json.dumps(obj)
        cur = self.db.execute("SELECT id, object FROM facts WHERE subject=? AND relation=? AND valid_to IS NULL",
                              (subject, relation)).fetchone()
        action = "add"
        if cur:
            if cur[1] == obj_s:
                self.db.execute("UPDATE facts SET helpful = helpful + 1 WHERE id=?", (cur[0],))
                self.db.commit()
                return {"action": "noop", "id": cur[0]}
            self.db.execute("UPDATE facts SET valid_to=? WHERE id=?", (episode, cur[0]))
            action = "supersede"
        c = self.db.execute(
            "INSERT INTO facts(subject,relation,object,tier,valid_from,source_agent,episode,evidence_ref,confidence,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (subject, relation, obj_s, tier, episode, source_agent, episode, evidence_ref, confidence, time.time()))
        self.db.commit()
        return {"action": action, "id": c.lastrowid, "superseded": cur[0] if cur else None}

    def vote(self, fact_id: int, delta: int) -> None:
        col = "helpful" if delta > 0 else "harmful"
        self.db.execute(f"UPDATE facts SET {col} = {col} + 1 WHERE id=?", (fact_id,))
        self.db.commit()

    def apply_deltas(self, deltas: list[dict], source_agent: str, episode: int) -> list[dict]:
        """Curator output: [{op: add|supersede|upvote|downvote, ...}]. 'supersede' is just 'add' with a new object."""
        out = []
        for d in deltas:
            op = d.get("op")
            if op in ("add", "supersede"):
                out.append(self.add(d["subject"], d["relation"], d["object"], d.get("tier", "fact"),
                                    source_agent, episode, d.get("evidence_ref"), float(d.get("confidence", 0.7))))
            elif op in ("upvote", "downvote") and d.get("id") is not None:
                m = re.search(r"(\d+)", str(d["id"]))   # briefing shows ids as F#3 / P#2 / I#4
                if not m:
                    continue
                self.vote(int(m.group(1)), 1 if op == "upvote" else -1)
                out.append({"action": op, "id": int(m.group(1))})
        self.prune()
        return out

    def prune(self, min_margin: int = 2) -> int:
        """Lean: retire insights whose harm outweighs their help. Retired = valid_to set, never deleted."""
        c = self.db.execute("UPDATE facts SET valid_to = -1 WHERE valid_to IS NULL AND tier='insight' AND harmful - helpful >= ?",
                            (min_margin,))
        self.db.commit()
        return c.rowcount

    # ---------------------------------------------------------------- reads
    def current(self) -> list[dict]:
        rows = self.db.execute("SELECT id,subject,relation,object,tier,valid_from,source_agent,episode,evidence_ref,confidence,helpful,harmful"
                               " FROM facts WHERE valid_to IS NULL ORDER BY id").fetchall()
        keys = ["id", "subject", "relation", "object", "tier", "valid_from", "source_agent", "episode", "evidence_ref", "confidence", "helpful", "harmful"]
        return [dict(zip(keys, r)) for r in rows]

    def history(self, subject: str, relation: str) -> list[dict]:
        rows = self.db.execute("SELECT id,object,valid_from,valid_to,source_agent FROM facts WHERE subject=? AND relation=? ORDER BY id",
                               (subject, relation)).fetchall()
        return [dict(zip(["id", "object", "valid_from", "valid_to", "source_agent"], r)) for r in rows]

    def retrieve(self, query: str, episode: int, k: int = 12, role: str | None = None) -> list[dict]:
        q = _tokens(query)
        scored = []
        for f in self.current():
            text = f"{f['subject']} {f['relation']} {f['object']} {f['tier']}"
            rel = len(q & _tokens(text)) / (len(q) + 1e-9)
            rel = 0.15 + rel  # every current fact gets a floor; facts/procedures are cheap to carry
            age = max(0, episode - f["valid_from"])
            decay = math.exp(-age / self.decay_window) if f["tier"] == "insight" else 1.0  # hard facts don't fade
            use = 1.0 + 0.25 * (f["helpful"] - f["harmful"])
            scored.append((rel * decay * use * f["confidence"], f))
        scored.sort(key=lambda x: -x[0])
        top = [f for _, f in scored[:k]]
        self.served.extend({"episode": episode, "role": role, "id": f["id"], "subject": f["subject"], "relation": f["relation"]} for f in top)
        return top

    def briefing(self, role: str, episode: int, query: str) -> str:
        """Compact text the planner/operator get as their LAST message (keeps the cached prefix stable)."""
        facts = self.retrieve(query, episode, role=role)
        if not facts:
            return "SHARED BRAIN: empty. You are the first shift."
        lines = [f"SHARED BRAIN (shift {episode}; {len(facts)} current entries, newest first):"]
        for f in sorted(facts, key=lambda f: -f["valid_from"]):
            tag = {"fact": "F", "procedure": "P", "insight": "I"}[f["tier"]]
            lines.append(f"- [{tag}#{f['id']} shift{f['valid_from']} by {f['source_agent']}] {f['subject']} {f['relation']} {f['object']}")
        lines.append("Cite entry ids you relied on. Report anything that turned out wrong so it can be superseded.")
        return "\n".join(lines)

    def stats(self) -> dict:
        cur = self.db.execute("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL").fetchone()[0]
        total = self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        sup = self.db.execute("SELECT COUNT(*) FROM facts WHERE valid_to IS NOT NULL AND valid_to >= 0").fetchone()[0]
        ret = self.db.execute("SELECT COUNT(*) FROM facts WHERE valid_to = -1").fetchone()[0]
        return {"current": cur, "total": total, "superseded": sup, "retired": ret}


if __name__ == "__main__":  # tiny self-test
    b = Brain(":memory:")
    print(b.add("bin_blue", "position_xy", [0.26, -0.18], "fact", "operator", 1, "call#4"))
    print(b.add("gripper", "needs_retry_on_close", "true", "procedure", "operator", 1, "call#6"))
    print(b.add("bin_blue", "position_xy", [0.26, -0.10], "fact", "operator", 3, "call#9"))  # supersede
    print(b.history("bin_blue", "position_xy"))
    print(b.briefing("operator", 4, "blue tube rack position gripper"))
    print(b.stats())
