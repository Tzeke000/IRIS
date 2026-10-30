"""
brain/iris_memory.py — Iris's memory store.

JSONL-backed durable log with a naive substring search. Semantic search is
brain/iris_semantic_memory.py (ChromaDB + bundled MiniLM), kept in step by a
write-through upsert/delete here and a reconcile at boot.

Every read/append/rewrite goes through brain/_jsonl_store.py — one flock'd
sidecar lock shared with iris_human_memory.record_revisit and
scripts/memory_decay.py, fsync'd appends that can't be swallowed by a torn
tail, per-line tolerant decoding, and mkstemp+replace rewrites.

The orb's memory tab contracts:
  GET    /api/v1/memory/mem0           → {ok, entries: [{id, text, ...}]}
  POST   /api/v1/memory/mem0/search    → {ok, results: [...]}
  DELETE /api/v1/memory/mem0/{id}      → {ok}

This module owns those entries. iris_runtime calls bootstrap on startup;
orb_http calls into the singleton for the three endpoints.

Storage: state/iris_memory.jsonl. One JSON object per line:
  {"id": "<12-hex>", "ts": <unix>, "iso": "<rfc3339>", "text": "<str>",
   "person_id": "zeke", "tags": ["..."], "category": "episodic",
   "importance": 0.5, "source": "manual"}

When the LLM-backed extraction lands, this file becomes a write-through
cache that mem0 also indexes — the JSONL is the durable log, mem0 is the
search layer over it.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from brain import _jsonl_store


_DEFAULT_PATH = "state/iris_memory.jsonl"
_DEFAULT_LIMIT = 200


class IrisMemory:
    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._path = self._base / _DEFAULT_PATH
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def _locked(self, fail_open: bool = False):
        """The shared cross-process lock on the canonical log (see _jsonl_store)."""
        return _jsonl_store.locked(self._path, fail_open=fail_open)

    def _read(self) -> "_jsonl_store.ReadResult":
        return _jsonl_store.read_jsonl(self._path)

    def _read_all(self) -> list[dict[str, Any]]:
        return self._read().entries

    def _rewrite_all(self, entries: list[dict[str, Any]],
                     salvage: Optional[list[bytes]] = None) -> None:
        _jsonl_store.rewrite_jsonl(self._path, entries, salvage=salvage)

    def add(
        self,
        text: str,
        person_id: str = "zeke",
        category: str = "episodic",
        importance: float = 0.5,
        source: str = "manual",
        tags: Optional[list[str]] = None,
        importance_level: str = "default",
    ) -> dict[str, Any]:
        """Append a new memory entry to iris_memory.jsonl.

        importance_level (added 2026-05-19 for decay support):
          - "permanent" — never decay/archive. Use for identity facts,
            relationship facts that don't degrade ("Zeke's job", "Wren is
            sibling AI on other machine").
          - "high" — slower decay, longer retention. Use for important but
            time-bounded facts.
          - "default" — standard decay per Ebbinghaus.
          - "low" — accelerated decay. Use for ephemeral observations that
            should fade unless reinforced.
        """
        text = str(text or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        # Phase 45: encoding-time arousal boost (amygdala/flashbulb pathway).
        # Memories created during high mood arousal encode more durably.
        # Read live mood arousal; boost importance up to +0.3.
        mood_arousal = 0.0
        mood_label = ""
        try:
            from brain import mood_core
            m = mood_core.load_mood_raw()
            weights = m.get("emotion_weights") or {}
            high_arousal_keys = ("excitement", "fear", "anger", "joy",
                                 "anxiety", "frustration", "horror",
                                 "surprise", "amusement", "distress")
            mood_arousal = sum(float(weights.get(k, 0.0)) for k in high_arousal_keys)
            mood_arousal = min(1.0, mood_arousal * 2.0)  # scale to 0-1
            # Find dominant for label.
            if weights:
                top = max(weights.items(), key=lambda kv: kv[1])
                mood_label = str(top[0])
        except Exception:
            pass
        # Boost: 0.3 * arousal added to base importance (capped at 1.0).
        encoding_boost = round(0.3 * mood_arousal, 3)
        final_importance = min(1.0, float(importance) + encoding_boost)

        # Normalize importance_level to known values, default to "default".
        _LEVELS = {"permanent", "high", "default", "low"}
        level = str(importance_level or "default").strip().lower()
        if level not in _LEVELS:
            level = "default"
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "iso": datetime.now().astimezone().isoformat(timespec="seconds"),
            "text": text[:2000],
            "person_id": str(person_id or "zeke"),
            "category": str(category or "episodic"),
            "importance": round(final_importance, 3),
            "base_importance": round(float(importance), 3),
            "importance_level": level,
            "encoding_arousal": round(mood_arousal, 3),
            "encoding_mood_label": mood_label,
            "source": str(source or "manual"),
            "tags": list(tags or []),
        }
        _jsonl_store.append_jsonl(self._path, [entry])
        # Phase 18: write-through to semantic index. Best-effort; if chroma
        # isn't ready, the JSONL write succeeded which is what matters.
        try:
            from brain import iris_semantic_memory
            iris_semantic_memory.upsert(entry)
        except Exception:
            pass
        return entry

    def list(self, limit: int = _DEFAULT_LIMIT) -> list[dict[str, Any]]:
        with self._locked(fail_open=True):
            entries = self._read_all()
        # Newest first.
        entries.sort(key=lambda e: float(e.get("ts") or 0.0), reverse=True)
        return entries[: max(0, int(limit))]

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        with self._locked(fail_open=True):
            entries = self._read_all()
        # A reconsolidated memory (record_revisit) stays in the log for audit,
        # but search returns only the version that superseded it.
        superseded = {str(e.get("supersedes")) for e in entries if e.get("supersedes")}
        scored: list[tuple[float, dict[str, Any]]] = []
        for e in entries:
            if str(e.get("id") or "") in superseded:
                continue
            text = str(e.get("text") or "").lower()
            tags = " ".join(str(t) for t in (e.get("tags") or [])).lower()
            score = 0.0
            if q in text:
                # crude: count occurrences, weight by inverse text length
                score += text.count(q) * 1.0
                score += 0.5 if text.startswith(q) else 0.0
            if q in tags:
                score += 0.5
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [e for _, e in scored[: max(0, int(limit))]]

    def delete(self, entry_id: str) -> bool:
        eid = str(entry_id or "")
        if not eid:
            return False
        with self._locked():
            res = self._read()
            entries = res.entries  # a whole-file read error => [] => nothing deleted
            kept = [e for e in entries if e.get("id") != eid]
            if len(kept) == len(entries):
                return False
            self._rewrite_all(kept, salvage=res.bad_raw)
        # Tombstone, so boot reconcile can tell a deliberate delete from a
        # memory the JSONL LOST (and remove it from Chroma even if the mirror
        # call below fails). Best-effort.
        try:
            _jsonl_store.append_jsonl(self._path.with_name("iris_memory_deleted.jsonl"),
                                      [{"id": eid, "ts": time.time(), "why": "deleted"}])
        except Exception:
            pass
        # Phase 18: mirror delete to semantic index.
        try:
            from brain import iris_semantic_memory
            iris_semantic_memory.delete(eid)
        except Exception:
            pass
        return True

    def count(self) -> int:
        with self._locked(fail_open=True):
            return len(self._read_all())


# ── singleton ────────────────────────────────────────────────────────────────

_SINGLETON: Optional[IrisMemory] = None


def get_iris_memory() -> Optional[IrisMemory]:
    return _SINGLETON


def bootstrap_iris_memory(g: dict[str, Any]) -> Optional[IrisMemory]:
    global _SINGLETON
    base = Path(g.get("BASE_DIR") or ".")
    mem = IrisMemory(base)
    # Touch the directory so the orb's first poll doesn't race a missing
    # parent dir.
    try:
        (base / "state").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _SINGLETON = mem
    g["_iris_memory"] = mem
    print(f"[iris_memory] ready (path={mem._path}, count={mem.count()})")
    return mem
