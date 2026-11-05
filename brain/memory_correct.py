"""
brain/memory_correct.py — correct a memory IN PLACE, in both stores.

record_revisit (memory_revisit) writes a NEW version that supersedes the old
one; it never edits an existing row, and it compares nothing against Chroma.
So when a memory's text was wrong (an invented premise, a bad caption) and the
JSONL got hand-fixed, Chroma kept embedding — and ranking — the old text, and
nothing could make it catch up. This module is that missing path:

  show(id)                   — JSONL text vs Chroma text, side by side
  correct(id, new_text, why) — rewrite the JSONL row's text (the old text +
                               reason go into its `corrections` audit list),
                               then delete + re-upsert the Chroma entry so it
                               is re-embedded from the new text
  resync(ids | None)         — wherever Chroma's document differs from the
                               live JSONL text (or is missing), re-embed it
                               from the JSONL. JSONL is the canonical log;
                               Chroma is only ever brought in line with it.

Must run INSIDE the runtime, which owns the Chroma client
(iris_semantic_memory._COLLECTION). Deliberately a NEW module that holds no
configured state of its own — it reads iris_semantic_memory's client and the
iris_memory path at call time — so it lands with iris_tool_reload alone and
never needs a hot-swap of the modules that do hold configured globals.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from brain import _jsonl_store

_PAGE = 500          # Chroma get() batch size
_PREVIEW = 160       # chars of old/new text echoed per resynced id


def _mem_path(base: Optional[Path] = None) -> Optional[Path]:
    if base is not None:
        return Path(base) / "state" / "iris_memory.jsonl"
    from brain import iris_memory
    mem = iris_memory.get_iris_memory()
    return Path(mem._path) if mem is not None else None


def _sm():
    """The configured semantic-memory module, or None if Chroma isn't up."""
    from brain import iris_semantic_memory as sm
    return sm if sm._ensure_client() else None


def _chroma_docs(sm, ids: list[str]) -> Optional[dict[str, str]]:
    """{id: document} for the ids that exist in Chroma. None on error (callers
    must not read an error as 'missing from Chroma')."""
    out: dict[str, str] = {}
    try:
        for i in range(0, len(ids), _PAGE):
            with sm._LOCK:
                got = sm._COLLECTION.get(ids=ids[i:i + _PAGE], include=["documents"])
            gids = got.get("ids") or []
            docs = got.get("documents") or []
            for j, gid in enumerate(gids):
                out[str(gid)] = str(docs[j] if j < len(docs) and docs[j] is not None else "")
        return out
    except Exception as e:
        print(f"[memory_correct] chroma get error: {e!r}")
        return None


def _reembed(sm, entry: dict[str, Any]) -> bool:
    """Drop the Chroma row and upsert it fresh from `entry`, so the vector is
    recomputed from the current text (not just the document string swapped)."""
    sm.delete(str(entry.get("id")))
    return bool(sm.upsert(entry))


def _load(path: Path) -> "_jsonl_store.ReadResult":
    return _jsonl_store.read_jsonl(path)


def _superseded_by(entries: list[dict[str, Any]], eid: str) -> Optional[str]:
    for e in entries:
        if str(e.get("supersedes") or "") == eid:
            return str(e.get("id"))
    return None


def show(memory_id: str, base: Optional[Path] = None) -> dict[str, Any]:
    eid = str(memory_id or "").strip()
    if not eid:
        return {"ok": False, "error": "id required"}
    path = _mem_path(base)
    if path is None or not path.is_file():
        return {"ok": False, "error": "iris_memory.jsonl not found"}
    entries = _load(path).entries
    row = next((e for e in entries if str(e.get("id")) == eid), None)
    sm = _sm()
    chroma_text: Optional[str] = None
    chroma_err = None
    if sm is None:
        chroma_err = "chroma not ready"
    else:
        docs = _chroma_docs(sm, [eid])
        if docs is None:
            chroma_err = "chroma read error"
        else:
            chroma_text = docs.get(eid)
    jsonl_text = None if row is None else str(row.get("text") or "")
    return {
        "ok": True,
        "id": eid,
        "in_jsonl": row is not None,
        "in_chroma": chroma_text is not None,
        "jsonl_text": jsonl_text,
        "chroma_text": chroma_text,
        "match": (row is not None and chroma_text is not None and jsonl_text == chroma_text),
        "superseded_by": _superseded_by(entries, eid),
        "corrections": (row or {}).get("corrections") or [],
        **({"chroma_error": chroma_err} if chroma_err else {}),
    }


def correct(memory_id: str, new_text: str, reason: str,
            base: Optional[Path] = None) -> dict[str, Any]:
    eid = str(memory_id or "").strip()
    text = str(new_text or "").strip()[:2000]
    why = str(reason or "").strip()[:300]
    if not eid or not text:
        return {"ok": False, "error": "id and new_text required"}
    if not why:
        return {"ok": False, "error": "reason required (it is the audit trail)"}
    path = _mem_path(base)
    if path is None or not path.is_file():
        return {"ok": False, "error": "iris_memory.jsonl not found"}
    with _jsonl_store.locked(path):
        res = _load(path)
        if res.error:
            return {"ok": False, "error": f"jsonl unreadable: {res.error}"}
        entries = res.entries
        newer = _superseded_by(entries, eid)
        if newer:
            return {"ok": False, "error": f"{eid} is superseded by {newer}; correct that one"}
        row = next((e for e in entries if str(e.get("id")) == eid), None)
        if row is None:
            return {"ok": False, "error": "memory not found in jsonl"}
        old = str(row.get("text") or "")
        changed = old != text
        if changed:
            audit = list(row.get("corrections") or [])
            audit.append({
                "ts": time.time(),
                "iso": datetime.now().astimezone().isoformat(timespec="seconds"),
                "old_text": old,
                "reason": why,
            })
            row["text"] = text
            row["corrections"] = audit
            _jsonl_store.rewrite_jsonl(path, entries, salvage=res.bad_raw)
    sm = _sm()
    reembedded = bool(sm is not None and _reembed(sm, row))
    return {
        "ok": True,
        "id": eid,
        "jsonl_changed": changed,
        "chroma_reembedded": reembedded,
        "old_text": old,
        "new_text": text,
        **({} if reembedded else
           {"warning": "JSONL updated but Chroma not re-embedded; run resync"}),
    }


def resync(memory_id: str = "", base: Optional[Path] = None,
           dry_run: bool = False) -> dict[str, Any]:
    """memory_id = one id, or 'all'. Only LIVE (non-superseded) JSONL rows are
    considered; stale Chroma rows for superseded/deleted ids are the boot
    reconcile's job, not this one."""
    want = str(memory_id or "").strip()
    if not want:
        return {"ok": False, "error": "id required (an id, or 'all')"}
    path = _mem_path(base)
    if path is None or not path.is_file():
        return {"ok": False, "error": "iris_memory.jsonl not found"}
    sm = _sm()
    if sm is None:
        return {"ok": False, "error": "chroma not ready"}
    res = _load(path)
    if res.error:
        return {"ok": False, "error": f"jsonl unreadable: {res.error}"}
    superseded = {str(e.get("supersedes")) for e in res.entries if e.get("supersedes")}
    live = {str(e["id"]): e for e in res.entries
            if e.get("id") and str(e["id"]) not in superseded and str(e.get("text") or "")}
    if want != "all":
        if want not in live:
            return {"ok": False,
                    "error": "not a live jsonl row" + (" (superseded)" if want in superseded else "")}
        live = {want: live[want]}
    docs = _chroma_docs(sm, list(live))
    if docs is None:
        return {"ok": False, "error": "chroma read error; nothing changed"}
    fixed: list[dict[str, Any]] = []
    failed: list[str] = []
    for eid, e in live.items():
        jt = str(e.get("text") or "")
        ct = docs.get(eid)
        if ct == jt:
            continue
        item = {"id": eid, "was": "missing" if ct is None else "differs",
                "chroma_old": None if ct is None else ct[:_PREVIEW], "jsonl": jt[:_PREVIEW]}
        if not dry_run and not _reembed(sm, e):
            failed.append(eid)
            continue
        fixed.append(item)
    return {"ok": not failed, "checked": len(live), "dry_run": bool(dry_run),
            "resynced": len(fixed), "items": fixed[:50], "failed": failed}
