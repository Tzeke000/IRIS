"""memory_correct — fix a memory IN PLACE in both stores (brain/memory_correct.py).

show    id                     — JSONL text vs Chroma text side by side (+ audit trail)
correct id new_text reason     — rewrite the JSONL row (old text + reason kept in its
                                 `corrections` list), then re-embed it in Chroma
resync  id | id=all [dry_run]  — re-embed from the JSONL wherever Chroma's text differs

Use this when a memory was WRONG (invented premise, bad caption). Use memory_revisit
when something genuinely CHANGED and the old version should stay as history.
"""
from __future__ import annotations

from typing import Any

from tools.tool_registry import register_tool


def _mc(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    from brain import memory_correct as mc
    a = str(params.get("action") or "show").lower()
    mid = str(params.get("id") or params.get("memory_id") or "")
    if a == "show":
        return mc.show(mid)
    if a == "correct":
        return mc.correct(mid, str(params.get("new_text") or ""), str(params.get("reason") or ""))
    if a == "resync":
        return mc.resync(mid, dry_run=bool(params.get("dry_run", False)))
    return {"ok": False, "error": f"unknown action {a!r} (show|correct|resync)"}


register_tool(
    "memory_correct",
    "CORRECT A MEMORY IN PLACE (JSONL + Chroma): show id (side by side) | correct id "
    "new_text reason (audit trail kept, re-embeds) | resync id|all [dry_run] (Chroma <- JSONL).",
    1,
    _mc,
)
