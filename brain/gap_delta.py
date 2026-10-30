"""
brain/gap_delta.py — read JSONL state files since a timestamp.

Single source of truth for the "what happened between then and now" pattern.
Previously duplicated in:
  - brain/iris_time.py:time_awareness_report (gap_evidence section)
  - scripts/voice_stop_hook.py:_time_orientation_block (off-thread section)

Both walked JSONL files looking for entries after a timestamp and produced
bullet-list summaries. This consolidates the read-since-timestamp logic
plus adds the tiered-fallback when the bullet list grows past a cap.

Stdlib-only (json, pathlib) so the Stop hook can import it without pulling
in brain/* dependencies. Resilient when iris_runtime isn't running.

Usage:

    from brain.gap_delta import read_since, summarize_for_bullet_list

    events = read_since(jsonl_path, since_ts)
    line = f"{len(events)} memory(ies) added"

    # Or with the cap-aware formatter:
    lines = summarize_for_bullet_list(items, max_items=15, max_chars=2000)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


_BLOCK = 64 * 1024
# The files are append-only, so entries are (nearly) in time order. Reading
# from the END and stopping once we are past the cutoff makes these calls cost
# O(entries since the cutoff) instead of O(whole file) (code review
# 2026-10-29: the Stop hook and time_awareness slowed down as logs grew).
# A few out-of-order lines near the cutoff are tolerated: we stop only after
# _STOP_AFTER_OLDER consecutive entries at/before the cutoff.
_STOP_AFTER_OLDER = 5


def _iter_lines_reverse(p: Path, block: int = _BLOCK):
    """Yield the file's lines (decoded, stripped) last-to-first."""
    with p.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        tail = b""
        while pos > 0:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            chunk = f.read(step) + tail
            parts = chunk.split(b"\n")
            tail = parts[0]          # may be a partial line; finish it next round
            for raw in reversed(parts[1:]):
                line = raw.strip()
                if line:
                    yield line.decode("utf-8", errors="replace")
        if tail.strip():
            yield tail.strip().decode("utf-8", errors="replace")


def _scan_since(p: Path, since_ts: float, ts_field: str,
                predicate: Callable[[dict], bool] | None):
    """Yield matching entries newest-first, stopping past the cutoff."""
    older_run = 0
    for line in _iter_lines_reverse(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if not isinstance(e, dict):
            continue
        try:
            ts = float(e.get(ts_field) or 0)
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue                      # no timestamp: skipped, doesn't end the scan
        if ts <= since_ts:
            older_run += 1
            if older_run >= _STOP_AFTER_OLDER:
                return
            continue
        older_run = 0
        if predicate is not None and not predicate(e):
            continue
        yield e


def read_since(
    path: str | Path,
    since_ts: float,
    *,
    ts_field: str = "ts",
    predicate: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Read a JSONL file and return entries whose `ts_field` > since_ts.

    Args:
        path: Path to the JSONL file. Missing file returns [].
        since_ts: Float timestamp (Unix epoch). Entries with ts <= this are skipped.
        ts_field: Name of the timestamp field in each entry. Defaults to "ts".
        predicate: Optional filter applied AFTER the ts check. Useful for
            filtering by status, sender, type, etc. Should return True to keep.

    Returns:
        List of dict entries, oldest first (file order). Empty list on any
        read/parse failure (silent — this is meant to be called from hot
        paths like the Stop hook).
    """
    p = Path(path)
    if not p.is_file():
        return []
    if since_ts <= 0:
        return []
    try:
        out = list(_scan_since(p, since_ts, ts_field, predicate))
    except Exception:
        return []
    out.reverse()
    return out


def count_since(
    path: str | Path,
    since_ts: float,
    *,
    ts_field: str = "ts",
    predicate: Callable[[dict], bool] | None = None,
) -> int:
    """Count entries since timestamp. Same args as read_since.

    Use when you only need the count (cheaper — doesn't accumulate the list).
    """
    p = Path(path)
    if not p.is_file() or since_ts <= 0:
        return 0
    n = 0
    try:
        for _ in _scan_since(p, since_ts, ts_field, predicate):
            n += 1
    except Exception:
        return 0
    return n


def summarize_for_bullet_list(
    lines: Iterable[str],
    *,
    max_items: int = 15,
    max_chars: int = 2000,
) -> list[str]:
    """Take an iterable of bullet-list strings and return a capped version.

    Tiered fallback when the list grows past max_items OR total chars > max_chars:
      - Keep the most-recent `keep` items as-is.
      - Collapse the older items into a single "(N more, oldest: ...)" summary.

    Designed for the off-thread / gap-evidence section of the Stop hook's
    time orientation block. Without a cap, a long gap could blow context
    with raw bullets. With this, the cap is graceful — the reader still
    sees that older items exist and gets one-line context on the oldest.

    Args:
        lines: Iterable of bullet-line strings (no leading bullet char —
            this returns plain strings the caller formats as needed).
        max_items: Cap on raw bullet count before collapsing.
        max_chars: Cap on total character count before collapsing.

    Returns:
        List of bullet-line strings, possibly with a trailing
        "(N more older items, oldest: ...)" summary if the cap was hit.
    """
    items = list(lines)
    if not items:
        return []

    total_chars = sum(len(s) for s in items)
    if len(items) <= max_items and total_chars <= max_chars:
        return items

    # Cap fired. Keep the most recent items, collapse the rest.
    # We assume the input is in some meaningful order (caller's choice).
    # If caller ordered oldest-first, we keep tail; if newest-first, keep head.
    # We default to keep-tail since most state-log inputs are append-only
    # (oldest-first). Caller can reverse before passing if they want the
    # opposite.
    keep = max(1, max_items // 2)
    kept = items[-keep:]
    collapsed = items[:-keep]
    if not collapsed:
        return kept
    oldest = collapsed[0]
    # Truncate the oldest preview so the summary line itself doesn't blow chars
    preview = oldest if len(oldest) <= 80 else (oldest[:77] + "...")
    summary = f"({len(collapsed)} older item(s), oldest: {preview})"
    return [summary] + kept
