"""
brain/iris_transcript.py — shared conversation log.

One durable JSONL at state/transcript.jsonl. Both voice and chat write here
so the orb's history view is unified across modalities.

Schema per line:
  {"id": "<12-hex>", "ts": <unix>, "iso": "<rfc3339>",
   "role": "user"|"assistant",
   "source": "zeke"|"iris",
   "modality": "voice"|"chat",
   "content": "<text>"}

Append-only. Bounded by trim_to_last(N) on read paths — we keep all turns on
disk forever (cheap), the orb just gets the most recent slice.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


_DEFAULT_PATH = "state/transcript.jsonl"
_LOCK = threading.Lock()
_BASE: Path | None = None


def configure(base_dir: Path | str) -> None:
    global _BASE
    _BASE = Path(base_dir)
    (_BASE / "state").mkdir(parents=True, exist_ok=True)


def _path() -> Path:
    base = _BASE if _BASE is not None else Path(".")
    return base / _DEFAULT_PATH


def append(
    role: str,
    content: str,
    source: str,
    modality: str,
) -> dict[str, Any]:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        # 10-29 code review: aware local time (RFC 3339 with offset), not a naive string.
        "iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "role": str(role),
        "source": str(source),
        "modality": str(modality),
        "content": str(content or "")[:8000],
    }
    p = _path()
    # 10-29 code review: ONE os.write on an O_APPEND fd. A buffered text write of a long line can
    # be split across syscalls, so two processes (runtime + body host) could interleave lines.
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            n = os.write(fd, data)
            if n < len(data):          # practically never for a regular file; finish the line
                os.write(fd, data[n:])
            # Round-2 fix 3.9 (2026-10-01): fsync each turn. The power-loss zero-fill that left
            # 12,889 NUL bytes mid-file (every reader silently skipped them) came from unflushed
            # pages; one fsync per conversation turn bounds the loss to the last entry.
            try:
                os.fsync(fd)
            except Exception:
                pass
        finally:
            os.close(fd)
    return entry


_TAIL_CHUNK = 64 * 1024


def _tail_lines(p: Path, n: int) -> list[str]:
    """Last n lines without reading the whole file (10-29 code review: recent() re-read the
    entire, ever-growing transcript on every call). Reads backwards in 64 KB chunks."""
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        pos = end
        data = b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(_TAIL_CHUNK, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = data.decode("utf-8", "replace").splitlines()
    if pos > 0 and lines:
        lines = lines[1:]          # first line may be partial
    return lines[-n:]


def recent(n: int = 100) -> list[dict[str, Any]]:
    p = _path()
    if not p.is_file():
        return []
    n = max(1, int(n))
    try:
        lines = _tail_lines(p, n)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if isinstance(e, dict):
                out.append(e)
        except Exception:
            pass
    return out
