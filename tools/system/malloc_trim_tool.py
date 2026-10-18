"""malloc_trim — hand freed heap memory back to the OS and report RSS before/after (2026-10-18).

Why: on 10-18 the runtime's RSS grew ~50–100 MB/min while the Python heap stayed small (heap_census:
numpy 0 GB). smaps showed the growth inside glibc malloc arenas (64 MB-aligned anon regions), the
classic signature of arena retention/fragmentation from large short-lived native buffers (video frames,
ONNX tensors), not of live Python objects. glibc's malloc_trim(0) releases free pages in every arena;
if RSS drops a lot, the "leak" was retention, not a real leak. Tool: malloc_trim(action='trim'|'status').
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

from tools.tool_registry import register_tool


def _rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return -1.0


def _malloc_trim(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "trim").lower()
    before = _rss_mb()
    if action == "status":
        return {"ok": True, "rss_mb": before}
    try:
        libc = ctypes.CDLL("libc.so.6")
        released = int(libc.malloc_trim(0))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e), "rss_mb": before}
    after = _rss_mb()
    return {"ok": True, "released_flag": released, "rss_before_mb": before, "rss_after_mb": after,
            "freed_mb": round(before - after, 1)}


_THREAD = "malloc_trim_guard"
_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "state", "runtime_rss.jsonl")


def _guard() -> None:
    """Every 5 min: trim, and log RSS before/after, so the history shows WHEN growth happens."""
    import json
    import time
    time.sleep(30)
    while True:
        try:
            r = _malloc_trim({"action": "trim"}, {})
            r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            r["threads"] = len(os.listdir(f"/proc/{os.getpid()}/task"))
            with open(_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            if os.path.getsize(_LOG) > 2_000_000:
                os.replace(_LOG, _LOG + ".1")
        except Exception:  # noqa: BLE001
            pass
        time.sleep(300)


def _ensure_guard() -> None:
    import sys
    import threading
    main = sys.modules.get("__main__")
    if main is None or not hasattr(main, "_iris_video_capture_loop"):
        return  # only inside the live runtime
    if any(t.name == _THREAD and t.is_alive() for t in threading.enumerate()):
        return
    threading.Thread(target=_guard, daemon=True, name=_THREAD).start()


register_tool(
    name="malloc_trim",
    description="Release freed glibc heap memory to the OS (malloc_trim(0)) and report runtime RSS before/after. "
                "action=trim|status. Safe: only returns already-free pages.",
    tier=1,
    handler=_malloc_trim,
)
_ensure_guard()
