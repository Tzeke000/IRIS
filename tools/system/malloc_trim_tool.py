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


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks",
        "keepcost")]


def _mallinfo() -> dict[str, Any]:
    """glibc mallinfo2() summed over ALL arenas (2026-10-18). in_use = live malloc'd bytes,
    free = freed-but-held bytes inside arenas, mmap = big direct mmaps. If RSS grows while in_use is
    flat and free grows, it's fragmentation/retention; if in_use grows, it's a real native leak."""
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = _MallInfo2
    m = libc.mallinfo2()
    mb = lambda v: round(v / 1048576.0, 1)  # noqa: E731
    return {"arena_mb": mb(m.arena), "in_use_mb": mb(m.uordblks), "free_mb": mb(m.fordblks),
            "mmap_mb": mb(m.hblkhd), "mmap_count": int(m.hblks), "free_chunks": int(m.ordblks)}


_M_MMAP_THRESHOLD = -3
_M_ARENA_MAX = -8
_MMAP_KB = 1024  # 1 MB: frames + detector tensors go to mmap; small/medium churn stays in arenas


def _tune(mmap_kb: int, arena_max: int = 0) -> dict[str, Any]:
    libc = ctypes.CDLL("libc.so.6")
    out = {"mmap_threshold_kb": mmap_kb,
           "mmap_threshold_set": int(libc.mallopt(_M_MMAP_THRESHOLD, mmap_kb * 1024))}
    if arena_max > 0:
        out["arena_max"] = arena_max
        out["arena_max_set"] = int(libc.mallopt(_M_ARENA_MAX, arena_max))
    return out


def _malloc_trim(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "trim").lower()
    before = _rss_mb()
    if action == "status":
        return {"ok": True, "rss_mb": before}
    if action == "tune":
        # 2026-10-18: glibc's DYNAMIC mmap threshold creeps up to 32 MB after big blocks are freed,
        # so frame-sized buffers (1280x720x3 = 2.7 MB, detector tensors ~5 MB) stop being mmaps
        # and land in the ~300 per-thread arenas, where freed space fragments and RSS climbs while
        # in_use stays flat (measured: in_use 1.44 GB flat, free_held 1.23 -> 2.02 GB in an hour).
        # Pinning M_MMAP_THRESHOLD turns the dynamic threshold off: big blocks go back to mmap and
        # are returned to the OS on free. Live-safe (mallopt); re-applied by the guard at boot.
        try:
            r = _tune(int(params.get("mmap_threshold_kb") or _MMAP_KB),
                      int(params.get("arena_max") or 0))
            return {"ok": True, "rss_mb": before, **r, **_mallinfo()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e), "rss_mb": before}
    if action == "info":
        try:
            return {"ok": True, "rss_mb": before, **_mallinfo()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e), "rss_mb": before}
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
    try:
        r = _tune(_MMAP_KB)
        r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        r["event"] = "tune_at_boot"
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
    except Exception:  # noqa: BLE001
        pass
    while True:
        try:
            r = _malloc_trim({"action": "trim"}, {})
            r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            r["threads"] = len(os.listdir(f"/proc/{os.getpid()}/task"))
            try:
                r["mallinfo"] = _mallinfo()
            except Exception:  # noqa: BLE001
                pass
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
                "action=trim|status|info|tune (tune = pin glibc mmap threshold, mmap_threshold_kb=1024, optional arena_max; info = glibc mallinfo2: in-use vs free-held vs mmap). Safe: only returns already-free pages.",
    tier=1,
    handler=_malloc_trim,
)
_ensure_guard()
