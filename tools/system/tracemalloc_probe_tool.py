"""tracemalloc_probe — name the Python line that is growing the runtime's RSS (2026-10-18).

SELF_ASSESSMENT: Tier 1 (read-only introspection; tracemalloc adds allocation overhead while
started, so stop it when done).

Why: after the 10-18 restart RSS still grew ~10-20 MB/min after malloc_trim. smaps put the growth
in ONE glibc arena, and a raw dump of that arena was full of transcript / chat-prompt / mood
strings, i.e. Python-made data, not ONNX/OpenCV buffers. heap_census only counts objects by type;
this diffs two tracemalloc snapshots so the answer is a file:line plus how many MB it gained.

Actions:
  start    tracemalloc.start(nframes) and take the baseline snapshot
  diff     snapshot now, top-N sites by growth since the baseline (by traceback, the leading frames)
  stop     tracemalloc.stop() (frees its bookkeeping)
  status   is it tracing, traced current/peak MB
If tracemalloc's traced total stays flat while RSS grows, the growth is NATIVE (C/C++ malloc),
and the Python view can't see it. That result is useful too.
"""
from __future__ import annotations

import os
import time
import tracemalloc
from typing import Any

from tools.tool_registry import register_tool

_BASE: dict[str, Any] = {"snap": None, "ts": 0.0, "rss": 0.0}


def _rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return -1.0


def _fmt_tb(tb: tracemalloc.Traceback, depth: int) -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = []
    # Traceback is oldest→newest; the allocating line is the LAST frame, so keep the innermost.
    for fr in reversed(list(tb)[-depth:]):
        fn = fr.filename.replace(root + os.sep, "")
        if "site-packages" in fn:
            fn = "…/" + fn.split("site-packages/")[-1]
        out.append(f"{fn}:{fr.lineno}")
    return out


def _probe(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "status").lower()
    if action == "start":
        nframes = int(params.get("nframes") or 8)
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start(nframes)
        _BASE.update(snap=tracemalloc.take_snapshot(), ts=time.time(), rss=_rss_mb())
        return {"ok": True, "started": True, "nframes": nframes, "rss_mb": _BASE["rss"]}
    if action == "stop":
        was = tracemalloc.is_tracing()
        tracemalloc.stop()
        _BASE.update(snap=None)
        return {"ok": True, "stopped": was}
    if action == "status":
        if not tracemalloc.is_tracing():
            return {"ok": True, "tracing": False, "rss_mb": _rss_mb()}
        cur, peak = tracemalloc.get_traced_memory()
        return {"ok": True, "tracing": True, "traced_mb": round(cur / 1e6, 1),
                "peak_mb": round(peak / 1e6, 1), "rss_mb": _rss_mb(),
                "since_s": round(time.time() - _BASE["ts"])}
    if action == "diff":
        if not tracemalloc.is_tracing() or _BASE["snap"] is None:
            return {"ok": False, "error": "not started — call action=start first"}
        top = int(params.get("top") or 15)
        depth = int(params.get("depth") or 4)
        key = "traceback" if depth > 1 else "lineno"
        snap = tracemalloc.take_snapshot().filter_traces(
            [tracemalloc.Filter(False, tracemalloc.__file__)])
        stats = snap.compare_to(_BASE["snap"], key)
        cur, peak = tracemalloc.get_traced_memory()
        mins = max(1e-6, (time.time() - _BASE["ts"]) / 60.0)
        rows = []
        for st in stats[:top]:
            rows.append({"growth_mb": round(st.size_diff / 1e6, 2), "now_mb": round(st.size / 1e6, 2),
                         "count_diff": st.count_diff, "where": _fmt_tb(st.traceback, depth)})
        rss = _rss_mb()
        return {"ok": True, "minutes": round(mins, 1), "rss_mb": rss,
                "rss_growth_mb": round(rss - _BASE["rss"], 1),
                "traced_total_mb": round(cur / 1e6, 1),
                "traced_growth_total_mb": round(sum(s.size_diff for s in stats) / 1e6, 1),
                "top": rows}
    return {"ok": False, "error": f"unknown action {action!r}"}


register_tool(
    name="tracemalloc_probe",
    description="Find the Python line growing runtime memory: action=start|diff|status|stop. "
                "diff lists top allocation sites by net growth since start (top=N, depth=frames). "
                "Adds overhead while tracing — stop when done.",
    tier=1,
    handler=_probe,
)
