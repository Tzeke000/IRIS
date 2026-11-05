"""MCP CALL LEDGER (2026-11-05, docket I6) — makes the next 180 s MCP hang name its own cause.

11-04 evening: five tool calls (wake_verdict, time_check, vr_temple follow, vr_temple speak ×2) hung until the CLI's
180 s timeout, and in every case the call issued right after completed in ~20 ms. The CLI's log
(~/.cache/claude-cli-nodejs/.../mcp-logs-iris) shows the calls were strictly sequential (nothing else in flight), so
"the loop was busy with another call" does not explain it — and the runtime kept no record of its own side, so the
three real possibilities could not be told apart:
  A. the request never reached a handler (lost/stalled in the stdio transport)  → no "start" row
  B. the handler ran and blocked (a lock, a socket, a disk stall)               → "start", no "end", + its stack
  C. the event loop itself was blocked (a sync tool inline, a CPU hog)          → the loop-lag probe fails
This wraps FastMCP's tool manager: one JSONL row per call start/end (state/mcp_calls.jsonl, capped), and a watchdog
thread that dumps EVERY thread's stack + whether the loop still answers into state/mcp_slow_calls.log the first time a
call passes SLOW_DUMP_S. Pure observation: it never changes a call's result or timing.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

SLOW_DUMP_S = 60.0
MAX_LOG_BYTES = 4_000_000

_lock = threading.Lock()
_inflight: dict[int, dict[str, Any]] = {}
_seq = 0
_loop: "asyncio.AbstractEventLoop | None" = None
_paths: dict[str, Path] = {}


def label(name: str, arguments: Any) -> str:
    """tool name + the bit that says WHICH call it was (iris_tool_call's tool, vr_temple's action)"""
    lab = str(name)
    try:
        if name == "iris_tool_call" and isinstance(arguments, dict):
            lab += ":" + str(arguments.get("name", "?"))
            p = arguments.get("params")
            if isinstance(p, dict) and "action" in p:
                lab += "." + str(p.get("action"))[:24]
    except Exception:  # noqa: BLE001
        pass
    return lab[:96]


def _write(path: Path, line: str) -> None:
    try:
        if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
            os.replace(path, path.with_suffix(path.suffix + ".1"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:  # noqa: BLE001  (the ledger must never cost a call)
        pass


def _row(d: dict[str, Any]) -> None:
    if "calls" in _paths:
        _write(_paths["calls"], json.dumps(d) + "\n")


def loop_lag_ms(timeout: float = 2.0) -> "float | None":
    """from ANOTHER thread: how long the event loop takes to run a no-op (None = it didn't within timeout = blocked)"""
    lp = _loop
    if lp is None or not lp.is_running():
        return None
    ev = threading.Event()
    t0 = time.monotonic()
    try:
        lp.call_soon_threadsafe(ev.set)
    except RuntimeError:
        return None
    return round((time.monotonic() - t0) * 1000.0, 1) if ev.wait(timeout) else None


def dump_slow(rec: dict[str, Any], now: float) -> str:
    lag = loop_lag_ms()
    out = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} SLOW MCP CALL #{rec['seq']} {rec['label']} — running "
           f"{now - rec['t0']:.0f} s; event loop {'answers in %.1f ms' % lag if lag is not None else 'DID NOT ANSWER in 2 s (BLOCKED)'}; "
           f"{len(_inflight)} call(s) in flight; {threading.active_count()} threads"]
    names = {t.ident: t.name for t in threading.enumerate()}
    for tid, fr in sys._current_frames().items():
        stack = traceback.extract_stack(fr)
        if not any("/IRIS/" in f.filename or "/mcp/" in f.filename or "anyio" in f.filename for f in stack):
            continue                                       # (skip idle pool threads + native workers: keep it readable)
        out.append(f"--- thread {names.get(tid, '?')} ({tid})")
        out.extend("  " + ln.rstrip() for ln in traceback.format_list(stack[-14:]))
    txt = "\n".join(out) + "\n"
    if "slow" in _paths:
        _write(_paths["slow"], txt)
    return txt


def _watch() -> None:
    while True:
        time.sleep(5.0)
        now = time.time()
        with _lock:
            due = [r for r in _inflight.values() if not r.get("dumped") and now - r["t0"] > SLOW_DUMP_S]
            for r in due:
                r["dumped"] = True
        for r in due:
            try:
                dump_slow(r, now)
                _row({"ev": "slow", "seq": r["seq"], "label": r["label"], "s": round(now - r["t0"], 1)})
            except Exception:  # noqa: BLE001
                pass


def note_loop(loop: "asyncio.AbstractEventLoop") -> None:
    global _loop
    _loop = loop


def install(tool_manager: Any, state_dir: Path) -> bool:
    """wrap tool_manager.call_tool (an instance attribute: FastMCP calls self._tool_manager.call_tool per request)"""
    global _seq
    if getattr(tool_manager, "_iris_ledger", False):
        return False
    _paths["calls"] = Path(state_dir) / "mcp_calls.jsonl"
    _paths["slow"] = Path(state_dir) / "mcp_slow_calls.log"
    orig = tool_manager.call_tool

    async def call_tool(name: str, arguments: dict[str, Any], *a: Any, **k: Any) -> Any:
        global _seq
        with _lock:
            _seq += 1
            rec = {"seq": _seq, "label": label(name, arguments), "t0": time.time()}
            _inflight[rec["seq"]] = rec
        _row({"ev": "start", "seq": rec["seq"], "label": rec["label"], "ts": round(rec["t0"], 3)})
        ok = "error"
        try:
            res = await orig(name, arguments, *a, **k)
            ok = "ok"
            return res
        except BaseException as e:  # noqa: BLE001
            ok = type(e).__name__
            raise
        finally:
            with _lock:
                _inflight.pop(rec["seq"], None)
            _row({"ev": "end", "seq": rec["seq"], "label": rec["label"], "ts": round(time.time(), 3),
                  "ms": round((time.time() - rec["t0"]) * 1000.0, 1), "result": ok})

    tool_manager.call_tool = call_tool
    tool_manager._iris_ledger = True
    threading.Thread(target=_watch, name="mcp-call-ledger", daemon=True).start()
    return True
