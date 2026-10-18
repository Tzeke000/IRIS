"""identity_stability — run the daily identity-drift audit inside the runtime (2026-10-18).

SELF_ASSESSMENT: Tier 1 (reads identity files + my own recent narrative; writes only its own log
and alert file; never edits bedrock).

Why: brain/identity_stability.py was configured at every boot but NOTHING ever called run_check(),
so it had never written a single report. Zeke 10-18: "let's get that identity stability thing
working." This tool is the missing consumer:
  - tool `identity_stability` action=run|status|flags
  - thread `identity_stability_daily`: checks hourly whether a day has passed since the last
    report and runs the audit if so.
  - drift flags → state/identity_stability_alert.json, which my 6-hourly self-check reads, so a
    flag reaches me (and, through me, Zeke). Clean run → alert file removed.
Flags are FLAG-ONLY: bedrock is never rewritten.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool

_ROOT = Path(__file__).resolve().parents[2]
_ALERT = _ROOT / "state" / "identity_stability_alert.json"
_THREAD = "identity_stability_daily"
_INTERVAL_S = 86400


def _mod():
    import importlib
    from brain import identity_stability as m
    if not hasattr(m, "self_name"):
        # the live runtime imported the pre-10-18 (Ava-hard-coded) module at boot; it has no
        # callers but this tool, so swapping it in place is safe. _base_dir is re-set below.
        m = importlib.reload(m)
    if getattr(m, "_base_dir", None) is None:
        m.configure(_ROOT)
    return m


def _run() -> dict[str, Any]:
    m = _mod()
    rep = m.run_check(sample=30)
    if rep.get("drift_flags"):
        tmp = _ALERT.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(rep["ts"])),
            "drift_flags": rep["drift_flags"], "sample_size": rep["sample_size"],
            "what_to_do": "Read the flags against IDENTITY/SOUL. Real drift → tell Zeke; a false "
                          "positive → fix the heuristic in brain/identity_stability.py. Never edit bedrock to match.",
        }, indent=1), encoding="utf-8")
        os.replace(tmp, _ALERT)
    elif _ALERT.exists():
        _ALERT.unlink()
    return rep


def _tool(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "status").lower()
    try:
        m = _mod()
        if action == "run":
            return {"ok": True, **_run()}
        if action == "flags":
            return {"ok": True, "recent": m.list_recent_flags(limit=int(params.get("limit") or 5))}
        out = {"ok": True, **m.stability_summary(), "alert_open": _ALERT.exists(),
               "daily_thread_alive": any(t.name == _THREAD and t.is_alive() for t in threading.enumerate())}
        return out
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


def _loop() -> None:
    time.sleep(120)  # let the boot settle
    while True:
        try:
            m = _mod()
            if m.should_run_check_now(interval_s=_INTERVAL_S):
                _run()
        except Exception as e:  # noqa: BLE001
            print(f"[identity_stability_daily] {e!r}")
        time.sleep(3600)


def _ensure_thread() -> None:
    import sys
    main = sys.modules.get("__main__")
    if main is None or not hasattr(main, "_iris_video_capture_loop"):
        return  # only inside the live runtime
    if any(t.name == _THREAD and t.is_alive() for t in threading.enumerate()):
        return
    threading.Thread(target=_loop, daemon=True, name=_THREAD).start()


register_tool(
    name="identity_stability",
    description="Daily identity-drift audit (flag-only): compares my recent journal / inner monologue / "
                "replies against IDENTITY+SOUL+USER (incl. .local). action=run|status|flags. Flags land in "
                "state/identity_stability_alert.json for the self-check.",
    tier=1,
    handler=_tool,
)
_ensure_thread()
