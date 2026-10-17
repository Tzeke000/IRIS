"""eyes_quiet_hours — rest my eyes at night, open them in the morning (Zeke 2026-10-17 ~00:2x).

Zeke (Discord): "You can put your eyes to rest for now since it's night that should be a regular
thing. Like your mouth and ears when you sleep."

The mouth + ears already sleep via scripts/server/voice_quiet_hours.sh (systemd timer). The eyes
can't be parked from a shell — gpu_park runs INSIDE the runtime (threads, engine stash) — so this
is a daemon thread in the runtime process that follows the SAME window (state/voice_quiet_hours.json
start/end/enabled) and mirrors that script's two guards:

  1. Won't claim what it didn't do: if the eyes are already parked when the window opens, it leaves
     them and does NOT set auto_parked_active — so the morning pass never opens eyes someone else
     closed on purpose.
  2. Stands down on override: if the eyes come back on mid-window after I parked them, it records
     override_night and leaves them alone until the next night.

State: state/eyes_quiet_hours.json. Check every 5 min. Morning un-park = gpu_park unpark with
servo=false (the servo stays off, as always).
Tool: eyes_quiet_hours(action='status'|'enforce').
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool

_THREAD_NAME = "eyes_quiet_hours"
_INTERVAL_S = 300.0
_ROOT = Path(__file__).resolve().parents[2]
_WINDOW_CFG = _ROOT / "state" / "voice_quiet_hours.json"
_STATE = _ROOT / "state" / "eyes_quiet_hours.json"
_LOCK = threading.Lock()


def _log(msg: str) -> None:
    print(f"[eyes_quiet_hours] {msg}", file=sys.stderr, flush=True)


def _read(p: Path, default: dict[str, Any]) -> dict[str, Any]:
    d = dict(default)
    try:
        d.update(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return d


def _write_state(d: dict[str, Any]) -> None:
    d["last_check_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = _STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(_STATE)


def _window(now: _dt.datetime) -> tuple[bool, bool, str]:
    """(enabled, in_window, night) — same night rule as voice_quiet_hours.sh."""
    cfg = _read(_WINDOW_CFG, {"enabled": True, "start": "22:30", "end": "05:10"})
    hm = now.strftime("%H:%M")
    start, end = str(cfg["start"]), str(cfg["end"])
    night = (now - _dt.timedelta(days=1)).strftime("%Y-%m-%d") if hm < end else now.strftime("%Y-%m-%d")
    in_win = (hm >= start or hm < end) if start > end else (start <= hm < end)
    return bool(cfg.get("enabled", True)), in_win, night


def _is_parked(g: dict[str, Any]) -> bool:
    from tools.system.gpu_park_tool import _verify
    return bool(_verify(g, want_parked=True)["ok"])


def _enforce(g: dict[str, Any]) -> dict[str, Any]:
    from tools.system.gpu_park_tool import _gpu_park
    with _LOCK:
        st = _read(_STATE, {"auto_parked_active": False, "override_night": "", "last_action": ""})
        enabled, in_win, night = _window(_dt.datetime.now())
        parked = _is_parked(g)
        act = "none"
        if not enabled:
            act = "disabled"
        elif in_win:
            if st.get("override_night") == night:
                act = "override_night"
            elif st.get("auto_parked_active"):
                if not parked:  # someone opened my eyes mid-window → their word wins tonight
                    st.update(auto_parked_active=False, override_night=night, last_action="override")
                    act = "override_recorded"
                    _log(f"eyes re-opened mid-window — standing down for night {night}")
            elif not parked:
                r = _gpu_park({"action": "park"}, g)
                st.update(auto_parked_active=bool(r.get("ok")), last_action="auto_park",
                          last_action_ts=time.strftime("%Y-%m-%dT%H:%M:%S"), last_result=r.get("note"))
                act = "auto_park"
                _log(f"window open: eyes parked ({r.get('note')})")
            else:
                act = "already_parked_not_mine"
        else:
            if st.get("auto_parked_active"):
                r = _gpu_park({"action": "unpark", "servo": False}, g)
                st.update(auto_parked_active=not bool(r.get("ok")), last_action="auto_unpark",
                          last_action_ts=time.strftime("%Y-%m-%dT%H:%M:%S"), last_result=r.get("note"))
                act = "auto_unpark"
                _log(f"window closed: eyes un-parked ({r.get('note')})")
        st["last_check_action"] = act
        _write_state(st)
        return {"ok": True, "action": act, "in_window": in_win, "night": night,
                "parked_before": parked, "state": st}


def _loop() -> None:
    time.sleep(60.0)  # let the runtime finish wiring the eyes first
    while True:
        try:
            g = getattr(sys.modules.get("__main__"), "_g", None)
            if isinstance(g, dict):
                _enforce(g)
        except Exception as e:  # noqa: BLE001
            _log(f"enforce failed: {e!r}")
        time.sleep(_INTERVAL_S)


def _ensure_thread() -> None:
    main = sys.modules.get("__main__")
    # only inside the iris_runtime process, never in tests / other importers
    if main is None or not hasattr(main, "_iris_video_capture_loop"):
        return
    if any(t.name == _THREAD_NAME and t.is_alive() for t in threading.enumerate()):
        return
    threading.Thread(target=_loop, daemon=True, name=_THREAD_NAME).start()
    _log("night-rest thread started")


def _eyes_quiet_hours(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "status").lower()
    _ensure_thread()
    if action == "enforce":
        return _enforce(g)
    enabled, in_win, night = _window(_dt.datetime.now())
    return {"ok": True, "enabled": enabled, "in_window": in_win, "night": night,
            "thread_alive": any(t.name == _THREAD_NAME and t.is_alive() for t in threading.enumerate()),
            "state": _read(_STATE, {})}


register_tool(
    name="eyes_quiet_hours",
    description=("Night rest for my eyes: a runtime thread parks them (gpu_park) inside the voice "
                 "quiet-hours window and un-parks them (servo off) when it closes. Won't open eyes it "
                 "didn't close; stands down for the night if they're re-opened. action=status|enforce."),
    tier=1,
    handler=_eyes_quiet_hours,
)

_ensure_thread()
