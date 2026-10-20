"""liveness_record — save CLEAN raw camera frames for the face-liveness eval (2026-10-20).

SELF_ASSESSMENT: Tier 1 (reads the in-process raw frame slot; writes JPEGs under scratch/liveness/ only).

Why: /api/v1/vision/latest_frame serves the DISPLAY frame (face boxes, landmarks, labels drawn on
it), which would poison an anti-spoof score. The eval (scripts/vision/liveness_eval.py) needs the
raw PIXY frames, and nobody may open the camera from outside the runtime. This tool reads
g["_raw_frame_slot"] in a background thread and writes frames into
scratch/liveness/<session>/<clip>/ at a fixed rate.

  liveness_record action=start clip=real_zeke_still_60s seconds=60 fps=5 [session=monday]
  liveness_record action=status
  liveness_record action=stop
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool

_ROOT = Path(__file__).resolve().parents[2]
_state: dict[str, Any] = {"running": False}
_stop = threading.Event()


def _loop(g: dict[str, Any], out: Path, seconds: float, fps: float) -> None:
    import cv2  # type: ignore
    period = 1.0 / fps
    t_end = time.time() + seconds
    last_ts = 0.0
    try:
        while time.time() < t_end and not _stop.is_set():
            slot = g.get("_raw_frame_slot")
            if slot and slot[0] > last_ts:
                last_ts, frame = slot[0], slot[1]
                name = out / f"{time.strftime('%H%M%S', time.localtime(last_ts))}_{int((last_ts % 1) * 1000):03d}.jpg"
                cv2.imwrite(str(name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                _state["saved"] = _state.get("saved", 0) + 1
            else:
                _state["stale_polls"] = _state.get("stale_polls", 0) + 1
            time.sleep(period)
    except Exception as e:  # noqa: BLE001
        _state["error"] = repr(e)
    finally:
        _state["running"] = False
        _state["ended_ts"] = time.time()


def _tool(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "status").lower()
    if action == "status":
        return {"ok": True, **_state}
    if action == "stop":
        _stop.set()
        return {"ok": True, "stopping": True, **_state}
    if action != "start":
        return {"ok": False, "error": f"unknown action {action!r}"}
    if _state.get("running"):
        return {"ok": False, "error": "already recording", **_state}
    if not g.get("_raw_frame_slot"):
        return {"ok": False, "error": "no _raw_frame_slot - capture loop not running (eyes parked?)"}
    clip = str(params.get("clip") or "")
    session = str(params.get("session") or "monday")
    if not re.fullmatch(r"(real|attack)_[a-z0-9_]{1,60}", clip) or not re.fullmatch(r"[a-z0-9_]{1,40}", session):
        return {"ok": False, "error": "clip must look like real_* / attack_* (lowercase), session [a-z0-9_]"}
    seconds = max(1.0, min(float(params.get("seconds") or 30), 180.0))
    fps = max(0.5, min(float(params.get("fps") or 5), 15.0))
    out = _ROOT / "scratch" / "liveness" / session / clip
    out.mkdir(parents=True, exist_ok=True)
    _stop.clear()
    _state.clear()
    _state.update(running=True, clip=clip, out=str(out), seconds=seconds, fps=fps,
                  started_ts=time.time(), saved=0, stale_polls=0)
    threading.Thread(target=_loop, args=(g, out, seconds, fps), daemon=True,
                     name="liveness-record").start()
    return {"ok": True, **_state}


register_tool(
    name="liveness_record",
    description="Record CLEAN raw PIXY frames (no overlays) for the face-liveness eval into "
                "scratch/liveness/<session>/<clip>/. action=start clip=real_*|attack_* seconds fps | status | stop.",
    tier=1,
    handler=_tool,
)
