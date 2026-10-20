"""liveness_gate — the LIVE use of face liveness (Zeke 2026-10-20: "it should definitely be ...
actually wired in").

The shadow runner (brain/face_liveness.py) scores every face the live engine reports and writes
its current per-track verdicts to state/face_liveness_now.json (2 Hz while faces are present).
This module is the tiny, dependency-free reader that the wake paths consult:

  * iris_body_host  — a committed face_appeared|unknown wake is turned into a quiet ledger line
                      when EVERY unnamed face in view is a confirmed fake (face_like / decoy:*).
  * unknown_capture — an unnamed face that is a confirmed fake is not "an unknown person", so it
                      doesn't open a draft or trigger the guest/away capture rules.

Rules (deliberately one-sided — a missed photo costs one extra wake, a missed PERSON costs a
stranger nobody looked at; UNKNOWN OUTRANKS KNOWN, Zeke 09-02):
  * Gate is OFF unless the flag file says  "gate": true  (state/face_liveness_shadow.json).
  * A stale/missing/unreadable snapshot => never suppress (fail-open).
  * Only 'face_like' and 'decoy:*' count as fake. 'visitor' and 'uncertain' always wake.
  * A recognised person is never touched (the gate only looks at unnamed faces).
Pure stdlib, no models: safe to import from the host process.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
NOW_PATH = ROOT / "state" / "face_liveness_now.json"
FLAG_PATH = ROOT / "state" / "face_liveness_shadow.json"
MAX_AGE_S = 4.0


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def gate_on(flag_path: Path = FLAG_PATH) -> bool:
    f = _read(flag_path) or {}
    return bool(f.get("on")) and bool(f.get("gate"))


def is_fake(verdict: Any) -> bool:
    v = str(verdict or "")
    return v == "face_like" or v.startswith("decoy:")


def _unnamed(pid: Any) -> bool:
    p = str(pid or "unknown")
    return p == "unknown" or p.startswith("_")


def fresh_tracks(max_age_s: float = MAX_AGE_S, now_path: Path = NOW_PATH,
                 now: Optional[float] = None) -> Optional[list[dict]]:
    """Live-engine tracks from a FRESH snapshot, or None when there is no trustworthy snapshot."""
    d = _read(now_path)
    if not isinstance(d, dict):
        return None
    t = time.time() if now is None else now
    try:
        if t - float(d.get("ts") or 0.0) > max_age_s:
            return None
    except Exception:
        return None
    out = []
    for tr in d.get("tracks") or []:
        if not isinstance(tr, dict) or tr.get("source") != "live":
            continue
        try:
            if t - float(tr.get("last_ts") or 0.0) > max_age_s:
                continue
        except Exception:
            continue
        out.append(tr)
    return out


def judge_unknown_wake(max_age_s: float = MAX_AGE_S, now_path: Path = NOW_PATH,
                       flag_path: Path = FLAG_PATH, now: Optional[float] = None) -> dict:
    """Should a face_appeared|unknown wake be quieted? suppress=True only when the gate is on,
    the snapshot is fresh, at least one unnamed face is in view, and ALL unnamed faces are fake."""
    try:
        if not gate_on(flag_path):
            return {"suppress": False, "reason": "gate off"}
        tracks = fresh_tracks(max_age_s, now_path, now)
        if tracks is None:
            return {"suppress": False, "reason": "no fresh liveness snapshot"}
        unnamed = [t for t in tracks if _unnamed(t.get("live_pid"))]
        if not unnamed:
            return {"suppress": False, "reason": "no unnamed face in the liveness snapshot"}
        verdicts = [str(t.get("would")) for t in unnamed]
        if all(is_fake(v) for v in verdicts):
            return {"suppress": True, "reason": "every unnamed face is a fake: " + ", ".join(verdicts),
                    "verdicts": verdicts}
        return {"suppress": False, "reason": "unnamed face not confirmed fake: " + ", ".join(verdicts),
                "verdicts": verdicts}
    except Exception as e:  # noqa: BLE001 - fail-open
        return {"suppress": False, "reason": "error: " + repr(e)[:120]}


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def is_fake_face(bbox, max_age_s: float = MAX_AGE_S, now_path: Path = NOW_PATH,
                 flag_path: Path = FLAG_PATH, now: Optional[float] = None) -> bool:
    """True when this live-engine face box matches a fresh track the runner judged fake."""
    try:
        if not bbox or not gate_on(flag_path):
            return False
        tracks = fresh_tracks(max_age_s, now_path, now) or []
        for t in tracks:
            if is_fake(t.get("would")) and t.get("bbox") and _iou(bbox, t["bbox"]) >= 0.3:
                return True
    except Exception:
        return False
    return False
