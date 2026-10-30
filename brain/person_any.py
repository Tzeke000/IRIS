"""person_any — follow ANYONE in the room, and notice a genuine stranger (Zeke, 2026-10-30).

Zeke's ask: the head should be able to FOLLOW ANYONE, not only people it recognises —
following an unknown person keeps their face in view, so recognition gets its chance when
they turn toward me. And a face that is looking STRAIGHT at the camera and is STILL
unrecognised is a genuine stranger: study it (photos via study_face) and DM him. SILENT
ONLY — "they're like higher important people ... no need to say hi" (Zeke 10-30). Nothing
in here can speak; nothing in here may ever call voice_speak.

Two pure pieces (offline-testable, synthetic detections — tests/test_person_any.py):

  AnyPicker     the `person:any` servo target. Each tick: a recognised OWNER (zeke) if
                present > else the largest (closest) REAL face > else a body track ONLY if a
                real face sat inside that track within CORROB_S (a body track alone is fooled
                by the statue/helmet — verify by FACE). Hysteresis so it never flips between
                people frame-to-frame; when the current person is missed it HOLDS (returns
                None = the servo's lost rails: zero vector, home after 8 s) — never hunts.

  StrangerGate  the frontal-unknown gate. A face counts as a fair recognition attempt only
                if it is unnamed, frontal (|yaw|,|pitch| <= 20 deg from the 3D pose, else a
                5-point-landmark symmetry estimate), >= 64 px, det_score >= 0.60, sharp
                enough, best similarity to every enrolled person < 0.30, and not a known fake
                (liveness gate: photo/screen/decoy:helmet) or a remembered static shape.
                >= 5 such observations of the same face over >= 2 s, with no recognition in
                between => a 'stranger' trigger. 'unrecognised' at an angle is NOT a stranger
                (CORE: "'unrecognised' = can't see the face from here") — that's why the
                frontal gate exists.

  AlertLimiter  one DM per stranger (keyed by the photographic-memory seen_id) per ALERT_COOLDOWN_S.

Wiring (all INERT until the runtime restarts / brain_hot_swap — see the commit):
  * brain/visual_attention.canonical_target: 'anyone'/'anybody'/'any'/'person:any' -> 'person:any'
  * brain/visual_attention.observe: person:any resolves through match_face() here
  * tools/system/attention_smooth_tool: person:any's fast path is servo_offset() here
  * brain/unknown_capture watcher: runs the StrangerGate at its 2 Hz poll, then study_face
    + a Discord DM (brain.host_hold.discord_send) + an `unknown_capture` signal
    (rule='stranger-frontal') so cognition can send the crop.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Optional

OWNER = "zeke"

# ── picker tuning ───────────────────────────────────────────────────────────────
MIN_DET_SCORE = 0.50        # a face detection weaker than this is texture, not a candidate
MIN_FACE_PX = 28            # mirrors insight_face_engine._MIN_FACE_PX (vent grilles)
OWNER_SWITCH_S = 0.4        # owner must be recognised this long before he takes the head
CHALLENGE_RATIO = 1.6       # a different face must be this much bigger (closer) ...
CHALLENGE_S = 1.2           # ... for this long, continuously, to take the head
BODY_CHALLENGE_S = 2.4      # while following a turned-away body, a new face needs longer
LOST_GRACE_S = 1.0          # current person missed: HOLD this long before picking anew
CORROB_S = 4.0              # a body track counts only if a real face was in it this recently
SAME_DIST_FRAC = 1.0        # same face if centres are within this many face-widths
TRACK_STALE_S = 1.0         # person_track snapshot older than this is ignored

# ── stranger gate tuning ────────────────────────────────────────────────────────
FRONTAL_MAX_YAW = 20.0
FRONTAL_MAX_PITCH = 20.0
STRANGER_MIN_FACE_PX = 64   # ArcFace sees 112 px; below ~64 recognition is not a fair try
STRANGER_MIN_DET = 0.60
STRANGER_MAX_SIM = 0.30     # = the engine's sticky floor; at/above it the face is ambiguous
STRANGER_MIN_SHARP = 25.0   # Laplacian variance of the 112x112 gray crop (None = not measured)
STRANGER_MIN_OBS = 5
STRANGER_MIN_SPAN_S = 2.0
STRANGER_GAP_RESET_S = 1.5  # streak dies if the face is not a fair attempt for this long
STRANGER_STUDY_COOLDOWN_S = 300.0
ALERT_COOLDOWN_S = 600.0


# ── small geometry helpers ──────────────────────────────────────────────────────

def _bbox(f: dict[str, Any]) -> Optional[list[float]]:
    bb = f.get("bbox") or f.get("box")
    try:
        if bb and len(bb) >= 4:
            x1, y1, x2, y2 = (float(v) for v in bb[:4])
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
    except Exception:
        pass
    return None


def _center(bb: list[float]) -> tuple[float, float]:
    return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0


def _iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _same_place(a: list[float], b: list[float], frac: float = SAME_DIST_FRAC) -> bool:
    (ax, ay), (bx, by) = _center(a), _center(b)
    w = max(a[2] - a[0], b[2] - b[0], 1.0)
    return math.hypot(ax - bx, ay - by) <= frac * w or _iou(a, b) >= 0.3


def is_unnamed(pid: Any) -> bool:
    p = str(pid or "unknown").lower()
    return p == "unknown" or p.startswith("_")


# ── fake / decoy predicates (live defaults; tests inject their own) ─────────────

def default_fake_fn(bbox: list[float]) -> bool:
    """Liveness gate: a face the runner CONFIRMED fake (photo/screen/decoy:helmet).
    Fail-open, same as every other reader of the gate."""
    try:
        from brain import liveness_gate
        return bool(liveness_gate.is_fake_face([int(v) for v in bbox]))
    except Exception:
        return False


def default_decoy_fn(bbox: list[float]) -> bool:
    """Face centre inside a remembered humanoid NON-person (brain/body_pose static
    shapes: the spartan-helmet statue, the speaker whose drivers read as eyes) at the
    current head bearing. Pixel boxes only transfer at the bearing they were marked at."""
    try:
        from brain import body_pose
        head = body_pose.current_head() or {}
        hp, ht = head.get("pan_deg"), head.get("tilt_deg")
        cx, cy = _center(bbox)
        for s in body_pose.static_shapes():
            sp, stt = s.get("head_pan"), s.get("head_tilt")
            if hp is not None and sp is not None and (
                    abs(float(hp) - float(sp)) > 4.0
                    or abs(float(ht or 0.0) - float(stt or 0.0)) > 4.0):
                continue
            sb = s.get("box") or []
            if len(sb) >= 4 and sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
                return True
    except Exception:
        return False
    return False


def real_face(f: dict[str, Any], *, fake_fn: Callable = default_fake_fn,
              decoy_fn: Callable = default_decoy_fn,
              min_det: float = MIN_DET_SCORE) -> tuple[bool, str]:
    """Is this face record worth pointing the head at? (owner faces included)."""
    bb = _bbox(f)
    if bb is None:
        return False, "no bbox"
    if bb[2] - bb[0] < MIN_FACE_PX or bb[3] - bb[1] < MIN_FACE_PX:
        return False, "tiny"
    ds = f.get("det_score")
    if ds is not None and float(ds) < min_det:
        return False, "weak detection"
    # Recognised people are never demoted by the decoy layers (liveness rule:
    # anti-spoof never demotes a recognised Zeke).
    if is_unnamed(f.get("person_id")):
        if fake_fn(bb):
            return False, "liveness: confirmed fake"
        if decoy_fn(bb):
            return False, "static decoy shape"
    return True, "ok"


# ── head pose ───────────────────────────────────────────────────────────────────

def face_angles(f: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """(yaw, pitch, source) in degrees. InsightFace's 3D pose is [pitch, yaw, roll]
    (insightface/model_zoo/landmark.py) — the engine writes [0,0,0] when it has no pose,
    so an exact zero triple means MISSING, not frontal. Fallback: the detector's 5 points
    (eyes, nose, mouth corners): nose offset from the eye midpoint / eye distance ~
    0.55*tan(yaw); nose height between eye line and mouth line ~0.5 when level.
    Rough on purpose — it only has to separate 'looking at me' from 'not'."""
    pose = f.get("pose")
    try:
        if pose is not None and len(pose) >= 2:
            p, y = float(pose[0]), float(pose[1])
            if not all(abs(float(v)) < 1e-6 for v in list(pose)[:3]):
                return y, p, "pose3d"
    except Exception:
        pass
    kps = f.get("kps")
    try:
        if kps is not None and len(kps) >= 5:
            (lex, ley), (rex, rey), (nx, ny), (lmx, lmy), (rmx, rmy) = [
                (float(k[0]), float(k[1])) for k in kps[:5]]
            ed = math.hypot(rex - lex, rey - ley)
            if ed < 1.0:
                return None, None, None
            emx, emy = (lex + rex) / 2.0, (ley + rey) / 2.0
            mmy = (lmy + rmy) / 2.0
            yaw = math.degrees(math.atan(((nx - emx) / ed) / 0.55))
            span = mmy - emy
            if span <= 1.0:
                return yaw, None, "kps"
            v = (ny - emy) / span
            pitch = math.degrees(math.atan((v - 0.5) / 0.5))
            return yaw, pitch, "kps"
    except Exception:
        pass
    return None, None, None


def is_frontal(f: dict[str, Any], max_yaw: float = FRONTAL_MAX_YAW,
               max_pitch: float = FRONTAL_MAX_PITCH) -> tuple[bool, dict[str, Any]]:
    yaw, pitch, src = face_angles(f)
    info = {"yaw": None if yaw is None else round(yaw, 1),
            "pitch": None if pitch is None else round(pitch, 1), "pose_source": src}
    if yaw is None or pitch is None:
        return False, info           # no angle = no claim of frontal
    return (abs(yaw) <= max_yaw and abs(pitch) <= max_pitch), info


# ── the person:any picker ───────────────────────────────────────────────────────

def _track_for(bb: list[float], tracks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Smallest body track whose box contains the face centre (person_track.bind_faces rule)."""
    cx, cy = _center(bb)
    best = None
    for t in tracks or []:
        try:
            x, y, w, h = (float(v) for v in t["tlwh"][:4])
        except Exception:
            continue
        if x <= cx <= x + w and y <= cy <= y + h:
            if best is None or w * h < best["tlwh"][2] * best["tlwh"][3]:
                best = t
    return best


class AnyPicker:
    """Chooses who `person:any` follows. Pure apart from time; thread-safe."""

    def __init__(self, owner: str = OWNER, *, fake_fn: Callable = default_fake_fn,
                 decoy_fn: Callable = default_decoy_fn) -> None:
        self.owner = str(owner or OWNER).lower()
        self.fake_fn = fake_fn
        self.decoy_fn = decoy_fn
        self._lock = threading.Lock()
        self.cur: Optional[dict[str, Any]] = None
        self.cur_seen_ts = 0.0
        self.owner_since: Optional[float] = None
        self.chal: Optional[dict[str, Any]] = None
        self.chal_since = 0.0
        self.track_face_ts: dict[Any, float] = {}
        self.track_face_up: dict[Any, float] = {}
        self.switches = 0
        self.last_reason = ""

    # candidates -------------------------------------------------------------
    def candidates(self, faces: list[dict[str, Any]], tracks: list[dict[str, Any]],
                   now: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        faced_tracks: set = set()
        for f in faces or []:
            if not isinstance(f, dict):
                continue
            ok, _why = real_face(f, fake_fn=self.fake_fn, decoy_fn=self.decoy_fn)
            if not ok:
                continue
            bb = _bbox(f)
            pid = str(f.get("person_id") or "unknown").lower()
            trk = _track_for(bb, tracks)
            tid = trk["id"] if trk is not None else None
            if tid is not None:
                faced_tracks.add(tid)
                self.track_face_ts[tid] = now
                x, y, w, h = (float(v) for v in trk["tlwh"][:4])
                up = (y + h) - _center(bb)[1]
                if up > 0:
                    self.track_face_up[tid] = up
            out.append({"kind": "face", "bbox": bb, "aim": _center(bb),
                        "size": bb[2] - bb[0], "pid": pid,
                        "owner": pid == self.owner, "track_id": tid,
                        "track": trk, "face": f})
        for t in tracks or []:
            tid = t.get("id")
            if tid in faced_tracks:
                continue
            seen = self.track_face_ts.get(tid)
            if seen is None or now - seen > CORROB_S:
                continue    # a body with no recent FACE is not somebody (statue/helmet scar)
            try:
                x, y, w, h = (float(v) for v in t["tlwh"][:4])
            except Exception:
                continue
            up = self.track_face_up.get(tid)
            ay = (y + h) - up if up and 0 < up <= h * 1.2 else y + h * 0.18
            out.append({"kind": "body", "bbox": [x, y, x + w, y + h],
                        "aim": (x + w / 2.0, ay), "size": w * 0.3, "pid": None,
                        "owner": False, "track_id": tid, "track": t, "face": None})
        # forget corroboration for long-dead tracks
        for tid in [k for k, v in self.track_face_ts.items() if now - v > 10 * CORROB_S]:
            self.track_face_ts.pop(tid, None)
            self.track_face_up.pop(tid, None)
        return out

    @staticmethod
    def _rank(c: dict[str, Any]) -> tuple:
        return (bool(c["owner"]), c["kind"] == "face", float(c["size"]))

    @staticmethod
    def _same(a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]) -> bool:
        if a is None or b is None:
            return False
        if a.get("track_id") is not None and a.get("track_id") == b.get("track_id"):
            return True
        if a["kind"] == "face" and b["kind"] == "face":
            return _same_place(a["bbox"], b["bbox"])
        return False

    def _set(self, c: dict[str, Any], now: float, reason: str) -> dict[str, Any]:
        if not self._same(self.cur, c):
            self.switches += 1
        self.cur, self.cur_seen_ts = c, now
        self.chal = None
        self.last_reason = reason
        return c

    # the decision -----------------------------------------------------------
    def pick(self, faces: list[dict[str, Any]], tracks: list[dict[str, Any]],
             now: Optional[float] = None) -> Optional[dict[str, Any]]:
        now = time.time() if now is None else float(now)
        with self._lock:
            cands = self.candidates(faces, tracks, now)
            owners = [c for c in cands if c["owner"]]
            m = next((c for c in cands if self._same(self.cur, c)), None)
            # 1) owner preference, debounced (a one-frame misrecognition can't grab the head)
            if owners:
                self.owner_since = self.owner_since or now
                best_owner = max(owners, key=lambda c: c["size"])
                if self.cur is not None and self.cur.get("owner") and m is not None:
                    return self._set(m, now, "owner (held)")
                if now - self.owner_since >= OWNER_SWITCH_S:
                    return self._set(best_owner, now, "owner preferred")
            else:
                self.owner_since = None
            # 2) keep the current person, unless a challenger has clearly earned it
            if m is not None:
                others = [c for c in cands if c is not m and c["kind"] == "face"
                          and not self._same(m, c)]
                best = max(others, key=lambda c: c["size"]) if others else None
                need_s = BODY_CHALLENGE_S if m["kind"] == "body" else CHALLENGE_S
                earned = best is not None and (
                    m["kind"] == "body" or best["size"] >= CHALLENGE_RATIO * m["size"])
                if earned and not m.get("owner"):
                    if self.chal is None or not self._same(self.chal, best):
                        self.chal, self.chal_since = best, now
                    else:
                        self.chal = best
                        if now - self.chal_since >= need_s:
                            return self._set(best, now, "challenger held")
                else:
                    self.chal = None
                self.cur, self.cur_seen_ts = m, now
                self.last_reason = "held current"
                return m
            # 3) current person missed: HOLD, never jump straight to someone else
            if self.cur is not None and now - self.cur_seen_ts <= LOST_GRACE_S:
                self.last_reason = "current missed - holding"
                return None
            self.cur = None
            if not cands:
                self.last_reason = "nobody"
                return None
            # 4) fresh pick (owner not yet debounced still wins on rank)
            return self._set(max(cands, key=self._rank), now, "fresh pick")

    def status(self) -> dict[str, Any]:
        c = self.cur
        return {"current": None if c is None else {
                    "kind": c["kind"], "pid": c["pid"], "owner": c["owner"],
                    "track_id": c["track_id"], "bbox": [round(v) for v in c["bbox"]]},
                "reason": self.last_reason, "switches": self.switches,
                "corroborated_tracks": len(self.track_face_ts)}


def aim_offset(cand: dict[str, Any], frame_shape: tuple[int, int],
               fps: float = 30.0) -> dict[str, Any]:
    """Servo-shaped offset for a candidate (same convention as person_track.target_offset):
    dx/dy normalized, right/down positive; vx/vy from the bound track's Kalman velocity."""
    h, w = frame_shape
    ax, ay = cand["aim"]
    vx = vy = 0.0
    trk = cand.get("track")
    if trk is not None:
        fps = max(5.0, min(60.0, float(fps)))
        try:
            vx = float(trk["vel"][0]) * fps / (w / 2.0)
            vy = ((float(trk["vel"][1]) + float(trk.get("vh") or 0.0) / 2.0)
                  * fps / (h / 2.0))
        except Exception:
            vx = vy = 0.0
    return {"dx": (ax - w / 2.0) / (w / 2.0), "dy": (ay - h / 2.0) / (h / 2.0),
            "vx": vx, "vy": vy, "source": "any_" + cand["kind"],
            "track_id": cand.get("track_id"), "pid": cand.get("pid")}


# ── live glue (one picker per process) ──────────────────────────────────────────
_PICKER: Optional[AnyPicker] = None
_PICKER_LOCK = threading.Lock()


def picker() -> AnyPicker:
    global _PICKER
    with _PICKER_LOCK:
        if _PICKER is None:
            _PICKER = AnyPicker()
        return _PICKER


def _live_tracks() -> tuple[list[dict[str, Any]], Optional[tuple[int, int]], float]:
    try:
        from brain import person_track
        snap = person_track.snapshot()
        if time.time() - float(snap.get("wall_ts") or 0.0) > TRACK_STALE_S:
            return [], snap.get("frame_shape"), float(snap.get("fps") or 30.0)
        return snap.get("tracks") or [], snap.get("frame_shape"), float(snap.get("fps") or 30.0)
    except Exception:
        return [], None, 30.0


def servo_offset(g: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The attention_smooth fast path for person:any. None = nobody (or holding)."""
    tracks, shape, fps = _live_tracks()
    if not shape:
        try:
            from brain import frame_store
            fr = frame_store.get_buffered_frame(max_age_sec=1.0).frame
            if fr is not None:
                shape = (int(fr.shape[0]), int(fr.shape[1]))
        except Exception:
            shape = None
    if not shape:
        return None
    c = picker().pick(list(g.get("_face_results") or []), tracks)
    if c is None:
        return None
    return aim_offset(c, shape, fps)


def match_face(g: dict[str, Any], faces: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """visual_attention.observe's resolver for person:any — a face-record-shaped dict."""
    tracks, _shape, _fps = _live_tracks()
    c = picker().pick(list(faces or []), tracks)
    if c is None:
        return None
    if c["kind"] == "face" and c.get("face") is not None:
        rec = dict(c["face"])
        rec["resolver"] = "person_any_face"
        return rec
    return {"bbox": [int(v) for v in c["bbox"]], "confidence": 0.0,
            "person_id": "unknown", "resolver": "person_any_body"}


def current_face() -> Optional[dict[str, Any]]:
    """Face dict the picker is on right now (TrackerVit fallback seed), else None."""
    c = picker().cur
    return c.get("face") if c and c.get("kind") == "face" else None


# ── the stranger gate ───────────────────────────────────────────────────────────

def fair_attempt(f: dict[str, Any], sharpness: Optional[float] = None, *,
                 fake_fn: Callable = default_fake_fn,
                 decoy_fn: Callable = default_decoy_fn) -> tuple[bool, dict[str, Any]]:
    """Was recognition given a FAIR chance on this face, and did it still come back unknown?"""
    info: dict[str, Any] = {}
    if not is_unnamed(f.get("person_id")):
        return False, {"why": "recognised"}
    bb = _bbox(f)
    if bb is None:
        return False, {"why": "no bbox"}
    info["face_px"] = [int(bb[2] - bb[0]), int(bb[3] - bb[1])]
    if min(bb[2] - bb[0], bb[3] - bb[1]) < STRANGER_MIN_FACE_PX:
        return False, dict(info, why="too small for a fair attempt")
    ds = f.get("det_score")
    info["det_score"] = None if ds is None else round(float(ds), 3)
    if ds is not None and float(ds) < STRANGER_MIN_DET:
        return False, dict(info, why="weak detection")
    fr, ang = is_frontal(f)
    info.update(ang)
    if not fr:
        return False, dict(info, why="not frontal")
    sim = float(f.get("confidence") or 0.0)
    info["best_sim"] = round(sim, 3)
    if sim >= STRANGER_MAX_SIM:
        return False, dict(info, why="ambiguous match (near an enrolled person)")
    if sharpness is not None:
        info["sharpness"] = round(float(sharpness), 1)
        if float(sharpness) < STRANGER_MIN_SHARP:
            return False, dict(info, why="blurry")
    if fake_fn(bb):
        return False, dict(info, why="liveness: confirmed fake (photo/screen/decoy)")
    if decoy_fn(bb):
        return False, dict(info, why="static decoy shape")
    return True, dict(info, why="fair attempt, still unknown")


class StrangerGate:
    """>= STRANGER_MIN_OBS fair-attempt observations of the same face over
    >= STRANGER_MIN_SPAN_S, no recognition at that spot in between => trigger."""

    def __init__(self, *, fake_fn: Callable = default_fake_fn,
                 decoy_fn: Callable = default_decoy_fn) -> None:
        self.fake_fn = fake_fn
        self.decoy_fn = decoy_fn
        self.streak: Optional[dict[str, Any]] = None
        self.cooldown_until = 0.0
        self.triggers = 0
        self.last_reject: Optional[dict[str, Any]] = None

    def observe(self, faces: list[dict[str, Any]], now: Optional[float] = None,
                sharp_fn: Optional[Callable[[list[float]], Optional[float]]] = None
                ) -> Optional[dict[str, Any]]:
        now = time.time() if now is None else float(now)
        st = self.streak
        # recognition at the streak's spot = it was somebody all along -> reset
        if st is not None:
            for f in faces or []:
                bb = _bbox(f) if isinstance(f, dict) else None
                if bb and not is_unnamed(f.get("person_id")) and _same_place(bb, st["bbox"]):
                    self.streak = st = None
                    break
        fair: list[tuple[dict[str, Any], list[float], dict[str, Any]]] = []
        for f in faces or []:
            if not isinstance(f, dict):
                continue
            bb = _bbox(f)
            if bb is None:
                continue
            sharp = sharp_fn(bb) if sharp_fn is not None else None
            ok, info = fair_attempt(f, sharp, fake_fn=self.fake_fn, decoy_fn=self.decoy_fn)
            if ok:
                fair.append((f, bb, info))
            elif is_unnamed(f.get("person_id")):
                self.last_reject = dict(info, ts=now)
        if st is not None and now - st["last_ts"] > STRANGER_GAP_RESET_S:
            self.streak = st = None
        if not fair:
            return None
        hit = None
        if st is not None:
            hit = next((x for x in fair if _same_place(x[1], st["bbox"])), None)
        if hit is None and st is None:
            f, bb, info = max(fair, key=lambda x: x[1][2] - x[1][0])
            self.streak = st = {"bbox": bb, "first_ts": now, "last_ts": now, "n": 1,
                                "info": info}
        elif hit is not None:
            f, bb, info = hit
            st.update({"bbox": bb, "last_ts": now, "n": st["n"] + 1, "info": info})
        if (st is not None and st["n"] >= STRANGER_MIN_OBS
                and now - st["first_ts"] >= STRANGER_MIN_SPAN_S
                and now >= self.cooldown_until):
            ev = {"bbox": [int(v) for v in st["bbox"]], "observations": st["n"],
                  "span_s": round(now - st["first_ts"], 2), "ts": now, **st["info"]}
            self.streak = None
            self.cooldown_until = now + STRANGER_STUDY_COOLDOWN_S
            self.triggers += 1
            return ev
        return None

    def status(self) -> dict[str, Any]:
        st = self.streak
        return {"streak": None if st is None else {
                    "n": st["n"], "span_s": round(st["last_ts"] - st["first_ts"], 2),
                    "bbox": [int(v) for v in st["bbox"]]},
                "cooldown_until": self.cooldown_until, "triggers": self.triggers,
                "last_reject": self.last_reject}


class AlertLimiter:
    """One alert per stranger key (seen_id) per cooldown."""

    def __init__(self, cooldown_s: float = ALERT_COOLDOWN_S) -> None:
        self.cooldown_s = float(cooldown_s)
        self.last: dict[str, float] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        k = str(key or "unidentified")
        t = self.last.get(k)
        if t is not None and now - t < self.cooldown_s:
            return False
        self.last[k] = now
        return True


def face_sharpness(frame: Any, bbox: list[float]) -> Optional[float]:
    """Laplacian variance of the face crop at 112x112 gray (ArcFace's input size)."""
    try:
        import cv2  # type: ignore
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return None


def stranger_dm_text(ev: dict[str, Any], study: dict[str, Any],
                     presence: Optional[dict[str, Any]] = None) -> str:
    """The Discord alert. Plain text; says outright that I stayed silent."""
    seen = (study or {}).get("seen") or {}
    crop = ((study or {}).get("crop") or {}).get("path")
    if seen.get("ok") and seen.get("seen_before"):
        memory = (f"I've seen this face before ({seen.get('seen_id')}, first "
                  f"{seen.get('first_seen_iso')}, {seen.get('sightings')} sightings"
                  + (f"; my note: {seen.get('note')}" if seen.get("note") else "") + ").")
    elif seen.get("ok"):
        memory = f"First time I've seen this face (filed as {seen.get('seen_id')})."
    else:
        memory = "I couldn't embed the face for my stranger memory."
    pres = (presence or {}).get("reason") or "presence unknown"
    yaw, pitch = ev.get("yaw"), ev.get("pitch")
    return ("Stranger in the room - I stayed SILENT and did not speak to them. "
            f"A face looked straight at my camera for {ev.get('span_s')}s "
            f"(yaw {yaw}, pitch {pitch}) and I don't recognise it "
            f"(best match {ev.get('best_sim')}). {memory} "
            f"Photos: {study.get('frames_saved', 0)} in {study.get('dir')}"
            + (f", crop {crop}" if crop else "") + f". Your phone: {pres}. "
            "I'll keep watching quietly; tell me who they are if you know.")
