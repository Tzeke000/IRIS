# SELF_ASSESSMENT: I tell a living face from a face-like THING (the helmet on the dresser, a photo, a screen) by appearance and anti-spoof liveness - in SHADOW MODE: I only log what I WOULD decide; wakes, recognition and the servo are untouched.
"""
face_liveness — shadow-mode decoy gallery + MiniFASNet anti-spoof (built 2026-10-18,
Zeke: "go for it"; he vetoed "doesn't move = not a person").

Card: profiles/vision/face_liveness.md. Logic: brain/face_liveness.py.

actions:
  status                      runner alive/stats, flag, decoy gallery, models
  start [persist=true]        start the shadow thread (persist=true also sets the
                              boot flag state/face_liveness_shadow.json on=true)
  stop  [persist=true]        stop it (persist=true sets on=false)
  set probe_full_frame=bool [probe_thresh=0.3]
                              also score face-like things the LIVE engine isn't
                              reporting (CPU full-frame detector every ~2 s)
  enroll_decoy name=helmet [seconds=25 samples=8 hint_bbox=[x1,y1,x2,y2] det_thresh=0.3 min_face_px=28]
                              (with a hint, the candidate's CENTRE must lie inside it)
                              capture crops+embeddings of ONE face-like detection
                              from the runtime's raw frame slot. Refuses samples that
                              look like an enrolled person (sim >= 0.30) or overlap a
                              recognised live face. Ambiguous (2+ candidates, no hint)
                              => refused.
  enroll_decoy name=helmet files=[...] hint_bbox=[...]
                              same, from saved RAW frames (e.g. faces/_drafts/*/*.jpg)
                              - for samples under other light.
  score_now                   score every live face + full-frame candidates now
  reload_code                 re-import brain/face_liveness.py - REFUSED while the shadow
                              thread runs (the module holds the runner singleton)
  shadow_summary [since_s=3600]
                              grouped counts / sims / anti-spoof from the shadow log

NEVER: writes g["_face_results"], fires signals, moves the head, opens the camera.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool


def _fl():
    from brain import face_liveness  # plain import; NEVER reload (holds the runner)
    return face_liveness


def _write_flag(**kw) -> dict:
    fl = _fl()
    cur = fl.flag()
    cur.update(kw)
    cur["updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fl.FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = fl.FLAG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    tmp.replace(fl.FLAG_PATH)
    return cur


def _scorer(g):
    fl = _fl()
    r = fl.runner(g)
    if r.scorer is None:
        r.scorer = fl.Scorer()
    return r.scorer


def _status(g) -> dict:
    fl = _fl()
    r = fl.get_runner()
    gal = fl.DecoyGallery().load(force=True)
    try:
        man = json.loads((fl.MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    except Exception as e:
        man = {"error": repr(e)}
    return {"ok": True, "mode": "SHADOW (logs only; changes no live decision)",
            "runner_alive": bool(r and r.alive()), "stats": (r.stats if r else None),
            "tracks": len(r.tracks) if r else 0, "flag": fl.flag(),
            "decoys": {k: len(v) for k, v in gal.samples.items()},
            "models": {"licence": man.get("licence"), "files": [m.get("file") for m in man.get("models", [])]},
            "log": str(fl.SHADOW_LOG), "log_bytes": fl.SHADOW_LOG.stat().st_size if fl.SHADOW_LOG.exists() else 0}


def _candidates(fl, sc, frame, live_faces, det_thresh, min_px=28):
    """Live faces + CPU full-frame detections. min_px mirrors the live engine's
    _MIN_FACE_PX: a box it would drop can never wake anyone, so it is not a candidate."""
    cands = []
    for f in live_faces:
        if isinstance(f, dict) and f.get("bbox"):
            cands.append({"bbox": [float(v) for v in f["bbox"][:4]], "kps": None,
                          "det_score": f.get("det_score"), "live_pid": str(f.get("person_id") or "unknown"),
                          "src": "live"})
    for d in sc.embedder.detect_full(frame, thresh=det_thresh):
        if min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) < min_px:
            continue
        if any(fl.iou(d["bbox"], c["bbox"]) > 0.3 for c in cands):
            continue
        cands.append({"bbox": d["bbox"], "kps": d["kps"], "det_score": d["det_score"],
                      "live_pid": "unknown", "src": "cpu_full"})
    return cands


def _pick(fl, cands, hint):
    unk = [c for c in cands if c["live_pid"] == "unknown"]
    if hint:
        # 10-18 scar: "nearest centre within 1.5x" enrolled the helmet's PLUME (a 23x33 box
        # just above the hint) as "helmet". Now the candidate's centre must lie INSIDE the
        # hint box; among those, highest IoU wins.
        best, bo = None, -1.0
        for c in unk:
            cx, cy, _ = fl.box_center_w(c["bbox"])
            if not (hint[0] <= cx <= hint[2] and hint[1] <= cy <= hint[3]):
                continue
            o = fl.iou(c["bbox"], hint)
            if o > bo:
                best, bo = c, o
        return best, (None if best else "no candidate centred inside hint_bbox")
    if len(unk) == 1:
        return unk[0], None
    return None, ("no unknown face-like detection in view" if not unk
                  else f"{len(unk)} candidates - pass hint_bbox=[x1,y1,x2,y2] to choose one")


def _crop(frame, bbox, pad=1.6, max_px=200):
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, pad * max(x2 - x1, y2 - y1) / 2
    c = frame[int(max(0, cy - r)):int(min(h, cy + r)), int(max(0, cx - r)):int(min(w, cx + r))]
    if c.size and max(c.shape[:2]) > max_px:
        s = max_px / max(c.shape[:2])
        c = cv2.resize(c, None, fx=s, fy=s)
    return c


def _enroll(params, g) -> dict:
    import cv2
    fl = _fl()
    name = str(params.get("name") or "").strip().lower()
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        return {"ok": False, "error": "name required (letters/digits/_/-), e.g. helmet"}
    samples = int(params.get("samples", 8))
    seconds = float(params.get("seconds", 25))
    det_thresh = float(params.get("det_thresh", 0.3))
    hint = params.get("hint_bbox")
    files = params.get("files") or []
    sc = _scorer(g)
    persons = fl.person_gallery(g)
    gal = fl.DecoyGallery()
    saved, rejected, notes = [], [], []

    def _one(frame, live_faces, origin):
        cands = _candidates(fl, sc, frame, live_faces, det_thresh, int(params.get("min_face_px", 28)))
        c, why = _pick(fl, cands, hint)
        if c is None:
            rejected.append({"origin": origin, "why": why})
            return
        # safety 1: never on top of a RECOGNISED live face
        for f in live_faces:
            pid = str(f.get("person_id") or "unknown")
            if pid != "unknown" and fl.iou(f.get("bbox") or [0, 0, 0, 0], c["bbox"]) > 0.2:
                rejected.append({"origin": origin, "why": f"overlaps recognised {pid}"})
                return
        e = sc.embedder.embed_at(frame, c["bbox"], kps=c["kps"], det_score=c["det_score"])
        if e is None:
            rejected.append({"origin": origin, "why": "no embedding (keypoints not found)"})
            return
        pname, psim = fl.best_match(e["emb"], persons)
        # safety 2: never enroll something that looks like an enrolled PERSON
        if psim >= fl.ENROLL_PERSON_SIM_MAX:
            rejected.append({"origin": origin, "why": f"looks like person {pname} ({psim:.3f})"})
            return
        a, _ = sc.antispoof.score(frame, c["bbox"])
        head = None
        if origin.startswith("live@"):   # a saved file's head pose is unknown, not "now"
            try:
                from brain import body_pose
                head = body_pose.current_head()
            except Exception:
                head = None
        info = {"bbox": [int(v) for v in c["bbox"]], "det_score": round(float(e["det_score"]), 4),
                "src": c["src"], "origin": origin, "nearest_person": pname, "person_sim": round(psim, 4),
                "antispoof": round(a, 4), "brightness": round(float(frame.mean()), 1),
                "head": head, "ts": time.time()}
        gal.add(name, e["emb"], _crop(frame, c["bbox"]), info)
        saved.append(info)

    if files:
        for p in files[:max(1, samples)]:
            img = cv2.imread(str(p))
            if img is None:
                rejected.append({"origin": str(p), "why": "unreadable"})
                continue
            _one(img, [], str(p))
    else:
        t_end = time.time() + seconds
        gap = max(0.5, seconds / max(1, samples))
        while time.time() < t_end and len(saved) < samples:
            frame, age = fl.raw_frame(g)
            if frame is None:
                notes.append(f"no fresh raw frame (age={age})")
                time.sleep(1.0)
                continue
            _one(frame, list(g.get("_face_results") or []), f"live@{time.strftime('%H:%M:%S')}")
            time.sleep(gap)
    gal.load(force=True)
    return {"ok": bool(saved), "name": name, "saved": len(saved), "rejected": len(rejected),
            "rejected_detail": rejected[:10], "samples": saved, "notes": notes[:5],
            "gallery_total": len(gal.samples.get(name, [])),
            "dir": str(fl.DECOY_DIR / name),
            "look": "crops are crop_*.jpg in that dir (<=200px) - LOOK before trusting the enrollment"}


def _score_now(params, g) -> dict:
    fl = _fl()
    sc = _scorer(g)
    frame, age = fl.raw_frame(g)
    if frame is None:
        return {"ok": False, "error": f"no fresh raw frame (age={age})"}
    live = list(g.get("_face_results") or [])
    out = []
    for c in _candidates(fl, sc, frame, live, float(params.get("det_thresh", 0.3)), int(params.get("min_face_px", 0))):
        s = sc.signals(frame, c["bbox"], fl.person_gallery(g), kps=c["kps"], det_score=c["det_score"])
        s.pop("_emb", None)
        out.append({"src": c["src"], "bbox": [int(v) for v in c["bbox"]], "live_pid": c["live_pid"],
                    "det_score": c["det_score"], **s,
                    "would_if_single_sample": fl.fuse(c["live_pid"], s.get("decoy_name"), bool(s.get("decoy_hit")),
                                                      "live" if s["antispoof"] >= fl.LIVE_HI else
                                                      "spoof" if s["antispoof"] <= fl.SPOOF_LO else "uncertain")})
    return {"ok": True, "frame_age_s": round(age, 3), "live_faces": len(live), "candidates": out}


def _handler(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    params = params or {}
    action = str(params.get("action") or "status")
    fl = _fl()
    try:
        if action == "status":
            return _status(g)
        if action == "start":
            if params.get("persist", True):
                _write_flag(on=True)
            return {**fl.runner(g).start(), "flag": fl.flag()}
        if action == "stop":
            if params.get("persist", True):
                _write_flag(on=False)
            r = fl.get_runner()
            return {**(r.stop() if r else {"ok": True, "note": "not running"}), "flag": fl.flag()}
        if action == "set":
            kw = {}
            if "probe_full_frame" in params:
                kw["probe_full_frame"] = bool(params["probe_full_frame"])
            if "probe_thresh" in params:
                kw["probe_thresh"] = float(params["probe_thresh"])
            return {"ok": True, "flag": _write_flag(**kw)}
        if action == "reload_code":
            # Safe ONLY while no shadow thread runs: the module's globals are the runner
            # singleton + a model cache, and nothing outside this tool reads them.
            r = fl.get_runner()
            if r and r.alive():
                return {"ok": False, "error": "shadow thread running - action=stop first, wait ~1 s"}
            import importlib
            import sys as _sys
            importlib.reload(_sys.modules["brain.face_liveness"])
            return {"ok": True, "reloaded": "brain.face_liveness", "runner": None}
        if action == "enroll_decoy":
            return _enroll(params, g)
        if action == "score_now":
            return _score_now(params, g)
        if action == "shadow_summary":
            since = time.time() - float(params.get("since_s", 3600))
            return {"ok": True, "since_s": params.get("since_s", 3600),
                    **fl.summarize(fl.read_log(since_ts=since))}
        return {"ok": False, "error": f"unknown action {action!r}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:500]}


register_tool(
    "face_liveness",
    "SHADOW-MODE face liveness: decoy gallery (appearance, e.g. the helmet) + MiniFASNet "
    "anti-spoof, logs what it WOULD decide to state/face_liveness_shadow.jsonl, changes "
    "nothing live. action=status|start|stop|set|enroll_decoy|score_now|shadow_summary|reload_code.",
    1,
    _handler,
)
