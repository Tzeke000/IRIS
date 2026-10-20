"""
face_liveness — "is this face-like thing a LIVING person?" (SHADOW MODE, built 2026-10-18)

Why: the Spartan helmet on the dresser (and photos, screens, masks) reads as an
"unknown face" and can wake me as a visitor. Zeke VETOED "doesn't move = not a
person" (a still, reading or sleeping person must still count). So there is NO
stillness logic and NO position-only logic here. Two appearance/liveness signals:

  1. DECOY GALLERY (appearance): `decoy:<name>` ArcFace embeddings, several
     samples, stored under state/faces_decoys/<name>/ — kept apart from faces/
     (faces/<pid>/ is what the recognizer loads as PEOPLE). A detection is a
     decoy only if it matches a decoy embedding at high similarity AND beats its
     nearest enrolled person by a margin. A person in the same spot has a
     different embedding, so they do not match. Position is logged, and only
     required if the gallery entry sets require_near=true.
  2. ANTI-SPOOF (MiniFASNet, minivision Silent-Face-Anti-Spoofing, Apache-2.0):
     two 80x80 ONNX models on CPU (scale 2.7 + 4.0 crops, their crop convention),
     real-score = mean softmax[class 1]. Rolling median over ~1.5 s + hysteresis.

FUSION (what a LIVE version would do — today it is only LOGGED):
  - a recognised enrolled person is NEVER demoted (anti-spoof can't touch them);
  - unknown + decoy match            -> "decoy:<name>"
  - unknown + liveness state 'spoof' -> "face_like" (something face-like, never "visitor")
  - unknown + liveness state 'live'  -> "visitor"
  - unknown + not enough samples     -> "uncertain"

SHADOW MODE: this module never writes g["_face_results"], never fires a signal,
never touches the servo, never changes recognition. It READS g["_face_results"]
and g["_raw_frame_slot"] (the capture loop's raw frame; never opens the camera)
and appends to state/face_liveness_shadow.jsonl. Its own models run on CPU
(no GPU contention), only while the live engine reports a face.

⚠ This module holds runtime state (the _RUNNER singleton) => it is a module WITH
configured globals: NEVER brain_hot_swap / importlib.reload it. Edits land on the
next stack restart; in between, the tool can only stop/start the existing runner.

Flag file state/face_liveness_shadow.json: {"on": true, "probe_full_frame": false,
                                            "probe_thresh": 0.3, "probe_min_px": 28}
  - on: autostart at boot (brain/iris_bootstrap.py calls maybe_autostart).
  - probe_full_frame: ALSO run a CPU full-frame detector every ~2 s so face-like
    things the live engine is not currently reporting get scored too (logged as
    source="probe", actual_verdict="not_detected_live").
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DECOY_DIR = ROOT / "state" / "faces_decoys"
SHADOW_LOG = ROOT / "state" / "face_liveness_shadow.jsonl"
FLAG_PATH = ROOT / "state" / "face_liveness_shadow.json"
NOW_PATH = ROOT / "state" / "face_liveness_now.json"   # live per-track verdicts (brain/liveness_gate.py)
MODEL_DIR = ROOT / "models" / "antispoof"
INSIGHT_DIR = Path(os.path.expanduser("~/.insightface/models/buffalo_l"))

# ── decoy match ──────────────────────────────────────────────────────────────
DECOY_SIM_MIN = 0.50       # cosine to a decoy sample to count as that decoy
DECOY_MARGIN = 0.10        # ...and must beat the nearest enrolled PERSON by this much
DECOY_NEAR_FRAC = 1.5      # "near the remembered spot": centre distance <= this x face width
ENROLL_PERSON_SIM_MAX = 0.30  # refuse to enroll a decoy sample that looks like an enrolled person
# ── liveness hysteresis on the median anti-spoof real-score ─────────────────
WINDOW_S = 1.5
MIN_SAMPLES = 3
LIVE_HI = 0.70             # median above -> 'live'
SPOOF_LO = 0.35            # median below -> 'spoof'; in between -> keep previous state
# ── runner cadence / log ─────────────────────────────────────────────────────
TICK_S = 0.5               # 2 Hz scoring while faces are present
EMB_EVERY_S = 1.0          # per-track embedding refresh
TRACK_TTL_S = 3.0
LOG_EVERY_S = 5.0          # per-track heartbeat record (plus every verdict change)
LOG_MAX_BYTES = 5_000_000  # rotate to .1 (one backup kept)
PROBE_EVERY_S = 2.0
CPU_THREADS = 2


# ═════════════════════════════ pure helpers (unit-tested) ══════════════════

def _np():
    import numpy as np  # local: keeps import of this module cheap
    return np


def cosine(a: Any, b: Any) -> float:
    np = _np()
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


def best_match(emb: Any, gallery: dict[str, list]) -> tuple[Optional[str], float]:
    """Max cosine over every sample of every entry -> (name, sim). (None, 0.0) if empty."""
    best_name, best = None, 0.0
    for name, samples in (gallery or {}).items():
        for s in samples or []:
            if s is None:
                continue
            sc = cosine(emb, s)
            if best_name is None or sc > best:
                best_name, best = name, sc
    return best_name, (best if best_name is not None else 0.0)


def box_center_w(b: list | tuple) -> tuple[float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def near_position(bbox, remembered: list | None, frac: float = DECOY_NEAR_FRAC) -> Optional[bool]:
    """Is bbox near any remembered decoy bbox (pixel space)? None if nothing remembered."""
    if not remembered:
        return None
    cx, cy, w = box_center_w(bbox)
    for rb in remembered:
        rx, ry, rw = box_center_w(rb)
        if ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 <= frac * max(w, rw):
            return True
    return False


def decoy_decision(decoy_sim: float, person_sim: float, near: Optional[bool] = None,
                   require_near: bool = False, sim_min: float = DECOY_SIM_MIN,
                   margin: float = DECOY_MARGIN) -> bool:
    """APPEARANCE gate first: a decoy only if it looks like the decoy (>= sim_min) AND
    looks more like the decoy than like any enrolled person (by margin). Position can
    only ADD a requirement (require_near), never make a match on its own."""
    if decoy_sim < sim_min:
        return False
    if decoy_sim - person_sim < margin:
        return False
    if require_near and near is not True:
        return False
    return True


class Hysteresis:
    """Rolling median of a 0..1 real-score over window_s, with a two-threshold latch.
    state: 'uncertain' (too few samples, initial) | 'live' | 'spoof'."""

    def __init__(self, window_s: float = WINDOW_S, hi: float = LIVE_HI, lo: float = SPOOF_LO,
                 min_samples: int = MIN_SAMPLES):
        self.window_s, self.hi, self.lo, self.min_samples = window_s, hi, lo, min_samples
        self.buf: deque = deque()
        self.state = "uncertain"
        self.median: Optional[float] = None

    def update(self, ts: float, value: float) -> str:
        self.buf.append((float(ts), float(value)))
        while self.buf and ts - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        if len(self.buf) < self.min_samples:
            self.median = statistics.median(v for _, v in self.buf) if self.buf else None
            return self.state
        self.median = float(statistics.median(v for _, v in self.buf))
        if self.median >= self.hi:
            self.state = "live"
        elif self.median <= self.lo:
            self.state = "spoof"
        return self.state


def fuse(live_pid: str, decoy_name: Optional[str], decoy_hit: bool, liveness_state: str) -> str:
    """The verdict a LIVE version WOULD give. Anti-spoof never demotes a recognised person."""
    pid = str(live_pid or "unknown")
    if pid != "unknown" and not pid.startswith("_"):
        return "person:" + pid
    if decoy_hit and decoy_name:
        return "decoy:" + decoy_name
    if liveness_state == "spoof":
        return "face_like"
    if liveness_state == "live":
        return "visitor"
    return "uncertain"


def actual_verdict(live_pid: str) -> str:
    """What the live path does with this face today: a named person, else a possible visitor
    (face_appeared|unknown -> wake)."""
    pid = str(live_pid or "unknown")
    if pid != "unknown" and not pid.startswith("_"):
        return "person:" + pid
    return "visitor"


def antispoof_box(src_w: int, src_h: int, bbox_xywh, scale: float) -> tuple[int, int, int, int]:
    """Silent-Face CropImage._get_new_box, verbatim semantics: expand the [x,y,w,h] box by
    `scale` about its centre (clamped to fit), shift to stay inside the image."""
    x, y, box_w, box_h = (float(v) for v in bbox_xywh)
    box_w, box_h = max(1.0, box_w), max(1.0, box_h)
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    lx, ly, rx, ry = cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2
    if lx < 0:
        rx -= lx; lx = 0
    if ly < 0:
        ry -= ly; ly = 0
    if rx > src_w - 1:
        lx -= rx - src_w + 1; rx = src_w - 1
    if ry > src_h - 1:
        ly -= ry - src_h + 1; ry = src_h - 1
    return int(lx), int(ly), int(rx), int(ry)


def xyxy_to_xywh(b) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    # Silent-Face's detector reports w = right-left+1
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


# ═════════════════════════════ models (CPU ONNX) ═══════════════════════════

def _cpu_session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = CPU_THREADS
    so.inter_op_num_threads = 1
    so.log_severity_level = 3   # det_10g's dynamic shapes spam VerifyOutputSizes warnings
    so.enable_cpu_mem_arena = False   # don't let the arena grow inside the runtime's RSS
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class AntiSpoof:
    """MiniFASNet ensemble (2.7 + 4.0 scale, 80x80). score() -> mean real-prob in 0..1."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        man = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.models = []
        for m in man["models"]:
            self.models.append((_cpu_session(model_dir / m["file"]), float(m["scale"]),
                                int(m["w"]), int(m["h"]), int(m.get("real_class", 1))))
        self.licence = man.get("licence")

    def score(self, frame, bbox_xyxy) -> tuple[float, list[float]]:
        import cv2
        np = _np()
        h, w = frame.shape[:2]
        xywh = xyxy_to_xywh(bbox_xyxy)
        per = []
        for sess, scale, ow, oh, real_cls in self.models:
            lx, ly, rx, ry = antispoof_box(w, h, xywh, scale)
            crop = frame[ly:ry + 1, lx:rx + 1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (ow, oh))
            x = crop.astype("float32").transpose(2, 0, 1)[None]  # BGR, 0..255, NO /255 (their to_tensor)
            logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
            e = np.exp(logits - logits.max())
            per.append(float(e[real_cls] / e.sum()))
        return (float(sum(per) / len(per)) if per else 0.0), per


class Embedder:
    """CPU copy of buffalo_l det_10g + w600k_r50 (same weights as the live GPU engine, so
    embeddings are comparable with the enrolled gallery). Never touches the live engine."""

    def __init__(self, det_thresh: float = 0.2):
        from insightface.model_zoo.retinaface import RetinaFace
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        self.det = RetinaFace(model_file=str(INSIGHT_DIR / "det_10g.onnx"),
                              session=_cpu_session(INSIGHT_DIR / "det_10g.onnx"))
        self.det.prepare(-1, det_thresh=det_thresh, input_size=(320, 320))
        self.rec = ArcFaceONNX(model_file=str(INSIGHT_DIR / "w600k_r50.onnx"),
                               session=_cpu_session(INSIGHT_DIR / "w600k_r50.onnx"))
        self.rec.prepare(-1)

    def detect_full(self, frame, thresh: float = 0.5) -> list[dict]:
        bboxes, kpss = self.det.detect(frame, input_size=(640, 640))
        out = []
        for i in range(bboxes.shape[0]):
            if float(bboxes[i, 4]) < thresh:
                continue
            out.append({"bbox": [float(v) for v in bboxes[i, :4]], "det_score": float(bboxes[i, 4]),
                        "kps": kpss[i] if kpss is not None else None})
        return out

    def embed_at(self, frame, bbox_xyxy, kps=None, det_score: Optional[float] = None,
                 full_dets: Optional[list] = None) -> Optional[dict]:
        """Embed the face at bbox.
        - With known 5-pt kps (a full-frame detection, or the live dict's 'kps' after the
          10-18 engine change lands) embed directly.
        - Otherwise re-run det_10g on the FULL frame at 640 - the exact preprocessing the
          live engine uses - and take the detection best overlapping bbox (IoU >= 0.3).
          Measured 10-18: crop-based re-detection is badly scale-sensitive on the helmet
          (found at one crop scale, lost at the next), which would silently skip the decoy
          check; the full-frame pass reproduces the live box. ~150 ms CPU, at most 1 Hz
          per track, only while a face is present.
        The low det_thresh only LOCATES keypoints for a box the caller already has."""
        from insightface.utils import face_align
        if kps is not None:
            k = _np().asarray(kps, dtype="float32").reshape(5, 2)
            aimg = face_align.norm_crop(frame, landmark=k, image_size=self.rec.input_size[0])
            return {"emb": self.rec.get_feat(aimg).flatten(), "det_score": None if det_score is None else float(det_score),
                    "bbox": [float(v) for v in bbox_xyxy[:4]], "iou": 1.0, "kps_from": "given"}
        dets = full_dets if full_dets is not None else self.detect_full(frame, thresh=0.2)
        best, best_iou = None, 0.0
        for d in dets:
            o = iou(d["bbox"], bbox_xyxy)
            if o > best_iou:
                best, best_iou = d, o
        if best is None or best_iou < 0.3 or best.get("kps") is None:
            return None
        aimg = face_align.norm_crop(frame, landmark=best["kps"], image_size=self.rec.input_size[0])
        return {"emb": self.rec.get_feat(aimg).flatten(), "det_score": float(best["det_score"]),
                "bbox": best["bbox"], "iou": best_iou, "kps_from": "cpu_full_frame"}


# ═════════════════════════════ decoy gallery ═══════════════════════════════

class DecoyGallery:
    """state/faces_decoys/<name>/{emb_*.npy, crop_*.jpg, meta.json}. NOT under faces/."""

    def __init__(self, root: Path = DECOY_DIR):
        self.root = root
        self.samples: dict[str, list] = {}
        self.meta: dict[str, dict] = {}
        self._sig = None

    def _signature(self):
        try:
            return tuple(sorted((p.name, p.stat().st_mtime) for p in self.root.glob("*/*")))
        except Exception:
            return None

    def load(self, force: bool = False) -> "DecoyGallery":
        sig = self._signature()
        if not force and sig == self._sig:
            return self
        np = _np()
        samples, meta = {}, {}
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                if not d.is_dir():
                    continue
                embs = []
                for p in sorted(d.glob("emb_*.npy")):
                    try:
                        embs.append(np.load(p))
                    except Exception:
                        pass
                if embs:
                    samples[d.name] = embs
                    try:
                        meta[d.name] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                    except Exception:
                        meta[d.name] = {}
        self.samples, self.meta, self._sig = samples, meta, sig
        return self

    def remembered_boxes(self, name: str) -> list:
        return [s.get("bbox") for s in (self.meta.get(name) or {}).get("samples", []) if s.get("bbox")]

    def require_near(self, name: str) -> bool:
        return bool((self.meta.get(name) or {}).get("require_near", False))

    def add(self, name: str, emb, crop_bgr, info: dict) -> Path:
        import cv2
        np = _np()
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        n = 0
        while (d / f"emb_{stamp}{'_' + str(n) if n else ''}.npy").exists():
            n += 1
        stamp = stamp + (f"_{n}" if n else "")
        np.save(d / f"emb_{stamp}.npy", np.asarray(emb, dtype="float32"))
        if crop_bgr is not None and getattr(crop_bgr, "size", 0):
            cv2.imwrite(str(d / f"crop_{stamp}.jpg"), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        mp = d / "meta.json"
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            m = {"name": name, "kind": "decoy", "require_near": False, "samples": []}
        m.setdefault("samples", []).append(dict(info, stamp=stamp))
        mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return d


def person_gallery(g: dict) -> dict[str, list]:
    """Read-only view of the LIVE engine's enrolled people (per-photo embeddings)."""
    ins = g.get("_insight_face")
    if ins is None:
        return {}
    try:
        multi = dict(getattr(ins, "_known_multi", {}) or {})
        if multi:
            return {k: list(v) for k, v in multi.items()}
        return {k: [v] for k, v in dict(getattr(ins, "_known", {}) or {}).items()}
    except Exception:
        return {}


def raw_frame(g: dict, max_age_s: float = 1.0):
    """Newest RAW (un-annotated) frame from the capture loop's slot. Never opens a camera."""
    slot = g.get("_raw_frame_slot")
    if not slot:
        return None, None
    ts, frame = slot
    age = time.time() - float(ts)
    if frame is None or age > max_age_s:
        return None, age
    return frame, age


# ═════════════════════════════ shadow log ══════════════════════════════════

class ShadowLog:
    def __init__(self, path: Path = SHADOW_LOG, max_bytes: int = LOG_MAX_BYTES):
        self.path, self.max_bytes = path, max_bytes
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                    os.replace(self.path, self.path.with_suffix(self.path.suffix + ".1"))
            except Exception:
                pass
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=float) + "\n")


def read_log(path: Path = SHADOW_LOG, since_ts: float = 0.0) -> list[dict]:
    out = []
    for p in (path.with_suffix(path.suffix + ".1"), path):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if float(r.get("ts", 0)) >= since_ts:
                    out.append(r)
    return out


def summarize(records: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for r in records:
        key = f"{r.get('source', 'live')}|actual={r.get('actual_verdict')}|would={r.get('would_verdict')}"
        s = by.setdefault(key, {"n": 0, "decoy_sim": [], "antispoof": [], "person_sim": [], "det": []})
        s["n"] += 1
        for k, f in (("decoy_sim", "decoy_sim"), ("antispoof", "antispoof_median"),
                     ("person_sim", "person_sim"), ("det", "det_score_shadow")):
            v = r.get(f)
            if v is not None:
                s[k].append(float(v))

    def st(v):
        if not v:
            return None
        return {"min": round(min(v), 3), "median": round(statistics.median(v), 3), "max": round(max(v), 3)}

    return {"records": len(records),
            "differs": sum(1 for r in records if r.get("differs")),
            "groups": {k: {"n": s["n"], "decoy_sim": st(s["decoy_sim"]), "antispoof_median": st(s["antispoof"]),
                           "person_sim": st(s["person_sim"]), "det_score_shadow": st(s["det"])}
                       for k, s in sorted(by.items())}}


# ═════════════════════════════ scoring one face ════════════════════════════

class Scorer:
    """Loads the CPU models once; scores (frame, bbox) -> signals. Shared by runner,
    tool (score_now / enroll) and the offline eval harness."""

    def __init__(self):
        self.antispoof = AntiSpoof()
        self.embedder = Embedder()
        self.decoys = DecoyGallery().load(force=True)

    def signals(self, frame, bbox, persons: dict[str, list], want_emb: bool = True,
                emb_cache: Optional[dict] = None, kps=None, det_score: Optional[float] = None,
                full_dets: Optional[list] = None) -> dict:
        t0 = time.time()
        a, per = self.antispoof.score(frame, bbox)
        out: dict[str, Any] = {"antispoof": round(a, 4), "antispoof_per_model": [round(x, 4) for x in per]}
        e = emb_cache
        if want_emb:
            e = self.embedder.embed_at(frame, bbox, kps=kps, det_score=det_score, full_dets=full_dets)
        if e is not None:
            self.decoys.load()
            pname, psim = best_match(e["emb"], persons)
            dname, dsim = best_match(e["emb"], self.decoys.samples)
            near = near_position(bbox, self.decoys.remembered_boxes(dname)) if dname else None
            hit = decoy_decision(dsim, psim, near, self.decoys.require_near(dname) if dname else False)
            out.update({"det_score_shadow": None if e.get("det_score") is None else round(e["det_score"], 4),
                        "kps_from": e.get("kps_from"), "person_name": pname,
                        "person_sim": round(psim, 4), "decoy_name": dname, "decoy_sim": round(dsim, 4),
                        "decoy_near": near, "decoy_hit": hit, "_emb": e})
        else:
            out.update({"det_score_shadow": None, "person_name": None, "person_sim": None,
                        "decoy_name": None, "decoy_sim": None, "decoy_near": None, "decoy_hit": False})
        out["ms"] = round((time.time() - t0) * 1000, 1)
        return out


# ═════════════════════════════ shadow runner ═══════════════════════════════

class ShadowRunner:
    def __init__(self, g: dict):
        self.g = g
        self.scorer: Optional[Scorer] = None
        self.log = ShadowLog()
        self.tracks: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next_id = 1
        self.stats = {"started_ts": None, "ticks": 0, "scored": 0, "logged": 0, "errors": 0,
                      "last_error": None, "last_ms": None, "avg_ms": None, "probe_runs": 0}
        self._last_probe = 0.0
        self._tick_dets: Optional[list] = None

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"ok": True, "already": True}
        self._stop.clear()   # the scorer (CPU models, ~5 s) loads INSIDE the thread: never blocks boot
        self.stats["started_ts"] = time.time()
        self._thread = threading.Thread(target=self._run, name="face_liveness_shadow", daemon=True)
        self._thread.start()
        return {"ok": True, "started": True}

    def stop(self) -> dict:
        self._stop.set()
        return {"ok": True, "stopping": True}

    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _flag(self) -> dict:
        try:
            return json.loads(FLAG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _track_for(self, bbox, now: float, source: str) -> dict:
        best, best_score = None, 0.0
        cx, cy, w = box_center_w(bbox)
        for t in self.tracks:
            if t["source"] != source:
                continue
            o = iou(bbox, t["bbox"])
            tx, ty, tw = box_center_w(t["bbox"])
            close = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 <= 0.5 * max(w, tw)
            sc = o if o >= 0.3 else (0.29 if close else 0.0)
            if sc > best_score:
                best, best_score = t, sc
        if best is None:
            best = {"id": self._next_id, "source": source, "bbox": list(bbox), "hyst": Hysteresis(),
                    "emb": None, "emb_ts": 0.0, "last_log": 0.0, "last_would": None, "first_ts": now}
            self._next_id += 1
            self.tracks.append(best)
        best["bbox"], best["last_ts"] = list(bbox), now
        return best

    def _score_face(self, frame, face: dict, now: float, frame_age, source: str) -> None:
        bbox = face["bbox"]
        t = self._track_for(bbox, now, source)
        want_emb = (now - t["emb_ts"]) >= EMB_EVERY_S or t["emb"] is None
        kps = face.get("kps")
        full = None
        if want_emb and kps is None:
            if self._tick_dets is None:   # once per tick, shared by every face in this frame
                self._tick_dets = self.scorer.embedder.detect_full(frame, thresh=0.2)
            full = self._tick_dets
        sig = self.scorer.signals(frame, bbox, person_gallery(self.g), want_emb=want_emb,
                                  emb_cache=None if want_emb else t["emb"],
                                  kps=kps, det_score=face.get("det_score"), full_dets=full)
        if want_emb and sig.get("_emb") is not None:
            t["emb"], t["emb_ts"] = sig["_emb"], now
        sig.pop("_emb", None)
        state = t["hyst"].update(now, sig["antispoof"])
        live_pid = str(face.get("person_id") or "unknown") if source == "live" else "unknown"
        would = fuse(live_pid, sig.get("decoy_name"), bool(sig.get("decoy_hit")), state)
        actual = actual_verdict(live_pid) if source == "live" else "not_detected_live"
        t["would"], t["state"], t["live_pid"] = would, state, live_pid   # read by write_now()
        self.stats["scored"] += 1
        ms = sig.get("ms")
        self.stats["last_ms"] = ms
        a = self.stats["avg_ms"]
        self.stats["avg_ms"] = ms if a is None else round(0.9 * a + 0.1 * ms, 1)
        if would != t["last_would"] or (now - t["last_log"]) >= LOG_EVERY_S:
            rec = {"ts": round(now, 3), "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                   "source": source, "track": t["id"], "track_age_s": round(now - t["first_ts"], 1),
                   "bbox": [int(v) for v in bbox], "frame_age_s": None if frame_age is None else round(frame_age, 3),
                   "det_score_live": face.get("det_score") if source == "live" else None, "live_pid": live_pid,
                   "live_conf": face.get("confidence"), "live_sticky": face.get("sticky"),
                   **sig, "antispoof_median": None if t["hyst"].median is None else round(t["hyst"].median, 4),
                   "liveness_state": state, "would_verdict": would, "actual_verdict": actual,
                   "differs": (would != actual) if source == "live" else None,
                   "verdict_changed": would != t["last_would"]}
            self.log.write(rec)
            self.stats["logged"] += 1
            t["last_log"], t["last_would"] = now, would

    def _write_now(self, now: float) -> None:
        """2026-10-20 (live gate): publish the current per-track verdicts for brain/liveness_gate.
        Written every tick while tracks exist, once more when they empty, then quiet."""
        if not self.tracks and not getattr(self, "_now_had_tracks", False):
            return
        snap = {"ts": round(now, 3), "tracks": [
            {"id": t["id"], "source": t["source"], "bbox": [int(v) for v in t["bbox"]],
             "live_pid": t.get("live_pid"), "would": t.get("would"), "state": t.get("state"),
             "last_ts": round(float(t.get("last_ts") or now), 3)}
            for t in self.tracks if t.get("would") is not None]}
        tmp = NOW_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap), encoding="utf-8")
        os.replace(tmp, NOW_PATH)
        self._now_had_tracks = bool(self.tracks)

    def _run(self) -> None:
        if self.scorer is None:
            try:
                self.scorer = Scorer()
            except Exception as e:
                self.stats["last_error"] = "scorer init: " + repr(e)[:280]
                self.stats["errors"] += 1
                return
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.stats["ticks"] += 1
                faces = list(self.g.get("_face_results") or [])
                self._tick_dets = None
                flag = self._flag()
                probe = bool(flag.get("probe_full_frame")) and (t0 - self._last_probe) >= PROBE_EVERY_S
                if faces or probe:
                    frame, age = raw_frame(self.g)
                    if frame is not None:
                        for f in faces:
                            if isinstance(f, dict) and f.get("bbox"):
                                self._score_face(frame, f, t0, age, "live")
                        if probe:
                            self._last_probe = t0
                            self.stats["probe_runs"] += 1
                            if self._tick_dets is None:
                                self._tick_dets = self.scorer.embedder.detect_full(frame, thresh=0.2)
                            pthr = float(flag.get("probe_thresh", 0.3))
                            pmin = int(flag.get("probe_min_px", 28))
                            for d in self._tick_dets:
                                if d["det_score"] < pthr:
                                    continue
                                if min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) < pmin:
                                    continue   # the live engine drops these (_MIN_FACE_PX) - can never wake
                                if any(iou(d["bbox"], f.get("bbox") or [0, 0, 0, 0]) > 0.3 for f in faces):
                                    continue   # already scored as a live face
                                self._score_face(frame, d, t0, age, "probe")
                self.tracks = [t for t in self.tracks if t0 - t.get("last_ts", t0) <= TRACK_TTL_S]
                self._write_now(t0)
            except Exception as e:  # never let the shadow die silently
                self.stats["errors"] += 1
                self.stats["last_error"] = repr(e)[:300]
                time.sleep(1.0)
            self._stop.wait(max(0.05, TICK_S - (time.time() - t0)))


_RUNNER: Optional[ShadowRunner] = None
_RUNNER_LOCK = threading.Lock()


def runner(g: dict) -> ShadowRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = ShadowRunner(g)
        else:
            _RUNNER.g = g
        return _RUNNER


def get_runner() -> Optional[ShadowRunner]:
    return _RUNNER


def flag() -> dict:
    try:
        return json.loads(FLAG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def maybe_autostart(g: dict) -> dict:
    """Boot hook (brain/iris_bootstrap.py). Default OFF: starts only if the flag says on."""
    if not flag().get("on"):
        return {"ok": True, "started": False, "reason": "flag off/absent"}
    return runner(g).start()
