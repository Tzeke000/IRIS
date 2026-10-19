"""iris_eyes — OFFLINE PROTOTYPE of the eyes process (docs/eyes_process_design.md).

Reads frames from a VIDEO FILE (never a camera), runs the SAME detectors the runtime
runs (imported from brain/, not copied), publishes frames to a /dev/shm ring
(shm_ring.py) and perception results on a Unix socket (eyes_bus.py), and logs
RSS / threads / glibc heap / VRAM / per-stage latency to a JSONL file.

    ~/venvs/iris-v100/bin/python scripts/eyes/iris_eyes.py \
        --video /data/iris_eyes_test/keyframes_30s.mp4 --loop --duration 240 \
        --metrics /data/iris_eyes_test/eyes_metrics.jsonl

Prototype scope — deliberately NOT here yet (see design §5/§7): enrollment hook,
face_tracking promotions (they write live state/), human_pose loop, liveness,
head-bearing input from the servo, sd_notify watchdog, camera capture.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# APPEND (never insert at 0): brain/ must be importable without letting the repo
# root shadow stdlib/site-packages names.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.insert(0, str(HERE))

import shm_ring  # noqa: E402
import eyes_bus  # noqa: E402

FRAME_W, FRAME_H = 1280, 720          # what the PIXY is asked for (MJPG 720p)


def _refuse_devices(path: str) -> str:
    p = str(path)
    if p.isdigit() or p.startswith("/dev/") or os.path.realpath(p).startswith("/dev/"):
        raise SystemExit("iris_eyes prototype: refusing a device path — video FILES only")
    if not os.path.isfile(p):
        raise SystemExit(f"no such video file: {p}")
    return p


# ── process self-measurement ─────────────────────────────────────────────────

def _proc_status() -> dict:
    out = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "Threads"):
                out[k] = int(v.split()[0])
    except OSError:
        pass
    return out


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
        "fsmblks", "uordblks", "fordblks", "keepcost")]


def _mallinfo() -> dict:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = _Mallinfo2
        m = libc.mallinfo2()
        return {"in_use_mb": round((m.uordblks + m.hblkhd) / 2**20, 1),
                "free_held_mb": round(m.fordblks / 2**20, 1),
                "mmap_mb": round(m.hblkhd / 2**20, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _vram_mb(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            p, _, mem = line.partition(",")
            if p.strip() == str(pid):
                return int(mem.strip())
        return 0
    except Exception:  # noqa: BLE001
        return None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    return round(float(np.percentile(np.asarray(xs), q)), 2)


# ── the event bus the brain/ modules fire into ───────────────────────────────

class ForwardingBus:
    """Stands in for g['_signal_bus'] inside the eyes process: brain/ modules
    (iris_hands, face_tracking, ...) fire exactly as they do today and the events
    are forwarded to subscribers, where the runtime shim re-fires them on the real
    in-process signal bus. Same .fire() signature as brain's bus."""

    def __init__(self, server: "eyes_bus.BusServer") -> None:
        self.server = server
        self.fired = 0

    def fire(self, name: str, data: dict | None = None, priority: str = "low", **_kw) -> None:
        self.fired += 1
        self.server.publish_event(name, {"data": data or {}, "priority": priority})


# ── the process ──────────────────────────────────────────────────────────────

class Eyes:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.g: dict = {"BASE_DIR": str(ROOT)}       # eyes-LOCAL g, never the runtime's
        self.lock = threading.Lock()
        self.slot: tuple | None = None               # (seq, capture_ts, frame) — in-process, no copy
        self.state = "starting"
        self.stop = threading.Event()
        self.results: dict = {}
        self.timings: dict[str, list[float]] = {}
        self.latency: dict[str, list[float]] = {}    # capture_ts -> result done
        self.ring_write_ms: list[float] = []
        self.frames = 0
        self.fps_ema = 0.0
        self.boot = {}
        self.ring = shm_ring.RingWriter(args.ring, nslots=4, slot_bytes=FRAME_W * FRAME_H * 3)
        self.bus = eyes_bus.BusServer(args.sock, self.snapshot, handlers={
            "status": lambda a: self.status(),
            "park": lambda a: self._set_state("parked"),
            "unpark": lambda a: self._set_state("streaming"),
        })
        self.g["_signal_bus"] = ForwardingBus(self.bus)
        self._prev_face_count = 0
        self._prev_pid = "unknown"

    # -- control ------------------------------------------------------------
    def _set_state(self, st: str) -> dict:
        self.state = st
        self.ring.set_state(shm_ring.STATE_PARKED if st == "parked" else shm_ring.STATE_STREAMING)
        self.bus.publish_event("eyes_state", {"state": st})
        return {"state": st}

    def status(self) -> dict:
        return {"state": self.state, "pid": os.getpid(), "frames": self.frames,
                "fps": round(self.fps_ema, 1), "boot": self.boot,
                "subscribers": self.bus.subscriber_count(), "bus_dropped": self.bus.dropped,
                **_proc_status()}

    def snapshot(self) -> dict:
        with self.lock:
            res = {k: dict(v) for k, v in self.results.items()}
            slot = self.slot
        return {"eyes_state": self.state, "pid": os.getpid(), "ts": time.time(),
                "frame": {"seq": slot[0] if slot else 0, "capture_ts": slot[1] if slot else None,
                          "fps": round(self.fps_ema, 1)},
                **res}

    def _publish(self, key: str, frame_seq: int, frame_ts: float, t0: float, value) -> None:
        done = time.time()
        with self.lock:
            self.results[key] = {"frame_seq": frame_seq, "frame_ts": frame_ts,
                                 "done_ts": done, "ms": round((done - t0) * 1000, 2),
                                 "value": value}
        self.bus.notify_result()
        self.timings.setdefault(key, []).append((done - t0) * 1000)
        self.latency.setdefault(key, []).append((done - frame_ts) * 1000)

    # -- boot ---------------------------------------------------------------
    def load_models(self) -> None:
        dets = set(self.args.detectors.split(","))
        if "face" in dets:
            t0 = time.time()
            from brain.insight_face_engine import InsightFaceEngine
            eng = InsightFaceEngine()
            ok = eng.initialize(ROOT / "faces")     # READ-only use of faces/
            self.g["_insight_face"] = eng if ok else None
            self.boot["face"] = {"ok": bool(ok), "s": round(time.time() - t0, 1),
                                 "provider": eng.provider() if ok else None,
                                 "known": eng.known_count() if ok else 0}
        if "body" in dets:
            t0 = time.time()
            from brain import person_track, yolox_person
            # warm the detector once so the first live frame isn't a 2 s outlier
            yolox_person.detect(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
            self.boot["body"] = {"ok": True, "s": round(time.time() - t0, 1)}
            self._person_track = person_track
        if "expr" in dets:
            from brain.expression_detector import bootstrap_expression_detector
            bootstrap_expression_detector(self.g)
            self.boot["expr"] = {"ok": bool(getattr(self.g.get("_expression_detector"), "available", False))}
        if "attn" in dets:
            from brain.eye_tracker import bootstrap_eye_tracker
            bootstrap_eye_tracker(self.g)
            self.boot["attn"] = {"ok": bool(getattr(self.g.get("_eye_tracker"), "available", False))}
        self.dets = dets

    # -- capture (video file stands in for the camera) ------------------------
    def capture_loop(self) -> None:
        import cv2
        period = 1.0 / self.args.fps
        cap = cv2.VideoCapture(self.args.video)
        last = 0.0
        while not self.stop.is_set():
            if self.state == "parked":
                self.ring.heartbeat()
                time.sleep(0.25)
                continue
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if not self.args.loop:
                    break
                cap.release()
                cap = cv2.VideoCapture(self.args.video)
                continue
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            cts = time.time()
            tw = time.perf_counter()
            seq = self.ring.write(frame, cts)
            self.ring_write_ms.append((time.perf_counter() - tw) * 1000)
            with self.lock:
                self.slot = (seq, cts, frame)
            self.frames += 1
            if last:
                gap = cts - last
                if 0 < gap < 2:
                    self.fps_ema = (0.95 * self.fps_ema + 0.05 / gap) if self.fps_ema else 1 / gap
            last = cts
            time.sleep(max(0.0, period - (time.time() - t0)))
        cap.release()
        self.stop.set()

    # -- perception workers (same 'newest frame, never twice' contract) -------
    def _worker(self, name: str, step) -> None:
        last_seq = 0
        while not self.stop.is_set():
            with self.lock:
                slot = self.slot
            if self.state != "streaming" or slot is None or slot[0] <= last_seq:
                time.sleep(0.004)
                continue
            last_seq = slot[0]
            t0 = time.time()
            try:
                step(*slot, t0)
            except Exception as e:  # noqa: BLE001 — seeing degrades, never raises
                print(f"[iris_eyes] {name} error: {e!r}", file=sys.stderr, flush=True)
                time.sleep(0.5)
            time.sleep(max(0.0, 1.0 / 30 - (time.time() - t0)))

    def _face_step(self, seq, cts, frame, t0) -> None:
        eng = self.g.get("_insight_face")
        res = eng.analyze_frame(frame) if eng is not None else []
        self.g["_face_results"] = res
        best = max(res, key=lambda r: r.get("confidence") or 0.0) if res else None
        pid = str(best.get("person_id") or "unknown") if best else "unknown"
        self._publish("faces", seq, cts, t0, res)
        with self.lock:
            self.results["recognized"] = {
                "frame_ts": cts, "frame_seq": seq,
                "value": {"person_id": pid,
                          "confidence": float(best.get("confidence") or 0.0) if best else 0.0,
                          "age": best.get("age", 0) if best else 0,
                          "gender": best.get("gender", "?") if best else "?"}}
        # transition events, as the runtime's face worker fires them today
        n = len(res)
        bus = self.g["_signal_bus"]
        if n and not self._prev_face_count:
            bus.fire("face_appeared", {"person_id": pid}, "medium")
        elif not n and self._prev_face_count:
            bus.fire("face_lost", {"prior_person_id": self._prev_pid}, "medium")
        elif pid != self._prev_pid and pid != "unknown":
            bus.fire("face_changed", {"from": self._prev_pid, "to": pid}, "medium")
        self._prev_face_count, self._prev_pid = n, pid

    def _body_step(self, seq, cts, frame, t0) -> None:
        pt = self._person_track
        pt.step(frame, self.g.get("_face_results") or [], cts)
        st = pt.status() if hasattr(pt, "status") else {}
        boxes, shape = pt.track_boxes()
        offsets = {}
        for label in list((st.get("bindings") or {}).keys()) if isinstance(st, dict) else []:
            try:
                offsets[label] = pt.target_offset(label)
            except Exception:  # noqa: BLE001
                pass
        self._publish("body", seq, cts, t0, {"boxes": boxes, "frame_shape": shape,
                                             "tracks": st.get("tracks"),
                                             "bindings": st.get("bindings"),
                                             "offsets": offsets})

    def _expr_step(self, seq, cts, frame, t0) -> None:
        et = self.g.get("_expression_detector")
        if et is None or not getattr(et, "available", False):
            return
        self._publish("expression", seq, cts, t0, et.detect_expression(frame))

    def _attn_step(self, seq, cts, frame, t0) -> None:
        ez = self.g.get("_eye_tracker")
        if ez is None or not getattr(ez, "available", False):
            return
        if not (self.g.get("_face_results") or []):
            self._publish("attention", seq, cts, t0, {"attention_state": "absent"})
            return
        az = ez.analyze(frame)
        self._publish("attention", seq, cts, t0, {k: az.get(k) for k in
                                                  ("attention_state", "looking_at_screen", "gaze_region")})

    # -- metrics ------------------------------------------------------------
    def metrics_loop(self) -> None:
        f = open(self.args.metrics, "a") if self.args.metrics else None
        t_start = time.time()
        while not self.stop.wait(self.args.metrics_every):
            row = {"t": round(time.time() - t_start, 1), "state": self.state,
                   "frames": self.frames, "fps": round(self.fps_ema, 1),
                   **_proc_status(), "heap": _mallinfo(), "vram_mb": _vram_mb(os.getpid()),
                   "ring_write_ms_p50": _pct(self.ring_write_ms[-300:], 50),
                   "ring_write_ms_p95": _pct(self.ring_write_ms[-300:], 95),
                   "subscribers": self.bus.subscriber_count(), "bus_dropped": self.bus.dropped}
            for k, xs in list(self.timings.items()):
                row[f"{k}_ms_p50"] = _pct(xs[-300:], 50)
                row[f"{k}_ms_p95"] = _pct(xs[-300:], 95)
            for k, xs in list(self.latency.items()):
                row[f"{k}_lat_ms_p50"] = _pct(xs[-300:], 50)
                row[f"{k}_lat_ms_p95"] = _pct(xs[-300:], 95)
            line = json.dumps(row)
            print(f"[iris_eyes] {line}", flush=True)
            if f:
                f.write(line + "\n")
                f.flush()

    def run(self) -> None:
        t0 = time.time()
        self.bus.start()
        self.load_models()
        self.boot["total_s"] = round(time.time() - t0, 1)
        print(f"[iris_eyes] models up: {json.dumps(self.boot)}", flush=True)
        self._set_state("streaming")
        threads = [threading.Thread(target=self.capture_loop, daemon=True, name="eyes-capture"),
                   threading.Thread(target=self.metrics_loop, daemon=True, name="eyes-metrics")]
        steps = {"face": self._face_step, "body": self._body_step,
                 "expr": self._expr_step, "attn": self._attn_step}
        for d in self.dets:
            if d in steps:
                threads.append(threading.Thread(target=self._worker, args=(d, steps[d]),
                                                daemon=True, name=f"eyes-{d}-worker"))
        for t in threads:
            t.start()
        try:
            self.stop.wait(self.args.duration if self.args.duration > 0 else None)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            time.sleep(0.3)
            self.ring.close()
            self.bus.stop()
            print("[iris_eyes] stopped; ring unlinked, socket removed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True, help="video FILE to stand in for the camera")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--detectors", default="face,body", help="comma list: face,body,expr,attn")
    ap.add_argument("--ring", default="iris_eyes_proto", help="/dev/shm name (NOT the future live name)")
    ap.add_argument("--sock", default=os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp",
                                                   "iris", "eyes_proto.sock"))
    ap.add_argument("--metrics", default="")
    ap.add_argument("--metrics-every", type=float, default=5.0)
    a = ap.parse_args()
    a.video = _refuse_devices(a.video)
    Eyes(a).run()


if __name__ == "__main__":
    main()
