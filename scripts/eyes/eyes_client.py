"""eyes_client — tiny consumer for the eyes prototype: latest frame + results.

Plays the role the runtime would: reads the newest frame from the /dev/shm ring at
~30 Hz into ONE preallocated buffer (as frame_store's ring mode would), mirrors the
perception state via a subscription, and measures what crossing the boundary costs.

    ~/venvs/iris-v100/bin/python scripts/eyes/eyes_client.py --seconds 60 \
        --out /data/iris_eyes_test/client_summary.json --save-frame /data/iris_eyes_test/latest.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import shm_ring  # noqa: E402
import eyes_bus  # noqa: E402


def _pct(xs, q):
    return round(float(np.percentile(np.asarray(xs), q)), 3) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", default="iris_eyes_proto")
    ap.add_argument("--sock", default=os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp",
                                                   "iris", "eyes_proto.sock"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-frame", default="", help="write one downscaled JPEG of the latest frame")
    a = ap.parse_args()

    ring = shm_ring.RingReader(a.ring)
    events: list[tuple[float, str]] = []
    snap_lat: list[float] = []           # NEW face result: source capture_ts -> arrival here (ms)
    xport_lat: list[float] = []          # NEW face result: eyes done_ts -> arrival here (transport only)
    snap_bytes: list[int] = []
    seen = {"face_seq": 0}

    def on_snap(s: dict) -> None:
        now = time.time()
        f = s.get("faces") or {}
        if f.get("frame_ts") and f.get("frame_seq", 0) > seen["face_seq"]:
            seen["face_seq"] = f["frame_seq"]          # count each face result once
            snap_lat.append((now - f["frame_ts"]) * 1000)
            xport_lat.append((now - f["done_ts"]) * 1000)
        snap_bytes.append(len(json.dumps(s)))

    sub = eyes_bus.Subscriber(a.sock, hz=a.hz, on_snapshot=on_snap,
                              on_event=lambda n, d: events.append((time.time(), n))).start()

    out = np.empty((720, 1280, 3), dtype=np.uint8)     # one buffer, reused forever
    copy_ms, age_ms, reasons = [], [], {}
    rtt_ms = []
    last_seq, new_frames, polls = 0, 0, 0
    t_end = time.time() + a.seconds
    next_ping = 0.0
    while time.time() < t_end:
        t0 = time.perf_counter()
        frame, meta = ring.read_latest(out=out, newer_than=last_seq, max_age_sec=2.0)
        dt = (time.perf_counter() - t0) * 1000
        polls += 1
        if frame is None:
            reasons[meta] = reasons.get(meta, 0) + 1
        else:
            copy_ms.append(dt)
            age_ms.append(meta.age_sec * 1000)
            last_seq = meta.seq
            new_frames += 1
        if time.time() >= next_ping:
            tp = time.perf_counter()
            r = eyes_bus.request("ping", sock_path=a.sock)
            if r.get("ok"):
                rtt_ms.append((time.perf_counter() - tp) * 1000)
            next_ping = time.time() + 1.0
        time.sleep(1.0 / a.hz)

    snap = sub.latest()
    pres = eyes_bus.presence_view(snap)
    if a.save_frame and last_seq:
        import cv2
        small = cv2.resize(out, (640, 360))
        cv2.imwrite(a.save_frame, small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    summary = {
        "seconds": a.seconds, "polls": polls, "new_frames": new_frames,
        "frame_copy_ms_p50": _pct(copy_ms, 50), "frame_copy_ms_p95": _pct(copy_ms, 95),
        "frame_copy_ms_max": round(max(copy_ms), 3) if copy_ms else None,
        "frame_age_at_read_ms_p50": _pct(age_ms, 50), "frame_age_at_read_ms_p95": _pct(age_ms, 95),
        "torn_read_retries": ring.retries, "no_frame_reasons": reasons,
        "face_capture_to_client_ms_p50": _pct(snap_lat, 50), "face_capture_to_client_ms_p95": _pct(snap_lat, 95),
        "face_transport_ms_p50": _pct(xport_lat, 50), "face_transport_ms_p95": _pct(xport_lat, 95),
        "face_results_rx": len(snap_lat),
        "snapshot_bytes_p50": _pct(snap_bytes, 50), "snapshot_bytes_max": max(snap_bytes) if snap_bytes else None,
        "snapshots_rx": len(snap_lat), "events_rx": len(events),
        "event_names": sorted({n for _, n in events}),
        "ping_rtt_ms_p50": _pct(rtt_ms, 50), "ping_rtt_ms_p95": _pct(rtt_ms, 95),
        "writer": ring.writer_status(), "presence_view": pres,
        "last_body_tracks": len(((snap or {}).get("body") or {}).get("value", {}).get("boxes") or []),
    }
    sub.stop()
    ring.close()
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
