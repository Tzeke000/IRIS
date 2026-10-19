"""Transport tests for the eyes-process prototype (scripts/eyes/shm_ring.py + eyes_bus.py).

No camera, no GPU, no models, no runtime import. Rings live in /dev/shm under a
per-test unique name and are unlinked afterwards; sockets live in a temp dir.

    /home/iris/venvs/iris-v100/bin/python tests/test_eyes_transport.py
(also collectable by pytest if it is ever installed)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import struct
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eyes"))   # the prototype dir only — never the repo root

import shm_ring as R  # noqa: E402
import eyes_bus as B  # noqa: E402


def _name() -> str:
    return f"iris_eyes_test_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _frame(val: int, h: int = 48, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), val % 256, dtype=np.uint8)


# ── ring: correctness ─────────────────────────────────────────────────────────

def test_roundtrip_and_meta():
    n = _name()
    w = R.RingWriter(n, nslots=4, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        f = np.random.RandomState(1).randint(0, 255, (48, 64, 3), dtype=np.uint8)
        ts = time.time()
        seq = w.write(f, ts)
        got, meta = r.read_latest()
        assert seq == 1 and meta.seq == 1
        assert got.shape == (48, 64, 3) and np.array_equal(got, f)
        assert abs(meta.capture_ts - ts) < 1e-6 and meta.freshness == "fresh"
        # the returned array is a private copy, not a view into shm
        w.write(_frame(9), time.time())
        assert np.array_equal(got, f)
    finally:
        r.close(); w.close()


def test_absent_and_empty():
    n = _name()
    r = R.RingReader(n)
    assert r.read_latest() == (None, "absent")
    assert r.writer_status()["alive"] is False
    w = R.RingWriter(n, nslots=2, slot_bytes=100)
    try:
        assert r.read_latest() == (None, "no_frame_yet")
    finally:
        r.close(); w.close()


def test_wraparound_returns_newest():
    n = _name()
    w = R.RingWriter(n, nslots=3, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        for i in range(1, 11):
            w.write(_frame(i), time.time())
        got, meta = r.read_latest()
        assert meta.seq == 10 and int(got[0, 0, 0]) == 10
    finally:
        r.close(); w.close()


def test_newer_than_guard():
    """Cross-process version of person_track's 'refuse a frame already seen'."""
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        w.write(_frame(1), time.time())
        _, m = r.read_latest()
        assert r.read_latest(newer_than=m.seq) == (None, "no_new_frame")
        w.write(_frame(2), time.time())
        got, m2 = r.read_latest(newer_than=m.seq)
        assert m2.seq == m.seq + 1
    finally:
        r.close(); w.close()


def test_staleness_bands():
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        w.write(_frame(1), time.time() - 5.0)        # an old capture
        assert r.read_latest(max_age_sec=1.0) == (None, "stale")
        got, meta = r.read_latest()                  # no limit: returned, but labelled
        assert got is not None and meta.freshness == "stale"
        assert R.classify(0.1) == "fresh" and R.classify(0.8) == "aging"
        assert R.classify(3.0) == "stale" and R.classify(None) == "unavailable"
    finally:
        r.close(); w.close()


def test_preallocated_out_buffer():
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        out = np.empty((48, 64, 3), dtype=np.uint8)
        w.write(_frame(77), time.time())
        got, _ = r.read_latest(out=out)
        assert np.shares_memory(got, out) and int(out[5, 5, 1]) == 77
    finally:
        r.close(); w.close()


def test_odd_lock_is_never_returned():
    """A slot whose seqlock is odd (writer mid-copy) must not be handed out."""
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        seq = w.write(_frame(3), time.time())
        sh = R.HDR_SIZE + (seq % 2) * R.SLOT_HDR
        lock = struct.unpack_from("<Q", w._buf, sh)[0]
        struct.pack_into("<Q", w._buf, sh, lock + 1)   # simulate a writer frozen mid-copy
        got, why = r.read_latest(max_tries=3)
        assert got is None and why == "contended" and r.retries == 3
        struct.pack_into("<Q", w._buf, sh, lock + 2)
        got, meta = r.read_latest()
        assert got is not None
    finally:
        r.close(); w.close()


def test_oversize_frame_rejected():
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=100)
    try:
        try:
            w.write(_frame(1), time.time())
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        w.close()


def test_writer_status_states():
    n = _name()
    w = R.RingWriter(n, nslots=2, slot_bytes=100)
    r = R.RingReader(n)
    try:
        st = r.writer_status()
        assert st["alive"] and st["state"] == "starting" and st["writer_pid"] == os.getpid()
        w.set_state(R.STATE_PARKED)
        assert r.writer_status()["state"] == "parked" and r.writer_status()["alive"]
        # heartbeat stops (process hung / SIGKILLed without cleanup) ⇒ dead
        struct.pack_into("<d", w._buf, R._OFF_HB, time.time() - 10.0)
        st = r.writer_status()
        assert st["alive"] is False and st["state"] == "dead"
    finally:
        w.close(unlink=False)
        assert r.writer_status()["state"] == "stopped"
        r.close()
        os.unlink(R.shm_path(n))


def test_reader_follows_writer_restart():
    """Eyes restart = new file (new inode). A long-lived reader must re-map, not
    keep reading the dead ring forever."""
    n = _name()
    w1 = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
    r = R.RingReader(n)
    try:
        w1.write(_frame(11), time.time())
        assert int(r.read_latest()[0][0, 0, 0]) == 11
        w1.close()                                    # unlinks
        assert r.read_latest()[1] == "absent"
        w2 = R.RingWriter(n, nslots=2, slot_bytes=64 * 48 * 3)
        w2.write(_frame(22), time.time())
        got, meta = r.read_latest()
        assert int(got[0, 0, 0]) == 22 and meta.seq == 1
        w2.close()
    finally:
        r.close()


def _stress_writer(name: str, seconds: float, ready) -> None:
    w = R.RingWriter(name, nslots=2, slot_bytes=120 * 160 * 3)   # 2 slots = max contention
    ready.set()
    t_end = time.time() + seconds
    i = 0
    base = np.empty((120, 160, 3), dtype=np.uint8)
    while time.time() < t_end:
        i += 1
        base.fill(i % 256)
        base[0, 0, :] = [(i >> 16) & 255, (i >> 8) & 255, i & 255]   # seq stamp
        w.write(base, time.time())
    w.close(unlink=False)


def test_no_torn_frames_under_contention():
    """Writer in another PROCESS hammering a 2-slot ring; every frame the reader
    accepts must be internally uniform and carry the seq it claims."""
    n = _name()
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_stress_writer, args=(n, 2.0, ready))
    p.start()
    try:
        assert ready.wait(10.0)
        r = R.RingReader(n)
        good = 0
        out = np.empty((120, 160, 3), dtype=np.uint8)
        t_end = time.time() + 1.8
        while time.time() < t_end:
            got, meta = r.read_latest(out=out)
            if got is None:
                continue
            stamp = (int(got[0, 0, 0]) << 16) | (int(got[0, 0, 1]) << 8) | int(got[0, 0, 2])
            assert stamp == meta.seq, (stamp, meta.seq)
            body = got[1:]
            assert int(body.min()) == int(body.max()) == meta.seq % 256, "torn frame"
            good += 1
        assert good > 100, good
        r.close()
    finally:
        p.join(10.0)
        try:
            os.unlink(R.shm_path(n))
        except FileNotFoundError:
            pass


# ── results bus ──────────────────────────────────────────────────────────────

def _bus(snap: dict):
    d = tempfile.mkdtemp(prefix="eyesbus_")
    path = os.path.join(d, "eyes.sock")
    srv = B.BusServer(path, lambda: dict(snap), handlers={"echo": lambda a: a})
    srv.start()
    return srv, path


def test_bus_request_response():
    snap = {"eyes_state": "streaming", "faces": {"frame_ts": time.time(), "value": []}}
    srv, path = _bus(snap)
    try:
        assert B.request("ping", sock_path=path)["result"]["pong"] is True
        assert B.request("snapshot", sock_path=path)["result"]["eyes_state"] == "streaming"
        assert B.request("echo", {"x": 1}, sock_path=path)["result"] == {"x": 1}
        bad = B.request("nope", sock_path=path)
        assert bad["ok"] is False and "unknown cmd" in bad["error"]
    finally:
        srv.stop()
    dead = B.request("ping", sock_path=path, timeout=0.3)
    assert dead["ok"] is False and dead["error"].startswith("eyes_unreachable")


def test_bus_subscription_events_and_link_loss():
    snap = {"eyes_state": "streaming", "faces": {"frame_ts": time.time(), "value": [{"person_id": "x"}]}}
    srv, path = _bus(snap)
    events = []
    sub = B.Subscriber(path, hz=20, on_event=lambda n, d: events.append((n, d))).start()
    try:
        t0 = time.time()
        while sub.latest() is None and time.time() - t0 < 3:
            time.sleep(0.02)
        assert sub.latest() is not None and sub.latest()["type"] == "snapshot"
        while srv.subscriber_count() == 0 and time.time() - t0 < 3:
            time.sleep(0.02)
        srv.publish_event("face_appeared", {"person_id": "x"})
        while not events and time.time() - t0 < 3:
            time.sleep(0.02)
        assert events and events[0][0] == "face_appeared"
    finally:
        srv.stop()
    # server gone: link must read as dead within LINK_DEAD_S (+ slack), and
    # latest() must stop returning the last snapshot
    t0 = time.time()
    while sub.latest() is not None and time.time() - t0 < B.LINK_DEAD_S + 2:
        time.sleep(0.05)
    assert sub.latest() is None and not sub.link_alive()
    sub.stop()


def test_push_on_new_result_not_on_clock():
    """A new result must reach a subscriber promptly (not wait for the 1 s
    heartbeat), and an idle server must still heartbeat."""
    state = {"n": 0}
    d = tempfile.mkdtemp(prefix="eyesbus_")
    path = os.path.join(d, "eyes.sock")
    srv = B.BusServer(path, lambda: {"eyes_state": "streaming", "n": state["n"]})
    srv.start()
    seen = []
    sub = B.Subscriber(path, hz=30, on_snapshot=lambda s: seen.append((time.time(), s["n"]))).start()
    try:
        t0 = time.time()
        while not seen and time.time() - t0 < 3:
            time.sleep(0.01)
        time.sleep(0.2)
        lat = []
        for i in range(1, 6):
            state["n"] = i
            ts = time.time()
            srv.notify_result()
            while not any(n == i for _, n in seen) and time.time() - ts < 2:
                time.sleep(0.001)
            lat.append(next(t for t, n in seen if n == i) - ts)
            time.sleep(0.1)
        assert max(lat) < 0.1, lat                # pushed, not waiting for heartbeat
        before = len(seen)
        time.sleep(2.3)                           # idle: heartbeats only
        assert 1 <= len(seen) - before <= 4, len(seen) - before
    finally:
        sub.stop(); srv.stop()


def test_presence_view_never_says_nobody_when_blind():
    now = time.time()
    fresh = {"eyes_state": "streaming", "faces": {"frame_ts": now - 0.1, "value": []}}
    assert B.presence_view(fresh, now) == {"seeing": True, "face_count": 0, "why": "fresh"}
    for snap, why in (
        (None, "no_link"),
        ({"eyes_state": "parked", "faces": {"frame_ts": now, "value": []}}, "eyes_parked"),
        ({"eyes_state": "streaming", "faces": {"frame_ts": now - 9, "value": []}}, "faces_stale"),
        ({"eyes_state": "streaming", "faces": {"frame_ts": None, "value": []}}, "faces_unavailable"),
    ):
        v = B.presence_view(snap, now)
        assert v["seeing"] is False and v["face_count"] is None and v["why"] == why, (v, why)
    assert B.classify_result(now + 5, now) == "stale"     # future stamps are never 'fresh'


def test_slow_subscriber_drops_oldest_not_blocks():
    srv = B.BusServer("/nonexistent/unused.sock", lambda: {})
    import queue
    q = queue.Queue(maxsize=3)
    srv._subs.append(q)
    for i in range(10):
        srv.publish_event("e", {"i": i})                 # must never block
    got = [q.get_nowait()["data"]["i"] for _ in range(3)]
    assert got == [7, 8, 9] and srv.dropped == 7


def _run_all() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({(time.time() - t0) * 1000:.0f} ms)")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_all())
