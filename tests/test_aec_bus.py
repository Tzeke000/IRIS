"""voice/aec_bus.py: sink reference -> UDP localhost -> RefSubscriber -> MicHub AEC, in real time.

No sound devices: a fake speaker publishes what it 'plays' (stamped like audio_sink does) and a fake
mic returns the room echo of that audio by CLOCK time, exactly the way the real pair shares the tower's
clock. Needs the livekit wheel for the AEC part (skips otherwise)."""
import os
import struct
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice"))
import aec_bus  # noqa: E402
import iris_aec as A  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_iris_aec import speechlike  # noqa: E402

SR = A.SR


def _has_webrtc():
    try:
        A._make_backend("webrtc")
        return True
    except Exception:
        return False


def test_reference_datagram_roundtrip_and_validation():
    tl = A.RenderTimeline()
    sub = aec_bus.RefSubscriber(tl, port=0)
    try:
        x = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        pkt = aec_bus._HDR.pack(aec_bus._MAGIC, 50.0, 16000, x.size) + x.astype("<f4").tobytes()
        assert sub.handle(pkt)
        got = tl.read(50.0 + 200 / SR, 160)
        assert np.abs(got).max() > 0.01
        # bad magic / truncated / absurd rate are rejected, never written
        assert not sub.handle(b"XXXX" + pkt[4:])
        assert not sub.handle(pkt[:-4])
        assert not sub.handle(aec_bus._HDR.pack(aec_bus._MAGIC, 1.0, 5, 1) + b"\0\0\0\0")
        assert sub.bad == 3
    finally:
        sub.stop()


def test_publisher_splits_and_never_raises_without_listener():
    pub = aec_bus.RefPublisher(port=9)        # discard port, nobody listening
    pub.publish(np.zeros(10000, np.float32), 44100, time.perf_counter())
    assert pub.sent + pub.errors == 3          # 4096 + 4096 + 1808


def test_play_start_estimate():
    t = aec_bus.RefPublisher.play_start_after_write(441, 44100, 0.1)
    assert abs(t - (time.perf_counter() + 0.09)) < 0.01


def test_session_reader_rechunks():
    import queue
    q = queue.Queue()
    for k in range(5):
        q.put(np.full(320, k, np.float32))
    r = aec_bus.session_reader(q, 512)
    a, b, c = next(r), next(r), next(r)
    assert a.size == b.size == c.size == 512
    assert a[0] == 0 and a[-1] == 1 and c[-1] == 4


class _Air:
    """What has come out of the speaker, by clock time (the room's memory)."""
    def __init__(self):
        self.tl = A.RenderTimeline(span_s=8.0)


class _FakeMic:
    def __init__(self, air, near, t0, delay_s, gain):
        self.air, self.near, self.t0, self.d, self.g = air, near, t0, delay_s, gain
        self.latency = 0.0
        self._t = None
        self.raw = []

    def __enter__(self):
        self._t = A.CLOCK()
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self._t += n / SR
        while A.CLOCK() < self._t:
            time.sleep(0.002)
        t_cap = self._t - n / SR
        echo = self.air.tl.read(t_cap - self.d, n) * self.g
        i = int(round((t_cap - self.t0) * SR))
        nn = np.zeros(n, np.float32)
        if 0 <= i < self.near.size:
            seg = self.near[i:i + n]
            nn[:seg.size] = seg
        self.raw.append(echo + nn)
        return (echo + nn).reshape(-1, 1), False


@pytest.mark.skipif(not _has_webrtc(), reason="livekit wheel not installed")
def test_hub_cancels_echo_end_to_end_realtime():
    far = speechlike(5.0, 41, f0=210)
    air = _Air()
    t0 = A.CLOCK() + 0.3
    near = np.zeros(int(7 * SR), np.float32)          # he stays silent: pure echo test
    mic = _FakeMic(air, near, t0, delay_s=0.07, gain=0.5)
    hub = aec_bus.MicHub(lambda sr, blk: mic, backend="webrtc", lead_ms=40, block=320, ref_port=0, log=lambda m: None)
    pub = aec_bus.RefPublisher(port=hub.ref.port)
    outs = []

    def speaker():
        # plays far at 44.1 kHz in 20 ms blocks: publishes the reference, then 'emits' it into the air
        from scipy.signal import resample_poly
        f44 = resample_poly(far, 441, 160).astype(np.float32)
        blk = 882
        t = t0
        for i in range(0, f44.size, blk):
            while A.CLOCK() < t - 0.06:                # the sink writes ~60 ms ahead of playback
                time.sleep(0.002)
            part = f44[i:i + blk]
            pub.publish(part, 44100, t)
            air.tl.write(resample_poly(part, 160, 441).astype(np.float32), t)
            t += part.size / 44100

    q = hub.subscribe()
    th = threading.Thread(target=speaker, daemon=True)
    th.start()
    rd = aec_bus.session_reader(q, 512, timeout_s=5)
    t_end = t0 + 5.0
    while A.CLOCK() < t_end:
        outs.append(next(rd))
    hub.unsubscribe(q)
    st = hub.stats()
    hub.close()
    assert st["ref_rx"] > 100 and st["ref_bad"] == 0
    raw = np.concatenate(mic.raw)
    out = np.concatenate(outs)
    n = min(raw.size, out.size)
    s = slice(int(2.3 * SR), n)                        # after convergence (playback starts at +0.3 s)
    erle = A.erle_db(raw[s], out[s])
    print(f"\nhub end-to-end ERLE {erle:.1f} dB, stats {st}")
    assert erle > 10.0, (erle, st)
