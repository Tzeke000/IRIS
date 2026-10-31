"""voice/iris_aec.py: echo canceller wrapper, render timeline, streaming resampler.

Synthetic audio only (no voices in the repo). The webrtc/speex cases skip when the backend
library isn't installed in the running venv (livekit wheel / libspeexdsp)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice"))
import iris_aec as A  # noqa: E402

SR = A.SR


def speechlike(seconds: float, seed: int, f0: float = 140.0) -> np.ndarray:
    """Harmonic source with a wandering pitch, syllable-rate amplitude envelope and pauses."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    pitch = f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(pitch) / SR
    x = sum((0.6 / k) * np.sin(k * ph + rng.uniform(0, 6)) for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6)), 0, None) ** 0.7
    gate = (np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6)) > -0.6).astype(float)
    x = x * env * gate + 0.01 * rng.standard_normal(n)
    return (0.4 * x / np.abs(x).max()).astype(np.float32)


def echo_path(x: np.ndarray, delay_ms: float, gain: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = int(delay_ms * SR / 1000)
    tail = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 500.0) * 0.2
    h = np.zeros(d + 2400)
    h[d] = 1.0
    h[d + 1:] += tail[1:]
    return (np.convolve(x, h)[:x.size] * gain).astype(np.float32)


def run(aec, mic, far, chunk=512):
    out = []
    for i in range(0, mic.size, chunk):
        aec.push_render(far[i:i + chunk])
        out.append(aec.process(mic[i:i + chunk]))
    return np.concatenate(out)


def _has(backend):
    try:
        A._make_backend(backend)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ plumbing (no backend needed)

@pytest.mark.parametrize("chunk", [1, 37, 160, 512, 1000])
def test_null_backend_is_exact_passthrough_with_fixed_latency(chunk):
    x = speechlike(1.0, 1)
    aec = A.EchoCanceller(backend="none")
    y = run(aec, x, np.zeros_like(x), chunk=chunk)
    assert y.size == x.size
    L = A.LATENCY
    q = np.round(x * 32767) / 32768          # int16 round trip inside the wrapper
    np.testing.assert_allclose(y[L:], q[:-L], atol=1.5 / 32768)
    assert np.all(y[:L] == 0)


def test_process_returns_same_length_every_call():
    aec = A.EchoCanceller(backend="none")
    for n in [5, 160, 511, 512, 3, 2000]:
        assert aec.process(np.zeros(n, np.float32)).size == n


def test_timeline_write_read_and_gaps():
    tl = A.RenderTimeline(span_s=2.0)
    a = np.arange(1, 161, dtype=np.float32)
    tl.write(a, t_start=10.0)
    np.testing.assert_array_equal(tl.read(10.0, 160), a)
    # half before the first write = zeros
    r = tl.read(10.0 - 80 / SR, 160)
    assert np.all(r[:80] == 0) and np.array_equal(r[80:], a[:80])
    # a later write leaves silence in the gap, never stale audio
    tl.write(a, t_start=10.5)
    assert np.all(tl.read(10.0 + 200 / SR, 160) == 0)
    np.testing.assert_array_equal(tl.read(10.5, 160), a)
    # beyond the newest sample: zeros
    assert np.all(tl.read(11.0, 160) == 0)


def test_timeline_ring_wrap_keeps_recent_audio():
    tl = A.RenderTimeline(span_s=0.1)       # 1600 samples
    for k in range(30):
        tl.write(np.full(160, k + 1, np.float32), t_start=1.0 + k * 0.01)
    assert np.all(tl.read(1.0 + 29 * 0.01, 160) == 30)
    assert np.all(tl.read(1.0, 160) == 0)    # fell off the ring


def test_resampler_24k_to_16k_streams_without_clicks():
    t = np.arange(24000) / 24000
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rs = A.StreamResampler(24000, 16000)
    rng = np.random.default_rng(0)
    parts, i = [], 0
    while i < x.size:
        n = int(rng.integers(1, 900))
        parts.append(rs.process(x[i:i + n]))
        i += n
    y = np.concatenate(parts)
    assert abs(y.size - 16000) <= 2
    tt = np.arange(y.size) / 16000
    # compare against the ideal 440 Hz tone after the filter's group delay
    best = min(range(0, 60), key=lambda d: float(np.mean((y[d + 200:15000] - 0.5 * np.sin(2 * np.pi * 440 * tt[200:15000 - d])) ** 2)))
    err = y[best + 200:15000] - 0.5 * np.sin(2 * np.pi * 440 * tt[200:15000 - best])
    assert 20 * np.log10(np.sqrt(np.mean(err ** 2)) / 0.354) < -30   # >30 dB clean, no block-edge clicks


def test_resampler_identity():
    rs = A.StreamResampler(16000, 16000)
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(rs.process(x), x)


# ------------------------------------------------------------------ real backends

@pytest.mark.skipif(not _has("webrtc"), reason="livekit wheel not installed")
def test_webrtc_cancels_echo_far_only():
    far = speechlike(8.0, 11, f0=210)
    mic = echo_path(far, 80, 0.5)
    aec = A.EchoCanceller(backend="webrtc")
    y = run(aec, mic, far)
    s = slice(2 * SR, 8 * SR)                # after convergence
    assert A.erle_db(mic[s], y[s]) > 15.0
    assert aec.stats()["erle_db"] > 10.0


@pytest.mark.skipif(not _has("webrtc"), reason="livekit wheel not installed")
def test_webrtc_keeps_near_end_when_no_render():
    near = speechlike(4.0, 5, f0=120)
    aec = A.EchoCanceller(backend="webrtc")
    y = run(aec, near, np.zeros_like(near))
    s = slice(SR, 4 * SR)
    lvl = lambda v: 10 * np.log10(np.mean(v.astype(np.float64) ** 2))  # noqa: E731
    assert abs(lvl(y[s]) - lvl(near[s])) < 2.0   # within 2 dB: near end passes


@pytest.mark.skipif(not _has("webrtc"), reason="livekit wheel not installed")
def test_webrtc_with_timeline_alignment():
    """The live wiring: render placed on a clock (as the sink plays it), capture stamped with its
    own clock; 150 ms acoustic delay beyond what the timeline knows."""
    far = speechlike(8.0, 21, f0=200)
    mic = echo_path(far, 150, 0.4)
    tl = A.RenderTimeline()
    t0 = 1000.0
    aec = A.EchoCanceller(backend="webrtc", timeline=tl)
    out, w = [], 0
    for i in range(0, mic.size, 512):
        while w < far.size and w <= i + 2048:   # the sink writes ~100 ms ahead of the mic, in real time
            tl.write(far[w:w + 1024], t0 + w / SR)
            w += 1024
        out.append(aec.process(mic[i:i + 512], t_capture=t0 + i / SR))
    y = np.concatenate(out)
    s = slice(2 * SR, 8 * SR)
    assert A.erle_db(mic[s], y[s]) > 12.0


@pytest.mark.skipif(not _has("speex"), reason="libspeexdsp not installed")
def test_speex_backend_runs_and_reduces_echo():
    far = speechlike(8.0, 31, f0=210)
    mic = echo_path(far, 20, 0.3)
    y = run(A.EchoCanceller(backend="speex", tail_ms=200), mic, far)
    s = slice(4 * SR, 8 * SR)
    assert A.erle_db(mic[s], y[s]) > 3.0


def test_auto_falls_back_to_something():
    aec = A.EchoCanceller(backend="auto")
    assert aec.name in ("webrtc", "speex", "none")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        A.EchoCanceller(backend="bogus")
