"""voice/iris_aec.py - acoustic echo cancellation so my ears can stay open while my mouth talks.

Full-duplex step (a), 2026-10-31. The mic (Zeke's headset on the tower, scripts/mic_source.py) and
the speaker (scripts/audio_sink.py, same tower) are the two ends an echo canceller needs. Without
AEC the only defences against hearing myself are the headphones assumption, the barge-in rms floor
and the voiceprint self-drop (wren_voice_core._self_voice_score) - fine on the headset, useless the
moment he switches the tower to speakers ("Zeke switches between headset and speakers", audio_sink).

Match-and-adapt, no home-grown adaptive filter:
  * backend "webrtc" = WebRTC AEC3 (+ optional NS/HPF) through the LiveKit Python SDK's
    rtc.AudioProcessingModule (pip `livekit`; prebuilt wheels for Linux AND win_amd64, Apache-2.0).
    AEC3 has its own render/capture delay estimator, so a stable-but-unknown device delay is fine.
  * backend "speex" = libspeexdsp's MDF echo canceller + residual-echo suppressor through ctypes
    (system libspeexdsp.so.1 / libspeexdsp.dll). No delay estimator: the filter tail must cover
    delay + room tail. Fallback / A-B only.
  * backend "none" = pass-through (the old behaviour), so callers never need two code paths.

Everything works on 16 kHz mono float32 in [-1, 1] (what the ears use) and accepts ANY chunk size
(the ears read 512-sample Silero chunks; AEC3 needs 10 ms = 160-sample frames) - EchoCanceller
buffers internally and returns exactly as many samples as it was given (output lags input by
< 10 ms of internal buffering, never more).

Alignment: RenderTimeline places far-end (what the speaker PLAYED) on a shared clock and hands the
AEC, for every capture frame, the render samples that were playing at that moment. Feeding render
"as it arrives" instead makes the effective delay = queue depth, which jumps every time the mouth
reconnects - AEC3 would re-converge on every sentence. Both tower processes stamp with
time.perf_counter() (QueryPerformanceCounter on Windows / CLOCK_MONOTONIC on Linux: system-wide).

Nothing here touches a device or the network; the sink/mic_source integration is in
voice/aec_bus.py, behind IRIS_AEC (default OFF).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
from collections import deque

import numpy as np

SR = 16000
FRAME = 160            # 10 ms at 16 kHz: the AEC3 frame
LATENCY = FRAME        # EchoCanceller.process output lags its input by exactly this many samples
CLOCK = time.perf_counter


def _f32_to_i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _i16_to_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


# ───────────────────────────────────────────────────────────────── backends (10 ms frames, int16)

class _NullBackend:
    name = "none"

    def render(self, frame_i16: np.ndarray) -> None:
        pass

    def capture(self, frame_i16: np.ndarray) -> np.ndarray:
        return frame_i16


class _WebRTCBackend:
    """WebRTC AEC3 via livekit.rtc.AudioProcessingModule (10 ms int16 frames, in-place)."""
    name = "webrtc"

    def __init__(self, noise_suppression: bool = False, high_pass: bool = True,
                 agc: bool = False, delay_hint_ms: "int | None" = None):
        from livekit import rtc   # lazy: only the processes that enable AEC need the wheel
        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True,
                                              noise_suppression=bool(noise_suppression),
                                              high_pass_filter=bool(high_pass),
                                              auto_gain_control=bool(agc))
        self._delay_hint = delay_hint_ms

    def render(self, frame_i16: np.ndarray) -> None:
        fr = self._rtc.AudioFrame(frame_i16.tobytes(), SR, 1, FRAME)
        self._apm.process_reverse_stream(fr)

    def capture(self, frame_i16: np.ndarray) -> np.ndarray:
        fr = self._rtc.AudioFrame(frame_i16.tobytes(), SR, 1, FRAME)
        if self._delay_hint is not None:
            # WebRTC wants the hint set before EVERY ProcessStream call
            self._apm.set_stream_delay_ms(int(self._delay_hint))
        self._apm.process_stream(fr)
        return np.frombuffer(bytes(fr.data), dtype=np.int16).copy()


class _SpeexBackend:
    """libspeexdsp MDF echo canceller + preprocessor residual-echo suppression (ctypes)."""
    name = "speex"
    _SET_SAMPLING_RATE = 24          # SPEEX_ECHO_SET_SAMPLING_RATE
    _PP_SET_ECHO_STATE = 24          # SPEEX_PREPROCESS_SET_ECHO_STATE
    _PP_SET_DENOISE = 0              # SPEEX_PREPROCESS_SET_DENOISE

    def __init__(self, tail_ms: int = 400, suppress: bool = True, denoise: bool = False):
        lib = None
        for cand in (os.environ.get("IRIS_SPEEXDSP_LIB"), "libspeexdsp.so.1",
                     ctypes.util.find_library("speexdsp"), "libspeexdsp.dll", "libspeexdsp-1.dll"):
            if not cand:
                continue
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise OSError("libspeexdsp not found (set IRIS_SPEEXDSP_LIB)")
        self._lib = lib
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_preprocess_state_init.restype = ctypes.c_void_p
        lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
        tail = max(FRAME, int(SR * tail_ms / 1000))
        self._st = lib.speex_echo_state_init(FRAME, tail)
        rate = ctypes.c_int(SR)
        lib.speex_echo_ctl(self._st, self._SET_SAMPLING_RATE, ctypes.byref(rate))
        self._pp = None
        if suppress:
            self._pp = lib.speex_preprocess_state_init(FRAME, SR)
            lib.speex_preprocess_ctl(self._pp, self._PP_SET_ECHO_STATE, ctypes.c_void_p(self._st))
            dn = ctypes.c_int(1 if denoise else 0)
            lib.speex_preprocess_ctl(self._pp, self._PP_SET_DENOISE, ctypes.byref(dn))
        self._play = deque()

    def render(self, frame_i16: np.ndarray) -> None:
        # speex_echo_cancellation takes (rec, play) together; keep render frames in step
        self._play.append(np.ascontiguousarray(frame_i16, dtype=np.int16))
        while len(self._play) > 50:          # never let a stalled capture side grow this
            self._play.popleft()

    def capture(self, frame_i16: np.ndarray) -> np.ndarray:
        play = self._play.popleft() if self._play else np.zeros(FRAME, np.int16)
        rec = np.ascontiguousarray(frame_i16, dtype=np.int16)
        out = np.zeros(FRAME, np.int16)
        self._lib.speex_echo_cancellation(self._st, rec.ctypes.data, play.ctypes.data, out.ctypes.data)
        if self._pp is not None:
            self._lib.speex_preprocess_run(self._pp, out.ctypes.data)
        return out

    def __del__(self):
        try:
            if self._pp:
                self._lib.speex_preprocess_state_destroy(self._pp)
            self._lib.speex_echo_state_destroy(self._st)
        except Exception:
            pass


def available_backends() -> list:
    out = ["none"]
    try:
        _WebRTCBackend()
        out.append("webrtc")
    except Exception:
        pass
    try:
        _SpeexBackend()
        out.append("speex")
    except Exception:
        pass
    return out


def _make_backend(name: str, **kw):
    name = (name or "auto").lower()
    if name == "auto":
        for cand in ("webrtc", "speex"):
            try:
                return _make_backend(cand, **kw)
            except Exception:
                continue
        return _NullBackend()
    if name == "webrtc":
        return _WebRTCBackend(**{k: v for k, v in kw.items()
                                 if k in ("noise_suppression", "high_pass", "agc", "delay_hint_ms")})
    if name == "speex":
        return _SpeexBackend(**{k: v for k, v in kw.items() if k in ("tail_ms", "suppress", "denoise")})
    if name in ("none", "off", "null"):
        return _NullBackend()
    raise ValueError(f"unknown AEC backend {name!r}")


# ───────────────────────────────────────────────────────────────── streaming resampler

class StreamResampler:
    """Stateful rational resampler: a windowed-sinc polyphase FIR evaluated only on the taps that
    meet real input samples (taps_per_phase multiply-adds per output, not up*taps), with carried
    history, so a 44.1/24 kHz mouth stream chopped into arbitrary blocks resamples to 16 kHz without
    the per-block edge clicks a stateless resample_poly would add. Group delay ~taps_per_phase/2 input
    samples (< 0.5 ms at 24+ kHz)."""

    def __init__(self, sr_in: int, sr_out: int, taps_per_phase: int = 24):
        from math import gcd
        g = gcd(int(sr_in), int(sr_out))
        self.up, self.down = int(sr_out) // g, int(sr_in) // g
        self.identity = self.up == 1 and self.down == 1
        if self.identity:
            return
        self._T = int(taps_per_phase)
        n = self._T * self.up
        cutoff = 0.95 / max(self.up, self.down)
        # windowed-sinc low-pass (== scipy.signal.firwin(n, cutoff), Hamming) in plain numpy: the tower's
        # venv needs only numpy + livekit for AEC
        m = np.arange(n) - (n - 1) / 2.0
        h = cutoff * np.sinc(cutoff * m) * np.hamming(n)
        self._h = (h / h.sum() * self.up).astype(np.float64)
        self._hist = np.zeros(self._T, np.float64)   # the last T input samples
        self._in = 0                                  # absolute index of the next input sample
        self._P = 0                                   # absolute upsampled index of the next output

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.identity:
            return x.copy()
        if x.size == 0:
            return np.zeros(0, np.float32)
        buf = np.concatenate([self._hist, x.astype(np.float64)])
        base = self._in - self._T                     # absolute index of buf[0]
        self._in += x.size
        last = self._in * self.up                     # outputs need input floor(P/up) < self._in
        P = np.arange(self._P, last, self.down, dtype=np.int64)
        if P.size:
            j = (P // self.up)[:, None] - np.arange(self._T)[None, :]       # contributing inputs
            k = P[:, None] - j * self.up                                     # their taps, in [0, n)
            rel = j - base
            vals = np.where(rel >= 0, buf[np.clip(rel, 0, buf.size - 1)], 0.0)
            y = (vals * self._h[k]).sum(axis=1)
            self._P = int(P[-1] + self.down)
        else:
            y = np.zeros(0)
        self._hist = buf[-self._T:]
        return y.astype(np.float32)


# ───────────────────────────────────────────────────────────────── render timeline

class RenderTimeline:
    """Far-end samples placed on an absolute clock (16 kHz sample index = round(t * SR)).

    write(samples, t_start): the speaker starts PLAYING samples[0] at clock time t_start.
    read(t_start, n): the n render samples that were playing from t_start (zeros where nothing
    played). Thread-safe; ring of `span_s` seconds, older audio falls off."""

    def __init__(self, span_s: float = 4.0):
        self._n = int(span_s * SR)
        self._ring = np.zeros(self._n, np.float32)
        self._hi = None            # absolute index one past the newest written sample
        self._lock = threading.Lock()
        self.writes = 0

    def write(self, samples: np.ndarray, t_start: float) -> None:
        s = np.asarray(samples, dtype=np.float32).reshape(-1)
        if s.size == 0:
            return
        if s.size > self._n:
            t_start += (s.size - self._n) / SR
            s = s[-self._n:]
        a = int(round(t_start * SR))
        with self._lock:
            if self._hi is not None and a > self._hi:
                gap = min(a - self._hi, self._n)          # silence between writes
                self._put(self._hi + max(0, (a - self._hi) - gap), np.zeros(gap, np.float32))
            self._put(a, s)
            self._hi = a + s.size if self._hi is None else max(self._hi, a + s.size)
            self.writes += 1

    def _put(self, a: int, s: np.ndarray) -> None:
        i = a % self._n
        k = min(s.size, self._n - i)
        self._ring[i:i + k] = s[:k]
        if k < s.size:
            self._ring[:s.size - k] = s[k:]

    def read(self, t_start: float, n: int) -> np.ndarray:
        a = int(round(t_start * SR))
        out = np.zeros(n, np.float32)
        with self._lock:
            if self._hi is None:
                return out
            lo = self._hi - self._n
            for j in range(n):           # n is 160: a plain loop is fine and handles both edges
                p = a + j
                if lo <= p < self._hi:
                    out[j] = self._ring[p % self._n]
        return out


# ───────────────────────────────────────────────────────────────── the canceller

class EchoCanceller:
    """Chunk-size-agnostic AEC on 16 kHz float32.

    Two ways to feed the far end:
      * push_render(x): render samples in capture order (offline benches, or a caller that
        already aligned them) - consumed one 10 ms frame per capture frame;
      * timeline=RenderTimeline + process(x, t_capture=...): the canceller reads, for each
        capture frame, what the speaker was playing at that time minus `lead_ms` (AEC3 needs
        render to LEAD the echo; it estimates the rest of the delay itself).

    Stats (for health + the bench): frames, render_active_frames, erle_db (running, over frames
    where render was active and the input was loud enough to measure)."""

    def __init__(self, backend: str = "auto", timeline: "RenderTimeline | None" = None,
                 lead_ms: float = 0.0, **kw):
        self.backend = _make_backend(backend, **kw)
        self.name = self.backend.name
        self.timeline = timeline
        self.lead_s = float(lead_ms) / 1000.0
        self._cap_buf = np.zeros(0, np.float32)
        # ONE frame of priming latency (10 ms): output = input delayed by exactly FRAME samples,
        # so the frame buffer can never underrun whatever chunk size the caller uses
        self._out_buf = deque([np.zeros(FRAME, np.float32)])
        self._out_len = FRAME
        self._ren_buf = np.zeros(0, np.float32)
        self._t_next = None          # clock time of the next unprocessed capture sample
        self.frames = 0
        self.render_active_frames = 0
        self._e_in = 0.0
        self._e_out = 0.0
        self._lock = threading.Lock()
        self.last_ts = 0.0

    def push_render(self, x: np.ndarray) -> None:
        with self._lock:
            self._ren_buf = np.concatenate([self._ren_buf, np.asarray(x, np.float32).reshape(-1)])
            cap = SR * 4
            if self._ren_buf.size > cap:
                self._ren_buf = self._ren_buf[-cap:]

    def _next_render(self) -> np.ndarray:
        if self.timeline is not None and self._t_next is not None:
            return self.timeline.read(self._t_next - self.lead_s, FRAME)
        if self._ren_buf.size >= FRAME:
            f, self._ren_buf = self._ren_buf[:FRAME], self._ren_buf[FRAME:]
            return f
        return np.zeros(FRAME, np.float32)

    def process(self, x: np.ndarray, t_capture: "float | None" = None) -> np.ndarray:
        """x: capture chunk (any length). t_capture: clock time its FIRST sample was captured
        (needed only with a timeline; defaults to now - duration). Returns len(x) samples."""
        x = np.asarray(x, np.float32).reshape(-1)
        with self._lock:
            if self.timeline is not None:
                if t_capture is None:
                    t_capture = CLOCK() - x.size / SR
                if self._t_next is None:
                    self._t_next = t_capture
                # re-anchor if the caller's clock and ours drift apart by > 50 ms (a dropped
                # block, a reconnect): never let a stale anchor misalign the render forever
                pending = self._cap_buf.size / SR
                if abs((self._t_next + pending) - t_capture) > 0.05:
                    self._t_next = t_capture - pending
            self._cap_buf = np.concatenate([self._cap_buf, x])
            while self._cap_buf.size >= FRAME:
                cap, self._cap_buf = self._cap_buf[:FRAME], self._cap_buf[FRAME:]
                ren = self._next_render()
                active = float(np.abs(ren).max()) > 1e-4
                self.backend.render(_f32_to_i16(ren))
                out = _i16_to_f32(self.backend.capture(_f32_to_i16(cap)))
                if active:
                    self.render_active_frames += 1
                    ein = float((cap * cap).sum())
                    if ein > FRAME * 1e-7:
                        self._e_in += ein
                        self._e_out += float((out * out).sum())
                self.frames += 1
                if self._t_next is not None:
                    self._t_next += FRAME / SR
                self._out_buf.append(out)
                self._out_len += FRAME
            # hand back exactly len(x) samples (constant LATENCY = FRAME samples, see __init__)
            need = x.size
            parts = []
            while need > 0 and self._out_buf:
                f = self._out_buf[0]
                if f.size <= need:
                    parts.append(self._out_buf.popleft())
                    need -= f.size
                else:
                    parts.append(f[:need])
                    self._out_buf[0] = f[need:]
                    need = 0
            self._out_len -= (x.size - need)
            if need > 0:          # unreachable with the priming frame; never return short
                parts.append(np.zeros(need, np.float32))
            self.last_ts = time.time()
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    @property
    def erle_db(self) -> "float | None":
        if self._e_in <= 0 or self._e_out <= 0:
            return None
        return 10.0 * float(np.log10(self._e_in / self._e_out))

    def stats(self) -> dict:
        e = self.erle_db
        return {"backend": self.name, "frames": self.frames,
                "render_active_frames": self.render_active_frames,
                "erle_db": None if e is None else round(e, 1),
                "timeline": self.timeline is not None, "last_ts": self.last_ts}


def erle_db(near_with_echo: np.ndarray, out: np.ndarray, mask: "np.ndarray | None" = None) -> float:
    """Echo Return Loss Enhancement over `mask` samples (far-end-only regions): 10 log10(Pin/Pout)."""
    a = np.asarray(near_with_echo, np.float64)
    b = np.asarray(out, np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    return 10.0 * float(np.log10((a * a).sum() / max((b * b).sum(), 1e-20)))
