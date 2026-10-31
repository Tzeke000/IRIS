"""voice/aec_bus.py - the wiring that lets the tower's mic_source cancel my voice out of his mic.

Topology (both ends on the TOWER, where the speaker and the mic share one clock):

  audio_sink.py --(UDP 127.0.0.1:8777, "what I just queued to the speaker, plays at t")--> RefSubscriber
                                                                                            |
  mic_source.py: MicHub = ONE InputStream -> EchoCanceller(timeline) -> fan-out to every ears session

Why one hub instead of an AEC per session: the ears open a fresh mic session for every listen and
every barge-in watch. AEC3 needs ~1-2 s of my speech to converge; a per-session canceller would
start deaf to echo at exactly the moment it matters (the start of each spoken turn). The hub keeps
one converged canceller alive across sessions and lingers LINGER_S after the last one closes.

Measured offline (scripts/duplex/aec_bench.py, 10-31): AEC3 removes echo when the reference LEADS the
echo by 0-450 ms and fails at >= 600 ms or when the reference is late by even 20 ms. So the sink stamps
each block with when it will PLAY (write-return time + PortAudio output latency - block duration),
the mic stamps each block with when it was CAPTURED (read-return time - input latency - block
duration), and the canceller reads the reference LEAD_MS early so small stamp errors stay on the safe
(causal) side.

Flags (all default OFF / harmless):
  IRIS_AEC=1                 mic_source runs the hub for 16 kHz sessions (header may opt out: "aec": false)
  IRIS_AEC_REF=1             audio_sink publishes the reference (cheap; nobody listening = dropped datagrams)
  IRIS_AEC_REF_PORT=8777     localhost only, both ends
  IRIS_AEC_BACKEND=webrtc    webrtc | speex | none
  IRIS_AEC_LEAD_MS=40
"""
from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

import numpy as np

try:
    from . import iris_aec  # type: ignore
except Exception:  # noqa: BLE001  (run as a plain module next to the tower scripts)
    import iris_aec  # type: ignore

REF_PORT = int(os.environ.get("IRIS_AEC_REF_PORT", "8777"))
_MAGIC = b"IRF1"
_HDR = struct.Struct("<4sdII")        # magic, t_play_start (perf_counter s), sample rate, n samples
MAX_SAMPLES = 4096                    # per datagram (16 KB payload, well under the UDP limit)
CLOCK = iris_aec.CLOCK


def flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "off", "false", "no", "")


# ───────────────────────────────────────────────────────────────── sink side

class RefPublisher:
    """Fire-and-forget reference datagrams to localhost. Never blocks, never raises into playback."""

    def __init__(self, port: int = REF_PORT):
        self._addr = ("127.0.0.1", int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self.sent = 0
        self.errors = 0

    def publish(self, samples: np.ndarray, sr: int, t_play_start: float) -> None:
        x = np.ascontiguousarray(np.asarray(samples, np.float32).reshape(-1))
        for i in range(0, x.size, MAX_SAMPLES):
            part = x[i:i + MAX_SAMPLES]
            try:
                self._sock.sendto(_HDR.pack(_MAGIC, float(t_play_start + i / sr), int(sr), part.size)
                                  + part.astype("<f4").tobytes(), self._addr)
                self.sent += 1
            except (BlockingIOError, OSError):
                self.errors += 1

    @staticmethod
    def play_start_after_write(n_samples: int, sr: int, output_latency_s: float) -> float:
        """A blocking PortAudio write returns once the block is IN the buffer; its last sample then
        plays ~output_latency later, so its first sample plays at now + latency - duration."""
        return CLOCK() + float(output_latency_s) - n_samples / float(sr)


# ───────────────────────────────────────────────────────────────── mic side

class RefSubscriber:
    """Receives reference datagrams, resamples to 16 kHz, places them on a RenderTimeline."""

    def __init__(self, timeline: "iris_aec.RenderTimeline", port: int = REF_PORT):
        self.timeline = timeline
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", int(port)))   # localhost ONLY: nothing remote can inject audio
        self._sock.settimeout(0.5)
        self._rs: "dict[int, iris_aec.StreamResampler]" = {}
        self._next_t: "dict[int, float]" = {}
        self.received = 0
        self.bad = 0
        self.last_rx = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="aec-ref-rx", daemon=True)

    @property
    def port(self) -> int:
        return self._sock.getsockname()[1]

    def start(self) -> "RefSubscriber":
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    def handle(self, data: bytes) -> bool:
        if len(data) < _HDR.size:
            self.bad += 1
            return False
        magic, t0, sr, n = _HDR.unpack_from(data)
        if magic != _MAGIC or len(data) != _HDR.size + 4 * n or not (8000 <= sr <= 96000) or n > MAX_SAMPLES:
            self.bad += 1
            return False
        x = np.frombuffer(data, dtype="<f4", count=n, offset=_HDR.size)
        rs = self._rs.get(sr)
        # a gap (> 250 ms) or a jump back in time = a new mouth session: fresh resampler state
        nt = self._next_t.get(sr)
        if rs is None or nt is None or abs(t0 - nt) > 0.25:
            rs = self._rs[sr] = iris_aec.StreamResampler(sr, iris_aec.SR)
        self._next_t[sr] = t0 + n / sr
        y = rs.process(x)
        if y.size:
            self.timeline.write(y, t0)
        self.received += 1
        self.last_rx = time.time()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, peer = self._sock.recvfrom(_HDR.size + 4 * MAX_SAMPLES + 16)
            except socket.timeout:
                continue
            except OSError:
                break
            if peer[0] != "127.0.0.1":
                self.bad += 1
                continue
            try:
                self.handle(data)
            except Exception:  # noqa: BLE001
                self.bad += 1


class MicHub:
    """One capture stream -> one EchoCanceller -> every subscribed session.

    open_stream(sr, blocksize) must return a context manager with .read(frames) -> (ndarray, overflow)
    and (optionally) a .latency attribute in seconds - sounddevice.InputStream fits, and so do fakes."""

    def __init__(self, open_stream, *, backend: str = "webrtc", lead_ms: float = 40.0,
                 block: int = 320, linger_s: float = 120.0, ref_port: "int | None" = REF_PORT,
                 log=print):
        self.timeline = iris_aec.RenderTimeline()
        self.aec = iris_aec.EchoCanceller(backend=backend, timeline=self.timeline, lead_ms=lead_ms)
        self.ref = RefSubscriber(self.timeline, port=ref_port).start() if ref_port is not None else None
        self._open = open_stream
        self._block = int(block)
        self._linger = float(linger_s)
        self._subs: "list[queue.Queue]" = []
        self._lock = threading.Lock()
        self._th: "threading.Thread | None" = None
        self._last_sub = time.time()
        self._log = log
        self.error: "str | None" = None
        self.blocks = 0
        self._closed = False

    def close(self) -> None:
        """Stop capture at the next block and release the reference port."""
        self._closed = True
        if self.ref is not None:
            self.ref.stop()

    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=400)          # ~8 s at 20 ms blocks
        with self._lock:
            self._subs.append(q)
            self._last_sub = time.time()
            if self._th is None or not self._th.is_alive():
                self.error = None
                self._th = threading.Thread(target=self._run, name="aec-mic-hub", daemon=True)
                self._th.start()
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)
            self._last_sub = time.time()

    def _run(self) -> None:
        try:
            with self._open(iris_aec.SR, self._block) as s:
                in_lat = float(getattr(s, "latency", 0.0) or 0.0)
                self._log(f"aec hub: capture open ({self.aec.name}, block {self._block}, in-latency {in_lat * 1000:.0f} ms)")
                last_report = time.time()
                while True:
                    with self._lock:
                        idle = not self._subs and time.time() - self._last_sub > self._linger
                        if idle or self._closed:
                            self._th = None
                            break
                    data, _ov = s.read(self._block)
                    x = np.asarray(data, np.float32).reshape(-1)
                    t_cap = CLOCK() - in_lat - x.size / iris_aec.SR
                    y = self.aec.process(x, t_capture=t_cap)
                    self.blocks += 1
                    with self._lock:
                        subs = list(self._subs)
                    for q in subs:
                        try:
                            q.put_nowait(y)
                        except queue.Full:                 # a stuck session never stalls the others
                            try:
                                q.get_nowait()
                                q.put_nowait(y)
                            except Exception:
                                pass
                    if time.time() - last_report > 60:
                        last_report = time.time()
                        self._log("aec hub: " + str(self.stats()))
            self._log("aec hub: capture closed (idle)")
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self._log(f"aec hub: capture error {e!r}")
            with self._lock:
                self._th = None
            for q in list(self._subs):
                try:
                    q.put_nowait(None)                     # wake sessions so they can fail over
                except Exception:
                    pass

    def stats(self) -> dict:
        st = self.aec.stats()
        st.update({"sessions": len(self._subs), "blocks": self.blocks, "error": self.error,
                   "ref_rx": self.ref.received if self.ref else None,
                   "ref_bad": self.ref.bad if self.ref else None,
                   "ref_age_s": (round(time.time() - self.ref.last_rx, 1)
                                 if self.ref and self.ref.last_rx else None)})
        return st


def session_reader(q: "queue.Queue", blocksize: int, timeout_s: float = 10.0):
    """Generator of exactly-blocksize float32 blocks from a hub subscription (re-chunks)."""
    buf = np.zeros(0, np.float32)
    while True:
        while buf.size < blocksize:
            item = q.get(timeout=timeout_s)
            if item is None:
                raise ConnectionError("aec hub capture failed")
            buf = np.concatenate([buf, item])
        out, buf = buf[:blocksize], buf[blocksize:]
        yield out
