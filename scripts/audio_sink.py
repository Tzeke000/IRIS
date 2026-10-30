"""scripts/audio_sink.py - play my voice on the TOWER's speakers when I live on the server.

Zeke 2026-10-07: "move your mouth and ears onto the server ... where it goes to the tower so I
can still hear you." The mouth (voice/wren_styletts_server.py) runs on the server's V100 and,
with IRIS_AUDIO_SINK=<tower>:8775, streams its output blocks here instead of to a local sound
device. This plays them on the tower's current default output.

Protocol (one TCP connection = one mouth session):
    first line : JSON header  {"sr": 24000, "ch": 1, "fmt": "f32le", "from": "iris-mouth"}
    then       : raw little-endian float32 PCM until the socket closes
A new connection preempts the old one once its header arrives (a bare probe never does). A closed socket stops playback at once (barge-in).

Security: only addresses in IRIS_AUDIO_SINK_ALLOW (default: localhost + the server VM) may
connect (default: localhost + `iris_home_host` from config/private.local.json) - otherwise
anyone on the wifi could make the tower talk in my voice.

usage:  .venv python scripts/audio_sink.py      (env IRIS_AUDIO_SINK_PORT, IRIS_AUDIO_SINK_ALLOW)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time

import numpy as np
import sounddevice as sd

PORT = int(os.environ.get("IRIS_AUDIO_SINK_PORT", "8775"))
BIND = os.environ.get("IRIS_AUDIO_SINK_BIND", "0.0.0.0")
from _private import priv  # config/private.local.json (git-ignored): network addresses never in tracked files
_DEFAULT_ALLOW = ",".join(a for a in ("127.0.0.1", priv("iris_home_host")) if a)
ALLOW = {a.strip() for a in os.environ.get("IRIS_AUDIO_SINK_ALLOW", _DEFAULT_ALLOW).split(",") if a.strip()}
BLOCK_BYTES = 4096  # 1024 float32 samples ~ 43 ms at 24 kHz

_current: "socket.socket | None" = None
_lock = threading.Lock()
# 2026-10-30 crash fix: held by a session for the WHOLE life of its PortAudio stream. PortAudio is not
# safe to terminate/re-initialise while another thread still has a stream open - that is exactly what a
# preempting session did (sd._terminate() under the old session's live stream.write) and pythonw died in
# ntdll (0xc0000005 / 0xc0000374 heap corruption) twice on 10-30, leaving my voice playing nowhere.
_stream_lock = threading.Lock()
HEADER_TIMEOUT_S = 5.0


_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state", "audio_sink.log")


def log(msg: str) -> None:
    line = f"[audio_sink {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    # pythonw (the scheduled task: no console window to flash over his games) has no stdout
    if sys.stdout is not None:
        try:
            print(line, flush=True)
        except Exception:
            pass
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _read_header(conn: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        ch = conn.recv(1)
        if not ch:
            raise ConnectionError("closed before header")
        buf += ch
        if len(buf) > 4096:
            raise ValueError("header too long")
    return json.loads(buf.decode("utf-8"))


def _serve(conn: socket.socket, peer: str) -> None:
    global _current
    stream = None
    have_stream_lock = False
    try:
        # A connection only preempts once it has sent a VALID header: a bare TCP probe (health checks
        # connect every minute) used to close the playing session - cutting me off mid-sentence.
        conn.settimeout(HEADER_TIMEOUT_S)
        hdr = _read_header(conn)
        conn.settimeout(None)
        sr, ch = int(hdr.get("sr", 24000)), int(hdr.get("ch", 1))
        if hdr.get("fmt", "f32le") != "f32le":
            raise ValueError(f"unsupported fmt {hdr.get('fmt')!r}")
        with _lock:
            old, _current = _current, conn
        if old is not None:
            log("new session preempts the old one")
            try:
                old.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        # wait for the old session to close ITS stream before touching PortAudio globally
        if not _stream_lock.acquire(timeout=3.0):
            raise TimeoutError("previous session still holds the audio stream")
        have_stream_lock = True
        # follow the OS default output (Zeke switches between headset and speakers) - safe now: no
        # other stream exists while we hold _stream_lock
        try:
            sd._terminate(); sd._initialize()
        except Exception:
            pass
        stream = sd.OutputStream(samplerate=sr, channels=ch, dtype="float32")
        stream.start()
        log(f"playing from {peer} ({hdr.get('from', '?')}, {sr} Hz) on '{sd.query_devices(kind='output').get('name')}'")
        pending = b""
        while True:
            if _current is not conn:      # preempted: stop at once, don't drain what's buffered
                break
            data = conn.recv(BLOCK_BYTES)
            if not data:
                break
            pending += data
            n = (len(pending) // (4 * ch)) * (4 * ch)
            if n:
                stream.write(np.frombuffer(pending[:n], dtype="<f4").reshape(-1, ch))
                pending = pending[n:]
        log(f"session from {peer} ended")
    except Exception as e:
        log(f"session from {peer} error: {e!r}")
    finally:
        if stream is not None:
            try:
                stream.abort(); stream.close()
            except Exception:
                pass
        if have_stream_lock:
            _stream_lock.release()
        try:
            conn.close()
        except Exception:
            pass
        with _lock:
            if _current is conn:
                _current = None


def main() -> int:
    global _current
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # EXCLUSIVE bind on Windows (2026-10-07): SO_REUSEADDR there lets a SECOND copy bind the same
    # port and silently share it (seen: two sinks on 8775 after an app relaunch). Exclusive = the
    # second copy fails to bind and exits; the first keeps serving.
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # small receive buffer on the LISTENING socket so accepted connections inherit it before the
    # window is negotiated - keeps the mouth <~0.5 s ahead of what is actually playing
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
    srv.bind((BIND, PORT))
    srv.listen(4)
    log(f"listening on {BIND}:{PORT}, allow={sorted(ALLOW)}")
    while True:
        conn, (ip, port) = srv.accept()
        if ip not in ALLOW:
            log(f"REFUSED {ip}:{port} (not in IRIS_AUDIO_SINK_ALLOW)")
            conn.close()
            continue
        threading.Thread(target=_serve, args=(conn, f"{ip}:{port}"), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
