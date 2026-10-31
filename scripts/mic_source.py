"""scripts/mic_source.py - stream Zeke's headset mic from the TOWER to my ears on the server.

Zeke 2026-10-07: mouth and ears live on the server; he talks to me through the headset that is
plugged into the tower. The ears (voice/wren_voice_core.py with IRIS_MIC_SOURCE=<tower>:8776,
see voice/net_audio.py) connect here, send a JSON header line, and receive raw float32 mono PCM
from the NAMED mic until they hang up. Several sessions may run at once (capture + barge-in
watch); each opens its own input stream.

Never the default input device: on 2026-07-13 the default was a loopback cable and my ears heard
my own mouth for 90 minutes. No named mic = refuse the session (deaf but honest).

Security: only localhost + the server VM (`iris_home_host` in config/private.local.json) may
connect - this is a live microphone in his room. IRIS_MIC_SOURCE_ALLOW overrides.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _private import priv  # noqa: E402  (git-ignored private config)

PORT = int(os.environ.get("IRIS_MIC_SOURCE_PORT", "8776"))
BIND = os.environ.get("IRIS_MIC_SOURCE_BIND", "0.0.0.0")
_DEFAULT_ALLOW = ",".join(a for a in ("127.0.0.1", priv("iris_home_host")) if a)
ALLOW = {a.strip() for a in os.environ.get("IRIS_MIC_SOURCE_ALLOW", _DEFAULT_ALLOW).split(",") if a.strip()}
MIC_SUBSTR = os.environ.get("WREN_MIC_SUBSTR", "Logitech PRO X")
_LOG_FILE = Path(__file__).resolve().parents[1] / "state" / "mic_source.log"

# Full-duplex echo cancellation (2026-10-31, default OFF). IRIS_AEC=1: 16 kHz sessions are served from
# ONE shared capture stream through ONE WebRTC AEC3 canceller fed by audio_sink's reference
# (IRIS_AEC_REF=1 on the sink) - so my own voice is removed from his mic before it leaves the tower and
# the ears can stay open while I talk. A session can opt out with {"aec": false} in its header. Any AEC
# failure falls back to the plain per-session stream (never deaf because of AEC).
AEC_ON = os.environ.get("IRIS_AEC", "0").strip().lower() not in ("0", "off", "false", "no", "")
_HUB = None
_HUB_LOCK = threading.Lock()


def _aec_hub():
    """The shared AEC capture hub, created on first use (None = AEC off or unavailable)."""
    global _HUB
    if not AEC_ON:
        return None
    with _HUB_LOCK:
        if _HUB is None:
            try:
                sys.path.append(str(Path(__file__).resolve().parents[1] / "voice"))
                import aec_bus

                def _open(sr, block):
                    idx = find_named_mic(MIC_SUBSTR)
                    if idx is None:
                        raise RuntimeError(f"named mic '{MIC_SUBSTR}' not found")
                    return sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=idx, blocksize=block)

                _HUB = aec_bus.MicHub(_open, backend=os.environ.get("IRIS_AEC_BACKEND", "webrtc"),
                                      lead_ms=float(os.environ.get("IRIS_AEC_LEAD_MS", "40")), log=log)
            except Exception as e:  # noqa: BLE001
                log(f"AEC unavailable, plain mic: {e!r}")
                return None
        return _HUB


def log(msg: str) -> None:
    line = f"[mic_source {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
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


def find_named_mic(substr: str):
    """EXACTLY voice/wren_listen.find_input_device: the FIRST input whose name contains substr
    (on Windows that is the MME entry, which resamples - the WASAPI entry rejects 16 kHz:
    'Invalid sample rate', seen 2026-10-07). Never the default device."""
    try:
        sd._terminate(); sd._initialize()   # pick up a headset plugged in after we started
    except Exception:
        pass
    low = substr.lower()
    for i, d in enumerate(sd.query_devices()):
        if low in d.get("name", "").lower() and d.get("max_input_channels", 0) > 0:
            return i
    return None


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
    try:
        hdr = _read_header(conn)
        sr, bs = int(hdr.get("sr", 16000)), int(hdr.get("blocksize", 512))
        idx = find_named_mic(MIC_SUBSTR)
        if idx is None:
            log(f"REFUSED session from {peer}: named mic '{MIC_SUBSTR}' not found (no default-device fallback)")
            return
        hub = _aec_hub() if (sr == 16000 and hdr.get("aec", True)) else None
        if hub is not None:
            import aec_bus
            log(f"streaming ECHO-CANCELLED mic idx={idx} to {peer} ({sr} Hz, block {bs}, {hub.aec.name})")
            q = hub.subscribe()
            try:
                for blk in aec_bus.session_reader(q, bs):
                    conn.sendall(blk.astype("<f4").tobytes())
            except ConnectionError as e:
                if hub.error is None:
                    raise                                   # the ears hung up: normal end
                log(f"AEC hub failed ({hub.error}); session {peer} falls back to the plain mic")
            finally:
                hub.unsubscribe(q)
        log(f"streaming mic idx={idx} '{sd.query_devices(idx)['name']}' to {peer} ({sr} Hz, block {bs})")
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=idx, blocksize=bs) as s:
            while True:
                data, _ov = s.read(bs)
                conn.sendall(data.tobytes())
    except (ConnectionError, OSError) as e:
        log(f"session {peer} ended ({e.__class__.__name__})")
    except Exception as e:
        log(f"session {peer} error: {e!r}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # EXCLUSIVE bind on Windows (2026-10-07): SO_REUSEADDR there lets a SECOND copy bind the same
    # port and silently share it (seen: two sinks on 8775 after an app relaunch). Exclusive = the
    # second copy fails to bind and exits; the first keeps serving.
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((BIND, PORT))
    srv.listen(8)
    log(f"listening on {BIND}:{PORT}, allow={sorted(ALLOW)}, mic='{MIC_SUBSTR}'")
    while True:
        conn, (ip, port) = srv.accept()
        if ip not in ALLOW:
            log(f"REFUSED {ip}:{port} (not in allowlist)")
            conn.close()
            continue
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=_serve, args=(conn, f"{ip}:{port}"), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
