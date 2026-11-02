"""iris_stream_asr.py - ears-side client for the streaming-ASR sidecar (Nemotron, 2026-11-02).

The sidecar (voice/iris_stream_asr_server.py, its own NeMo venv, unit iris-stream-asr) listens on a
localhost-only UNIX socket. The ears daemon opens ONE StreamSession per utterance, feeds it the same
mic frames it captures, and gets back partials (live words) and, after flush(), a final transcript.

NEVER BLOCKS CAPTURE: connect + send happen on the session's own sender thread behind a bounded
queue; feed() is put_nowait and a full queue just marks the session broken. Any error / timeout ->
the session goes quiet and the caller falls back to whisper. Pure stdlib + numpy (ears venv).

Wire format (both ways): 1 type byte + uint32 big-endian length + payload.
  client -> server  b"O" open  {json}     b"A" audio float32 LE 16 kHz mono
                    b"F" flush {}         b"C" cancel          b"P" ping
  server -> client  b"J" {json}: {"type": "partial"|"final"|"error"|"pong", ...}
"""
from __future__ import annotations

import json
import os
import queue
import re
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HDR = struct.Struct(">cI")
MAX_MSG = 8 << 20
SAMPLE_RATE = 16000


def default_sock_path() -> str:
    """$IRIS_STREAM_ASR_SOCK, else $XDG_RUNTIME_DIR/iris-stream-asr.sock (per-user, 0700 dir), else
    ~/.cache/iris/iris-stream-asr.sock. Server and client both call this, so they always agree."""
    p = os.environ.get("IRIS_STREAM_ASR_SOCK")
    if p:
        return p
    rd = os.environ.get("XDG_RUNTIME_DIR") or ""
    if not rd and os.path.isdir(f"/run/user/{os.getuid()}"):
        rd = f"/run/user/{os.getuid()}"
    if rd:
        return os.path.join(rd, "iris-stream-asr.sock")
    return str(Path.home() / ".cache" / "iris" / "iris-stream-asr.sock")


def send_msg(sock, kind: bytes, payload: bytes = b"") -> None:
    sock.sendall(HDR.pack(kind, len(payload)) + payload)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b:
            raise ConnectionError("closed")
        buf += b
    return bytes(buf)


def recv_msg(sock):
    kind, n = HDR.unpack(_recv_exact(sock, HDR.size))
    if n > MAX_MSG:
        raise ValueError(f"message too large ({n})")
    return kind, (_recv_exact(sock, n) if n else b"")


# ── "Iris" post-fix (11-02) ─────────────────────────────────────────────────────────────────────
# Bench: "Hey hey Iris, is it pushing..." -> "E I risk if uh is it pushing". NeMo phrase boosting
# (greedy_batch boosting_tree, key phrase "Iris") was tried at alpha 0.5 / 1 / 2 on the 31 clips: it
# never fixed that clip and changed other words ("They start" -> "I start"), so it ships OFF and this
# tiny map handles only the obvious cases, at the START of an utterance: "I risk"/"Irish" after one to
# three address words ("hey", "hi", "hello", "yo", incl. "hey" misheard as "E"/"A"), or on its own as a
# vocative ("I risk." / "Irish, ..."). "I risk losing it", "so I risk it all" and anything mid-sentence
# are left exactly as heard.
_GREET = r"(?:hey|hi|hello|yo|e|a|ay|eh)"          # address words (+ how "hey" gets misheard)
_IRIS_AFTER_GREETING = re.compile(
    rf"^(\W*(?:{_GREET}[\s,.!]+){{1,3}})(?:i\s+risk|irish)\b", re.IGNORECASE)
_IRIS_ALONE = re.compile(r"^(\W*)(?:i\s+risk|irish)(?=\s*(?:[,.!?]|$))", re.IGNORECASE)
_MISHEARD_HEY = re.compile(r"^(\W*)(?:e|a|ay|eh)\b", re.IGNORECASE)


def postfix(text: str) -> str:
    """Apply the narrow "Iris" map (see above). Pure; returns text unchanged when nothing matches."""
    if not text:
        return text
    m = _IRIS_AFTER_GREETING.match(text)
    if m:
        return _MISHEARD_HEY.sub(lambda g: g.group(1) + "Hey", m.group(1), count=1) + "Iris" + text[m.end():]
    m = _IRIS_ALONE.match(text)
    if m:
        return m.group(1) + "Iris" + text[m.end():]
    return text


# ── health (read by voice_status / health_snapshot via the daemon's status + a state file) ──────
_H_LOCK = threading.Lock()
_HEALTH: dict = {"reachable": None, "last_ok_ts": None, "last_error": None, "last_error_ts": None,
                 "sessions": 0, "finals_used": 0, "finals_late": 0, "finals_missing": 0,
                 "broken_sessions": 0, "last_final_ms": None, "server": None}
_SKIP_UNTIL = 0.0          # after a connect failure, don't try again for a few seconds (no churn)
SKIP_AFTER_FAIL_S = 5.0


def _note(**kw) -> None:
    with _H_LOCK:
        for k, v in kw.items():
            if k.startswith("inc_"):
                _HEALTH[k[4:]] = (_HEALTH.get(k[4:]) or 0) + v
            else:
                _HEALTH[k] = v


def _note_error(err: str) -> None:
    global _SKIP_UNTIL
    _note(last_error=str(err)[:200], last_error_ts=round(time.time(), 3))


def health() -> dict:
    with _H_LOCK:
        return dict(_HEALTH)


def ping(sock_path: "str | None" = None, timeout: float = 0.5) -> "dict | None":
    """{"type": "pong", "warm": ..., ...} or None. Updates health()."""
    global _SKIP_UNTIL
    s = None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(sock_path or default_sock_path())
        send_msg(s, b"P")
        kind, payload = recv_msg(s)
        d = json.loads(payload.decode("utf-8")) if kind == b"J" else None
        ok = bool(d and d.get("type") == "pong" and d.get("warm"))
        _note(reachable=ok, server=d, **({"last_ok_ts": round(time.time(), 3)} if ok else {}))
        if ok:
            _SKIP_UNTIL = 0.0
        return d
    except Exception as e:
        _note(reachable=False)
        _note_error(f"ping: {e!r}")
        _SKIP_UNTIL = time.time() + SKIP_AFTER_FAIL_S
        return None
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def recently_failed() -> bool:
    return time.time() < _SKIP_UNTIL


class StreamSession:
    """One utterance on the sidecar. Thread-safe feed/flush/close; never raises to the caller."""

    def __init__(self, on_partial=None, sock_path: "str | None" = None, connect_timeout: float = 0.3,
                 max_queue: int = 400, source: str = ""):
        self.on_partial = on_partial
        self.sock_path = sock_path or default_sock_path()
        self.connect_timeout = connect_timeout
        self.source = source
        self.sid = f"{source or 'utt'}-{int(time.time() * 1000)}"
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)   # 400 x 100 ms = 40 s of backlog
        self.partial = ""
        self.partial_ts = None
        self.final = None                 # dict once the final arrived
        self.broken = None                # reason string once the session is unusable
        self.fed_samples = 0
        self.t_flush = None
        self._final_evt = threading.Event()
        self._sock = None
        self._closed = False
        _note(inc_sessions=1)
        self._tx = threading.Thread(target=self._sender, daemon=True, name="stream-asr-tx")
        self._tx.start()

    # -- caller side --------------------------------------------------------------------------
    def feed(self, frame) -> None:
        if self.broken or self._closed or self.t_flush is not None:
            return
        try:
            a = np.ascontiguousarray(np.asarray(frame, dtype=np.float32).reshape(-1))
            self.q.put_nowait(("A", a.astype("<f4", copy=False).tobytes()))
            self.fed_samples += a.size
        except queue.Full:
            self._break("send queue full (sidecar slow or hung)")
        except Exception as e:
            self._break(f"feed: {e!r}")

    def flush(self, timeout_s: float = 0.15) -> "dict | None":
        """Ask for the final and wait up to timeout_s. Returns {"text", "flush_ms", ...} or None."""
        if self.final is not None:
            return self.final
        if self.broken or self._closed:
            _note(inc_finals_missing=1)
            return None
        if self.t_flush is None:
            self.t_flush = time.time()
            try:
                self.q.put_nowait(("F", b""))
            except queue.Full:
                self._break("send queue full at flush")
                _note(inc_finals_missing=1)
                return None
        self._final_evt.wait(max(0.0, timeout_s - (time.time() - self.t_flush)))
        if self.final is not None:
            _note(inc_finals_used=1, last_final_ms=self.final.get("client_ms"))
            return self.final
        _note(inc_finals_late=1 if not self.broken else 0, inc_finals_missing=1 if self.broken else 0)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        s = self._sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass

    # -- internals ----------------------------------------------------------------------------
    def _break(self, why: str) -> None:
        if self.broken is None:
            self.broken = why
            _note(inc_broken_sessions=1)
            _note_error(why)
            print(f"[stream_asr] session {self.sid} -> whisper fallback: {why}", file=sys.stderr, flush=True)
        self._final_evt.set()
        self.close()

    def _sender(self) -> None:
        global _SKIP_UNTIL
        s = None
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect(self.sock_path)
            # Blocking from here on: ONE timeout would cover recv too, and a quiet stretch with no
            # partial is normal. Capture still never blocks (feed is put_nowait; a wedged sidecar fills
            # the queue -> broken), flush waits only its own timeout, and close() shuts the socket
            # down, which wakes a blocked send/recv in these threads.
            s.settimeout(None)
            self._sock = s
            send_msg(s, b"O", json.dumps({"sid": self.sid, "source": self.source}).encode())
            threading.Thread(target=self._reader, args=(s,), daemon=True, name="stream-asr-rx").start()
            _note(reachable=True)
        except Exception as e:
            _SKIP_UNTIL = time.time() + SKIP_AFTER_FAIL_S
            _note(reachable=False)
            self._break(f"connect: {e!r}")
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
            return
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "A":
                    send_msg(s, b"A", payload)
                elif kind == "F":
                    send_msg(s, b"F")
            if self.final is None and self.t_flush is None:
                try:
                    send_msg(s, b"C")
                except Exception:
                    pass
        except Exception as e:
            if not self._closed:
                self._break(f"send: {e!r}")
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _reader(self, s) -> None:
        try:
            while True:
                kind, payload = recv_msg(s)
                if kind != b"J":
                    continue
                d = json.loads(payload.decode("utf-8"))
                t = d.get("type")
                if t == "partial":
                    txt = postfix((d.get("text") or "").strip())
                    if txt and txt != self.partial:
                        self.partial = txt
                        self.partial_ts = time.time()
                        if self.on_partial is not None:
                            try:
                                self.on_partial(txt)
                            except Exception:
                                pass
                elif t == "final":
                    d["raw_text"] = d.get("text") or ""
                    d["text"] = postfix(d["raw_text"].strip())
                    if self.t_flush is not None:
                        d["client_ms"] = round((time.time() - self.t_flush) * 1000, 1)
                    self.final = d
                    _note(last_ok_ts=round(time.time(), 3))
                    self._final_evt.set()
                    return
                elif t == "error":
                    self._break(f"server: {d.get('msg')}")
                    return
        except Exception as e:
            if not self._closed and self.final is None:
                self._break(f"recv: {e!r}")
