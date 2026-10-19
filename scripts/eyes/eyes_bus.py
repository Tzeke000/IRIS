"""eyes_bus — perception RESULTS transport for the eyes process (prototype).

Design: docs/eyes_process_design.md §4.3. Same wire format as the ears daemon
(voice/wren_voice_daemon.py: newline-delimited JSON, {"cmd","args"} -> {"ok","result"}),
but on a Unix domain socket instead of TCP (local-only by construction, no port to
collide with, file permissions = access control).

Two shapes on one socket:
  * request/response  — ping, status, snapshot, park, unpark, ... (one JSON line each way)
  * subscribe          — after {"cmd":"subscribe","args":{"hz":30}} the server PUSHES
                         {"type":"snapshot",...} lines at `hz` and {"type":"event",...}
                         lines (face_appeared / face_lost / face_changed /
                         expression_changed / attention_changed / eyes_state) as they
                         happen. A slow subscriber loses old snapshots (bounded queue),
                         never blocks the eyes.

Staleness: every perception field carries the capture_ts of the frame it was computed
from (`frame_ts`) and its own `done_ts`. The CLIENT judges freshness with its own clock
(same host, same wall clock), so a frozen eyes process can never look fresh:
`classify_result(...)` -> fresh / aging / stale / unavailable, and stale perception
must be read as "I can't see" — never as "nobody is there".
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from typing import Any, Callable, Optional

DEFAULT_SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "iris", "eyes.sock")

# Result freshness (seconds since the SOURCE FRAME's capture_ts).
RESULT_FRESH_S = 0.35
RESULT_AGING_S = 1.20
# No line from a subscription for this long ⇒ link is down (snapshots come at >=1 Hz).
LINK_DEAD_S = 2.0


def classify_result(frame_ts: Optional[float], now: Optional[float] = None) -> str:
    if not frame_ts:
        return "unavailable"
    age = (now if now is not None else time.time()) - float(frame_ts)
    if age < -0.5:          # from the future = clock skew/bug; never call it fresh
        return "stale"
    if age <= RESULT_FRESH_S:
        return "fresh"
    if age <= RESULT_AGING_S:
        return "aging"
    return "stale"


def presence_view(snap: Optional[dict], now: Optional[float] = None) -> dict:
    """The rule every 'is someone here?' consumer must use.

    Returns {"seeing": bool, "face_count": int|None, "why": str}.
    face_count is None (UNKNOWN) — not 0 — whenever the eyes are down, parked or stale.
    """
    if not snap:
        return {"seeing": False, "face_count": None, "why": "no_link"}
    st = snap.get("eyes_state")
    if st != "streaming":
        return {"seeing": False, "face_count": None, "why": f"eyes_{st}"}
    faces = snap.get("faces") or {}
    fr = classify_result(faces.get("frame_ts"), now)
    if fr not in ("fresh", "aging"):
        return {"seeing": False, "face_count": None, "why": f"faces_{fr}"}
    return {"seeing": True, "face_count": len(faces.get("value") or []), "why": fr}


# ── server ────────────────────────────────────────────────────────────────────

class BusServer:
    """Runs inside the eyes process. `handlers` maps cmd -> fn(args) -> jsonable.
    `snapshot_fn()` returns the current perception snapshot dict."""

    def __init__(self, sock_path: str, snapshot_fn: Callable[[], dict],
                 handlers: Optional[dict[str, Callable[[dict], Any]]] = None) -> None:
        self.sock_path = sock_path
        self.snapshot_fn = snapshot_fn
        self.handlers = dict(handlers or {})
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._stop = threading.Event()
        self._srv: Optional[socket.socket] = None
        self.dropped = 0
        # Result-driven push: notify_result() bumps _gen; subscriptions send a
        # snapshot as soon as a NEW result exists (capped at the subscriber's hz),
        # plus a >=1 Hz heartbeat snapshot so a quiet/parked eyes still proves life.
        self._cond = threading.Condition()
        self._gen = 0

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.sock_path), mode=0o700, exist_ok=True)
        try:
            os.unlink(self.sock_path)          # stale socket from a crashed predecessor
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.sock_path)
        os.chmod(self.sock_path, 0o600)
        s.listen(16)
        s.settimeout(0.5)
        self._srv = s
        threading.Thread(target=self._accept_loop, daemon=True, name="eyes-bus-accept").start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._srv:
                self._srv.close()
        finally:
            try:
                os.unlink(self.sock_path)
            except FileNotFoundError:
                pass

    def publish_event(self, name: str, data: dict) -> None:
        self._fanout({"type": "event", "name": name, "ts": time.time(), "data": data})
        with self._cond:
            self._cond.notify_all()

    def notify_result(self) -> None:
        """Call after any perception result changes — wakes push subscriptions."""
        with self._cond:
            self._gen += 1
            self._cond.notify_all()

    def subscriber_count(self) -> int:
        with self._subs_lock:
            return len(self._subs)

    # -- internals ----------------------------------------------------------
    def _fanout(self, msg: dict) -> None:
        with self._subs_lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:                       # drop the OLDEST, keep the newest
                    q.get_nowait()
                    self.dropped += 1
                    q.put_nowait(msg)
                except Exception:
                    pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._conn_loop, args=(conn,), daemon=True,
                             name="eyes-bus-conn").start()

    def _conn_loop(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    cmd, args = str(req.get("cmd", "")), (req.get("args") or {})
                except Exception as e:
                    _send(conn, {"ok": False, "error": f"parse error: {e!r}"})
                    continue
                if cmd == "subscribe":
                    self._serve_subscription(conn, float(args.get("hz") or 10.0))
                    return
                try:
                    if cmd == "ping":
                        res: Any = {"pong": True, "pid": os.getpid()}
                    elif cmd == "snapshot":
                        res = self.snapshot_fn()
                    elif cmd in self.handlers:
                        res = self.handlers[cmd](args)
                    else:
                        raise ValueError(f"unknown cmd: {cmd!r}")
                    _send(conn, {"ok": True, "result": res})
                except Exception as e:
                    _send(conn, {"ok": False, "error": repr(e)})
        except OSError:
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _serve_subscription(self, conn: socket.socket, hz: float) -> None:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._subs_lock:
            self._subs.append(q)
        min_gap = 1.0 / max(1.0, min(hz, 60.0))   # rate cap
        heartbeat = 1.0                            # snapshot at least this often
        try:
            _send(conn, {"ok": True, "result": {"subscribed": True, "hz": 1.0 / min_gap}}, strict=True)
            last_sent, sent_gen = 0.0, -1
            while not self._stop.is_set():
                while True:                         # drain events first (they are small)
                    try:
                        _send(conn, q.get_nowait(), strict=True)
                    except queue.Empty:
                        break
                now = time.time()
                due_new = self._gen != sent_gen and now - last_sent >= min_gap
                if due_new or now - last_sent >= heartbeat:
                    sent_gen = self._gen
                    snap = self.snapshot_fn()
                    snap["type"] = "snapshot"
                    _send(conn, snap, strict=True)
                    last_sent = now
                    continue
                with self._cond:
                    if self._gen != sent_gen:       # new result, but rate-capped
                        wait = min_gap - (now - last_sent)
                    else:
                        wait = heartbeat - (now - last_sent)
                    if q.empty():
                        self._cond.wait(timeout=max(0.001, wait))
        except OSError:
            pass                            # subscriber went away
        finally:
            with self._subs_lock:
                if q in self._subs:
                    self._subs.remove(q)


def _send(conn: socket.socket, payload: dict, strict: bool = False) -> None:
    try:
        conn.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    except OSError:
        if strict:
            raise


# ── client ────────────────────────────────────────────────────────────────────

def request(cmd: str, args: Optional[dict] = None, sock_path: str = DEFAULT_SOCK,
            timeout: float = 1.0) -> dict:
    """One-shot request. Returns {"ok":..., "result"|"error":...}; never raises on
    a dead eyes process (returns ok=False, error='eyes_unreachable: ...')."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(sock_path)
            s.sendall((json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf.split(b"\n", 1)[0])
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"eyes_unreachable: {e!r}"}


class Subscriber:
    """Background mirror of the eyes' perception state — what the runtime's
    compatibility shim would run (one thread, auto-reconnect).

    .latest()  -> last snapshot dict or None if the link is dead
    on_event   -> callback(name, data) for transition events
    """

    def __init__(self, sock_path: str = DEFAULT_SOCK, hz: float = 30.0,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 on_snapshot: Optional[Callable[[dict], None]] = None) -> None:
        self.sock_path, self.hz = sock_path, hz
        self.on_event, self.on_snapshot = on_event, on_snapshot
        self._latest: Optional[dict] = None
        self._last_rx = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.reconnects = 0
        self.rx_lines = 0

    def start(self) -> "Subscriber":
        self._t = threading.Thread(target=self._run, daemon=True, name="eyes-subscriber")
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)

    def link_alive(self) -> bool:
        return self._connected and (time.time() - self._last_rx) <= LINK_DEAD_S

    def latest(self) -> Optional[dict]:
        return self._latest if self.link_alive() else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(self.sock_path)
                    self._connected = True
                    s.sendall((json.dumps({"cmd": "subscribe", "args": {"hz": self.hz}}) + "\n").encode())
                    buf = b""
                    while not self._stop.is_set():
                        try:
                            chunk = s.recv(262144)
                        except socket.timeout:
                            if not self.link_alive():
                                break       # silent server — reconnect
                            continue
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._handle(line)
            except OSError:
                pass
            self._connected = False
            self._latest = None
            self.reconnects += 1
            self._stop.wait(0.5)            # backoff, then reconnect

    def _handle(self, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return
        self._last_rx = time.time()
        self.rx_lines += 1
        t = msg.get("type")
        if t == "snapshot":
            self._latest = msg
            if self.on_snapshot:
                self.on_snapshot(msg)
        elif t == "event" and self.on_event:
            try:
                self.on_event(str(msg.get("name")), msg.get("data") or {})
            except Exception:
                pass
