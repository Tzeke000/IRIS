"""iris_stream_asr_server.py - streaming-ASR sidecar for the ears (Nemotron Speech Streaming 0.6B).

Runs in its OWN NeMo venv (never iris-v100), as the systemd --user unit iris-stream-asr
(scripts/server/units/iris-stream-asr.service). Loads the model fp32 on the V100, cache-aware 160 ms
chunks (att_context [70,1]), warms it, THEN binds a UNIX socket (so "socket answers" == "warm").
Client + wire format: voice/iris_stream_asr.py. Engine: voice/iris_stream_asr_engine.py.

  * localhost only: an AF_UNIX socket in $XDG_RUNTIME_DIR, mode 0600 - nothing on the network.
  * one session per utterance, newest wins: a new "open" pre-empts any older live session.
  * every GPU step runs on ONE worker thread; socket readers only enqueue, so a slow step never
    blocks reading and the ears' sender never blocks on us.
  * idle = blocked on a queue / accept: ~0 CPU, no polling.
  * robust restart: a CUDA error or a step stuck > STUCK_S exits non-zero -> systemd restarts us;
    the ears fall back to whisper meanwhile (their session just breaks).

Env knobs: IRIS_STREAM_ASR_SOCK, IRIS_STREAM_ASR_MODEL, IRIS_STREAM_ASR_RC (right context in 80 ms
frames: 1 = 160 ms chunks, 6 = 560 ms), IRIS_STREAM_ASR_BOOST (comma list of phrases; default "" = off),
IRIS_STREAM_ASR_BOOST_ALPHA, IRIS_STREAM_ASR_MAX_S (cap per utterance, default 40 s).
"""
from __future__ import annotations

import json
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iris_stream_asr as proto   # noqa: E402  (pure stdlib + numpy: wire format + socket path)

import numpy as np  # noqa: E402

STUCK_S = float(os.environ.get("IRIS_STREAM_ASR_STUCK_S", "10"))
MAX_S = float(os.environ.get("IRIS_STREAM_ASR_MAX_S", "40"))


def log(msg: str) -> None:
    print(f"[stream-asr] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class Conn:
    """One client connection = one utterance session."""

    def __init__(self, sock):
        self.sock = sock
        self.sid = None
        self.sess = None          # engine StreamSession, created lazily on the worker thread
        self.dead = False
        self.audio_s = 0.0
        self.t_open = time.time()
        self.lock = threading.Lock()

    def send(self, d: dict) -> None:
        if self.dead:
            return
        try:
            with self.lock:
                proto.send_msg(self.sock, b"J", json.dumps(d).encode("utf-8"))
        except Exception:
            self.dead = True


class Server:
    def __init__(self, streamer):
        self.st = streamer
        self.jobs: queue.Queue = queue.Queue()
        self.active: "Conn | None" = None
        self.alock = threading.Lock()
        self.t0 = time.time()
        self.n_sessions = 0
        self.n_errors = 0
        self.last_step_ms = None
        self.last_flush_ms = None
        self.busy_since = None    # worker started a job at (watchdog)

    # -- worker (the ONLY thread that touches the GPU) ---------------------------------------
    def worker(self) -> None:
        while True:
            conn, kind, data = self.jobs.get()
            if conn.dead and kind != "close":
                continue
            self.busy_since = time.time()
            try:
                if kind == "audio":
                    # coalesce whatever else is already queued for this conn (catch up in one push)
                    chunks = [data]
                    while True:
                        try:
                            nxt = self.jobs.queue[0]
                        except IndexError:
                            break
                        if nxt[0] is conn and nxt[1] == "audio":
                            chunks.append(self.jobs.get_nowait()[2])
                        else:
                            break
                    if conn.sess is None:
                        conn.sess = self.st.session()
                    a = np.frombuffer(b"".join(chunks), dtype="<f4")
                    if conn.audio_s + a.size / proto.SAMPLE_RATE > MAX_S:
                        continue          # cap: ignore audio past MAX_S, still finalize on flush
                    conn.audio_s += a.size / proto.SAMPLE_RATE
                    if conn.sess.push(a):
                        conn.send({"type": "partial", "sid": conn.sid, "text": conn.sess.text,
                                   "t": round(conn.audio_s, 3)})
                    if conn.sess.step_ms:
                        self.last_step_ms = round(conn.sess.step_ms[-1], 1)
                elif kind == "flush":
                    t = time.perf_counter()
                    if conn.sess is None:
                        conn.sess = self.st.session()
                    text = conn.sess.finish()
                    ms = round((time.perf_counter() - t) * 1000, 1)
                    self.last_flush_ms = ms
                    conn.send({"type": "final", "sid": conn.sid, "text": text, "flush_ms": ms,
                               "audio_s": round(conn.audio_s, 3), "steps": conn.sess.step,
                               "words": [[w, round(a, 3), round(b, 3)] for w, a, b in conn.sess.words()]})
                    conn.sess = None
                elif kind == "close":
                    conn.sess = None
            except Exception as e:
                self.n_errors += 1
                msg = f"{type(e).__name__}: {e}"
                log(f"step error ({conn.sid}): {msg}\n{traceback.format_exc(limit=4)}")
                conn.send({"type": "error", "sid": conn.sid, "msg": msg[:300]})
                conn.dead = True
                conn.sess = None
                if "CUDA" in msg or "cuda" in msg or "CUBLAS" in msg:
                    log("CUDA error -> exiting so systemd restarts a clean process")
                    os._exit(3)
            finally:
                self.busy_since = None

    def watchdog(self) -> None:
        while True:
            time.sleep(1.0)
            b = self.busy_since
            if b is not None and time.time() - b > STUCK_S:
                log(f"worker stuck > {STUCK_S:.0f}s on one step -> exiting for a clean restart")
                os._exit(4)

    # -- connections ------------------------------------------------------------------------
    def pong(self) -> dict:
        b = self.busy_since
        busy_s = round(time.time() - b, 2) if b is not None else 0.0
        # a step running > 2 s means the GPU worker is wedged: say so (the watchdog exits at STUCK_S)
        return {"type": "pong", "warm": busy_s < 2.0, "busy_s": busy_s,
                "model": self.st.model_id, "chunk_ms": self.st.chunk_ms,
                "boost": self.st.boost_phrases, "uptime_s": round(time.time() - self.t0, 1),
                "sessions": self.n_sessions, "errors": self.n_errors, "pid": os.getpid(),
                "active": bool(self.active and not self.active.dead), "queue": self.jobs.qsize(),
                "last_step_ms": self.last_step_ms, "last_flush_ms": self.last_flush_ms}

    def handle(self, sock) -> None:
        conn = Conn(sock)
        try:
            while True:
                kind, payload = proto.recv_msg(sock)
                if kind == b"P":
                    conn.send(self.pong())
                elif kind == b"O":
                    try:
                        conn.sid = json.loads(payload.decode("utf-8")).get("sid")
                    except Exception:
                        conn.sid = None
                    with self.alock:
                        old, self.active = self.active, conn
                    if old is not None and old is not conn and not old.dead:
                        old.send({"type": "error", "sid": old.sid, "msg": "preempted by a newer session"})
                        old.dead = True
                        self.jobs.put((old, "close", None))
                    self.n_sessions += 1
                elif kind == b"A":
                    if conn.sid is not None and not conn.dead:
                        self.jobs.put((conn, "audio", payload))
                elif kind == b"F":
                    if conn.sid is not None and not conn.dead:
                        self.jobs.put((conn, "flush", None))
                elif kind == b"C":
                    conn.dead = True
                    self.jobs.put((conn, "close", None))
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            if not conn.dead:
                conn.dead = True
                self.jobs.put((conn, "close", None))
            with self.alock:
                if self.active is conn:
                    self.active = None
            try:
                sock.close()
            except Exception:
                pass


def main() -> int:
    path = proto.default_sock_path()
    model_id = os.environ.get("IRIS_STREAM_ASR_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b")
    rc = int(os.environ.get("IRIS_STREAM_ASR_RC", "1"))
    # phrase boosting OFF by default: on the 11-02 clips it never fixed "I risk" and changed other
    # words (see iris_stream_asr.postfix, which handles that case on the ears side instead)
    boost = [p.strip() for p in os.environ.get("IRIS_STREAM_ASR_BOOST", "").split(",") if p.strip()]
    alpha = float(os.environ.get("IRIS_STREAM_ASR_BOOST_ALPHA", "1.0"))
    torch_threads = int(os.environ.get("IRIS_STREAM_ASR_THREADS", "2"))
    t = time.time()
    import torch  # (import here so startup errors land in the journal)
    torch.set_num_threads(torch_threads)
    from iris_stream_asr_engine import NemoStreamer
    st = NemoStreamer(model_id=model_id, right_ctx=rc, boost_phrases=boost, boost_alpha=alpha)
    warm_ms = st.warm()
    log(f"model {model_id} rc={rc} ({st.chunk_ms} ms chunks) boost={boost}@{alpha} loaded+warm in "
        f"{time.time() - t:.1f}s (last warm utterance {warm_ms:.0f} ms)")

    srv = Server(st)
    threading.Thread(target=srv.worker, daemon=True, name="gpu-worker").start()
    threading.Thread(target=srv.watchdog, daemon=True, name="watchdog").start()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(path)          # stale socket from a crashed run
    except FileNotFoundError:
        pass
    ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o177)
    try:
        ls.bind(path)
    finally:
        os.umask(old_umask)
    os.chmod(path, 0o600)
    ls.listen(8)
    log(f"listening on {path}")

    def _bye(*_a):
        try:
            os.unlink(path)
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    while True:
        c, _ = ls.accept()
        threading.Thread(target=srv.handle, args=(c,), daemon=True, name="conn").start()


if __name__ == "__main__":
    sys.exit(main())
