"""Streaming-ASR sidecar (11-02): client <-> server over a real unix socket with a FAKE model, and the
ears' stream paths (listen turn-end, tiered barge) with a fake session. No NeMo, no GPU, no mic.

    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_stream_asr.py
"""
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "voice"))

sasr = pytest.importorskip("iris_stream_asr")
server_mod = pytest.importorskip("iris_stream_asr_server")
core = pytest.importorskip("wren_voice_core")
import wren_voice_status  # noqa: E402


# ───────────────────────────── fake model behind the REAL server ──────────────────────────────
class FakeEngineSession:
    """Emits one word per 0.3 s of audio; finish() sleeps `finish_s` (a slow GPU)."""
    WORDS = "hello there this is a streaming test".split()

    def __init__(self, finish_s=0.0):
        self.n = 0
        self.text = ""
        self.step_ms = []
        self.step = 0
        self.finish_s = finish_s

    def push(self, a):
        self.n += a.size
        before = self.text
        k = min(len(self.WORDS), self.n // 4800)
        self.text = " ".join(self.WORDS[:k])
        self.step += 1
        self.step_ms.append(1.0)
        return self.text != before

    def finish(self):
        time.sleep(self.finish_s)
        k = min(len(self.WORDS), -(-self.n // 4800))
        self.text = " ".join(self.WORDS[:k])
        return self.text

    def words(self):
        return [(w, i * 0.3, i * 0.3 + 0.2) for i, w in enumerate(self.text.split())]


class FakeStreamer:
    model_id = "fake"
    chunk_ms = 160
    boost_phrases = []

    def __init__(self, finish_s=0.0):
        self.finish_s = finish_s

    def session(self):
        return FakeEngineSession(self.finish_s)


@pytest.fixture
def sidecar(monkeypatch):
    """Start the real Server (worker + accept loop) on a temp socket with a fake streamer."""
    d = tempfile.mkdtemp(prefix="sasr")
    path = os.path.join(d, "s.sock")
    monkeypatch.setenv("IRIS_STREAM_ASR_SOCK", path)
    monkeypatch.setattr(sasr, "_SKIP_UNTIL", 0.0)
    import socket
    made = {}

    def start(finish_s=0.0):
        srv = server_mod.Server(FakeStreamer(finish_s))
        threading.Thread(target=srv.worker, daemon=True).start()
        ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ls.bind(path)
        ls.listen(8)
        made["ls"] = ls

        def _acc():
            while True:
                try:
                    c, _ = ls.accept()
                except OSError:
                    return
                threading.Thread(target=srv.handle, args=(c,), daemon=True).start()
        threading.Thread(target=_acc, daemon=True).start()
        return srv
    yield path, start
    if "ls" in made:
        made["ls"].close()


def _feed_seconds(sess, seconds, frame_s=0.1):
    f = np.full(int(16000 * frame_s), 0.05, np.float32)
    for _ in range(int(round(seconds / frame_s))):
        sess.feed(f)


def test_partials_and_final_within_150ms(sidecar):
    path, start = sidecar
    start()
    got = []
    s = sasr.StreamSession(on_partial=got.append, sock_path=path, source="t")
    _feed_seconds(s, 1.0)
    t = time.time()
    while (not got or got[-1] != "hello there this") and time.time() - t < 2:
        time.sleep(0.01)
    # the server coalesces queued audio, so an intermediate partial may be skipped - never reordered
    assert got[-1] == "hello there this"
    assert all("hello there this".startswith(g) for g in got) and got == sorted(got, key=len)
    fin = s.flush(0.15)
    assert fin is not None and fin["text"] == "hello there this is"
    assert fin["client_ms"] < 150 and fin["words"][0][0] == "hello"
    s.close()


def test_long_quiet_stretch_does_not_break_the_session(sidecar):
    """Regression (e2e 11-02): a socket timeout also covered recv, so >2 s without a partial killed it."""
    path, start = sidecar
    start()
    s = sasr.StreamSession(sock_path=path)
    time.sleep(2.5)                          # he hasn't said anything yet: no partials at all
    _feed_seconds(s, 0.6)
    fin = s.flush(0.5)
    assert s.broken is None and fin is not None and fin["text"] == "hello there"
    s.close()


def test_final_after_150ms_is_late_and_returns_none(sidecar):
    path, start = sidecar
    start(finish_s=0.4)
    s = sasr.StreamSession(sock_path=path)
    _feed_seconds(s, 0.5)
    late0 = sasr.health()["finals_late"]
    t = time.time()
    assert s.flush(0.15) is None
    assert time.time() - t < 0.3
    assert sasr.health()["finals_late"] == late0 + 1
    s.close()


def test_sidecar_down_breaks_fast_and_never_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(sasr, "_SKIP_UNTIL", 0.0)
    s = sasr.StreamSession(sock_path=str(tmp_path / "nobody.sock"))
    t = time.time()
    _feed_seconds(s, 2.0)
    assert s.flush(0.15) is None
    assert time.time() - t < 0.5
    assert s.broken and "connect" in s.broken
    assert sasr.recently_failed()
    assert sasr.ping(str(tmp_path / "nobody.sock")) is None and sasr.health()["reachable"] is False


def test_hung_sidecar_never_blocks_feed_and_flush_times_out(monkeypatch, tmp_path):
    import socket
    monkeypatch.setattr(sasr, "_SKIP_UNTIL", 0.0)
    path = str(tmp_path / "hung.sock")
    ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ls.bind(path)
    ls.listen(1)                       # accepts at the kernel level, never reads, never answers
    try:
        s = sasr.StreamSession(sock_path=path, max_queue=20)
        t = time.time()
        _feed_seconds(s, 30.0)         # far more than the socket buffer + queue hold
        assert time.time() - t < 0.5   # feed is put_nowait: never blocks capture
        assert s.flush(0.15) is None
        assert time.time() - t < 1.0
        assert s.broken                # queue overflow or send timeout -> whisper
    finally:
        ls.close()


def test_newer_session_preempts_older(sidecar):
    path, start = sidecar
    start()
    a = sasr.StreamSession(sock_path=path)
    _feed_seconds(a, 0.3)
    time.sleep(0.1)
    b = sasr.StreamSession(sock_path=path)
    _feed_seconds(b, 0.6)
    t = time.time()
    while not a.broken and time.time() - t < 2:
        time.sleep(0.01)
    assert a.broken and "preempted" in a.broken
    fin = b.flush(0.5)
    assert fin and fin["text"] == "hello there"
    a.close()
    b.close()


def test_ping_reports_warm(sidecar):
    path, start = sidecar
    start()
    p = sasr.ping(path)
    assert p["type"] == "pong" and p["warm"] is True and p["chunk_ms"] == 160


# ───────────────────────────── ears side (wren_voice_core) ──────────────────────────────
class FakeSess:
    """Stands in for iris_stream_asr.StreamSession inside the ears."""
    made: list = []

    def __init__(self, on_partial, source, text="hello there iris", delay=0.0, broken=None):
        self.on_partial, self.source = on_partial, source
        self.text, self.delay, self.broken = text, delay, broken
        self.fed = []
        self.closed = False
        FakeSess.made.append(self)

    def feed(self, f):
        self.fed.append(np.asarray(f))
        if len(self.fed) == 3 and self.on_partial:
            self.on_partial("hello")

    def flush(self, timeout):
        if self.broken:
            return None
        if self.delay > timeout:
            time.sleep(timeout)
            return None
        time.sleep(self.delay)
        return {"text": self.text, "client_ms": self.delay * 1000, "flush_ms": 5.0}

    def close(self):
        self.closed = True


class W:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


def _sess_factory(**kw):
    def f(on_partial, source):
        return FakeSess(on_partial, source, **kw)
    return f


@pytest.fixture
def ears(monkeypatch, tmp_path):
    FakeSess.made = []
    monkeypatch.setattr(core, "STREAM_ASR_LOG", tmp_path / "turns.jsonl")
    monkeypatch.setattr(core._sasr, "_SKIP_UNTIL", 0.0)
    calls = {"whisper": 0}

    def _whisper(audio):
        calls["whisper"] += 1
        return "hello there Irish", [W(" hello", 0.0, 0.3), W(" there", 0.35, 0.6), W(" Irish", 0.7, 1.0)]
    monkeypatch.setattr(core._prosody, "transcribe", _whisper)
    return calls


@pytest.mark.parametrize("heard,want", [
    ("E I risk if uh is it pushing", "Hey Iris if uh is it pushing"),     # the bench clip
    ("Hey hey I risk, you there?", "Hey hey Iris, you there?"),
    ("hey Irish can you", "hey Iris can you"),
    ("I risk, can you hear me", "Iris, can you hear me"),
    ("I risk losing it if I go", "I risk losing it if I go"),             # a real "I risk": untouched
    ("so I risk it all", "so I risk it all"),
    ("But Web View Two Iris is using", "But Web View Two Iris is using"),
    ("A lot I risk", "A lot I risk"),
])
def test_iris_postfix_only_obvious_cases(heard, want):
    assert sasr.postfix(heard) == want


def test_postfix_applies_to_partials_and_finals(sidecar, monkeypatch):
    path, start = sidecar
    monkeypatch.setattr(FakeEngineSession, "WORDS", "E I risk if uh".split())
    start()
    got = []
    s = sasr.StreamSession(on_partial=got.append, sock_path=path)
    _feed_seconds(s, 1.5)
    fin = s.flush(0.5)
    assert fin["text"] == "Hey Iris if uh" and fin["raw_text"] == "E I risk if uh"
    assert any(g.startswith("Hey Iris") for g in got)
    s.close()


def test_align_keeps_every_stream_word():
    ww = [W(" hello", 0.0, 0.3), W(" there", 0.35, 0.6), W(" Irish", 0.7, 1.0)]
    out = core._align_stream_words("Hello there, Iris", ww)
    assert [w.word.strip() for w in out] == ["Hello", "there,", "Iris"]
    assert out[0].start == 0.0 and out[1].end == 0.6
    assert 0.6 <= out[2].start <= 1.0                       # unmatched word: interpolated in the gap
    assert core._align_stream_words("totally different words here", ww) == []   # too few matches


def _speech(seconds=1.0, tail=0.4):
    a = np.concatenate([np.full(int(16000 * seconds), 0.1, np.float32), np.zeros(int(16000 * tail), np.float32)])
    return a


def test_turn_end_uses_stream_final_and_skips_whisper(monkeypatch, ears):
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory())
    st = core._StreamFeed("listen")
    st.feed(np.zeros(1600, np.float32))
    audio = _speech()
    p = core._StreamingPartials()
    p.update("hello there Irish", [W(" hello", 0.0, 0.3), W(" there", 0.35, 0.6), W(" Irish", 0.7, 1.0)],
             audio.size, 0.2)                                  # the pause pass covered everything
    st.t_end = time.time()
    text, words = core._transcribe_stream_turn(audio, st, p, None, 0.15)
    assert text == "hello there iris" and ears["whisper"] == 0
    assert [w.word.strip() for w in words] == ["hello", "there", "iris"]
    assert FakeSess.made[0].closed


def test_turn_end_stale_pass_goes_out_unmarked_not_late(monkeypatch, ears):
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory())
    st = core._StreamFeed("listen")
    st.feed(np.zeros(1600, np.float32))
    audio = _speech(2.0)
    p = core._StreamingPartials()
    p.update("hello", [W(" hello", 0.0, 0.3)], 8000, 0.2)    # he spoke again after this pass
    st.t_end = time.time()
    text, words = core._transcribe_stream_turn(audio, st, p, None, 0.15)
    assert text == "hello there iris" and words == [] and ears["whisper"] == 0


def test_turn_end_late_final_falls_back_to_whisper(monkeypatch, ears):
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory(delay=0.4))
    st = core._StreamFeed("listen")
    st.feed(np.zeros(1600, np.float32))
    st.t_end = time.time()
    t = time.time()
    text, words = core._transcribe_stream_turn(_speech(), st, core._StreamingPartials(), None, 0.15)
    assert time.time() - t < 0.35
    assert text == "hello there Irish" and ears["whisper"] == 1 and len(words) == 3


def test_turn_end_empty_final_falls_back_to_whisper(monkeypatch, ears):
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory(text=""))
    st = core._StreamFeed("listen")
    st.feed(np.zeros(1600, np.float32))
    st.t_end = time.time()
    text, _ = core._transcribe_stream_turn(_speech(), st, core._StreamingPartials(), None, 0.15)
    assert text == "hello there Irish" and ears["whisper"] == 1


def test_turn_end_open_failure_falls_back(monkeypatch, ears):
    def boom(on_partial, source):
        raise OSError("no sidecar")
    monkeypatch.setattr(core, "_open_stream_session", boom)
    st = core._StreamFeed("listen")
    st.feed(np.zeros(1600, np.float32))
    assert st.broken
    st.t_end = time.time()
    text, _ = core._transcribe_stream_turn(_speech(), st, core._StreamingPartials(), None, 0.15)
    assert text == "hello there Irish" and ears["whisper"] == 1


def test_stream_ctl_flag_and_skip(monkeypatch):
    monkeypatch.setattr(core._sasr, "_SKIP_UNTIL", 0.0)
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    assert core._stream_ctl()["on"] is False
    monkeypatch.setattr(wren_voice_status, "read_control",
                        lambda: {"stream_asr": "on", "stream_asr_final_ms": 90})
    c = core._stream_ctl()
    assert c["on"] is True and abs(c["final_s"] - 0.09) < 1e-9
    monkeypatch.setattr(core._sasr, "_SKIP_UNTIL", time.time() + 5)   # sidecar just failed
    assert core._stream_ctl()["on"] is False


# ── capture: the sidecar gets pre-roll + every captured frame, never blocking ──
class FakeMic100:
    """100 ms frames; each read advances the shared fake clock by one frame."""

    def __init__(self, frames, clock):
        self._it = iter(frames)
        self.clock = clock

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.clock["t"] += 0.1
        time.sleep(0.02)               # real time passes too, so the pause-pass worker gets to run
        try:
            c = next(self._it)
        except StopIteration:
            c = np.zeros(n, np.float32)
        return c.reshape(-1, 1), False


class CapCtx:
    mic_dev_idx = "net"
    speaking = False


def test_capture_feeds_preroll_and_runs_one_pause_pass(monkeypatch, ears):
    loud = np.full(1600, 0.1, np.float32)
    quiet = np.zeros(1600, np.float32)
    frames = [quiet] * 5 + [loud] * 10 + [quiet] * 12
    monkeypatch.setattr(core, "_trusted_mic_or_none", lambda ctx: "net")
    clock = {"t": 1000.0}
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: FakeMic100(frames, clock))
    monkeypatch.setattr(core, "set_state", lambda *a, **k: None)
    monkeypatch.setattr(core, "is_muted", lambda: False)
    monkeypatch.setattr(core.time, "time", lambda: clock["t"])
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory())
    st = core._StreamFeed("listen", preroll_s=0.3)
    p = core._StreamingPartials()
    wout = {}
    audio = core._capture_utterance(CapCtx(), 5.0, 20.0, 1.1, partials=p,
                                    transcribe_fn=lambda a: ("hello", [W(" hello", 0, 0.3)]),
                                    asr_stream=st, prosody_on_pause=True, worker_out=wout)
    assert audio is not None and audio.size == 21 * 1600          # 10 voiced + 11 silent (1.1 s end-silence)
    sess = FakeSess.made[0]
    assert len(sess.fed) == 3 + 21                                 # 0.3 s pre-roll + every captured frame
    t = wout.get("thread")
    assert t is not None
    t.join(2.0)
    lt = p.latest()
    assert lt is not None and lt[0] == "hello" and lt[2] >= 10 * 1600   # the pause pass covered the speech


# ── tiered barge: his words during my speech come from the sidecar ──
def _barge_rig(monkeypatch, script, heard="whisper words"):
    import test_bargein_tiered_daemon as tb
    monkeypatch.setattr(core._req, "urlopen", lambda u, timeout=0: _Ok())
    monkeypatch.setattr(core, "is_muted", lambda: False)
    monkeypatch.setattr(core, "_trusted_mic_or_none", lambda ctx: "net")
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: tb.FakeMic(script))
    calls = {"whisper": 0}

    def _w(a):
        calls["whisper"] += 1
        return heard
    monkeypatch.setattr(core.wl, "_transcribe_array", _w)
    monkeypatch.setattr(core.wl, "append_transcript", lambda t: None)
    monkeypatch.setattr(core, "_self_voice_score", lambda a, bar: None)
    monkeypatch.setattr(core, "_voice_id_verdict", lambda a: {"status": "matched", "speaker": "zeke", "score": 0.8})
    monkeypatch.setattr(core, "_voice_id_line", lambda a, v=None: "")
    monkeypatch.setattr(core, "_save_utt_wav", lambda a, tag: None)
    monkeypatch.setattr(core, "set_state", lambda *a, **k: None)
    return tb, calls


class _Ok:
    def read(self):
        return b"ok"


def test_tiered_barge_uses_stream_final(monkeypatch, ears, tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    tb, calls = _barge_rig(monkeypatch, None)
    script = [tb.LOUD] * 50 + [tb.QUIET] * 25
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: tb.FakeMic(script))
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {"stream_asr": "on"})
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory(text="no use the other render"))
    out = core._cmd_bargein_watch_tiered(tb.Ctx(), {"tiers": True}, None)
    assert out.endswith("no use the other render") and calls["whisper"] == 0
    assert len(FakeSess.made) == 1 and len(FakeSess.made[0].fed) >= 50   # pre-roll + episode


def test_tiered_barge_broken_stream_falls_back_to_whisper(monkeypatch, ears):
    sys.path.insert(0, str(ROOT / "tests"))
    tb, calls = _barge_rig(monkeypatch, None, heard="no use the other render")
    script = [tb.LOUD] * 50 + [tb.QUIET] * 25
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: tb.FakeMic(script))
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {"stream_asr": "on"})
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory(broken="send queue full"))
    out = core._cmd_bargein_watch_tiered(tb.Ctx(), {"tiers": True}, None)
    assert out.endswith("no use the other render") and calls["whisper"] == 1


def test_tiered_barge_flag_off_never_opens_a_session(monkeypatch, ears):
    sys.path.insert(0, str(ROOT / "tests"))
    tb, calls = _barge_rig(monkeypatch, None, heard="no use the other render")
    script = [tb.LOUD] * 50 + [tb.QUIET] * 25
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: tb.FakeMic(script))
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    monkeypatch.setattr(core, "_open_stream_session", _sess_factory())
    out = core._cmd_bargein_watch_tiered(tb.Ctx(), {"tiers": True}, None)
    assert out.endswith("no use the other render") and calls["whisper"] == 1 and FakeSess.made == []


# ── health_snapshot probe ──
def _hs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("health_snapshot_sasr", ROOT / "scripts" / "server" / "health_snapshot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _vc(root, d):
    import json
    (root / "scratch").mkdir(parents=True, exist_ok=True)
    (root / "scratch" / "voice_control.json").write_text(json.dumps(d))


def test_health_probe_off_ok_and_down(sidecar, tmp_path, monkeypatch):
    m = _hs()
    path, start = sidecar
    root = tmp_path / "root"
    _vc(root, {})
    c = {"root": root, "voice_off": False, "stream_asr_sock": path}
    assert m.probe_stream_asr(c)["status"] == "off"                       # flag off -> deliberately off
    _vc(root, {"stream_asr": "on"})
    monkeypatch.setattr(m, "unit_active_seconds", lambda unit: None)
    r = m.probe_stream_asr(c)
    assert r["status"] == "red" and "ping failed" in r["error"]           # flag on, nothing listening
    monkeypatch.setattr(m, "unit_active_seconds", lambda unit: 30.0)
    assert m.probe_stream_asr(c)["status"] == "warn"                      # unit up, still loading
    start()
    r = m.probe_stream_asr(c)
    assert r["status"] == "ok" and r["detail"]["chunk_ms"] == 160
    assert "stream_asr" in m.PROBES
