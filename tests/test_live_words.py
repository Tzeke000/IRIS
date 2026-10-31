"""Live words (full-duplex chunk 3, 2026-10-29): display-only rolling partials.

Covers the atomic writer + LiveWords lifecycle, the barge accumulator's capture tap, the
listen/barge wrappers never changing a turn's return value, and the orb read endpoint.

    /home/iris/venvs/iris-v100/bin/python tests/test_live_words.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "voice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import iris_live_words as lw  # noqa: E402


def _tmpfile() -> Path:
    return Path(tempfile.mkdtemp(prefix="live_words_")) / "sub" / "live_partial.json"


def test_write_atomic_leaves_no_temp() -> None:
    p = _tmpfile()
    assert lw.write_atomic(p, {"text": "héllo"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"text": "héllo"}
    assert [x.name for x in p.parent.iterdir()] == ["live_partial.json"]


def test_partials_stream_then_final() -> None:
    p = _tmpfile()
    calls = []

    def fake_stt(buf):
        calls.append(buf.size)
        return f"words {buf.size}"

    w = lw.LiveWords("room-barge", path=p, transcribe_fn=fake_stt, cadence_s=0.1,  # floored to 0.5
                     min_audio_s=0.1, lock=threading.Lock())
    assert w.cadence_s == lw.MIN_CADENCE_S
    for _ in range(10):
        w.feed(np.zeros(1600, dtype=np.float32))
    deadline = time.time() + 3.0
    while not w.published and time.time() < deadline:
        time.sleep(0.05)
    d = json.loads(p.read_text())
    assert d["final"] is False and d["text"].startswith("words ") and d["source"] == "room-barge"
    assert d["utterance_id"] == w.utterance_id and d["seq"] >= 1 and d["dropped"] is None
    w.final_text = "the real words"
    w.finish()
    d = json.loads(p.read_text())
    assert d["final"] is True and d["text"] == "the real words"
    assert not w.publish("late partial")                          # never overwrites the final
    assert json.loads(p.read_text())["text"] == "the real words"
    assert lw.last()["text"] == "the real words"


def test_dropped_turn_clears_display_and_silence_writes_nothing() -> None:
    p = _tmpfile()
    w = lw.LiveWords("room-listen", path=p)
    w.publish("so let me")                                        # a partial was shown...
    w.finish(dropped="[bargein_watch] self_voice_dropped (iris 0.81)")
    d = json.loads(p.read_text())
    assert d["final"] is True and d["text"] == "" and "self_voice" in d["dropped"]
    p2 = _tmpfile()
    lw.LiveWords("room-listen", path=p2).finish(dropped="[voice_listen] no speech within 45s")
    assert not p2.exists()                                        # silent windows don't churn the file


def test_disabled_publishes_nothing() -> None:
    p = _tmpfile()
    w = lw.LiveWords("room-listen", path=p, enabled=False, transcribe_fn=lambda b: "x")
    w.feed(np.zeros(16000, dtype=np.float32))
    w.final_text = "hi"
    w.finish()
    assert not p.exists()


def test_bargein_accumulator_taps_every_captured_chunk() -> None:
    import wren_voice_core as core
    frames = [(np.full(4, i, dtype=np.float32), 0.9 if 2 <= i <= 6 else 0.0) for i in range(12)]
    tapped = []
    out = core._bargein_accumulate(iter(frames), lambda: True, lambda: None, threshold=0.5, need=2,
                                   preroll_chunks=2, end_silence_chunks=2, max_chunks=100,
                                   on_capture=lambda c: tapped.append(c))
    assert out is not None and tapped
    assert np.array_equal(np.concatenate(tapped), out)            # display sees exactly the capture
    # a throwing tap never breaks the capture
    out2 = core._bargein_accumulate(iter(frames), lambda: True, lambda: None, threshold=0.5, need=2,
                                    preroll_chunks=2, end_silence_chunks=2, max_chunks=100,
                                    on_capture=lambda c: 1 / 0)
    assert np.array_equal(out, out2)


def test_wrappers_never_change_the_turn() -> None:
    import wren_voice_core as core
    p = _tmpfile()
    orig_new, orig_listen, orig_barge = core._new_live, core._cmd_listen_impl, core._cmd_bargein_watch_impl
    made = []

    def fake_new(source, ctx, transcribe_fn=None, cadence_s=0.6):
        made.append(lw.LiveWords(source, path=p, transcribe_fn=transcribe_fn))
        return made[-1]

    def fake_listen(ctx, args, live):
        live.publish("hello the")
        live.final_text = "hello there"
        return {"text": "hello there", "status": "matched"}

    def fake_barge(ctx, args, live):
        live.publish("so")
        return "[bargein_watch] self_voice_dropped (iris 0.80)"

    core._new_live, core._cmd_listen_impl, core._cmd_bargein_watch_impl = fake_new, fake_listen, fake_barge
    try:
        assert core.cmd_listen(object(), {"structured": True}) == {"text": "hello there", "status": "matched"}
        d = json.loads(p.read_text())
        assert d["final"] and d["text"] == "hello there" and d["source"] == "room-listen"
        assert core.cmd_bargein_watch(object(), {}) == "[bargein_watch] self_voice_dropped (iris 0.80)"
        d = json.loads(p.read_text())
        assert d["final"] and d["text"] == "" and "self_voice_dropped" in d["dropped"]
    finally:
        core._new_live, core._cmd_listen_impl, core._cmd_bargein_watch_impl = orig_new, orig_listen, orig_barge
    assert core.cmd_live_partial(None, {})["source"] == "room-barge"


def test_live_whisper_text_never_loads_a_model() -> None:
    import wren_voice_core as core
    old = getattr(core.wl, "_WHISPER", None)
    core.wl._WHISPER = None
    try:
        assert core._live_whisper_text(np.zeros(16000, dtype=np.float32)) == ""
    finally:
        core.wl._WHISPER = old


def test_orb_endpoint_reads_both_files() -> None:
    import brain.orb_http as o
    room, vr = _tmpfile(), _tmpfile()
    lw.write_atomic(room, {"utterance_id": "room-listen-1", "text": "hey", "final": False})
    old = o._LIVE_ROOM_FILE, o._LIVE_VR_FILE
    o._LIVE_ROOM_FILE, o._LIVE_VR_FILE = room, vr                 # vr file missing -> null
    try:
        r = o.voice_live_partial()
        assert r["ok"] and r["room"]["text"] == "hey" and r["vr"] is None
        room.write_text("{not json", encoding="utf-8")
        assert o.voice_live_partial()["room"] is None             # garbage -> null, never 500
        room.write_text(json.dumps({"text": "x" * 70000}), encoding="utf-8")
        assert o.voice_live_partial()["room"] is None             # oversized -> null
        routes = {getattr(rt, "path", None): rt for rt in o.app.routes}
        assert routes["/api/v1/voice/live_partial"].methods == {"GET"}
    finally:
        o._LIVE_ROOM_FILE, o._LIVE_VR_FILE = old


def test_vr_hear_per_speaker_partials_then_final() -> None:
    import importlib.util
    import os
    import types
    os.environ["IRIS_VR_HEAR_LDPATH"] = "1"                       # don't let the module re-exec the test
    spec = importlib.util.spec_from_file_location("vr_hear_t", str(ROOT / "scripts" / "vr" / "vr_hear.py"))
    vh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vh)
    seg = types.SimpleNamespace(text=" hey Iris can you", no_speech_prob=0.0, avg_logprob=-0.1, start=0.0,
                                words=None)
    e = object.__new__(vh.Ears)                                   # skip model load
    e.model = types.SimpleNamespace(transcribe=lambda x, **k: ([seg], None))
    e.live_on, e.live, e.live_last = True, {}, {}
    e.live_path = _tmpfile()
    e.live_path.parent.mkdir(parents=True)
    x = (0.3 * np.sin(np.arange(16000) / 5.0)).astype(np.float32)
    e.buf = {"zeke": {"x": x, "meta": {"t_start": 1.0, "display_name": "Z"}, "t_end": 2.0, "n": 1, "stem": "s1"}}
    e.live_partial("zeke")
    d = json.loads(e.live_path.read_text())["zeke"]
    assert d["final"] is False and d["text"] == "hey Iris can you" and d["utterance_id"] == "vr-s1"
    e.live_partial("zeke")                                        # throttled: no second decode
    assert json.loads(e.live_path.read_text())["zeke"]["seq"] == 1
    e._live_set("zeke", e.buf["zeke"], "hey Iris can you hear me", final=True)
    e._live_set("zeke", e.buf["zeke"], "late", final=False)
    d = json.loads(e.live_path.read_text())["zeke"]
    assert d["final"] and d["text"] == "hey Iris can you hear me"




def test_tail_silence_measures_trailing_quiet() -> None:
    sr = 16000
    rng = np.random.default_rng(0)
    speech = (0.2 * rng.standard_normal(sr)).astype(np.float32)
    quiet = (0.001 * rng.standard_normal(int(0.8 * sr))).astype(np.float32)
    assert lw.tail_silence_s(speech) < 0.05
    assert abs(lw.tail_silence_s(np.concatenate([speech, quiet])) - 0.8) <= 0.07
    assert lw.tail_silence_s(np.zeros(10, np.float32)) == 0.0


def test_unchanged_text_republishes_when_silence_grows() -> None:
    p = _tmpfile()
    sr = 16000
    lwo = lw.LiveWords("t", path=p, transcribe_fn=lambda b: "what's it called", cadence_s=0.5)
    rng = np.random.default_rng(1)
    lwo.feed((0.2 * rng.standard_normal(sr)).astype(np.float32))
    time.sleep(0.8)
    first = json.loads(p.read_text(encoding="utf-8"))
    lwo.feed(np.zeros(int(0.7 * sr), np.float32))
    time.sleep(0.8)
    later = json.loads(p.read_text(encoding="utf-8"))
    lwo.stop()
    assert first["text"] == later["text"] == "what's it called"
    assert later["seq"] > first["seq"] and later["tail_silence_s"] >= 0.6 > first["tail_silence_s"]


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok", _n)
