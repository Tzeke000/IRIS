"""2026-11-01 docaudit small fixes: chat-side deadline + abandoned status, voice latency stage stamps,
delegated tasks -> act verification ledger.

    /home/iris/venvs/iris-v100/bin/python -m pytest -q tests/test_smallfix_1101.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import iris_chat  # noqa: E402
from brain.iris_paths import paths as _paths  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_paths():
    """iris_chat.configure() re-points the GLOBAL brain.iris_paths root; put it back after each test
    so other test modules' bridges stay in their own temp dirs."""
    prev_root, prev_base = _paths._root, iris_chat._BASE
    yield
    iris_chat._BASE = prev_base
    if prev_root is not None:
        _paths.configure(prev_root)


def _chat_tmp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="iris_smallfix_chat_"))
    iris_chat.configure(d)
    return iris_chat._chat_dir()


# ---------------------------------------------------------------- 1. chat deadline + abandoned
def test_chat_submit_writes_deadline_default_is_flat_ttl() -> None:
    _chat_tmp()
    rid = iris_chat.submit("hi")
    d = iris_chat.get(rid)
    assert abs((d["deadline_ts"] - d["ts"]) - iris_chat._REQUEST_TTL_S) < 1e-6
    rid2 = iris_chat.submit("hi", deadline_s=30)
    d2 = iris_chat.get(rid2)
    assert abs((d2["deadline_ts"] - d2["ts"]) - 30.0) < 1e-6


def test_chat_past_deadline_expires_and_clears_flag() -> None:
    _chat_tmp()
    rid = iris_chat.submit("short", deadline_s=0.05)
    assert iris_chat.has_pending()
    time.sleep(0.1)
    assert iris_chat.list_pending() == []
    d = iris_chat.get(rid)
    assert d["status"] == "expired" and d["expired_reason"] == "deadline"
    assert not iris_chat.has_pending()


def test_chat_legacy_file_without_deadline_uses_flat_ttl() -> None:
    cdir = _chat_tmp()
    now = time.time()
    legacy = {"id": "legacy1", "ts": now - 30, "user_text": "old format", "status": "pending",
              "reply": None, "answered_ts": None}
    (cdir / "legacy1.json").write_text(json.dumps(legacy))
    assert [r["id"] for r in iris_chat.list_pending()] == ["legacy1"]
    legacy["ts"] = now - iris_chat._REQUEST_TTL_S - 5
    (cdir / "legacy1.json").write_text(json.dumps(legacy))
    assert iris_chat.list_pending() == []
    assert iris_chat.get("legacy1")["status"] == "expired"


def test_chat_deadline_never_extends_past_flat_ttl() -> None:
    _chat_tmp()
    rid = iris_chat.submit("long", deadline_s=10 * iris_chat._REQUEST_TTL_S)
    p = iris_chat._request_path(rid)
    d = json.loads(p.read_text())
    d["ts"] = time.time() - iris_chat._REQUEST_TTL_S - 5
    p.write_text(json.dumps(d))
    assert iris_chat.list_pending() == []


def test_chat_wait_abandons_only_when_asked_and_answer_wins() -> None:
    _chat_tmp()
    keep = iris_chat.submit("orb style")
    assert iris_chat.wait_for_reply(keep, timeout_s=0.3) is None
    assert iris_chat.is_pending(keep)                     # default: stays deliverable (orb)
    gone = iris_chat.submit("only waiter")
    assert iris_chat.wait_for_reply(gone, timeout_s=0.3, abandon_on_timeout=True) is None
    d = iris_chat.get(gone)
    assert d["status"] == "abandoned" and d["abandoned_reason"] == "waiter timed out"
    assert [r["id"] for r in iris_chat.list_pending()] == [keep]
    assert not iris_chat.is_pending(gone)                 # the host drops it at fire time
    assert iris_chat.mark_answered(keep, "late but answered")
    assert not iris_chat.abandon(keep)                    # an answer is never overwritten
    assert iris_chat.get(keep)["status"] == "answered"
    assert not iris_chat.has_pending()


def test_chat_wait_returns_early_on_abandoned() -> None:
    _chat_tmp()
    rid = iris_chat.submit("x")
    assert iris_chat.abandon(rid, "test")
    t0 = time.monotonic()
    assert iris_chat.wait_for_reply(rid, timeout_s=5.0) is None
    assert time.monotonic() - t0 < 1.0


def test_stop_hook_skips_chat_past_its_deadline() -> None:
    spec = importlib.util.spec_from_file_location("vsh_smallfix", ROOT / "scripts" / "voice_stop_hook.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cdir = _chat_tmp()
    m.CHAT_DIR = cdir
    live = iris_chat.submit("live")
    dead = iris_chat.submit("dead")
    p = iris_chat._request_path(dead)
    d = json.loads(p.read_text())
    d["ts"] -= 1.0                      # older => would be picked first (FIFO) if not skipped
    d["deadline_ts"] = time.time() - 1.0
    p.write_text(json.dumps(d))
    assert m._next_pending_chat()["id"] == live


# ---------------------------------------------------------------- 2. voice latency stage stamps
import asyncio  # noqa: E402
import importlib  # noqa: E402
import queue as _queue  # noqa: E402
import sqlite3  # noqa: E402
import types  # noqa: E402

if str(ROOT / "voice") not in sys.path:
    sys.path.insert(0, str(ROOT / "voice"))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

from brain import iris_events as ev  # noqa: E402

FISH_REPLY = "[voice_speak] [fish-s2pro] spoke 42 chars in 3.5s (first-audio 0.62s, rtf 0.40)"


def _core():
    return pytest.importorskip("wren_voice_core")


def test_core_first_audio_parsed_from_the_mouth_reply() -> None:
    core = _core()
    # t_done - total + fa : mouth started at 10.0, first sample 0.62 s later
    assert core._first_audio_ts(FISH_REPLY, 9.9, 13.5) == pytest.approx(10.62)
    assert core._first_audio_ts("[fish-s2pro] spoke 3 chars in 1.0s (first-audio Nones)", 0, 1) is None
    assert core._first_audio_ts("[voice_speak] warm voice server not reachable", 0, 1) is None


def test_core_play_one_records_stamp_only_for_stamped_items(monkeypatch) -> None:
    core = _core()
    monkeypatch.setattr(core, "_post_speak", lambda ctx, text, e=None, i=None: FISH_REPLY)
    ctx = types.SimpleNamespace(now_playing=None, played_chunks=[], bargein_fired=False)
    core.play_one(ctx, "plain legacy string")
    core.play_one(ctx, {"text": "no stamp", "emotion": None, "intensity": None})
    assert not getattr(ctx, "play_stamps", None)
    core.play_one(ctx, {"text": "turn one", "emotion": None, "intensity": None, "stamp": "host-turn:1@9"})
    core.play_one(ctx, {"text": "turn two", "emotion": "warm", "intensity": 0.5, "stamp": "host-turn:2@9"})
    allst = core.cmd_play_stamps(ctx, {})
    assert [r["stamp"] for r in allst] == ["host-turn:1@9", "host-turn:2@9"]
    one = core.cmd_play_stamps(ctx, {"stamp": "host-turn:2@9"})
    assert len(one) == 1 and one[0]["first_audio_ts"] is not None
    assert ctx.played_chunks[-1] == "turn two"


def test_core_cmd_speak_carries_stamp_and_keeps_old_shape(monkeypatch) -> None:
    core = _core()
    import wren_voice_status
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    monkeypatch.setattr(core, "_cancel_stall_bridge", lambda ctx: None)
    ctx = types.SimpleNamespace(play_queue=_queue.Queue(), speaking=False, bargein_fired=False)
    core.cmd_speak(ctx, {"text": "hello"})
    core.cmd_speak(ctx, {"text": "hi", "stamp": "host-turn:3@1"})
    assert ctx.play_queue.get_nowait() == "hello"                       # unstamped: legacy str
    item = ctx.play_queue.get_nowait()
    assert item["stamp"] == "host-turn:3@1" and core._item_fields(item)[0] == "hi"


def test_core_with_speaker_adds_stamps_only_when_structured() -> None:
    core = _core()
    timing = {"speech_end_ts": 100.0, "stt_done_ts": 100.4}
    verdict = {"speaker": "zeke", "status": "matched", "score": 0.9}
    assert core._with_speaker("hi", verdict, {}, timing) == "hi"
    out = core._with_speaker("hi", verdict, {"structured": True}, timing)
    assert out["speech_end_ts"] == 100.0 and out["stt_done_ts"] == 100.4 and out["speaker"] == "zeke"
    assert "speech_end_ts" not in core._with_speaker("hi", verdict, {"structured": True})


import iris_body_host as h  # noqa: E402


@pytest.fixture()
def host_env(tmp_path, monkeypatch):
    db = tmp_path / "v3" / "events.sqlite"
    monkeypatch.setenv("IRIS_EVENTS_DB", str(db))
    monkeypatch.setenv("IRIS_V3_EVENTS", "1")
    monkeypatch.setattr(h, "_blog", lambda *a, **k: None)
    h._EV.update(store=None, turn_n=0, owner=None, trace=None, turn_evs=[], first_speak_ts=None)
    h._EV_IDS.clear()
    yield types.SimpleNamespace(db=db)
    try:
        if isinstance(h._EV["store"], ev.EventStore):
            h._EV["store"].close()
    except Exception:
        pass
    h._EV.update(store=None, turn_n=0, owner=None, trace=None, turn_evs=[], first_speak_ts=None)
    h._EV_IDS.clear()


def _meta_of(db, text):
    c = sqlite3.connect(str(db))
    try:
        (mj,) = c.execute("SELECT meta_json FROM events WHERE text=?", (text,)).fetchone()
        return json.loads(mj or "{}")
    finally:
        c.close()


def test_host_voice_split_keeps_daemon_stamps() -> None:
    text, meta = h._voice_split({"text": "hi", "speaker": "zeke", "status": "matched",
                                 "speech_end_ts": 50.0, "stt_done_ts": 50.3})
    assert text == "hi" and meta["speech_end_ts"] == 50.0 and meta["stt_done_ts"] == 50.3
    _t, old = h._voice_split({"text": "hi", "speaker": "zeke", "status": "matched"})
    assert "speech_end_ts" not in old                       # older daemon: no keys invented


def test_host_voice_turn_records_all_four_stamps(host_env, monkeypatch) -> None:
    t_end = time.time() - 2.0
    meta = json.dumps({"speaker": "zeke", "status": "matched", "score": 0.9, "threshold": 0.4,
                       "speech_end_ts": t_end, "stt_done_ts": t_end + 0.4})
    calls = []
    fa = {"ts": 0.0}

    def fake_daemon(cmd, args=None, timeout=5.0):
        calls.append((cmd, dict(args or {})))
        if cmd == "play_stamps":
            return json.dumps({"ok": True, "result": [
                {"stamp": args["stamp"], "first_audio_ts": fa["ts"] + 0.3},
                {"stamp": args["stamp"], "first_audio_ts": fa["ts"]}]})
        return json.dumps({"ok": True, "result": "[queued]"})
    monkeypatch.setattr(h, "daemon_cmd", fake_daemon)

    class _SyncThread:   # run the post-turn fill inline so the test can read its result
        def __init__(self, target, args=(), **kw):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)
    monkeypatch.setattr(h.threading, "Thread", _SyncThread)

    async def run():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("voice", "what time is it", None, meta))
        entry = q.get_nowait()
        h._ev_turn_begin(h._ev_take(entry), "voice", None, meta)
    asyncio.run(run())
    owner = h._EV["owner"]
    stamp = h._ev_speak_stamp()
    assert stamp == owner and h._EV["first_speak_ts"] is not None
    first_speak = h._EV["first_speak_ts"]
    fa["ts"] = first_speak + 1.1
    h.speak("It is late.", None, None, stamp)
    assert h._ev_speak_stamp() == owner and h._EV["first_speak_ts"] == first_speak   # first one sticks
    h._ev_turn_end(True)
    assert ("speak", {"text": "It is late.", "stamp": owner}) in calls
    m = _meta_of(host_env.db, "what time is it")
    assert m["speech_end_ts"] == pytest.approx(t_end) and m["stt_done_ts"] == pytest.approx(t_end + 0.4)
    assert m["first_speak_ts"] == pytest.approx(first_speak)
    assert m["first_audio_ts"] == pytest.approx(first_speak + 1.1)          # the earliest chunk
    assert json.loads(m["item_sender"])["speaker"] == "zeke"               # speaker meta untouched
    assert h._EV["first_speak_ts"] is None


def test_host_first_audio_fill_old_daemon_records_first_speak_only(host_env, monkeypatch) -> None:
    st = h._ev_store()
    eid = st.insert("voice", "x", status="admitted", meta={"item_sender": "{}"}).id
    monkeypatch.setattr(h, "daemon_cmd", lambda *a, **k: json.dumps({"ok": False, "error": "unknown cmd"}))
    t0 = time.monotonic()
    h._ev_first_audio_fill([eid], "host-turn:1@1", 123.0, wait_s=30)
    assert time.monotonic() - t0 < 2.0                                       # gives up at once
    m = st.get(eid)["meta"]
    assert m["first_speak_ts"] == 123.0 and "first_audio_ts" not in m


def test_no_stamp_outside_a_turn_or_with_store_off(host_env, monkeypatch) -> None:
    assert h._ev_speak_stamp() is None                     # no turn open
    monkeypatch.delenv("IRIS_V3_EVENTS")
    h._EV["turn_evs"] = [{"id": "x"}]
    h._EV["owner"] = "host-turn:1@1"
    assert h._ev_speak_stamp() is None                     # flag off


def test_event_store_merge_meta(tmp_path) -> None:
    st = ev.EventStore(tmp_path / "e.sqlite")
    eid = st.insert("voice", "x", status="admitted", meta={"a": 1}).id
    assert st.merge_meta(eid, {"b": 2})
    assert st.merge_meta(eid, {"a": 3})
    assert st.get(eid)["meta"] == {"a": 3, "b": 2}
    assert st.get(eid)["status"] == "admitted"
    assert not st.merge_meta("ev_missing", {"b": 1})
    assert not st.merge_meta(eid, {})
    st.close()


def test_latency_report_voice_stages(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("lr_smallfix", ROOT / "scripts" / "server" / "latency_report.py")
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)
    now = [1_800_014_400.0]
    st = ev.EventStore(tmp_path / "ev.sqlite", clock=lambda: now[0])
    base = now[0]
    for i in range(4):
        se = base + i * 60
        st.insert("voice", f"v{i}", status="admitted", received_ts=se + 0.5,
                  meta={"speech_end_ts": se, "stt_done_ts": se + 0.4,
                        "first_speak_ts": se + 3.0, "first_audio_ts": se + 4.0 + i})
    st.insert("voice", "older daemon", status="admitted", received_ts=base + 1000)
    st.insert("discord", "d", status="admitted", received_ts=base + 1001)
    rep = lr.build(st._conn, days=1, now=base + 3000)
    (day,) = rep["events"].keys()
    v = rep["events"][day]["voice"]["stages"]
    assert v["speech_end->stt_done"]["n"] == 4 and v["speech_end->stt_done"]["p50"] == pytest.approx(0.4)
    assert v["first_speak->first_audio"]["n"] == 4 and v["first_speak->first_audio"]["max"] == pytest.approx(4.0)
    assert v["speech_end->first_audio"]["max"] == pytest.approx(7.0)
    assert "speech_end->stt_done" not in rep["events"][day]["discord"]["stages"]   # shown only where recorded
    assert "first_speak->first_audio" in lr.render_text(dict(rep, db="x"))
    st.close()
