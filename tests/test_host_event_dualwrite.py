"""V3 Phase 1 step 1.2: the body host's dual-write into the durable event store (IRIS_V3_EVENTS=1).

Covers: flag off => no store writes at all; flag on => admitted -> assigned -> output_pending(+reply_ref)
-> committed; crash AFTER a reply => the boot replay labels it "already answered" (P1-T3e); crash BEFORE a
reply => still "[unhandled before my restart]"; voice/VR re-offer < 10 min with age, older cancelled
(P1-R1b); a store that raises never touches delivery; the helper file-lease hook is log-only; the stream
heartbeat file."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

import iris_body_host as h  # noqa: E402
from brain import iris_events as ev  # noqa: E402

DM = str(h.DISCORD_CHANNEL_ID or "")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "v3" / "events.sqlite"
    monkeypatch.setenv("IRIS_EVENTS_DB", str(db))
    monkeypatch.setenv("IRIS_V3_EVENTS", "1")
    logs = []
    monkeypatch.setattr(h, "_blog", lambda *a, **k: logs.append(a))
    monkeypatch.setattr(h, "HOST_STREAM_PATH", str(tmp_path / "host_stream.json"))
    monkeypatch.setattr(h, "_orb_claim", lambda mid: True)
    h._EV.update(store=None, turn_n=0, owner=None, trace=None, turn_evs=[])
    h._EV_IDS.clear()
    h._INBOX.update(queue=None, turn=None, turn_ids=[])
    ns = types.SimpleNamespace(db=db, logs=logs, tmp=tmp_path)
    yield ns
    try:
        if isinstance(h._EV["store"], ev.EventStore):
            h._EV["store"].close()
    except Exception:
        pass
    h._EV.update(store=None, turn_n=0, owner=None, trace=None, turn_evs=[])
    h._EV_IDS.clear()
    h._INBOX.update(queue=None, turn=None, turn_ids=[])


def _rows(db):
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    try:
        return {r["idempotency_key"]: dict(r) for r in c.execute("SELECT * FROM events")}
    finally:
        c.close()


def _new_process():
    """Simulate a host restart: drop the in-memory store handle + queue maps (the DB file survives)."""
    if h._EV["store"] is not None:
        h._EV["store"].close()
    h._EV.update(store=None, turn_n=0, owner=None, trace=None, turn_evs=[])
    h._EV_IDS.clear()


def _discord_reply(msg_id, sent_id, chat=DM):
    return {"tool_name": "mcp__plugin_discord_discord__reply",
            "tool_input": {"chat_id": chat, "text": "hi", "reply_to": str(msg_id)},
            "tool_response": [{"type": "text", "text": "sent (id: " + str(sent_id) + ")"}]}


def _turn_from_queue(q):
    """What the main loop does for one entry: take its event ids, begin the turn."""
    entry = q.get_nowait()
    entry = h._vr_coalesce(q, entry)
    eids = h._ev_take(entry)
    item = entry[2]
    h._ev_turn_begin(eids, item[0], item[2], item[3] if len(item) > 3 else None)
    turn = types.SimpleNamespace(active=True, guest=False)
    h._INBOX.update(queue=q, turn=turn)
    return entry, eids


# ── flag off ────────────────────────────────────────────────────────────────
def test_flag_off_writes_nothing(env, monkeypatch):
    monkeypatch.delenv("IRIS_V3_EVENTS")

    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "hello", "111"))
        await h._enqueue(q, ("voice", "hi iris", None, json.dumps({"status": "matched"})))
        entry = q.get_nowait()
        assert entry[2] == ("voice", "hi iris", None, json.dumps({"status": "matched"})) or entry[2][0] == "discord"
        assert len(entry) == 4 and abs(entry[3] - time.time()) < 5
        h._ev_turn_begin(h._ev_take(entry), "discord", "111", None)
        h._INBOX["turn"] = types.SimpleNamespace(active=True, guest=False)
        h._ev_mark_reply(_discord_reply("111", "999"))
        h._ev_turn_end(True)
        await h._ev_boot_replay(q, time.time() + 10, [], None)
        h._host_stream_beat(h._INBOX["turn"], force=True)
    asyncio.run(go())
    assert not env.db.exists(), "flag off must not even create the DB"
    assert h._EV["store"] is None and not h._EV_IDS
    assert h._ev_discord_replay_prefix("111") == "[unhandled before my restart] "
    assert not os.path.exists(h.HOST_STREAM_PATH)
    assert h._tr({"a": 1}) == {"a": 1}
    assert not any(a[0] == "turn" for a in env.logs)
    hooks = h._inbox_hooks()
    if hooks is not None:
        assert len(hooks["PreToolUse"]) == 1, "no lease hook registered with the flag off"


# ── flag on: the full lifecycle ─────────────────────────────────────────────
def test_lifecycle_enqueue_assigned_replied_committed(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "hello", "111"))
        r = _rows(env.db)["discord:111"]
        assert r["status"] == "admitted" and r["trusted"] == 1 and r["priority"] == h._SOURCE_PRIORITY["discord"]
        assert r["channel_msg_id"] == "111"
        _turn_from_queue(q)
        r = _rows(env.db)["discord:111"]
        assert r["status"] == "assigned" and r["owner"].startswith("host-turn:1@")
        # an orb message arrives mid-turn and is handed over at a tool boundary
        await h._enqueue(q, ("orb", "and this", "orb-7"))
        out = await h._inbox_post_tool_hook({"tool_name": "Bash"}, "t1", None)
        assert "INBOX CHECKPOINT" in out["hookSpecificOutput"]["additionalContext"]
        r2 = _rows(env.db)["orb:orb-7"]
        assert r2["status"] == "assigned" and r2["owner"] == r["owner"]
        # the discord reply goes out -> output_pending + reply_ref (NOT committed: parity with the JSON ack)
        await h._inbox_post_tool_hook(_discord_reply("111", "999"), "t2", None)
        r = _rows(env.db)["discord:111"]
        assert r["status"] == "output_pending" and r["reply_ref"] == "discord:999"
        assert _rows(env.db)["orb:orb-7"]["reply_ref"] is None, "a discord reply doesn't answer the orb"
        await h._inbox_post_tool_hook({"tool_name": "mcp__iris__chat_reply",
                                       "tool_input": {"request_id": "orb-7", "text": "yes"},
                                       "tool_response": [{"type": "text", "text": '{"ok": true}'}]}, "t3", None)
        assert _rows(env.db)["orb:orb-7"]["reply_ref"] == "orb:orb-7"
        h._ev_turn_end(True)
        rows = _rows(env.db)
        assert rows["discord:111"]["status"] == "committed" and rows["orb:orb-7"]["status"] == "committed"
        # trace: the turn's log lines carry the root event id
        starts = [a for a in env.logs if a[0] == "turn" and a[1] == "start"]
        assert starts and starts[0][2]["trace"] == rows["discord:111"]["event_id"]
        cps = [a for a in env.logs if a[0] == "inbox" and a[1] == "checkpoint delivered"]
        assert cps and cps[0][2]["trace"] == rows["discord:111"]["event_id"]
    asyncio.run(go())


def test_failed_reply_does_not_mark(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "hello", "222"))
        _turn_from_queue(q)
        bad = _discord_reply("222", "x")
        bad["tool_response"] = [{"type": "text", "text": "reply failed: 50013 missing permissions"}]
        h._ev_mark_reply(bad)
        assert _rows(env.db)["discord:222"]["reply_ref"] is None
        h._ev_mark_reply({"tool_name": "mcp__iris__iris_tool_call",
                          "tool_input": {"name": "voice_speak", "params": {"text": "x"}},
                          "tool_response": '{"ok": false, "error": "mouth down"}'})
        assert _rows(env.db)["discord:222"]["reply_ref"] is None
    asyncio.run(go())


# ── P1-T3e: crash after vs before the reply ─────────────────────────────────
def test_crash_after_reply_boot_does_not_reanswer_blind(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "what time is it", "333"))
        _turn_from_queue(q)
        h._ev_mark_reply(_discord_reply("333", "4444"))
        # CRASH: no turn end, discord_inbox.json still lists 333 as unacked
    asyncio.run(go())
    _new_process()
    pre = h._ev_discord_replay_prefix("333")
    assert pre.startswith("[I already answered this before the restart: discord:4444")
    assert "unhandled" not in pre

    async def boot():
        q = asyncio.PriorityQueue()
        start = time.time() + 1
        # what main() does: re-queue the JSON unacked item with the chosen prefix, then the store replay
        await h._enqueue(q, ("discord", pre + "what time is it", "333"))
        await h._ev_boot_replay(q, start, ["333"], "333")
        items = []
        while not q.empty():
            items.append(q.get_nowait()[2])
        return items
    items = asyncio.run(boot())
    assert len(items) == 1, "exactly one re-queue (from discord_inbox.json), none from the store"
    assert items[0][1].startswith("[I already answered this before the restart")
    assert len(_rows(env.db)) == 1, "the replay re-used the original row (idempotency key)"


def test_crash_before_reply_still_requeued_unhandled(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "are you there", "555"))
        _turn_from_queue(q)
        # CRASH before any reply tool ran
    asyncio.run(go())
    _new_process()
    assert h._ev_discord_replay_prefix("555") == "[unhandled before my restart] "
    assert h._ev_discord_replay_prefix("never-seen") == "[unhandled before my restart] "


def test_boot_reconciles_discord_rows_the_json_already_acked(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "old", "100"))
        await h._enqueue(q, ("discord", "past cursor", "900"))
    asyncio.run(go())
    _new_process()
    asyncio.run(h._ev_boot_replay(asyncio.PriorityQueue(), time.time() + 1, [], "500"))
    rows = _rows(env.db)
    assert rows["discord:100"]["status"] == "committed", "acked in JSON + behind the cursor -> committed"
    assert rows["discord:900"]["status"] == "admitted", "past the cursor -> left for the poller"


# ── P1-R1b: voice / VR / letters ────────────────────────────────────────────
def _old_store(env, age_s):
    t0 = time.time() - age_s
    h._EV["store"] = ev.EventStore(env.db, clock=lambda: t0, busy_timeout_s=2)


def test_voice_older_than_10_min_cancelled_fresh_reoffered_with_age(env):
    meta = json.dumps({"speaker": "zeke", "status": "matched", "score": 0.9, "threshold": 0.4})

    async def seed():
        q = asyncio.PriorityQueue()
        _old_store(env, 700)
        await h._enqueue(q, ("voice", "stale words", None, meta))
        _new_process()
        _old_store(env, 120)
        await h._enqueue(q, ("voice", "fresh words", None, meta))
        await h._enqueue(q, ("vr", "[VR TEMPLE header]\nZeke: hi iris", "vr-1", "Zeke"))
        await h._enqueue(q, ("letter", "dear iris", "L-1", "wren"))
        await h._enqueue(q, ("orb", "orb thing", "orb-1"))
        _new_process()
        _old_store(env, 60)
        await h._enqueue(q, ("voice", "already answered", None, meta))
        e = q.get_nowait()   # voice band first
        while e[2][1] != "already answered":
            e = q.get_nowait()
        h._ev_turn_begin(h._ev_take(e), "voice", None, meta)
        h._INBOX["turn"] = types.SimpleNamespace(active=True, guest=False)
        h._ev_mark_reply({"tool_name": "mcp__iris__voice_speak", "tool_input": {"text": "sure"},
                          "tool_response": [{"type": "text", "text": '{"ok": true, "engine": "fish"}'}]})
        # crash
    asyncio.run(seed())
    _new_process()

    async def boot():
        q = asyncio.PriorityQueue()
        await h._ev_boot_replay(q, time.time(), [], None)
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out
    out = asyncio.run(boot())
    rows = {r["text"]: r for r in _rows(env.db).values()}
    assert rows["stale words"]["status"] == "cancelled" and "never replayed as live" in rows["stale words"]["note"]
    assert rows["already answered"]["status"] == "committed", "answered aloud -> not re-offered"
    assert rows["orb thing"]["status"] == "cancelled"
    by_src = {e[2][0]: e for e in out}
    assert set(by_src) == {"voice", "vr", "letter"}
    v = by_src["voice"]
    assert v[2][1].startswith("[said 2 min ago, BEFORE my restart - not live")
    assert v[2][3] == meta, "speaker meta survives the replay (the trust gate still applies)"
    assert 110 < time.time() - v[3] < 200, "the entry carries the ORIGINAL capture time for the stale marker"
    vr = by_src["vr"]
    assert vr[2][1].startswith("[VR TEMPLE header]\n[said 2 min ago") and vr[2][2] == "vr-1" and vr[2][3] == "Zeke"
    assert by_src["letter"][2][1].startswith("[arrived 2 min ago, unhandled before my restart] dear iris")
    # the re-offer re-uses the original rows (no duplicates) and they flow on normally
    assert len(_rows(env.db)) == 6
    eids = h._ev_take(v)
    assert eids == [rows["fresh words"]["event_id"]]


def test_vr_coalesce_carries_all_event_ids(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("vr", "[VR]\nZeke: one", "a", "Zeke"))
        await h._enqueue(q, ("vr", "[VR]\nZeke: two", "b", "Zeke"))
        _turn_from_queue(q)
        h._ev_turn_end(True)
        assert {r["status"] for r in _rows(env.db).values()} == {"committed"}
    asyncio.run(go())


# ── fail open ───────────────────────────────────────────────────────────────
class _Boom:
    last_error = "boom"

    def __getattr__(self, name):
        def f(*a, **k):
            raise RuntimeError("store exploded: " + name)
        return f


def test_store_raising_never_breaks_delivery(env):
    h._EV["store"] = _Boom()

    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "still delivered", "777"))
        await h._enqueue(q, ("voice", "me too", None, "{}"))
        assert q.qsize() == 2
        entry, eids = _turn_from_queue(q)
        assert entry[2][0] == "voice" and eids == []
        await h._enqueue(q, ("discord", "mid-turn", "778"))
        out = await h._inbox_post_tool_hook(_discord_reply("778", "1"), "t", None)
        assert "mid-turn" in out["hookSpecificOutput"]["additionalContext"]
        h._ev_turn_end(True)
        assert h._ev_discord_replay_prefix("777") == "[unhandled before my restart] "
        await h._ev_boot_replay(q, time.time(), [], None)
        assert await h._lease_pre_tool_hook({"agent_id": "a1", "tool_name": "Write",
                                             "tool_input": {"file_path": "/x"}}, "t", None) == {}
    asyncio.run(go())


def test_corrupt_db_file_delivery_unaffected(env):
    env.db.parent.mkdir(parents=True, exist_ok=True)
    env.db.write_bytes(b"this is not a database" * 100)

    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "after corruption", "888"))
        assert q.qsize() == 1 and q.get_nowait()[2][1] == "after corruption"
    asyncio.run(go())
    assert any(p.name.startswith("events.sqlite.corrupt-") for p in env.db.parent.iterdir())


# ── leases (log-only) + heartbeat ───────────────────────────────────────────
def test_lease_hook_logs_conflicts_never_denies(env, tmp_path):
    st = ev.EventStore(env.db, lease_root=tmp_path)
    tid_other = st.create_task("someone else's job", kind="t").id
    f = st.claim(tid_other, "worker-x")
    assert st.acquire_leases(tid_other, [str(tmp_path / "brain" / "*.py")], fence=f)["ok"]
    st.close()

    async def go():
        out = await h._lease_pre_tool_hook({"agent_id": "a1", "tool_name": "Edit",
                                            "tool_input": {"file_path": str(tmp_path / "brain" / "x.py")}}, "t", None)
        assert out == {}
        out = await h._lease_pre_tool_hook({"tool_name": "Edit",   # main thread: not checked at all
                                            "tool_input": {"file_path": str(tmp_path / "brain" / "x.py")}}, "t", None)
        assert out == {}
    h._EV["store"] = ev.EventStore(env.db, lease_root=tmp_path)
    asyncio.run(go())
    hits = [a for a in env.logs if a[0] == "lease"]
    assert len(hits) == 1 and hits[0][2]["held_by"] == [tid_other]


def test_stream_heartbeat_atomic_and_throttled(env):
    turn = types.SimpleNamespace(active=True)
    h._STREAM.pop("hb_ts", None)
    h._STREAM["turn_started_ts"] = 123.0
    h._host_stream_beat(turn)
    d = json.loads(Path(h.HOST_STREAM_PATH).read_text())
    assert d["pid"] == os.getpid() and d["turn_active"] is True and d["turn_started_ts"] == 123.0
    first = d["ts"]
    turn.active = False
    h._host_stream_beat(turn)                       # throttled (< 2 s)
    assert json.loads(Path(h.HOST_STREAM_PATH).read_text())["ts"] == first
    h._host_stream_beat(turn, force=True)
    assert json.loads(Path(h.HOST_STREAM_PATH).read_text())["turn_active"] is False
    h._STREAM["turn_started_ts"] = None


# ── P1-H1: the host stubs a checkpoint before it interrupts a working turn ───────────────────
def test_interrupt_stub_records_where_the_turn_stood(env):
    async def go():
        q = asyncio.PriorityQueue()
        await h._enqueue(q, ("discord", "build the thing", "901"))
        _, eids = _turn_from_queue(q)
        h._INBOX["last_tool_ts"] = time.time() - 120
        h._INBOX["last_tool_name"] = "Bash"
        return eids
    eids = asyncio.run(go())
    tid = h._ev_interrupt_stub("look-up phrase waited 90 s with no tool boundary")
    assert tid
    cp = h._ev_store().read_checkpoint(tid)
    assert cp["auto_stub"] is True and cp["events"] == eids and cp["last_tool"] == "Bash"
    assert 110 <= cp["last_tool_age_s"] <= 130 and "look-up" in cp["reason"]
    assert h._ev_store().get_task(tid)["kind"] == "interrupted_turn"
    pre = h._ev_interrupt_note_prefix()
    assert tid in pre and "task_checkpoint" in pre
    assert h._ev_interrupt_note_prefix() == "", "one-shot"
    assert h._ev_interrupt_stub("again") == tid, "a double fire in the same minute reuses the stub"
    h._INBOX.pop("interrupt_note", None)


def test_interrupt_stub_flag_off_is_a_noop(env, monkeypatch):
    monkeypatch.setenv("IRIS_V3_EVENTS", "0")
    h._INBOX.pop("interrupt_note", None)
    assert h._ev_interrupt_stub("x") is None
    assert h._ev_interrupt_note_prefix() == ""
