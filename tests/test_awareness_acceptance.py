"""Acceptance tests - Heads-Down Awareness brief (Project 1), the parts that can run offline.

What these cover (each maps to a row in the private acceptance checklist):
  * a separate receiver keeps taking + durably queuing owner DMs while a turn works
  * the PostToolUse checkpoint hands waiting messages to the WORKING turn (several at once, in order)
  * handing a message over is not an ack: it stays unacked until the turn that got it ends
  * the look-up phrase is recognised, marked PRIORITY, and interrupts a turn that has no tool
    boundary for LOOKUP_INTERRUPT_S - but only for an authenticated sender
  * a restart re-delivers what was unhandled (durable inbox) and a corrupt inbox fails open

Nothing here touches the live host, Discord, or state/: the inbox path is a tmp file and the
network + transcript calls are stubbed. Never imports avaagent (a stub is registered).
"""
from __future__ import annotations

import asyncio
import json
import sys
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

OWNER = "100000000000000001"      # fake ids only - this repo is public
STRANGER = "100000000000000002"


class _Stop(BaseException):
    """Ends an infinite host loop from inside a stubbed sleep."""


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "DISCORD_INBOX_PATH", str(tmp_path / "discord_inbox.json"))
    monkeypatch.setattr(h, "DISCORD_OWNER_ID", OWNER)
    monkeypatch.setattr(h, "DISCORD_CHANNEL_ID", "200000000000000001")
    monkeypatch.setattr(h, "_blog", lambda *a, **k: None)
    monkeypatch.setattr(h, "transcript_append", lambda *a, **k: None)
    monkeypatch.setitem(h._DISCORD_INBOX, "last_id", None)
    monkeypatch.setitem(h._DISCORD_INBOX, "unacked", [])
    saved = dict(h._INBOX)
    h._INBOX.update({"queue": None, "turn": None, "client": None, "last_tool_ts": 0.0, "turn_ids": []})
    yield
    h._INBOX.clear()
    h._INBOX.update(saved)


def _msg(mid, content, author=OWNER, bot=False):
    import datetime as dt
    return {"id": str(mid), "content": content, "author": {"id": author, "bot": bot},
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}


def _turn(guest=False):
    return types.SimpleNamespace(active=True, guest=guest)


async def _queue_with(*items):
    q = asyncio.PriorityQueue()
    for it in items:
        await h._enqueue(q, it)
    return q


# ── receiver: keeps receiving + durably queuing while work continues ─────────────────────────
def test_poller_queues_owner_dms_durably_and_drops_everyone_else(monkeypatch):
    batches = [[_msg(105, "routine: how's it going?"), _msg(103, "", author=OWNER),
                _msg(104, "hi", author=STRANGER), _msg(106, "bot says", bot=True),
                {"id": "not-a-number"}, "garbage", _msg(107, "second routine")]]
    monkeypatch.setattr(h, "discord_get_messages", lambda token, after=None, limit=20:
                        batches.pop(0) if batches else [])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 2:
            raise _Stop()

    async def go():
        q = asyncio.PriorityQueue()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        with pytest.raises(_Stop):
            await h.discord_poller("tok", q, asyncio.get_running_loop(), baseline_id="100")
        return [q.get_nowait()[2] for _ in range(q.qsize())]

    got = asyncio.run(go())
    assert [g[0] for g in got] == ["discord", "discord"]
    assert [g[2] for g in got] == ["105", "107"], "owner text DMs only, oldest first"
    on_disk = json.loads(Path(h.DISCORD_INBOX_PATH).read_text())
    assert on_disk["last_id"] == "107", "cursor persisted past everything seen"
    assert [u["id"] for u in on_disk["unacked"]] == ["105", "107"], "queued == durably unacked"


# ── checkpoint: the working turn gets an actual opportunity to consider the message ──────────
def test_checkpoint_hands_routine_dm_to_working_turn_without_acking_it():
    async def go():
        q = await _queue_with(("discord", "routine midway message", "555"))
        h._DISCORD_INBOX["unacked"] = [{"id": "555", "content": "routine midway message", "ts": 1.0}]
        h._INBOX.update({"queue": q, "turn": _turn()})
        out = await h._inbox_post_tool_hook({"tool_name": "Bash"}, "t1", None)
        return q, out

    q, out = asyncio.run(go())
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert ctx.startswith("[INBOX CHECKPOINT") and "routine midway message" in ctx
    assert "message_id 555" in ctx and "★ PRIORITY" not in ctx
    assert q.qsize() == 0, "handed over = removed from the queue (no duplicate turn later)"
    assert h._INBOX["turn_ids"] == ["555"], "acked when THIS turn ends"
    assert [u["id"] for u in h._DISCORD_INBOX["unacked"]] == ["555"], \
        "a hand-over is not a read receipt: still unacked until the turn finishes"


def test_checkpoint_delivers_several_queued_messages_in_order_and_holds_guest_vr():
    async def go():
        q = await _queue_with(("discord", "first", "1"), ("vr", "guest talking", None, "Visitor"),
                              ("discord", "second", "2"), ("discord", "third", "3"))
        h._INBOX.update({"queue": q, "turn": _turn()})
        out = await h._inbox_post_tool_hook({"tool_name": "Read"}, "t", None)
        left = [q.get_nowait()[2] for _ in range(q.qsize())]
        return out, left

    out, left = asyncio.run(go())
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "3 message(s)" in ctx
    assert ctx.index("first") < ctx.index("second") < ctx.index("third")
    assert "guest talking" not in ctx, "guest VR speech never enters a full-tool turn"
    assert [x[0] for x in left] == ["vr"]
    assert h._INBOX["turn_ids"] == ["1", "2", "3"]


def test_no_checkpoint_outside_an_active_turn():
    async def go():
        q = await _queue_with(("discord", "while idle", "9"))
        h._INBOX.update({"queue": q, "turn": types.SimpleNamespace(active=False, guest=False)})
        out = await h._inbox_post_tool_hook({"tool_name": "Bash"}, "t", None)
        return out, q.qsize()

    out, n = asyncio.run(go())
    assert out == {} and n == 1, "idle: the main loop serves it as its own turn"


def test_checkpoint_hook_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("drain exploded")

    monkeypatch.setattr(h, "_inbox_drain", boom)
    h._INBOX["turn"] = _turn()
    assert asyncio.run(h._inbox_post_tool_hook({"tool_name": "Bash"}, "t", None)) == {}


# ── the owner's look-up phrase ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", ["Iris, look up for a moment", "iris look up", "IRIS -- look  up please",
                                  "hey Iris, look up"])
def test_lookup_phrase_matches(text):
    assert h._LOOKUP_RE.search(text)


@pytest.mark.parametrize("text", ["can you look it up", "Iris, looking up the docs now?", "iris lookup table",
                                  "look up, Iris"])
def test_lookup_phrase_does_not_fire_on_ordinary_text(text):
    assert not h._LOOKUP_RE.search(text)


def test_lookup_line_is_marked_priority_for_owner_channels_only():
    assert h._inbox_line("discord", "Iris, look up for a moment", "1", None, 0).startswith("★ PRIORITY")
    assert "PRIORITY" not in h._inbox_line("vr", "Iris, look up", None, "Visitor", 0)
    assert "PRIORITY" not in h._inbox_line("discord", "routine", "1", None, 0)


def _run_watcher(monkeypatch, items, last_tool_ts, interrupt_after_s, max_ticks=4):
    """Run lookup_watcher for a few 5 s ticks (stubbed) and return how often it interrupted."""
    calls = []

    class Client:
        async def interrupt(self):
            calls.append(1)

    ticks = []

    async def fake_sleep(s):
        ticks.append(s)
        if len(ticks) > max_ticks:
            raise _Stop()

    async def go():
        q = await _queue_with(*items)
        h._INBOX.update({"queue": q, "turn": _turn(), "client": Client(), "last_tool_ts": last_tool_ts})
        monkeypatch.setattr(h, "LOOKUP_INTERRUPT_S", interrupt_after_s)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        with pytest.raises(_Stop):
            await h.lookup_watcher()
        return q.qsize()

    left = asyncio.run(go())
    return calls, left


def test_lookup_interrupts_a_turn_stuck_in_one_long_tool_call(monkeypatch):
    calls, left = _run_watcher(monkeypatch, [("discord", "Iris, look up for a moment", "77")],
                               last_tool_ts=0.0, interrupt_after_s=0.0, max_ticks=1)
    assert calls == [1], "no tool boundary for LOOKUP_INTERRUPT_S => interrupt()"
    assert left == 1, "the watcher never consumes the message; the main loop serves it next"


def test_lookup_waits_while_tool_boundaries_keep_coming(monkeypatch):
    import time as _t
    calls, _ = _run_watcher(monkeypatch, [("discord", "Iris, look up for a moment", "77")],
                            last_tool_ts=_t.time() + 3600, interrupt_after_s=90.0)
    assert calls == [], "a recent tool boundary means the checkpoint will hand it over - no interrupt"


def test_routine_message_never_interrupts(monkeypatch):
    calls, _ = _run_watcher(monkeypatch, [("discord", "routine, whenever you get a sec", "78")],
                            last_tool_ts=0.0, interrupt_after_s=0.0)
    assert calls == []


def test_lookup_from_an_unverified_voice_never_interrupts(monkeypatch):
    meta = json.dumps({"status": "unknown", "speaker": None, "score": 0.1, "closest": "someone"})
    calls, _ = _run_watcher(monkeypatch, [("voice", "Iris, look up for a moment", None, meta)],
                            last_tool_ts=0.0, interrupt_after_s=0.0)
    assert calls == [], "authenticated owner only"


def test_lookup_from_a_voice_with_no_verdict_needs_require_match_off(monkeypatch):
    meta = json.dumps({"status": "skipped", "reason": "voice-id off"})
    monkeypatch.setattr(h, "VOICE_REQUIRE_MATCH", True)
    calls, _ = _run_watcher(monkeypatch, [("voice", "Iris, look up", None, meta)],
                            last_tool_ts=0.0, interrupt_after_s=0.0)
    assert calls == [], "strict mode: no voiceprint verdict = not the owner"


# ── restart: nothing handled-but-unacked is lost, nothing corrupt is fatal ───────────────────
def test_unacked_dms_survive_a_restart_and_ack_is_durable():
    h._DISCORD_INBOX["last_id"] = "500"
    h._DISCORD_INBOX["unacked"] = [{"id": "498", "content": "a", "ts": 1.0}, {"id": "499", "content": "b", "ts": 2.0}]
    h._discord_inbox_save()
    h._DISCORD_INBOX["last_id"], h._DISCORD_INBOX["unacked"] = None, []      # "process died"
    h._discord_inbox_load()
    assert h._DISCORD_INBOX["last_id"] == "500"
    assert [u["id"] for u in h._DISCORD_INBOX["unacked"]] == ["498", "499"]
    h._discord_ack(["498"])
    on_disk = json.loads(Path(h.DISCORD_INBOX_PATH).read_text())
    assert [u["id"] for u in on_disk["unacked"]] == ["499"]


def test_corrupt_inbox_file_fails_open():
    Path(h.DISCORD_INBOX_PATH).write_text('{"last_id": "5", "unack')
    h._discord_inbox_load()   # must not raise
    assert h._DISCORD_INBOX["unacked"] == []
