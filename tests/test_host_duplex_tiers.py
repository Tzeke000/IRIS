"""Host side of the full-duplex tiers (IRIS_DUPLEX_TIERS, default OFF).

- off: bargein_watch is asked exactly as before ({"structured": true}) and only '[barge-in]' counts
- on: the host asks for tiers, a '[barge-defer]' becomes a voice turn that does NOT interrupt mine,
  an '[bargein_watch] ack: ...' never becomes a turn from the watch itself - since 11-04 it is buffered
  and delivered after my utterance (tests/test_every_yes_1104.py)
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

import iris_body_host as h  # noqa: E402

h._blog = lambda *a, **k: None  # type: ignore[assignment]


def _reply(text):
    return json.dumps({"ok": True, "result": {"text": text, "status": "skipped", "speaker": None}})


def _watch(monkeypatch, text, tiers):
    sent = {}

    def fake(cmd, args=None, timeout=0):
        sent["args"] = dict(args or {})
        return _reply(text)
    monkeypatch.setattr(h, "daemon_cmd", fake)
    monkeypatch.setattr(h, "DUPLEX_TIERS", tiers)
    return h._daemon_bargein_watch_once(), sent["args"]


def test_off_by_default_asks_exactly_as_before(monkeypatch):
    import importlib
    monkeypatch.delenv("IRIS_DUPLEX_TIERS", raising=False)
    assert importlib.reload(h).DUPLEX_TIERS is False
    h._blog = lambda *a, **k: None
    r, args = _watch(monkeypatch, "[barge-in] [cut] stop", False)
    assert args == {"structured": True} and r[0].startswith("[barge-in]")


def test_off_ignores_new_prefixes(monkeypatch):
    r, _ = _watch(monkeypatch, "[barge-defer] [talked over me] hi", False)
    assert r is None


def test_on_asks_for_tiers_and_returns_defer(monkeypatch):
    r, args = _watch(monkeypatch, "[barge-defer] [talked over me; my reply kept playing] hello", True)
    assert args == {"structured": True, "tiers": True}
    assert r[0].startswith("[barge-defer]")


def test_ack_is_never_a_turn(monkeypatch):
    h._PENDING_ACKS.clear()
    h._RECENT_SPOKEN.clear()
    r, _ = _watch(monkeypatch, "[bargein_watch] ack: Mm-hmm.", True)
    assert r is None
    assert [a[1] for a in h._PENDING_ACKS] == ["Mm-hmm."]    # buffered for after my utterance (11-04)
    h._PENDING_ACKS.clear()


class _Turn:
    active = True
    speak_out = True


class _Client:
    def __init__(self):
        self.interrupts = 0

    async def interrupt(self):
        self.interrupts += 1


def _run_reader_once(monkeypatch, barge_text):
    """Drive voice_reader through ONE barge window and report (queued items, interrupts)."""
    q: asyncio.Queue = asyncio.Queue()
    client = _Client()
    h._PENDING_ACKS.clear()
    monkeypatch.setattr(h, "DUPLEX_TIERS", True)
    monkeypatch.setattr(h, "BARGEIN_ON", True)
    monkeypatch.setattr(h, "BARGEIN_STOPS_TURN", True)
    monkeypatch.setattr(h, "_voice_flag_says_off", lambda p: False)
    calls = {"n": 0}

    def fake_daemon(cmd, args=None, timeout=0):
        if cmd == "bargein_watch":
            calls["n"] += 1
            if calls["n"] == 1:
                return _reply(barge_text)
            raise SystemExit   # end the reader after one window
        return json.dumps({"ok": True, "result": {}})
    monkeypatch.setattr(h, "daemon_cmd", fake_daemon)
    h._INBOX["turn"], h._INBOX["client"] = _Turn(), client

    async def main():
        gate = asyncio.Event()           # cleared = a speaking turn is in flight -> barge watch
        loop = asyncio.get_running_loop()
        try:
            await h.voice_reader(q, loop, gate)
        except SystemExit:
            pass
    try:
        asyncio.run(main())
    finally:
        h._INBOX["turn"], h._INBOX["client"] = None, None
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items, client.interrupts


def test_reader_defer_queues_without_interrupt(monkeypatch):
    items, n_int = _run_reader_once(monkeypatch, "[barge-defer] [talked over me] are you the robot")
    assert n_int == 0
    assert len(items) == 1 and "[barge-defer] [talked over me] are you the robot" in repr(items[0])


def test_reader_barge_still_interrupts(monkeypatch):
    items, n_int = _run_reader_once(monkeypatch, "[barge-in] [cut mid-sentence] stop")
    assert n_int == 1 and len(items) == 1


def test_question_flag_rides_to_the_daemon_and_clears_once_answered(monkeypatch):
    import time as _t
    h._RECENT_SPOKEN.clear()
    h._RECENT_SPOKEN.append((_t.time(), "Is it bigger than a breadbox?"))
    r, args = _watch(monkeypatch, "[barge-defer] [said while I was still talking, right after I asked "
                                  "a question - treat as his answer] No.", True)
    assert args.get("question") is True and r[0].endswith("No.")
    assert not h._asked_a_question_recently()       # answered: a later "mm-hm" is an ack again
    _r2, args2 = _watch(monkeypatch, "[bargein_watch] no_barge", True)
    assert "question" not in args2
