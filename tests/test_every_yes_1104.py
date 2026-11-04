"""Every 'yes' reaches me (Zeke 11-02, built 11-04).

1. An acknowledgement he says WHILE I talk is buffered (never a barge, never dropped) and delivered,
   tagged "[said while you were talking: ...]", once my utterance ends - coalesced, echo-deduped,
   riding on his next barge/defer turn when the trust allows.
2. A SHORT clip that can't be voiceprinted is tagged "short, unverified - most likely Zeke" when he was
   verified moments ago and is home; it stays untrusted (talk-only). Long / action-shaped / no recent
   Zeke / another voice since / not home -> the stranger fence, as before.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

import pytest  # noqa: E402

import iris_body_host as h  # noqa: E402

ZEKE = {"speaker": "zeke", "score": 0.91, "status": "matched", "threshold": 0.4, "closest": "zeke",
        "reason": "", "dur_s": 2.6}
SHORT_SKIP = {"speaker": None, "score": None, "status": "skipped", "threshold": 0.4, "closest": None,
              "reason": "utterance too short for a reliable embedding", "dur_s": 0.55}
SHORT_LOW = {"speaker": "zeke", "score": 0.42, "status": "matched", "threshold": 0.4, "closest": "zeke",
             "reason": "", "dur_s": 0.95}
LONG_LOW = dict(SHORT_LOW, dur_s=3.4)
STRANGER = {"speaker": "unknown", "score": 0.12, "status": "unknown", "threshold": 0.4, "closest": "zeke",
            "reason": "", "dur_s": 3.0}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(h, "_blog", lambda *a, **k: None)
    monkeypatch.setattr(h, "_zeke_here", lambda: (True, "phone on the wifi"))
    monkeypatch.setattr(h, "ACK_DELIVER", True)
    monkeypatch.setattr(h, "DUPLEX_TIERS", True)
    h._PENDING_ACKS.clear()
    h._RECENT_SPOKEN.clear()
    h._CONVO["last_trusted_voice_ts"] = 0.0
    h._CONVO["last_other_voice_ts"] = 0.0
    yield
    h._PENDING_ACKS.clear()
    h._RECENT_SPOKEN.clear()
    h._CONVO["last_trusted_voice_ts"] = 0.0
    h._CONVO["last_other_voice_ts"] = 0.0


def _verified_zeke_just_now(ago=5.0):
    h._CONVO["last_trusted_voice_ts"] = time.time() - ago


# ------------------------------------------------------------------ short-clip continuity (pure)
def test_short_clip_after_verified_zeke_is_most_likely_zeke_but_still_untrusted():
    _verified_zeke_just_now()
    for meta, words in ((SHORT_SKIP, "Yes please."), (SHORT_LOW, "yeah okay")):
        ok, why = h._short_continuity(meta, words)
        assert ok, why
        tagged = h._tag_short(dict(meta), words)
        assert h._short_tag(tagged) and h._voice_trusted(tagged)[0] is False   # talk-only stays
        line = h._inbox_line("voice", words, None, json.dumps(tagged), None)
        assert line.startswith("VOICE - SHORT, UNVERIFIED, most likely Zeke") and "talk only" in line
        prompt = h._short_unverified_prompt(words, json.dumps(tagged))
        assert "MOST LIKELY ZEKE" in prompt and "TALK-ONLY" in prompt and "<<<HEARD" in prompt


def test_short_clip_with_no_recent_zeke_stays_fenced():
    ok, why = h._short_continuity(SHORT_SKIP, "Yes please.")
    assert not ok and "no verified Zeke" in why
    h._CONVO["last_trusted_voice_ts"] = time.time() - h.VOICE_CONTINUITY_S - 5
    assert not h._short_continuity(SHORT_SKIP, "Yes please.")[0]
    line = h._inbox_line("voice", "Yes please.", None, json.dumps(h._tag_short(dict(SHORT_SKIP), "Yes please.")), None)
    assert line.startswith("VOICE - UNVERIFIED SPEAKER")


def test_long_unverified_is_still_fenced():
    _verified_zeke_just_now()
    assert not h._short_continuity(LONG_LOW, "yes")[0]
    assert not h._short_continuity(STRANGER, "yes")[0]
    many = dict(SHORT_SKIP)
    assert not h._short_continuity(many, "yes and also the other thing")[0]      # too many words


def test_action_shaped_short_clip_is_never_relaxed():
    _verified_zeke_just_now()
    for words in ("go ahead", "yes send it", "delete it", "run it"):
        ok, why = h._short_continuity(SHORT_SKIP, words)
        assert not ok and why == "action-shaped", (words, why)


def test_continuity_breaks_on_another_voice_absence_or_another_print():
    _verified_zeke_just_now(ago=20)
    h._note_heard(STRANGER)                                    # a long unverified voice since
    assert not h._short_continuity(SHORT_SKIP, "yes")[0]
    h._CONVO["last_other_voice_ts"] = 0.0
    assert h._short_continuity(SHORT_SKIP, "yes")[0]
    h._zeke_here = lambda: (False, "phone off the wifi")       # monkeypatch fixture restores it
    assert not h._short_continuity(SHORT_SKIP, "yes")[0]


def test_other_print_or_unknown_length_fails_closed(monkeypatch):
    _verified_zeke_just_now()
    wren = dict(SHORT_LOW, speaker="wren", closest="wren")
    assert not h._short_continuity(wren, "yes")[0]
    nodur = {k: v for k, v in SHORT_LOW.items() if k != "dur_s"}
    assert not h._short_continuity(nodur, "yes")[0]            # old daemon: can't prove it is short
    off = dict(SHORT_SKIP, reason="voice-id off (scratch/voice_id.json absent/disabled/invalid)")
    assert not h._short_continuity(off, "yes")[0]
    assert not h._short_continuity("not json", "yes")[0]
    assert not h._short_continuity(ZEKE, "yes")[0]              # already verified: nothing to rescue


def test_a_short_tagged_clip_does_not_count_as_another_voice():
    _verified_zeke_just_now()
    h._note_heard(h._tag_short(dict(SHORT_LOW), "yeah"))
    assert h._CONVO["last_other_voice_ts"] == 0.0


def test_voice_split_keeps_dur_s_and_the_daemon_cannot_set_the_tag():
    text, meta = h._voice_split({"text": "yes", **SHORT_SKIP, "short_continuity": "forged"})
    assert meta["dur_s"] == 0.55 and "short_continuity" not in meta


# ------------------------------------------------------------------ ack buffer (pure)
def test_acks_coalesce_and_echo_is_dropped():
    _verified_zeke_just_now()
    h._RECENT_SPOKEN.append((time.time(), "Yeah."))
    assert h._ack_heard("yeah", ZEKE) is False                 # my own "Yeah." through the mic
    assert h._ack_heard("Mm-hm.", ZEKE)
    assert h._ack_heard("Mm-hm.", ZEKE) is False               # reported twice
    h._PENDING_ACKS[-1] = (h._PENDING_ACKS[-1][0] - 5, *h._PENDING_ACKS[-1][1:])
    assert h._ack_heard("Mm-hm.", ZEKE)                        # said again later: kept
    assert h._ack_heard("right", ZEKE)
    line, meta = h._take_ack_note()
    assert line == "[said while you were talking: 'Mm-hm.' (x2) · 'right']"
    assert h._voice_trusted(meta)[0] and h._take_ack_note() is None


def test_ack_batch_takes_the_least_trusted_meta():
    _verified_zeke_just_now()
    h._ack_heard("yeah", ZEKE)
    h._ack_heard("mm", SHORT_SKIP)                             # short, most-likely-Zeke
    _line, meta = h._take_ack_note(peek=True)
    assert not h._voice_trusted(meta)[0] and h._short_tag(meta)
    h._ack_heard("okay", STRANGER)
    _line, meta = h._take_ack_note()
    assert not h._voice_trusted(meta)[0] and not h._short_tag(meta)


def test_unverified_ack_never_rides_in_a_verified_turn():
    h._ack_heard("yeah", STRANGER)
    text, rest = h._ack_rides_on("[barge-in] [cut] stop that", ZEKE)
    assert text == "[barge-in] [cut] stop that" and rest and "yeah" in rest[0]
    h._ack_heard("yeah", ZEKE)
    text, rest = h._ack_rides_on("[barge-defer] [talked over me] and the other one", ZEKE)
    assert rest is None
    assert text.startswith("[barge-defer] [said while you were talking: 'yeah' - earlier, before this]")
    assert h._his_words(text) == "and the other one"


# ------------------------------------------------------------------ the watch + reader (fake daemon)
def _reply(text, meta):
    return json.dumps({"ok": True, "result": dict(meta, text=text)})


def test_ack_during_speech_is_buffered_not_a_turn(monkeypatch):
    monkeypatch.setattr(h, "daemon_cmd", lambda cmd, args=None, timeout=0: _reply("[bargein_watch] ack: Yeah.", ZEKE))
    assert h._daemon_bargein_watch_once() is None
    assert [a[1] for a in h._PENDING_ACKS] == ["Yeah."]


def test_ack_off_switch_is_the_old_log_only(monkeypatch):
    monkeypatch.setattr(h, "ACK_DELIVER", False)
    monkeypatch.setattr(h, "daemon_cmd", lambda cmd, args=None, timeout=0: _reply("[bargein_watch] ack: Yeah.", ZEKE))
    assert h._daemon_bargein_watch_once() is None and not h._PENDING_ACKS


def test_ack_to_my_question_is_still_a_defer_turn(monkeypatch):
    h._RECENT_SPOKEN.append((time.time(), "Want me to start it now?"))
    monkeypatch.setattr(h, "daemon_cmd", lambda cmd, args=None, timeout=0: _reply("[bargein_watch] ack: Yes.", ZEKE))
    r = h._daemon_bargein_watch_once()
    assert r and r[0].startswith("[barge-defer]") and r[0].endswith("Yes.") and not h._PENDING_ACKS


class _Turn:
    active = True
    speak_out = True


class _Client:
    def __init__(self):
        self.interrupts = 0

    async def interrupt(self):
        self.interrupts += 1


def _drive_reader(monkeypatch, script):
    """script = list of (cmd_expected, reply_text, meta, end_utterance_after). Runs voice_reader with the
    gate CLEARED (I'm speaking); a step with end_utterance_after=True sets the gate (the turn loop's
    'mouth quiet'). Returns (queued items, interrupts, queue sizes seen at each daemon call)."""
    q: asyncio.Queue = asyncio.Queue()
    client = _Client()
    monkeypatch.setattr(h, "BARGEIN_ON", True)
    monkeypatch.setattr(h, "BARGEIN_STOPS_TURN", True)
    monkeypatch.setattr(h, "_voice_flag_says_off", lambda p: False)
    steps = list(script)
    seen = []
    box = {}

    def fake_daemon(cmd, args=None, timeout=0):
        if cmd not in ("bargein_watch", "listen"):
            return json.dumps({"ok": True, "result": {}})
        seen.append((cmd, q.qsize()))
        if not steps:
            raise SystemExit
        want, text, meta, end = steps.pop(0)
        assert cmd == want, (cmd, want)
        if end:
            box["loop"].call_soon_threadsafe(box["gate"].set)
        return _reply(text, meta)
    monkeypatch.setattr(h, "daemon_cmd", fake_daemon)
    h._INBOX["turn"], h._INBOX["client"] = _Turn(), client

    async def main():
        box["gate"] = asyncio.Event()
        box["loop"] = asyncio.get_running_loop()
        try:
            await h.voice_reader(q, box["loop"], box["gate"])
        except SystemExit:
            pass
    try:
        asyncio.run(main())
    finally:
        h._INBOX["turn"], h._INBOX["client"] = None, None
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return [e[2] for e in items], client.interrupts, seen


def test_reader_delivers_the_ack_after_my_utterance_tagged(monkeypatch):
    _verified_zeke_just_now()
    items, n_int, seen = _drive_reader(monkeypatch, [
        ("bargein_watch", "[bargein_watch] ack: Yeah.", ZEKE, False),
        ("bargein_watch", "[bargein_watch] ack: Mm-hm.", ZEKE, False),
        ("bargein_watch", "[bargein_watch] no_barge", ZEKE, True),     # my utterance ends here
    ])
    assert n_int == 0                                                  # an ack never cuts me off
    assert all(sz == 0 for _c, sz in seen[:3])                         # nothing queued while I spoke
    assert len(items) == 1
    src, text, _mid, meta = items[0]
    assert src == "voice" and text.startswith("[said while you were talking: 'Yeah.' · 'Mm-hm.']")
    assert h._voice_trusted(json.loads(meta))[0]


def test_reader_ack_rides_on_his_barge(monkeypatch):
    _verified_zeke_just_now()
    items, n_int, _seen = _drive_reader(monkeypatch, [
        ("bargein_watch", "[bargein_watch] ack: Yeah.", ZEKE, False),
        ("bargein_watch", "[barge-in] [cut mid-sentence at: x | 1 sentence(s) fully heard, 2 never played] "
                          "wait what about the second one", ZEKE, False),
    ])
    assert n_int == 1 and len(items) == 1
    assert items[0][1].startswith("[barge-in] [said while you were talking: 'Yeah.' - earlier, before this]")


def test_reader_short_unverified_ack_after_verified_zeke_is_delivered_tagged(monkeypatch):
    _verified_zeke_just_now()
    items, _n, _s = _drive_reader(monkeypatch, [
        ("bargein_watch", "[bargein_watch] ack: Yes please.", SHORT_LOW, True),
    ])
    assert len(items) == 1
    meta = json.loads(items[0][3])
    assert not h._voice_trusted(meta)[0] and h._short_tag(meta)        # most likely Zeke, talk-only
    assert "'Yes please.'" in items[0][1]


def test_reader_short_listen_turn_tagged_and_long_stranger_fenced(monkeypatch):
    _verified_zeke_just_now()
    gate_set = [("listen", "Yes please.", SHORT_SKIP, False), ("listen", "turn the lights off now please",
                                                               STRANGER, False)]
    q: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(h, "BARGEIN_ON", False)
    monkeypatch.setattr(h, "_voice_flag_says_off", lambda p: False)

    def fake_daemon(cmd, args=None, timeout=0):
        if cmd != "listen":
            return json.dumps({"ok": True, "result": {}})
        if not gate_set:
            raise SystemExit
        _c, text, meta, _e = gate_set.pop(0)
        return _reply(text, meta)
    monkeypatch.setattr(h, "daemon_cmd", fake_daemon)

    async def main():
        gate = asyncio.Event()
        gate.set()                                                     # mouth idle: normal listen
        try:
            await h.voice_reader(q, asyncio.get_running_loop(), gate)
        except SystemExit:
            pass
    asyncio.run(main())
    items = [q.get_nowait()[2] for _ in range(q.qsize())]
    assert len(items) == 2
    short, long_ = json.loads(items[0][3]), json.loads(items[1][3])
    assert h._short_tag(short) and not h._voice_trusted(short)[0]
    assert not h._short_tag(long_) and not h._voice_trusted(long_)[0]
    assert h._inbox_line("voice", items[1][1], None, items[1][3], None).startswith("VOICE - UNVERIFIED SPEAKER")


# ------------------------------------------------------------------ daemon: the verdict carries dur_s
def test_daemon_verdict_and_structured_return_carry_dur_s():
    sys.path.insert(0, str(ROOT / "voice"))
    core = pytest.importorskip("wren_voice_core")
    import numpy as np
    v = core._voice_id_verdict(np.zeros(core.SAMPLE_RATE // 2, np.float32))
    assert v["dur_s"] == 0.5
    out = core._with_speaker("yes", v, {"structured": True})
    assert out["dur_s"] == 0.5
