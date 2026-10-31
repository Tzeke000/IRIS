"""Turn-taking fixes from Zeke's live duplex test 2026-10-31.

1. Cut off mid-thought: the continuation merge (host), the grace-window knob, and the stranded
   _StreamingPartials.latest() that sent every in-call turn down the plain-whisper fallback.
2. Backchannels while he talks (IRIS_BACKCHANNEL, default OFF): policy, gates, the mouth player,
   and the capture-loop wiring incl. the self-voice window.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "voice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import iris_backchannel as bcm  # noqa: E402
import iris_live_words as lw  # noqa: E402

core = pytest.importorskip("wren_voice_core")
import wren_voice_status  # noqa: E402

_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)
import iris_body_host as h  # noqa: E402

h._blog = lambda *a, **k: None  # type: ignore[assignment]

ZEKE = {"status": "matched", "speaker": "zeke", "score": 0.9, "threshold": 0.3}
STRANGER = {"status": "unknown", "speaker": "unknown", "score": 0.1}


# ── 1a. the stranded latest() ────────────────────────────────────────────────────
def test_streaming_partials_has_latest_again():
    p = core._StreamingPartials()
    assert p.latest() is None
    p.update("hello there", [{"w": 1}], 16000, 0.1)
    assert p.latest() == ("hello there", [{"w": 1}], 16000)


def test_transcribe_in_call_uses_the_partial(monkeypatch):
    p = core._StreamingPartials()
    p.update("covered words", [], 16000, 0.1)
    audio = np.concatenate([np.full(16000, 0.2, np.float32), np.zeros(8000, np.float32)])

    def boom(_a):
        raise AssertionError("full pass must not run on a HIT")
    monkeypatch.setattr(core._prosody, "transcribe", boom)
    assert core._transcribe_in_call(audio, p) == ("covered words", [])


# ── 1b. grace knob ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ctl,want", [({}, core.GRACE_RESUME_TIMEOUT), ({"grace_resume_s": 3.5}, 3.5),
                                      ({"grace_resume_s": 99}, 8.0), ({"grace_resume_s": "x"},
                                                                       core.GRACE_RESUME_TIMEOUT)])
def test_grace_resume_knob(monkeypatch, ctl, want):
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: ctl)
    assert core._grace_resume_s() == want


# ── 1c. continuation merge (host) ──────────────────────────────────────────────────
PREV = ("Yeah, my thought is that you probably will step in too early because You don't really "
        "know if I have like stopped on the middle of a thought maybe\n"
        "[pace: about your usual pace (140 wpm) | 16.3s]")
BARGE_1250 = ('[barge-in] [cut mid-sentence at: "You\'re right that I go by silence, not meaning." '
              '| 0 sentence(s) fully heard, 5 never played] and then I would probably you know')


def test_the_1250_incident_merges_into_one_turn(monkeypatch):
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 20.0)
    last = {"text": PREV, "meta": ZEKE, "ts": 100.0}
    out = h._merge_continuation(last, BARGE_1250, ZEKE, 109.0)
    assert out is not None
    text, words = out
    assert text.startswith("[barge-in] [continuation")
    assert "0 sentence(s) fully heard" in text and "[pace:" not in text
    assert words == ("Yeah, my thought is that you probably will step in too early because You don't "
                     "really know if I have like stopped on the middle of a thought maybe ... and then "
                     "I would probably you know")


@pytest.mark.parametrize("barge", [
    "[barge-in] [cut between sentences | 0 heard, 3 never played] wait",
    "[barge-in] [barge landed before the reply started playing — nothing had been spoken; the "
    "pending reply was dropped unheard] and also",
])
def test_other_unheard_markers_merge(monkeypatch, barge):
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 20.0)
    assert h._merge_continuation({"text": "so I was", "meta": ZEKE, "ts": 0}, barge, ZEKE, 5) is not None


@pytest.mark.parametrize("barge,now,meta", [
    # he heard a sentence of my reply -> a real interruption, not a continuation
    ('[barge-in] [cut mid-sentence at: "x" | 1 sentence(s) fully heard, 2 never played] no', 5, ZEKE),
    ("[barge-in] [reply had already finished playing — all 4 sentence(s) heard] yes", 5, ZEKE),
    (BARGE_1250, 40, ZEKE),                      # too long after his turn
    (BARGE_1250, 5, STRANGER),                   # different trust -> never merged
    ("[barge-defer] [talked over me] " + BARGE_1250[11:], 5, ZEKE),
])
def test_no_merge_cases(monkeypatch, barge, now, meta):
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 20.0)
    assert h._merge_continuation({"text": PREV, "meta": ZEKE, "ts": 0}, barge, meta, now) is None


def test_merge_disabled_and_no_previous(monkeypatch):
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 0.0)
    assert h._merge_continuation({"text": PREV, "meta": ZEKE, "ts": 0}, BARGE_1250, ZEKE, 1) is None
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 20.0)
    assert h._merge_continuation(None, BARGE_1250, ZEKE, 1) is None


def test_chained_merges_stay_clean(monkeypatch):
    monkeypatch.setattr(h, "CONTINUATION_MERGE_S", 20.0)
    t1, w1 = h._merge_continuation({"text": PREV, "meta": ZEKE, "ts": 0}, BARGE_1250, ZEKE, 5)
    b2 = "[barge-in] [cut between sentences | 0 heard, 2 never played] start over"
    t2, w2 = h._merge_continuation({"text": t1, "words": w1, "meta": ZEKE, "ts": 5}, b2, ZEKE, 9)
    assert w2.endswith("you know ... start over") and w2.startswith("Yeah, my thought")
    assert t2.count("[continuation") == 1


# ── 2a. policy (pure) ──────────────────────────────────────────────────────────────
def _talk(pol, t0, secs, frame=0.1):
    t = t0
    for _ in range(int(round(secs / frame))):
        pol.on_frame(t, True, frame)
        t += frame
    return t


def test_policy_waits_for_a_long_utterance_and_a_real_pause():
    pol = bcm.BackchannelPolicy()
    t = _talk(pol, 0.0, 2.0)
    assert not pol.want_check(t + 0.5, 0.5, t, 1.1)          # only 2.5 s in
    t = _talk(pol, t, 3.0)
    assert not pol.want_check(t + 0.2, 0.2, t, 1.1)          # breath, not a pause
    assert not pol.want_check(t + 0.9, 0.9, t, 1.1)          # too close to my turn-end
    assert pol.want_check(t + 0.5, 0.5, t, 1.1)
    assert not pol.want_check(t + 0.6, 0.6, t, 1.1)          # one check per pause


def test_policy_rate_limit_and_new_speech():
    pol = bcm.BackchannelPolicy()
    t = _talk(pol, 0.0, 5.0)
    assert pol.want_check(t + 0.5, 0.5, t, 1.1)
    pol.fired(t + 0.5)
    t2 = _talk(pol, t + 0.6, 2.0)                             # new speech, but only ~3 s later
    assert not pol.want_check(t2 + 0.5, 0.5, t2, 1.1)
    # 6+ s later but almost no new speech (long quiet pauses) -> still no
    pol2 = bcm.BackchannelPolicy()
    t = _talk(pol2, 0.0, 5.0)
    pol2.want_check(t + 0.5, 0.5, t, 1.1)
    pol2.fired(t + 0.5)
    t3 = _talk(pol2, t + 7.0, 0.5)
    assert not pol2.want_check(t3 + 0.5, 0.5, t3, 1.1)
    t4 = _talk(pol2, t3 + 0.6, 2.0)
    assert pol2.want_check(t4 + 0.5, 0.5, t4, 1.1)


# ── 2b. gates ──────────────────────────────────────────────────────────────────────
NOON = dt.datetime(2026, 10, 31, 12, 0)


def test_allowed_gates(monkeypatch, tmp_path):
    flag = tmp_path / "voice_deliberately_off.json"
    monkeypatch.delenv("IRIS_BACKCHANNEL", raising=False)
    assert bcm.allowed({}, NOON, flag) == (False, "IRIS_BACKCHANNEL off")
    monkeypatch.setenv("IRIS_BACKCHANNEL", "1")
    assert bcm.allowed({}, NOON, flag) == (True, "")
    assert not bcm.allowed({"backchannel": False}, NOON, flag)[0]
    assert bcm.allowed({"mouth_muted": True}, NOON, flag) == (False, "mouth_muted")
    assert bcm.allowed({}, dt.datetime(2026, 10, 31, 23, 0), flag) == (False, "quiet hours")
    assert bcm.allowed({}, dt.datetime(2026, 10, 31, 5, 0), flag)[1] == "quiet hours"
    flag.write_text('{"off": true}')
    assert bcm.allowed({}, NOON, flag)[1] == "quiet hours"
    flag.write_text('{"off": false}')
    assert bcm.allowed({}, NOON, flag)[0]


def test_config_overrides_are_hot_and_safe():
    c = bcm.config({"backchannel_cfg": {"min_gap_s": 9, "bogus": 1, "min_pause_s": "x"}})
    assert c["min_gap_s"] == 9.0 and "bogus" not in c and c["min_pause_s"] == bcm.DEFAULTS["min_pause_s"]


# ── 2c. player ─────────────────────────────────────────────────────────────────────
class _R:
    def __init__(self, body):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _wait(pred, s=2.0):
    import time
    t0 = time.time()
    while not pred() and time.time() - t0 < s:
        time.sleep(0.01)
    return pred()


@pytest.fixture
def fresh_player():
    bcm.Player._disabled_reason = None
    yield
    bcm.Player._disabled_reason = None


def test_player_posts_and_closes_the_self_window(fresh_player):
    import threading
    seen, release = [], threading.Event()

    def opener(req, timeout):
        seen.append((req.full_url, req.get_method()))
        release.wait(2.0)                               # the clip is "playing"
        return _R(b'{"text": "Mm-hm.", "dur_s": 0.4}')
    p = bcm.Player("http://mouth.test", opener=opener)
    assert p.fire(now=1000.0)
    assert p.in_self_window(1000.5)                    # assumed clip length until the reply
    assert not p.fire(now=1000.1)                      # never two at once
    release.set()
    assert _wait(lambda: not p.busy)
    assert not p.in_self_window(p.win_end + p.cfg["self_tail_s"] + 0.01)
    assert seen[0] == ("http://mouth.test/backchannel", "POST")
    assert p.last_info["text"] == "Mm-hm."


def test_player_disables_on_a_mouth_without_the_endpoint(fresh_player):
    from urllib.error import HTTPError

    def opener(req, timeout):
        raise HTTPError(req.full_url, 404, "nf", {}, None)
    p = bcm.Player("http://mouth.test", opener=opener)
    assert p.fire()
    assert _wait(lambda: not p.busy)
    assert not bcm.Player.available() and not p.fire()


# ── 2d. capture-loop wiring ────────────────────────────────────────────────────────
class _Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class _Mic:
    def __init__(self, frames, clock):
        self.frames, self.clock, self.i = frames, clock, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.clock.t += 0.1
        f = self.frames[self.i] if self.i < len(self.frames) else np.zeros(n, np.float32)
        self.i += 1
        return f.reshape(-1, 1), False


class _StubPlayer:
    def __init__(self, clip_s=0.5):
        self.fired_at, self.busy, self.clip_s = [], False, clip_s
        self.cfg = dict(bcm.DEFAULTS)

    def in_self_window(self, now):
        return any(t <= now <= t + self.clip_s + self.cfg["self_tail_s"] for t in self.fired_at)

    def fire(self, now=None):
        self.fired_at.append(now)
        return True


def _frames(spec):
    out = []
    for level, secs in spec:
        out += [np.full(1600, level, np.float32)] * int(round(secs / 0.1))
    return out


@pytest.fixture
def cap(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(core, "time", types.SimpleNamespace(time=clock.time, sleep=clock.sleep))
    monkeypatch.setattr(core, "_trusted_mic_or_none", lambda ctx: "net")
    monkeypatch.setattr(core, "set_state", lambda *a, **k: None)
    monkeypatch.setattr(core, "is_muted", lambda: False)
    monkeypatch.setattr(core, "EARLY_FINALIZE", False)
    monkeypatch.setenv("IRIS_BACKCHANNEL", "1")
    monkeypatch.setattr(bcm, "in_quiet_hours", lambda now=None: False)
    monkeypatch.setattr(bcm, "_voice_flag_off", lambda path=None: False)
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    st = {"prob": 0.5}
    monkeypatch.setattr(core, "_smartturn", types.SimpleNamespace(
        predict_endpoint=lambda a: {"prob": st["prob"]}))

    def run(spec):
        monkeypatch.setattr(core._net_audio, "open_mic",
                            lambda sd, sr, idx, bs: _Mic(_frames(spec), clock))
        ctx = types.SimpleNamespace(speaking=False, server_url="http://mouth.test")
        lane = core._Backchannel(ctx, bcm.config({}))
        lane.player = _StubPlayer()
        audio = core._capture_utterance(ctx, 5.0, 30.0, 1.1, backchannel=lane)
        return audio, lane
    return run, st


def test_backchannel_fires_once_in_a_mid_thought_pause(cap):
    run, st = cap
    audio, lane = run([(0.2, 5.0), (0.0, 0.6), (0.2, 2.0), (0.0, 1.5)])
    assert len(lane.player.fired_at) == 1 and lane.policy.fires == 1
    assert audio is not None and audio.size >= 16000 * 7     # the pause did not end his turn


def test_no_backchannel_when_smart_turn_says_done(cap):
    run, st = cap
    st["prob"] = 0.75
    _audio, lane = run([(0.2, 5.0), (0.0, 0.6), (0.2, 2.0), (0.0, 1.5)])
    assert lane.player.fired_at == []


def test_no_backchannel_on_a_short_utterance(cap):
    run, _ = cap
    _audio, lane = run([(0.2, 2.0), (0.0, 0.6), (0.2, 1.0), (0.0, 1.5)])
    assert lane.player.fired_at == []


def test_my_clip_never_becomes_his_words(cap):
    run, _ = cap
    # pause -> clip fires at ~0.5 s of silence; "bleed" at 0.03 (below resume_peak 0.05) while it
    # plays must be zeroed and must not hold the turn open; the turn still ends.
    audio, lane = run([(0.2, 5.0), (0.0, 0.5), (0.03, 0.5), (0.0, 1.5)])
    assert len(lane.player.fired_at) == 1
    bleed = audio[int(16000 * 5.5): int(16000 * 6.0)]
    assert bleed.size and float(np.abs(bleed).max()) == 0.0


def test_lane_is_off_without_the_flag(monkeypatch):
    monkeypatch.delenv("IRIS_BACKCHANNEL", raising=False)
    assert core._new_backchannel(types.SimpleNamespace(server_url="http://x")) is None


# ── 2e. live partial reader ────────────────────────────────────────────────────────
def test_read_current(tmp_path):
    p = tmp_path / "live_partial.json"
    assert lw.read_current(p) is None
    p.write_text(json.dumps({"text": "so I was thinking", "final": False, "ts": 100.0}))
    assert lw.read_current(p, now=101.0)["text"] == "so I was thinking"
    assert lw.read_current(p, now=110.0) is None                     # stale
    p.write_text(json.dumps({"text": "done", "final": True, "ts": 100.0}))
    assert lw.read_current(p, now=100.5) is None                     # finished utterance
