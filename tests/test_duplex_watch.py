"""voice/iris_duplex_watch.py: tiered barge-in accumulation + mouth actions (synthetic frames)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice"))
import iris_duplex_watch as W  # noqa: E402
from iris_interrupt_tiers import Observation, OverlapClassifier, Tier  # noqa: E402

LOUD = np.full(512, 0.1, np.float32)       # -20 dBFS
FAINT = np.full(512, 0.005, np.float32)    # -46 dBFS
QUIET = np.zeros(512, np.float32)


def frames(spec):
    """spec: list of (n_chunks, chunk, prob)."""
    for n, c, p in spec:
        for _ in range(n):
            yield c, p


def acts():
    return W.MouthActions(duck=lambda g: None, unduck=lambda: None, pause=lambda: None, resume=lambda: None)


def test_no_episode_returns_none_when_mouth_finishes():
    n = [0]

    def speaking():
        n[0] += 1
        return n[0] < 20
    a = acts()
    audio, clf = W.tiered_accumulate(frames([(100, QUIET, 0.0)]), speaking, a)
    assert audio is None and clf is None and a.calls == []


def test_faint_voiced_never_starts_an_episode():
    a = acts()
    st = {}
    n = [0]
    audio, clf = W.tiered_accumulate(frames([(60, FAINT, 0.9)]), lambda: (n.__setitem__(0, n[0] + 1) or n[0] < 60),
                                     a, min_rms=0.02, stats=st)
    assert audio is None and st["faint_voiced"] >= 50 and a.calls == []


def test_short_blip_fizzles_and_watch_continues():
    # 5 voiced chunks (160 ms) then silence: episode starts (>=120 ms) but never reaches DUCK (200 ms)
    a = acts()
    st = {}
    spec = [(5, LOUD, 0.9), (25, QUIET, 0.0), (30, QUIET, 0.0)]
    n = [0]

    def speaking():
        n[0] += 1
        return n[0] < 55
    audio, clf = W.tiered_accumulate(frames(spec), speaking, a, stats=st)
    assert audio is None and st.get("fizzled") == 1 and "pause" not in a.calls


def test_long_speech_ducks_then_pauses_then_barges_and_returns_audio():
    a = acts()
    spec = [(50, LOUD, 0.9), (20, QUIET, 0.0)]
    audio, clf = W.tiered_accumulate(frames(spec), lambda: True, a)
    assert audio is not None and clf.tier == Tier.BARGE
    assert a.calls[:2] == ["duck", "unduck"] and "pause" in a.calls
    assert a.state == "paused"


def test_backchannel_partial_never_pauses():
    a = acts()
    spec = [(14, LOUD, 0.9), (20, QUIET, 0.0)]          # 448 ms "mm-hm"
    audio, clf = W.tiered_accumulate(frames(spec), lambda: True, a, text_fn=lambda: "mm-hm")
    assert audio is not None and "pause" not in a.calls
    d = clf.update(Observation(text="Mm-hmm.", final=True))
    a.settle(d)
    assert d.tier == Tier.ACK and a.state == "normal" and a.calls[-1] == "unduck"
    assert W.final_marker(d) == "[bargein_watch] ack"


def test_explicit_stop_pauses_immediately_from_partial():
    a = acts()
    seen = []
    spec = [(8, LOUD, 0.9), (20, QUIET, 0.0)]

    def text():
        seen.append(1)
        return "stop" if len(seen) > 2 else None
    audio, clf = W.tiered_accumulate(frames(spec), lambda: True, a, text_fn=text)
    assert clf.tier == Tier.BARGE and "pause" in a.calls
    # paused within a few chunks of the stop partial, not after the 700 ms ladder
    assert a.calls.index("pause") <= 3


def test_echo_residual_keeps_my_own_echo_from_triggering():
    a = acts()
    n = [0]

    def speaking():
        n[0] += 1
        return n[0] < 80
    audio, clf = W.tiered_accumulate(frames([(80, LOUD, 0.9)]), speaking, a, residual_fn=lambda: -22.0)
    # level -20 dBFS is within 6 dB of the -22 dBFS predicted residual -> never more than NONE
    assert "pause" not in a.calls and "duck" not in a.calls


def test_settle_resume_after_pause_on_self_voice():
    a = acts()
    clf = OverlapClassifier()
    a.apply(clf.update(Observation(voiced_ms=800, level_dbfs=-18)))
    assert a.state == "paused"
    d = clf.update(Observation(speaker="self", final=True))
    a.settle(d)
    assert a.state == "normal" and a.calls[-1] == "resume"
    assert W.final_marker(d) == "[bargein_watch] no_turn"


def test_defer_marker_and_unduck():
    a = acts()
    clf = OverlapClassifier()
    a.apply(clf.update(Observation(voiced_ms=300, level_dbfs=-18, speaker="unknown")))
    assert a.state == "ducked"
    d = clf.update(Observation(text="is this the AI", final=True))
    a.settle(d)
    assert d.tier == Tier.DEFER and W.final_marker(d) == "[barge-defer]" and a.state == "normal"


def test_failed_mouth_action_never_raises():
    def boom(*a):
        raise RuntimeError("mouth down")
    a = W.MouthActions(duck=boom, unduck=boom, pause=boom, resume=boom)
    clf = OverlapClassifier()
    a.apply(clf.update(Observation(voiced_ms=800, level_dbfs=-18)))
    assert a.state == "paused"


def test_speaker_class_mapping():
    assert W.speaker_class({"status": "matched", "speaker": "Zeke"}, False) == "owner"
    assert W.speaker_class({"status": "matched", "speaker": "q"}, False) == "unknown"
    assert W.speaker_class({"status": "unknown"}, False) == "unknown"
    assert W.speaker_class({"status": "skipped"}, False) is None
    assert W.speaker_class({"status": "matched", "speaker": "zeke"}, True) == "self"


# ------------------------------------------------------------------ mouth-side duck

def test_ducker_passthrough_by_default():
    d = W.MouthDucker(44100)
    x = np.ones(2205, np.float32)
    assert d.apply(x, now=0.0) is x


def test_ducker_ramps_down_without_click_then_expires():
    d = W.MouthDucker(44100, ramp_ms=30)
    d.set(-12, ttl_s=2.0, now=100.0)
    y = d.apply(np.ones(2205, np.float32), now=100.0)          # 50 ms block
    g = 10 ** (-12 / 20)
    assert abs(y[0] - 1.0) < 0.01 and abs(y[-1] - g) < 1e-3
    assert np.max(np.abs(np.diff(y))) < 0.001                  # smooth ramp, no step
    y2 = d.apply(np.ones(2205, np.float32), now=101.0)
    assert np.allclose(y2, g, atol=1e-4)                       # held while the duck is fresh
    y3 = d.apply(np.ones(2205, np.float32), now=102.5)         # TTL passed: back to full volume
    assert abs(y3[-1] - 1.0) < 1e-3 and y3[0] < 0.9


def test_ducker_clamps_and_undo():
    d = W.MouthDucker(44100)
    assert d.set(-200, ttl_s=999, now=0.0) == {"db": -40.0, "ttl_s": 10.0}
    assert d.set(5, now=0.0)["db"] == 0.0
    d.set(-12, now=0.0)
    d.apply(np.ones(4410, np.float32), now=0.1)
    d.set(0, now=0.2)
    y = d.apply(np.ones(4410, np.float32), now=0.2)
    assert abs(y[-1] - 1.0) < 1e-3


def test_ducker_stereo_blocks():
    d = W.MouthDucker(44100)
    d.set(-6, now=0.0)
    y = d.apply(np.ones((4410, 2), np.float32), now=0.0)
    assert y.shape == (4410, 2) and abs(y[-1, 0] - 10 ** (-6 / 20)) < 1e-3
