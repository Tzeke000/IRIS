"""Tier classifier for talk-over-me episodes (voice/iris_interrupt_tiers.py). Pure, no audio."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice"))

import pytest  # noqa: E402

from iris_interrupt_tiers import (Observation as O, OverlapClassifier, Tier, TierConfig,  # noqa: E402
                                  addressed, is_backchannel, real_words, stop_phrase)


def run(seq, **ctx):
    c = OverlapClassifier(**ctx)
    out = [c.update(o) for o in seq]
    return c, out


# ------------------------------------------------------------------ lexical helpers

@pytest.mark.parametrize("t", ["mm-hm", "Mm-hmm.", "yeah", "Yeah yeah.", "uh huh", "right", "okay",
                               "Oh wow!", "gotcha", "haha", "i see", "makes sense"])
def test_backchannels(t):
    assert is_backchannel(t)


@pytest.mark.parametrize("t", ["yeah but wait", "no", "what", "okay so the thing is", "", "turn it off"])
def test_not_backchannels(t):
    assert not is_backchannel(t)


@pytest.mark.parametrize("t,p", [("Stop.", "stop"), ("Iris, stop!", "stop"), ("wait wait wait", "wait wait"),
                                 ("hold on a second", "hold on"), ("hey Iris hang on", "hang on"),
                                 ("No, wait, I meant the other one", "no wait"), ("shh", "shh")])
def test_stop_phrases(t, p):
    assert stop_phrase(t) == p


@pytest.mark.parametrize("t", ["I can't wait to see it", "the bus stop is close", "don't stop", ""])
def test_stop_phrase_not_mid_sentence(t):
    assert stop_phrase(t) is None


def test_addressed_and_real_words():
    assert addressed("Iris, what's the time")
    assert addressed("what do you think iris")
    assert not addressed("turn the lights off")
    assert real_words("yeah okay so the render") == 3


# ------------------------------------------------------------------ the ladder

def test_ambient_too_faint_never_triggers():
    c, out = run([O(voiced_ms=300, level_dbfs=-30, residual_dbfs=-28),
                  O(voiced_ms=900, level_dbfs=-29, residual_dbfs=-28),
                  O(voiced_ms=1600, level_dbfs=-29, residual_dbfs=-28)])
    assert all(d.tier == Tier.NONE for d in out)


def test_absolute_floor_without_residual_estimate():
    c, out = run([O(voiced_ms=2000, level_dbfs=-45)])
    assert out[-1].tier == Tier.NONE


def test_short_blip_is_nothing():
    c, out = run([O(voiced_ms=64, level_dbfs=-20), O(ended=True)])
    assert out[-1].tier == Tier.NONE and not out[-1].provisional


def test_duration_ladder_duck_pause_barge():
    c, out = run([O(voiced_ms=150, level_dbfs=-18), O(voiced_ms=250), O(voiced_ms=800), O(voiced_ms=1500)])
    assert [d.tier for d in out] == [Tier.NONE, Tier.DUCK, Tier.PAUSE, Tier.BARGE]
    assert out[1].gain_db < 0
    assert out[-1].interrupt_turn and out[-1].enqueue


def test_backchannel_acks_and_resumes():
    c, out = run([O(voiced_ms=250, level_dbfs=-18), O(voiced_ms=400, text="mm-hm"),
                  O(voiced_ms=500, text="Mm-hmm.", final=True)])
    assert out[0].tier == Tier.DUCK
    assert out[-1].tier == Tier.ACK and out[-1].resume and not out[-1].interrupt_turn
    assert not out[-1].enqueue and out[-1].ack_text


def test_backchannel_after_pause_resumes():
    c, out = run([O(voiced_ms=750, level_dbfs=-18), O(voiced_ms=900, text="yeah yeah", final=True)])
    assert out[0].tier == Tier.PAUSE
    assert out[1].tier == Tier.ACK and out[1].resume


def test_long_speech_transcribed_as_backchannel_is_not_an_ack():
    # 2 s of speech that whisper rendered as just "yeah" = mis-heard real speech: give him the floor
    c, out = run([O(voiced_ms=2000, level_dbfs=-18, text="yeah", final=True)])
    assert out[-1].tier == Tier.BARGE


def test_explicit_stop_is_immediate_even_short():
    c, out = run([O(voiced_ms=180, level_dbfs=-18, text="stop")])
    assert out[-1].tier == Tier.BARGE and out[-1].interrupt_turn and not out[-1].provisional


def test_stop_beats_critical_line():
    c, out = run([O(voiced_ms=300, level_dbfs=-18, text="Iris stop")], critical=True)
    assert out[-1].tier == Tier.BARGE


def test_stop_from_unverified_only_pauses_and_is_fenced():
    c, out = run([O(voiced_ms=300, level_dbfs=-18, text="stop", speaker="unknown"),
                  O(final=True)])
    assert out[0].tier == Tier.PAUSE and not out[0].interrupt_turn
    assert out[1].tier == Tier.PAUSE and out[1].enqueue and out[1].fenced


def test_three_words_barge_before_duration():
    c, out = run([O(voiced_ms=500, level_dbfs=-18, text="no the other render")])
    assert out[-1].tier == Tier.BARGE


def test_addressed_one_word_barge():
    c, out = run([O(voiced_ms=400, level_dbfs=-18, text="Iris, question")])
    assert out[-1].tier == Tier.BARGE


def test_short_real_interjection_final_barges():
    c, out = run([O(voiced_ms=450, level_dbfs=-18, text="what?", final=True)])
    assert out[-1].tier == Tier.BARGE


def test_self_voice_resolves_to_none_and_resumes():
    c, out = run([O(voiced_ms=800, level_dbfs=-18), O(speaker="self", text="so let me", final=True)])
    assert out[0].tier == Tier.PAUSE
    assert out[1].tier == Tier.NONE and out[1].resume and not out[1].enqueue


def test_cough_no_words_resumes():
    c, out = run([O(voiced_ms=750, level_dbfs=-18), O(text="", final=True)])
    assert out[1].tier == Tier.NONE and out[1].resume


def test_unverified_speaker_defers_not_barges():
    c, out = run([O(voiced_ms=300, level_dbfs=-18, speaker="unknown"),
                  O(voiced_ms=2500, text="hey are you the robot everyone talks about"),
                  O(final=True)])
    assert out[0].tier == Tier.DUCK
    assert out[1].tier == Tier.DEFER and not out[1].interrupt_turn
    assert out[2].tier == Tier.DEFER and out[2].enqueue and out[2].fenced


def test_critical_line_defers_owner_speech():
    c, out = run([O(voiced_ms=2000, level_dbfs=-18, text="what about the other thing we said")],
                 critical=True)
    assert out[-1].tier == Tier.DEFER and not out[-1].interrupt_turn


def test_never_deescalates_without_resolution():
    c, out = run([O(voiced_ms=800, level_dbfs=-18), O(voiced_ms=820)])
    assert out[1].tier == Tier.PAUSE


def test_closed_episode_is_inert():
    c, out = run([O(voiced_ms=180, level_dbfs=-18, text="stop"), O(voiced_ms=3000, text="more")])
    assert out[1].tier == Tier.BARGE and out[1].reason == "episode closed"


def test_mouth_idle_is_not_an_overlap():
    c, out = run([O(voiced_ms=2000, level_dbfs=-18)], mouth_speaking=False)
    assert out[0].tier == Tier.NONE and not out[0].provisional


def test_config_from_control_hot_overrides():
    cfg = TierConfig.from_control({"tier_pause_ms": "900", "tier_barge_words": 4, "tier_bogus": 1,
                                   "tier_duck_db": "x"})
    assert cfg.pause_ms == 900 and cfg.barge_words == 4 and cfg.duck_db == -12.0
    c, out = run([O(voiced_ms=800, level_dbfs=-18)], cfg=cfg)
    assert out[0].tier == Tier.DUCK
