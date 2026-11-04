"""Semantic end-of-turn (Zeke 11-02, built 11-04): voice/iris_eot.py + its wiring in wren_voice_core.

1. The lexical judge on >= 40 realistic utterance endings (finished / unfinished / either-is-fine).
2. plan_grace: unfinished extends (bounded), clearly-finished shortens, unsure = legacy exactly.
3. Latency: lexical well under the 30 ms budget; the optional Namo model too (skipped if not on disk).
4. Integration: _cmd_listen_impl over a fake mic + fake clock - with IRIS_SEMANTIC_EOT=1 a pause after
   "is it" is waited out and his continuation is merged; a clearly-finished question under a smart-turn
   HOLD ends sooner; with the flag off the behaviour is the legacy one. Every decision is logged.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "voice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import iris_eot as eot  # noqa: E402

# ── 1. the judge ─────────────────────────────────────────────────────────────────
UNFINISHED = [
    "is it", "Is it...", "is it…", "Is it -", "and uh", "And uh, do you currently...",
    "so I was thinking that", "I went to the store and", "what do you think about the",
    "can you check if", "Thinking. My question is, is it", "the thing is", "I want to",
    "let's go ahead and um", "because", "I'm gonna", "and then", "but", "we could put it in the",
    "I was wondering if", "the reason is that", "I think the next logical step is to", "uh",
    "what's the", "where are", "it was kind of", "So...", "so um", "do you know",
    "Okay, so the thing I wanted to ask about is,", "and you were", "I mean", "is there a",
]
FINISHED = [
    "yes", "Yes.", "no", "No.", "okay", "Okay.", "yeah", "thanks", "thank you", "stop", "go ahead",
    "hey Iris", "What time is it?", "Can you turn the lights off?", "Turn the lights off.",
    "Is it a boat?", "Hey, Iris, what's up?", "That's all.", "I think so.", "Who are you talking to?",
    "What's it about?", "I like that.", "Play some music.", "Good night Iris.",
    "How do we get to real live listening then?", "Yeah, why?", "never mind", "sounds good",
    "Can you list it for me?", "It's almost like the barge in didn't work.", "Do it!",
]
NOT_UNFINISHED = [   # finished or unsure are both fine - the acoustic signal decides
    "what time is it", "turn it on", "check it out", "Is it a building", "let's give it a try then",
    "what i said though", "and so on", "who is it for", "this thing larger than an average house",
    "I think so", "what is that", "1258", "God.",
]


def test_case_count():
    assert len(UNFINISHED) + len(FINISHED) + len(NOT_UNFINISHED) >= 40


@pytest.mark.parametrize("text", UNFINISHED)
def test_unfinished(text):
    v = eot.lexical_verdict(text)
    assert v.label == "unfinished", (text, v)
    assert v.p_finished <= eot.UNFINISHED_BAR


@pytest.mark.parametrize("text", FINISHED)
def test_finished(text):
    v = eot.lexical_verdict(text)
    assert v.label == "finished", (text, v)


@pytest.mark.parametrize("text", NOT_UNFINISHED)
def test_not_unfinished(text):
    assert eot.lexical_verdict(text).label != "unfinished", text


def test_strong_vs_medium_bands():
    assert eot.lexical_verdict("and uh").p_finished == eot.P_UNFINISHED_STRONG
    assert eot.lexical_verdict("is it").p_finished == eot.P_UNFINISHED_MED
    assert eot.lexical_verdict("yes").p_finished == eot.P_FINISHED_STRONG
    assert eot.lexical_verdict("Can you turn the lights off?").p_finished == eot.P_FINISHED_STRONG
    # whisper adds '.' mid-utterance too, so a period is only medium evidence
    assert eot.lexical_verdict("Turn the lights off.").p_finished == eot.P_FINISHED_MED


def test_empty_and_garbage_never_raise():
    for t in ("", "   ", None, "!!!", "...", "—", "12:58", "\U0001f600"):
        v = eot.judge(t)
        assert v.label in ("unfinished", "unsure", "finished")
    assert eot.judge("").label == "unsure"


# ── 2. combining with the acoustic verdict ───────────────────────────────────────────
def _v(text):
    return eot.lexical_verdict(text)


def test_plan_unfinished_extends_bounded():
    strong = eot.plan_grace(False, _v("and uh"), 1.2, max_extend_s=1.5)
    assert not strong.complete and strong.grace_s == pytest.approx(2.7)
    med = eot.plan_grace(True, _v("is it"), 1.2, max_extend_s=1.5)
    assert not med.complete and med.grace_s == pytest.approx(1.95)
    # even when smart-turn says the waveform is done, dangling words keep the mic open
    assert not eot.plan_grace(False, _v("is it"), 1.2).complete
    # the extension is capped by the knob, never more
    assert eot.plan_grace(False, _v("and uh"), 1.2, max_extend_s=0.0).grace_s == pytest.approx(1.2)


def test_plan_finished_ends_promptly():
    assert eot.plan_grace(False, _v("yes"), 1.2).complete
    short = eot.plan_grace(True, _v("Can you turn the lights off?"), 1.2, finished_grace_s=0.3)
    assert not short.complete and short.grace_s == pytest.approx(0.3)
    assert eot.plan_grace(True, _v("yes"), 1.2, finished_grace_s=0.0).complete


def test_plan_unsure_is_exactly_legacy():
    for hold in (True, False):
        p = eot.plan_grace(hold, _v("Is it a building"), 1.2)
        assert p.complete is (not hold)
        assert hold is False or p.grace_s == pytest.approx(1.2)
    # medium-finished ('.' close) also defers to the acoustic verdict
    assert not eot.plan_grace(True, _v("Turn the lights off."), 1.2).complete


def test_knobs_clamped():
    k = eot.knobs({"semantic_eot_max_extend_s": 99, "semantic_eot_finished_grace_s": -1})
    assert k == {"max_extend_s": 3.0, "finished_grace_s": 0.0}
    assert eot.knobs({"semantic_eot_max_extend_s": "x"})["max_extend_s"] == eot.DEFAULT_MAX_EXTEND_S
    assert eot.knobs(None)["finished_grace_s"] == eot.DEFAULT_FINISHED_GRACE_S


def test_env_flag_default_off(monkeypatch):
    monkeypatch.delenv("IRIS_SEMANTIC_EOT", raising=False)
    assert not eot.env_enabled()
    monkeypatch.setenv("IRIS_SEMANTIC_EOT", "1")
    assert eot.env_enabled()


class _FakeModel:
    def __init__(self, p):
        self.p, self.calls = p, []

    def predict(self, text):
        self.calls.append(text)
        return self.p


def test_model_only_refines_unsure_long_text():
    m = _FakeModel(0.0)
    assert eot.judge("no", m).label == "finished" and not m.calls          # rules win, model not asked
    assert eot.judge("and uh", m).label == "unfinished" and not m.calls
    v = eot.judge("Is it a building", m)
    assert v.label == "unfinished" and v.p_finished == eot.P_UNFINISHED_MED and v.source == "model"
    assert eot.judge("what time is it", _FakeModel(0.999)).p_finished == eot.P_FINISHED_MED
    assert eot.judge("ok then", _FakeModel(0.0)).label != "unfinished"     # < MODEL_MIN_WORDS
    # a mid-range model answer changes nothing; a broken model (None) changes nothing
    assert eot.judge("Is it a building", _FakeModel(0.5)).label == "unsure"
    assert eot.judge("Is it a building", _FakeModel(None)).label == "unsure"


def test_model_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("IRIS_SEMANTIC_EOT_MODEL", raising=False)
    assert eot.model_from_env() is None
    monkeypatch.setenv("IRIS_SEMANTIC_EOT_MODEL", str(tmp_path))      # no model files there
    m = eot.model_from_env()
    assert m is not None and m.predict("is it a building") is None    # fails soft
    assert eot.judge("Is it a building", m).label == "unsure"


# ── 3. latency ───────────────────────────────────────────────────────────────────────
def test_lexical_latency_budget():
    texts = UNFINISHED + FINISHED + NOT_UNFINISHED
    lat = []
    for _ in range(20):
        for t in texts:
            t0 = time.perf_counter()
            eot.judge(t)
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p99 = lat[int(len(lat) * 0.99)]
    print(f"lexical judge: p50 {lat[len(lat) // 2]:.3f} ms, p99 {p99:.3f} ms over {len(lat)} calls")
    assert p99 < 5.0   # budget 30 ms; measured ~0.02 ms


NAMO_DIR = os.environ.get("IRIS_TEST_NAMO_DIR", "/data/models/namo_eot_en")


@pytest.mark.skipif(not os.path.isfile(os.path.join(NAMO_DIR, "model.onnx")),
                    reason="Namo model not on disk")
def test_namo_model_latency_and_safety():
    m = eot.ModelJudge(NAMO_DIR, threads=2)
    assert m.predict("I think the next logical step is to") < 0.05
    assert m.predict("so that's all I have for today") > 0.95
    lat = []
    for t in UNFINISHED + FINISHED + NOT_UNFINISHED:
        t0 = time.perf_counter()
        eot.judge(t, m)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"judge+model: p50 {lat[len(lat) // 2]:.1f} ms, max {lat[-1]:.1f} ms")
    assert lat[int(len(lat) * 0.9)] <= 30.0
    # the model can never flip a rule-decided case
    for t in UNFINISHED:
        assert eot.judge(t, m).label == "unfinished", t
    for t in FINISHED:
        assert eot.judge(t, m).label == "finished", t


# ── 4. integration: the listen path over a fake mic + fake clock ────────────────────────────
core = pytest.importorskip("wren_voice_core")
import wren_voice_status  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def perf_counter(self):
        return time.perf_counter()

    def monotonic(self):
        return self.t


class _TimelineMic:
    """Every open_mic() reads the SAME timeline: level 0.2 inside a speech span, else silence."""

    def __init__(self, clock, spans):
        self.clock, self.spans = clock, spans

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.clock.t += 0.1
        rel = self.clock.t - 1000.0
        lvl = 0.2 if any(a <= rel < b for a, b in self.spans) else 0.0
        return np.full((n, 1), lvl, np.float32), False


@pytest.fixture
def listen(monkeypatch, tmp_path):
    clock = _Clock()
    log = tmp_path / "eot.jsonl"
    monkeypatch.setattr(core, "time", clock)
    monkeypatch.setattr(core, "EOT_LOG", log)
    monkeypatch.setattr(core, "_trusted_mic_or_none", lambda ctx: "net")
    monkeypatch.setattr(core, "set_state", lambda *a, **k: None)
    monkeypatch.setattr(core, "is_muted", lambda: False)
    monkeypatch.setattr(core, "_speaker_end_silence", lambda: None)
    monkeypatch.setattr(core, "_smart_extend_cfg", lambda: None)
    monkeypatch.setattr(core, "_new_backchannel", lambda ctx: None)
    monkeypatch.setattr(core, "_is_phantom", lambda t, a: False)
    monkeypatch.setattr(core, "_self_voice_score", lambda a, bar: None)
    monkeypatch.setattr(core, "_voice_id_verdict", lambda a: None)
    monkeypatch.setattr(core, "_voice_id_line", lambda a, v: "")
    monkeypatch.setattr(core, "_save_utt_wav", lambda a, tag: None)
    monkeypatch.setattr(core, "_with_speaker", lambda r, v, args, timing: r)
    monkeypatch.setattr(core.wren_pace, "analyze", lambda a, sr, t: ({}, ""))
    monkeypatch.setattr(core.wl, "append_transcript", lambda t: None)
    monkeypatch.setattr(core.wl, "pop_last_rich", lambda: None)
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {"grace_resume_s": 1.2})
    monkeypatch.setattr(core, "_HAVE_SMARTTURN", True)
    st = {"prob": 0.5}   # smart-turn: < 0.6 = HOLD
    monkeypatch.setattr(core, "_smartturn", types.SimpleNamespace(
        predict_endpoint=lambda a: {"prob": st["prob"]}))
    graces = []
    real_capture = core._capture_utterance

    def capture(ctx, timeout_s, *a, **k):
        if k.get("partials") is None and "early_finalize_fn" not in k:
            graces.append(round(timeout_s, 3))       # a grace-round capture
        return real_capture(ctx, timeout_s, *a, **k)
    monkeypatch.setattr(core, "_capture_utterance", capture)

    def run(spans, texts_by_len, eot_on, prob=0.5):
        st["prob"] = prob
        monkeypatch.setenv("IRIS_SEMANTIC_EOT", "1" if eot_on else "0")
        monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: _TimelineMic(clock, spans))

        def transcribe(audio):   # text depends on how much audio was captured (grace merged or not)
            secs = audio.size / core.SAMPLE_RATE
            return [t for lim, t in texts_by_len if secs <= lim][0]
        monkeypatch.setattr(core.wl, "_transcribe_array", transcribe)
        ctx = types.SimpleNamespace(speaking=False, call_warm=False, mic_lock=threading.Lock(),
                                    server_url="http://mouth.test")
        graces.clear()
        clock.t = 1000.0                                  # spans are relative to this
        t0 = clock.t
        out = core._cmd_listen_impl(ctx, {"timeout_seconds": 5, "end_silence_seconds": 1.1})
        recs = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        log.unlink(missing_ok=True)
        return out, list(graces), clock.t - t0, recs
    return run


# "is it" ... a 1.8 s thinking pause after the turn closes ... "the lighthouse?"
# Turn closes at ~3.45 s (0.35 settle + 2.0 speech + 1.1 end-silence); he resumes at 5.2 s.
IS_IT = [(0.4, 2.4), (5.2, 6.4)]
IS_IT_TEXTS = [(4.0, "is it"), (99, "is it the lighthouse?")]


def test_integration_unfinished_grace_extends_and_merges(listen):
    out, graces, _dt, recs = listen(IS_IT, IS_IT_TEXTS, eot_on=True)
    assert out.splitlines()[0] == "is it the lighthouse?"
    assert graces[0] == pytest.approx(1.95)            # 1.2 base + 0.75 (medium unfinished)
    stages = [r["stage"] for r in recs]
    assert stages[0] == "turn-end" and recs[0]["semantic"] == "unfinished"
    assert recs[0]["acoustic"] == "hold" and recs[0]["text"] == "is it"
    assert "grace-1" in stages and any(r.get("outcome") == "resumed" for r in recs)
    last = [r for r in recs if r["stage"].startswith("after-grace")][-1]
    # merged text closes with '?': finished, so the remaining smart-turn HOLD only gets the short grace
    assert last["semantic"] == "finished" and last["grace_s"] == pytest.approx(0.3)
    assert graces == [pytest.approx(1.95), pytest.approx(0.3)]


def test_integration_flag_off_is_legacy_and_cuts_him_off(listen):
    out, graces, _dt, recs = listen(IS_IT, IS_IT_TEXTS, eot_on=False)
    assert out.splitlines()[0] == "is it"              # the 11-02 bug, reproduced with the flag off
    assert graces == [pytest.approx(1.2)]
    assert recs == []                                  # no decision log when off


def test_integration_finished_question_ends_sooner(listen):
    spans = [(0.4, 2.4)]
    texts = [(99, "Can you turn the lights off?")]
    out_on, g_on, dt_on, recs = listen(spans, texts, eot_on=True, prob=0.5)    # smart-turn HOLD
    out_off, g_off, dt_off, _ = listen(spans, texts, eot_on=False, prob=0.5)
    assert out_on.splitlines()[0] == out_off.splitlines()[0] == "Can you turn the lights off?"
    assert g_off == [pytest.approx(1.2)] and g_on == [pytest.approx(0.3)]
    assert dt_off - dt_on == pytest.approx(0.9, abs=0.15)
    assert recs[0]["why"] == "semantic-finished-short-grace" and recs[0]["legacy_complete"] is False


def test_integration_strong_unfinished_waits_long_pause(listen):
    spans = [(0.4, 2.4), (5.9, 7.0)]                    # a 2.45 s gap after the turn closes
    texts = [(4.0, "and uh"), (99, "and uh, play the next song.")]
    out, graces, _dt, _r = listen(spans, texts, eot_on=True, prob=0.8)   # smart-turn says DONE
    assert graces[0] == pytest.approx(2.7)              # 1.2 + the full 1.5 cap
    assert out.splitlines()[0] == "and uh, play the next song."


def test_integration_complete_on_both_no_grace(listen):
    out, graces, _dt, recs = listen([(0.4, 1.4)], [(99, "yes")], eot_on=True, prob=0.8)
    assert out.splitlines()[0] == "yes" and graces == []
    assert recs[0]["complete"] is True and recs[0]["why"] == "both-finished"


def test_early_finalize_veto(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "EOT_LOG", tmp_path / "e.jsonl")
    p = core._StreamingPartials()
    assert core._eot_veto_early_finalize(p) is False            # no partial -> no veto
    p.update("and the thing is, is it", [], 16000, 0.1)
    assert core._eot_veto_early_finalize(p) is True
    p.update("what time is it?", [], 16000, 0.1)
    assert core._eot_veto_early_finalize(p) is False


def test_eot_decide_judge_error_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "EOT_LOG", tmp_path / "e.jsonl")
    monkeypatch.setattr(core, "_HAVE_SMARTTURN", True)
    monkeypatch.setattr(core, "_smartturn", types.SimpleNamespace(predict_endpoint=lambda a: {"prob": 0.9}))
    monkeypatch.setattr(core._eot, "judge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert core._eot_decide(np.zeros(160, np.float32), "and", 1.2, "t") == (False, 1.2)
    assert core._eot_decide(np.zeros(160, np.float32), "done now", 1.2, "t") == (True, 1.2)
