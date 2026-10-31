"""brain/iris_convo_layer.py - WordAssist decisions with fake live words / LLM / mouth."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain import iris_convo_layer as cl  # noqa: E402


class Live:
    def __init__(self):
        self.cur = None

    def set(self, uid, text, sil=0.8):
        self.cur = {"utterance_id": uid, "text": text, "tail_silence_s": sil}

    def __call__(self):
        return self.cur


def _wa(live, word="geometry nodes", gate=(True, ""), on_suggest=None):
    spoken, offers = [], cl.Offers()

    def suggest(p, ctx):
        if on_suggest:
            on_suggest()
        return word, 0.1
    wa = cl.WordAssist(live, suggest, spoken.append, lambda: gate, offers=offers, log=lambda m: None)
    return wa, spoken, offers


def test_offers_the_word_after_a_pause():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called", sil=0.1)
    wa, spoken, offers = _wa(live)
    assert wa.step(100.0) is None                    # his audio isn't quiet yet: wait for the pause
    live.set("u1", "do the trees with the node thing in blender, what's it called", sil=0.7)
    assert wa.step(100.8) == "geometry nodes?" and spoken == ["geometry nodes?"]
    assert wa.step(102.0) is None                    # once per utterance
    note = offers.take_note(103.0)
    assert "geometry nodes?" in note and offers.take_note(103.0) == ""


def test_gated_says_nothing():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called")
    wa, spoken, _ = _wa(live, gate=(False, "quiet hours"))
    wa.step(100.0)
    assert wa.step(101.0) is None and spoken == [] and wa.stats["gated"] == 1


def test_no_guess_says_nothing():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called")
    wa, spoken, _ = _wa(live, word=None)
    wa.step(100.0)
    assert wa.step(101.0) is None and spoken == []


def test_he_found_it_while_the_model_thought():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called")
    wa, spoken, _ = _wa(live, on_suggest=lambda: live.set(
        "u1", "do the trees with the node thing in blender, what's it called, geometry nodes"))
    wa.step(100.0)
    assert wa.step(101.0) is None and spoken == [] and wa.stats["stale"] == 1


def test_he_started_talking_again_while_the_model_thought():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called")
    wa, spoken, _ = _wa(live, on_suggest=lambda: live.set(
        "u1", "do the trees with the node thing in blender, what's it called", sil=0.0))
    wa.step(100.0)
    assert wa.step(101.0) is None and spoken == [] and wa.stats["stale"] == 1


def test_turn_ended_while_the_model_thought():
    live = Live()
    live.set("u1", "do the trees with the node thing in blender, what's it called")
    wa, spoken, _ = _wa(live, on_suggest=lambda: setattr(live, "cur", None))
    wa.step(100.0)
    assert wa.step(101.0) is None and spoken == [] and wa.stats["ended"] == 1


def test_offers_note_expires():
    o = cl.Offers(keep_s=10)
    o.add("word offer", "pesto?", "the green sauce", now=0.0)
    assert o.take_note(now=50.0) == ""


def test_may_speak_now_gates():
    from datetime import datetime
    noon = datetime(2026, 10, 31, 12, 0)
    night = datetime(2026, 10, 31, 23, 0)
    missing = ROOT / "tests" / "_no_such_flag.json"
    assert cl.may_speak_now({}, now=noon, flag_path=missing)[0]
    assert not cl.may_speak_now({"word_assist": False}, now=noon, flag_path=missing)[0]
    assert not cl.may_speak_now({"mouth_muted": True}, now=noon, flag_path=missing)[0]
    assert not cl.may_speak_now({}, now=night, flag_path=missing)[0]


def test_little_brain_park(tmp_path):
    f = tmp_path / "park.json"
    assert not cl.little_brain_parked(f)
    f.write_text(json.dumps({"off": True}))
    assert cl.little_brain_parked(f)
    f.write_text(json.dumps({"off": False}))
    assert not cl.little_brain_parked(f)


def test_flags_default_off(monkeypatch):
    import importlib
    monkeypatch.delenv("IRIS_WORD_ASSIST", raising=False)
    monkeypatch.delenv("IRIS_CONVO_LOOKUPS", raising=False)
    m = importlib.reload(cl)
    assert m.WORD_ASSIST_ON is False and m.LOOKUPS_ON is False
