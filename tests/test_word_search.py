"""voice/iris_word_search.py - cue detection, the one-offer policy, the suggestion parser (pure)."""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "voice"))
import iris_word_search as ws  # noqa: E402


def test_detects_common_cues():
    for t in ["we used the node thing in blender, what's it called",
              "the guy who painted clocks, what's his name",
              "I can't remember the name of that place",
              "it's on the tip of my tongue",
              "the thing that keeps power on, what do you call it",
              "what's the word for when you can't sleep",
              "i forgot what it's called"]:
        assert ws.detect_cue(t), t


def test_ignores_ordinary_speech():
    for t in ["what is it", "I can't wait", "uh, so, um, the thing is I'm tired",
              "what's it like outside", "do you remember the name I gave you", ""]:
        assert ws.detect_cue(t) is None, t


def test_cue_split_before_after():
    c = ws.detect_cue("use the node thing in blender, what's it called, the")
    assert c["before"].startswith("use the node thing") and c["after"] == "the"


def _pol():
    return ws.WordSearchPolicy(dict(ws.DEFAULTS))


def test_fires_once_on_real_audio_silence():
    p = _pol()
    p.observe("u1", "the node thing in blender, what's it called", 10.0, tail_sil=0.1)
    assert p.due(12.0) is None                      # text unchanged for 2 s but his AUDIO isn't quiet
    p.observe("u1", "the node thing in blender, what's it called", 12.1, tail_sil=0.7)
    assert p.due(12.1) is not None
    assert p.due(13.5) is None                      # one offer per utterance


def test_fallback_without_tail_silence_waits_past_the_partial_cadence():
    p = _pol()
    p.observe("u1", "the node thing in blender, what's it called", 10.0)
    assert p.due(10.7) is None                      # 0.7 s unchanged partial is NOT a pause (10-31 e2e bug)
    assert p.due(11.5) is not None


def test_not_when_he_keeps_going():
    p = _pol()
    p.observe("u1", "the node thing, what's it called, oh right geometry nodes you know", 10.0)
    assert p.due(11.0) is None


def test_needs_context_before_cue():
    p = _pol()
    p.observe("u1", "what's it called", 10.0)
    assert p.due(11.0) is None


def test_pause_resets_on_new_words_and_new_utterance_resets():
    p = _pol()
    p.observe("u1", "the node thing, what's it called", 10.0)
    p.observe("u1", "the node thing, what's it called, the", 10.5)
    assert p.due(11.2) is None and p.due(12.0) is not None
    p.observe("u2", "something else entirely", 12.0)
    assert p.cue is None and not p.fired


def test_too_late_is_noise():
    p = _pol()
    p.observe("u1", "the node thing, what's it called", 10.0)
    assert p.due(30.0) is None


def test_still_relevant():
    p = _pol()
    c = ws.detect_cue("the node thing, what's it called")
    assert p.still_relevant("the node thing, what's it called", c, "geometry nodes")
    assert not p.still_relevant("the node thing, what's it called, geometry nodes", c, "geometry nodes")
    assert not p.still_relevant("the node thing, what's it called, never mind I'll look it up later", c, "x")
    assert not p.still_relevant("the node thing, what's it called", c, "geometry nodes", tail_sil=0.05)
    assert p.still_relevant("the node thing, what's it called", c, "geometry nodes", tail_sil=1.0)


def test_parse_suggestion():
    assert ws.parse_suggestion("Geometry Nodes\n") == "Geometry Nodes"
    assert ws.parse_suggestion('"Dali."') == "Dali"
    assert ws.parse_suggestion("Word: pesto") == "pesto"
    assert ws.parse_suggestion("NONE") is None
    assert ws.parse_suggestion("") is None
    assert ws.parse_suggestion("it is probably the thing that people use for that") is None


def test_suggest_uses_post_fn_and_glossary():
    seen = {}

    def post(url, payload, timeout):
        seen.update(payload)
        return {"response": "Geometry Nodes"}
    w, dt = ws.suggest("blender node thing, what's it called", ["Zeke: trees"], post_fn=post,
                       glossary=["geometry nodes", "EEVEE"])
    assert w == "Geometry Nodes" and dt >= 0
    assert "geometry nodes; EEVEE" in seen["prompt"] and seen["options"]["temperature"] == 0


def test_suggest_error_is_none():
    def boom(*a):
        raise OSError("down")
    assert ws.suggest("x what's it called", post_fn=boom)[0] is None


def test_glossary_file_loads_and_is_public_safe(tmp_path):
    g = ws.load_glossary()
    assert "geometry nodes" in g and not any(x.startswith("#") for x in g)
    assert ws.load_glossary(tmp_path / "missing.txt") == []


def test_offer_is_a_question():
    assert ws.offer_line("pesto") == "pesto?" and ws.offer_line("pesto?") == "pesto?"
