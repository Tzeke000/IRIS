"""VR-191 (Zeke 10-29, VR): my ears missed him - "Okay, hey, I am back!" had no name ⇒ never routed to me.
A player I'm WITH (following them / they told me to wait / we've been talking) reaches me without my name for the
lines that are plausibly to me; strangers and other chatter still need it; the 90 s name window is unchanged."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import tools.tool_registry  # noqa: F401
except Exception:  # noqa: BLE001
    reg = types.ModuleType("tools.tool_registry")
    reg.register_tool = lambda *a, **k: (lambda f: f)
    pkg = sys.modules.get("tools") or types.ModuleType("tools")
    pkg.__path__ = [str(ROOT / "tools")]
    sys.modules["tools"] = pkg
    sys.modules["tools.tool_registry"] = reg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


os.environ.setdefault("IRIS_VR_HEAR_LDPATH", "1")
vh = _load("vr_hear_with", "scripts/vr/vr_hear.py")
vt = _load("vr_temple_tool_with", "tools/system/vr_temple_tool.py")


class _Seg:
    def __init__(self, text):
        self.text, self.no_speech_prob, self.avg_logprob, self.start, self.words = text, 0.0, -0.2, 0.0, None


class _Model:
    def __init__(self):
        self.text = ""

    def transcribe(self, x, **kw):
        return [_Seg(self.text)], None


@pytest.fixture
def ears(tmp_path, monkeypatch):
    monkeypatch.setattr(vh, "FOLLOW_FILE", tmp_path / "follow.json")
    e = vh.Ears.__new__(vh.Ears)
    e.spool, e.out, e.state_path = tmp_path / "spool", tmp_path / "heard.jsonl", tmp_path / "att.json"
    e.keep_wav, e.gestures, e.live_on, e.live, e.live_last = None, None, False, {}, {}
    e.bc, e.attn, e.model = vh.VRBackchannel(), {}, _Model()
    return e


X = (0.3 * np.sin(np.arange(32000) * 2 * np.pi * 220 / 16000)).astype(np.float32)    # 2 s, loud
T0 = 1_800_000_000.0


def say(e, who, text, t):
    e.model.text = text
    line = e.transcribe({"speaker": who, "t_start": t, "t_end": t + 2, "dist_m": 3.0, "atten_db": -6}, X, "s%d" % int(t))
    assert line is not None, text
    return line


# ─────────────────────────────────────── the exact 10-29 transcript
def test_10_29_stay_here_then_back_six_minutes_later_is_routed(ears):
    assert say(ears, "Zeke", "Iris, come look at this.", T0)["addressed"]
    stay = say(ears, "Zeke", "Stay here, I'll be back.", T0 + 20)          # inside the 90 s name window
    assert stay["attending"] and stay["with"] == "wait"
    back = say(ears, "Zeke", "Okay, hey, I am back!", T0 + 20 + 6 * 60)    # 6 min later: no name
    assert back["attending"] and not back["addressed"] and back["routed"] == "with:wait:return"
    again = say(ears, "Zeke", "I hope you hear me, I am back.", T0 + 20 + 6 * 60 + 5)
    assert again["attending"]


def test_stay_here_with_the_name_in_it(ears):
    say(ears, "Zeke", "Iris, stay here, I'll be back.", T0)
    assert say(ears, "Zeke", "Okay, hey, I am back!", T0 + 25 * 60)["attending"]


def test_the_wait_lasts_thirty_minutes_not_forever(ears):
    say(ears, "Zeke", "Iris, stay here, I'll be back.", T0)
    late = say(ears, "Zeke", "Okay, hey, I am back!", T0 + 31 * 60)
    assert not late["attending"] and late["with"] is None


# ─────────────────────────────────────── strangers + other chatter
def test_a_strangers_hey_is_not_routed(ears):
    say(ears, "Zeke", "Iris, stay here, I'll be back.", T0)
    for txt in ("Hey!", "Hey, I'm back.", "Are you there?"):
        d = say(ears, "Bee", txt, T0 + 300)
        assert not d["attending"] and d["with"] is None, txt


def test_plain_chatter_from_a_player_im_with_is_not_routed(ears):
    say(ears, "Zeke", "Iris, what's that over there", T0)
    for i, txt in enumerate(("That tower is huge.", "I think the server lagged earlier.")):
        assert not say(ears, "Zeke", txt, T0 + 200 + i * 10)["attending"], txt


def test_greeting_and_question_need_my_name_when_someone_else_is_talking(ears):
    say(ears, "Zeke", "Iris, stay here.", T0)
    say(ears, "Bee", "Yo what's up", T0 + 300)                               # another player, 20 s before
    assert not say(ears, "Zeke", "Hey, what's up?", T0 + 320)["attending"]
    assert say(ears, "Zeke", "Okay I'm back.", T0 + 330)["attending"]       # a return is direct: still mine


def test_greeting_and_question_route_when_its_just_us(ears):
    say(ears, "Zeke", "Iris, stay here.", T0)
    g = say(ears, "Zeke", "Hey!", T0 + 300)
    assert g["attending"] and g["routed"] == "with:wait:greeting"
    q = say(ears, "Zeke", "Where did everyone go?", T0 + 400)
    assert q["attending"] and q["routed"].endswith(":question")


# ─────────────────────────────────────── WITH ends
@pytest.mark.parametrize("bye", ["Okay, bye!", "See you later.", "Good night.", "Alright I'm logging off."])
def test_with_ends_on_bye(ears, bye):
    say(ears, "Zeke", "Iris, stay here, I'll be back.", T0)
    b = say(ears, "Zeke", bye, T0 + 120)
    assert b["attending"] and b["with"] is None           # the goodbye itself reaches me, then WITH is over
    assert not say(ears, "Zeke", "Hey, I'm back.", T0 + 300)["attending"]


def test_bye_with_a_promise_to_return_keeps_waiting(ears):
    say(ears, "Zeke", "Iris, bye for now, I'll be back.", T0)
    assert say(ears, "Zeke", "Okay I'm back!", T0 + 600)["attending"]


def test_following_makes_me_with_them_and_stopping_ends_it(ears, tmp_path):
    (tmp_path / "follow.json").write_text(json.dumps({"name": "zeke", "on": True, "t": T0}))
    d = say(ears, "Zeke", "You still there?", T0 + 60)
    assert d["attending"] and d["routed"] == "with:follow:check"
    assert not say(ears, "Bee", "You still there?", T0 + 70)["attending"]   # following Zeke, not Bee
    (tmp_path / "follow.json").write_text(json.dumps({"name": "", "on": False, "t": T0 + 100}))
    ears.withs.end("Zeke")
    assert not say(ears, "Zeke", "You still there?", T0 + 200)["attending"]


def test_conversation_window_ten_minutes(ears):
    say(ears, "Zeke", "Iris, how are you", T0)
    assert say(ears, "Zeke", "Hey, you there?", T0 + 9 * 60)["attending"]
    assert not say(ears, "Zeke", "Hey, you there?", T0 + 9 * 60 + 11 * 60)["attending"]


# ─────────────────────────────────────── the old path is unchanged
def test_name_window_unchanged_for_non_with_speakers(ears):
    ears.withs.with_.clear()
    d = say(ears, "Bee", "That's cool", T0)
    assert not d["attending"] and not d["addressed"]
    say(ears, "Bee", "Iris look", T0 + 10)
    assert ears.attn["Bee"] == T0 + 12 + vh.ATTEND_S
    assert say(ears, "Bee", "That's cool", T0 + 12 + vh.ATTEND_S - 1)["attending"]      # inside 90 s: as before


def test_state_file_round_trips_with(ears):
    say(ears, "Zeke", "Iris, stay here.", time.time())
    ears.save_state()
    st = json.loads(ears.state_path.read_text())
    assert st["with"]["Zeke"]["why"] == "wait" and st["with_now"] == ["Zeke"] and st["attend_s"] == vh.ATTEND_S
    e2 = vh.Ears.__new__(vh.Ears)
    e2.state_path = ears.state_path
    assert e2.withs.is_with("Zeke", time.time()) == "wait"


# ─────────────────────────────────────── vr_temple follow writes the ears' copy
def test_vr_temple_follow_writes_follow_file(tmp_path, monkeypatch):
    f = tmp_path / "follow.json"
    monkeypatch.setattr(vt, "_FOLLOW", str(f))
    vt._note_follow("follow", {"name": "Zeke", "dist": 3}, {"ok": True})
    assert json.loads(f.read_text())["name"] == "Zeke" and json.loads(f.read_text())["on"] is True
    vt._note_follow("follow", {}, {"ok": True, "mode": "follow"})             # status only: unchanged
    assert json.loads(f.read_text())["on"] is True
    vt._note_follow("follow", {"name": "Bee"}, {"ok": False})                 # refused: unchanged
    assert json.loads(f.read_text())["name"] == "Zeke"
    vt._note_follow("move", {"to": [0, 0, 0]}, {"ok": True})                  # an explicit move ends follow
    assert json.loads(f.read_text())["on"] is False
    vt._note_follow("follow", {"name": "Zeke", "on": False}, {"ok": True})
    assert json.loads(f.read_text())["on"] is False


@pytest.mark.parametrize("txt,want", [
    ("Okay, hey, I am back!", "return"), ("I hope you hear me, I am back", "return"), ("im back", "return"),
    ("Can you hear me?", "check"), ("Iris? You there?", "check"),
])
def test_direct_phrases(txt, want):
    w = vh.WithTracker({"Z": {"until": T0 + 999, "why": "wait"}}, follow_path=Path("/nonexistent"))
    w.heard["Other"] = T0                                                     # direct phrases ignore other chatter
    assert w.route("Z", txt, T0) == "with:wait:%s" % want


@pytest.mark.parametrize("txt", ["They're back on the server", "I'm going to build here", "you know what I mean",
                                 "the heyday of it", "back there", "this is wild"])
def test_not_to_me(txt):
    w = vh.WithTracker({"Z": {"until": T0 + 999, "why": "convo"}}, follow_path=Path("/nonexistent"))
    w.heard["Other"] = T0
    assert w.route("Z", txt, T0) is None, txt


def test_i_can_see_you_does_not_end_with(ears):
    say(ears, "Zeke", "Iris, stay here, I'll be back.", T0)
    say(ears, "Zeke", "I can see you from here", T0 + 60)
    assert say(ears, "Zeke", "Okay I'm back!", T0 + 600)["attending"]
