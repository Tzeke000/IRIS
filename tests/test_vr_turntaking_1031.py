"""VR turn-taking (2026-10-31, Zeke: the room's turn-taking fixes must also work in the temple).

1. continuation merge   - host _vr_turntaking: he went on before hearing ANY of my temple reply -> one turn
2. backchannels         - vr_hear VRBackchannel: my "mm-hm" in a mid-thought pause; never my own echo
3. "yes" after my "?"   - host: an ack while I talked = no turn, unless I'd just asked -> his answer;
                          vr_hear: a short "okay" right after my question is not dropped as noise
4. graceful stop        - vr_temple hush fade_ms (bot side: harness unit/voice_fade_unit.gd); the room
                          mouth's FISH_HUSH_FADE_MS ramp (run under the Fish venv, skipped without it)
Nothing here touches live state: every path is monkeypatched to tmp_path.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "voice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


vt = _load("vr_temple_tool_tt", "tools/system/vr_temple_tool.py")
os.environ.setdefault("IRIS_VR_HEAR_LDPATH", "1")      # vr_hear re-execs itself for CUDA libs unless this is set
vh = _load("vr_hear_tt", "scripts/vr/vr_hear.py")

_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)
import iris_body_host as h  # noqa: E402

h._blog = lambda *a, **k: None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch, tmp_path):
    for k in ("IRIS_VR_TURNTAKING", "IRIS_VR_BACKCHANNEL", "IRIS_VR_HUSH_FADE_MS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(vt, "_CONTROL", str(tmp_path / "voice_control.json"))
    monkeypatch.setattr(vt, "_SPEECH", str(tmp_path / "my_speech.json"))
    monkeypatch.setattr(vh, "CONTROL", tmp_path / "voice_control.json")
    monkeypatch.setattr(vh, "MY_SPEECH", tmp_path / "my_speech.json")
    monkeypatch.setattr(h, "VR_CONTROL_PATH", str(tmp_path / "voice_control.json"))
    monkeypatch.setattr(h, "VR_SPEECH_PATH", str(tmp_path / "my_speech.json"))


# ───────────────────────────────────────────── switches
def test_everything_is_off_by_default(tmp_path):
    c = vt.vr_cfg()
    assert c == {"turntaking": False, "backchannel": False, "hush_fade_ms": 0.0}
    assert not h._vr_turntaking_on()
    assert not vh.switch("IRIS_VR_BACKCHANNEL", "vr_backchannel")


def test_switches_are_hot_in_voice_control(tmp_path):
    (tmp_path / "voice_control.json").write_text(json.dumps(
        {"vr_turntaking": True, "vr_backchannel": True, "vr_hush_fade_ms": 9999}))
    c = vt.vr_cfg()
    assert c["turntaking"] and c["backchannel"] and c["hush_fade_ms"] == 2000.0   # capped
    assert h._vr_turntaking_on()
    assert vh.switch("IRIS_VR_BACKCHANNEL", "vr_backchannel")


def test_env_forces_either_way(monkeypatch, tmp_path):
    (tmp_path / "voice_control.json").write_text(json.dumps({"vr_turntaking": True}))
    monkeypatch.setenv("IRIS_VR_TURNTAKING", "0")
    assert not h._vr_turntaking_on() and not vt.vr_cfg()["turntaking"]
    monkeypatch.setenv("IRIS_VR_TURNTAKING", "1")
    (tmp_path / "voice_control.json").write_text("{}")
    assert h._vr_turntaking_on()


# ───────────────────────────────────────────── vr_temple speak: the record of what played
class FakeBody:
    def __init__(self):
        self.sent = []

    def __call__(self, req, timeout=4.0):
        self.sent.append(req)
        if req["cmd"] == "voice":
            return {"ok": True, "seconds": 2.0}
        return {"ok": True}


class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def speak_env(monkeypatch, tmp_path):
    body = FakeBody()
    monkeypatch.setattr(vt, "_send", body)
    monkeypatch.setattr(vt, "_mouth_ready", lambda: True)
    monkeypatch.setattr(vt, "_VOICE_DIR", str(tmp_path / "voice"))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: _Resp(b"RIFFfake"))
    return body


def test_speak_barged_before_anything_played_records_zero_heard(monkeypatch, speak_env, tmp_path):
    calls = {"n": 0}

    def barge(t0):
        calls["n"] += 1
        return {"speaker": "Zeke", "dist_m": 1.5} if calls["n"] >= 2 else None
    monkeypatch.setattr(vt, "_barge_since", barge)
    r = vt._speak("Okay so the plan is this. Then we build the tower. Sound good?")
    assert r["interrupted"] and r["sentences_heard"] == 0
    rec = json.loads((tmp_path / "my_speech.json").read_text())
    assert rec["state"] == "interrupted" and rec["by"] == "Zeke" and rec["heard_n"] == 0
    assert len(rec["spans"]) == 1 and rec["sentences"][-1].endswith("?")
    assert speak_env.sent[-1] == {"cmd": "hush"}              # no fade configured: the old hard cut


def test_speak_hush_fades_when_configured(monkeypatch, speak_env, tmp_path):
    (tmp_path / "voice_control.json").write_text(json.dumps({"vr_hush_fade_ms": 400}))
    monkeypatch.setattr(vt, "_barge_since", lambda t0: {"speaker": "Zeke", "dist_m": 1.0})
    vt._speak("Hello there, how are you doing?")
    assert speak_env.sent[-1] == {"cmd": "hush", "fade_ms": 400.0}


def test_hush_action_gets_the_default_fade(monkeypatch, speak_env, tmp_path):
    vt._vr_temple({"action": "hush"}, {})
    assert speak_env.sent[-1] == {"cmd": "hush"}
    (tmp_path / "voice_control.json").write_text(json.dumps({"vr_hush_fade_ms": 350}))
    vt._vr_temple({"action": "hush"}, {})
    assert speak_env.sent[-1] == {"cmd": "hush", "fade_ms": 350.0}
    vt._vr_temple({"action": "hush", "fade_ms": 0}, {})          # an explicit value wins
    assert speak_env.sent[-1] == {"cmd": "hush", "fade_ms": 0}


def test_speak_to_the_end_records_done(monkeypatch, speak_env, tmp_path):
    monkeypatch.setattr(vt, "_barge_since", lambda t0: None)
    monkeypatch.setattr(vt, "_send", lambda req, timeout=4.0: {"ok": True, "seconds": 0.05})
    r = vt._speak("This is the first sentence. Another sentence here.")
    rec = json.loads((tmp_path / "my_speech.json").read_text())
    assert r.get("sentences") == 2 and rec["state"] == "done" and rec["heard_n"] == 2


# ───────────────────────────────────────────── host: continuation merge + ack after a question
T = 1_000_000.0


def line(text, ts, speaker="Zeke", dur=1.5, **kw):
    d = {"speaker": speaker, "text": text, "ts": ts, "ts_end": ts + dur, "attending": True,
         "t_start": dt.datetime.fromtimestamp(ts).isoformat()}
    d.update(kw)
    return d


def speech(t0, sentences, spans, state, **kw):
    d = {"t0": t0, "sentences": sentences, "spans": spans, "state": state}
    d.update(kw)
    return d


def test_mid_thought_continuation_is_one_turn():
    first = [line("so I was thinking about the tower", T)]
    out, last, _ = h._vr_turntaking(first, None, None, T + 2.5)
    assert out == first and last["speaker"] == "Zeke"
    # I start answering at T+4; he goes on at T+5 - my first sentence never finished playing
    sp = speech(T + 4, ["Oh nice, a tower!", "What kind?"], [[T + 4.2, T + 6.0]], "interrupted",
                cut=T + 5.7, by="Zeke", heard_n=0)
    out, last2, notes = h._vr_turntaking([line("and then I would probably you know build it tall", T + 5.0)],
                                         sp, last, T + 7.5)
    assert len(out) == 1 and "continuation" in out[0]["_pre"]
    assert out[0]["text"] == "so I was thinking about the tower ... and then I would probably you know build it tall"
    assert any("continuation merge" in n for n in notes)
    prompt = h._vr_heard_prompt(out)
    assert "[continuation" in prompt and "<<<TEMPLE so I was thinking" in prompt
    assert prompt.index("[continuation") < prompt.index("<<<TEMPLE so I was")   # the tag is OUTSIDE the fence
    # a chain stays clean: a third piece joins the already-joined words
    sp3 = speech(T + 9, ["Got it."], [[T + 9.2, T + 10]], "interrupted", cut=T + 9.5, by="Zeke", heard_n=0)
    out3, _, _ = h._vr_turntaking([line("with a flag on top", T + 9.4)], sp3, last2, T + 11)
    assert out3[0]["text"].endswith("build it tall ... with a flag on top") and out3[0]["text"].count("[") == 0


@pytest.mark.parametrize("change", ["heard_one", "other_speaker", "too_late", "old_speech", "not_barged",
                                    "barged_by_someone_else"])
def test_no_merge(change):
    _, last, _ = h._vr_turntaking([line("so I was thinking", T)], None, None, T + 2.5)
    sp = speech(T + 4, ["Nice."], [[T + 4.2, T + 5.0]], "interrupted", cut=T + 5.7, by="Zeke", heard_n=0)
    nxt = line("and another thing", T + 5.0)
    if change == "heard_one":
        sp["heard_n"] = 1
    elif change == "other_speaker":
        nxt["speaker"] = "Guest"
        sp["by"] = "Guest"
    elif change == "too_late":
        nxt = line("and another thing", T + 1.5 + 21)
    elif change == "old_speech":
        sp["t0"] = T - 30
    elif change == "not_barged":
        sp["state"] = "done"
    elif change == "barged_by_someone_else":
        sp["by"] = "Guest"
    out, _, _ = h._vr_turntaking([nxt], sp, last, T + 7)
    assert out == [nxt]


def test_ack_while_i_talk_is_no_turn():
    sp = speech(T, ["The tower is done.", "It has six floors."], [[T, T + 2], [T + 2, T + 4]], "speaking")
    out, last, notes = h._vr_turntaking([line("yeah", T + 1.0, dur=0.4)], sp, None, T + 2)
    assert out == [] and any("no turn" in n for n in notes)


def test_ack_after_my_question_is_their_answer():
    sp = speech(T, ["Want me to build it?", "I can start now."], [[T, T + 1.8], [T + 1.8, T + 3.5]], "speaking")
    out, _, notes = h._vr_turntaking([line("yeah", T + 2.2, dur=0.4)], sp, None, T + 3)
    assert len(out) == 1 and "treat as their answer" in out[0]["_pre"]
    assert "treat as their answer" in h._vr_heard_prompt(out)


def test_ack_outside_my_speech_is_a_normal_turn():
    sp = speech(T, ["Done."], [[T, T + 1]], "done")
    ln = line("okay", T + 5)
    assert h._vr_turntaking([ln], sp, None, T + 6)[0] == [ln]


def test_real_words_while_i_talk_still_wake_me():
    sp = speech(T, ["The tower is done."], [[T, T + 3]], "speaking")
    ln = line("wait stop, put the flag on the left", T + 1)
    assert h._vr_turntaking([ln], sp, None, T + 2)[0] == [ln]


def test_mixed_batch_keeps_the_real_line():
    sp = speech(T, ["Look at this."], [[T, T + 3]], "speaking")
    ack, real = line("mm-hm", T + 0.5, dur=0.3), line("what is that thing", T + 2)
    assert h._vr_turntaking([ack, real], sp, None, T + 3)[0] == [real]


def test_gesture_lines_pass_untouched():
    sp = speech(T, ["Look."], [[T, T + 3]], "speaking")
    g = line("yeah", T + 1, kind="gesture")
    assert h._vr_turntaking([g], sp, None, T + 2)[0] == [g]


# ───────────────────────────────────────────── vr_hear: an "okay" answer survives the noise filter
class _Seg:
    def __init__(self, text, start=0.0, end=0.4, nsp=0.05, lp=-0.3):
        self.text, self.start, self.end, self.no_speech_prob, self.avg_logprob, self.words = text, start, end, nsp, lp, []


class _Model:
    def __init__(self, text):
        self.text = text

    def transcribe(self, x, **kw):
        return [_Seg(self.text)], None


def _ears(text):
    e = object.__new__(vh.Ears)
    e.model, e.attn, e.keep_wav = _Model(text), {"Zeke": 1e12}, None
    return e


def _tone(secs=0.6, amp=0.1):
    t = np.arange(int(16000 * secs)) / 16000.0
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_short_okay_after_my_question_is_kept(tmp_path, monkeypatch):
    import time as _t
    now = _t.time()
    meta = {"speaker": "Zeke", "t_start": now, "t_end": now + 0.6, "dist_m": 2.0, "atten_db": 0.0}
    assert _ears("Okay.").transcribe(meta, _tone(), "s1") is None          # off: still dropped as noise
    (tmp_path / "voice_control.json").write_text(json.dumps({"vr_turntaking": True}))
    assert _ears("Okay.").transcribe(meta, _tone(), "s1") is None          # on, but I asked nothing
    (tmp_path / "my_speech.json").write_text(json.dumps(speech(now - 5, ["Shall I build it?"], [[now - 5, now - 3]], "done")))
    got = _ears("Okay.").transcribe(meta, _tone(), "s1")
    assert got and got["text"] == "Okay."
    (tmp_path / "my_speech.json").write_text(json.dumps(speech(now - 200, ["Shall I?"], [[now - 200, now - 199]], "done")))
    assert _ears("Okay.").transcribe(meta, _tone(), "s1") is None          # the question was too long ago
    (tmp_path / "my_speech.json").write_text(json.dumps(speech(now - 5, ["Shall I build it?"], [[now - 5, now - 3]], "done")))
    assert _ears("Thanks for watching.").transcribe(meta, _tone(), "s1") is None   # only answer-words are spared


# ───────────────────────────────────────────── vr_hear: backchannels
DAY = dt.datetime(2026, 10, 31, 14, 0)
NIGHT = dt.datetime(2026, 10, 31, 23, 0)
ON = {"vr_backchannel": True}


def _buf(voiced_s=5.0, dist=2.0, stem="u1"):
    x = np.concatenate([_tone(voiced_s), np.zeros(int(16000 * 0.6), np.float32)])
    return {"x": x, "meta": {"dist_m": dist}, "t_end": 0.0, "n": 3, "stem": stem}


@pytest.fixture
def bc(tmp_path):
    sent = []
    b = vh.VRBackchannel(send=lambda req: (sent.append(req), {"ok": True, "seconds": 0.5})[1],
                         predict=lambda a: 0.2)
    clip = tmp_path / "bc_0.wav"
    clip.write_bytes(b"x")
    b.clips = [(str(clip), 0.5)]
    b.sent = sent
    return b


def test_backchannel_plays_in_a_mid_thought_pause(bc):
    r = bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=None, now_dt=DAY)
    assert r.startswith("played") and bc.sent[-1]["cmd"] == "voice" and bc.sent[-1]["path"].endswith("bc_0.wav")
    assert bc.windows[-1] == (100.0, 100.5)


@pytest.mark.parametrize("why,kw,ctl,now_dt", [
    ("vr_backchannel off", {}, {}, DAY),
    ("mouth_muted", {}, {"vr_backchannel": True, "mouth_muted": True}, DAY),
    ("quiet hours", {}, ON, NIGHT),
    ("not a backchannel speaker", {"who": "Guest"}, ON, DAY),
    ("not talking to me", {"attending": False}, ON, DAY),
    ("too far", {"buf": _buf(dist=9.0)}, ON, DAY),
    ("not long enough", {"buf": _buf(voiced_s=2.0)}, ON, DAY),
])
def test_backchannel_gates(bc, why, kw, ctl, now_dt):
    r = bc.consider(kw.get("who", "Zeke"), kw.get("buf", _buf()), kw.get("attending", True),
                    now=100.0, ctl=ctl, sp=None, now_dt=now_dt)
    assert r.startswith(("off: " + why, why)) and not bc.sent


def test_guests_only_when_configured(bc):
    ctl = {"vr_backchannel": True, "backchannel_cfg": {"vr_speakers": "*"}}
    assert bc.consider("Guest", _buf(), True, now=100.0, ctl=ctl, sp=None, now_dt=DAY).startswith("played")


def test_no_backchannel_when_they_sound_done(bc):
    bc._predict = lambda a: 0.9
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=None, now_dt=DAY).startswith("they sound done")


def test_no_backchannel_without_smart_turn(bc):
    def boom(a):
        raise RuntimeError("no model")
    bc._predict = boom
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=None, now_dt=DAY) == "no smart-turn"


def test_no_backchannel_while_i_talk(bc):
    sp = speech(95.0, ["Long line."], [[95.0, 103.0]], "speaking")
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=sp, now_dt=DAY) == "I'm talking"
    sp_done = speech(95.0, ["Long line."], [[95.0, 100.2]], "done")       # still playing out
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=sp_done, now_dt=DAY) == "I'm talking"
    stale = speech(0.0, ["x"], [[0.0, 1.0]], "speaking")                  # a crashed speak never blocks forever
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=stale, now_dt=DAY).startswith("played")


def test_rate_limits(bc):
    b = _buf()
    assert bc.consider("Zeke", b, True, now=100.0, ctl=ON, sp=None, now_dt=DAY).startswith("played")
    assert bc.consider("Zeke", b, True, now=103.0, ctl=ON, sp=None, now_dt=DAY) == "rate limit"
    # 6 s later but no new speech in the same utterance: never two in a row
    assert bc.consider("Zeke", b, True, now=107.0, ctl=ON, sp=None, now_dt=DAY).startswith("no new speech")
    b["x"] = np.concatenate([b["x"], _tone(2.0)])
    assert bc.consider("Zeke", b, True, now=108.0, ctl=ON, sp=None, now_dt=DAY).startswith("played")


def test_my_clip_coming_back_is_not_their_words(bc):
    bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=None, now_dt=DAY)     # plays 100.0-100.5
    assert bc.is_my_echo(100.3, 100.9)              # short, inside the window + round trip
    assert not bc.is_my_echo(100.3, 103.0)          # long: that's them talking
    assert not bc.is_my_echo(102.0, 102.4)          # after the window
    assert not bc.is_my_echo(99.0, 100.4)           # started before my clip did


def test_no_clips_yet_primes_instead_of_playing(bc, monkeypatch):
    bc.clips = []
    started = []
    monkeypatch.setattr(bc, "prime_async", lambda: started.append(1))
    assert bc.consider("Zeke", _buf(), True, now=100.0, ctl=ON, sp=None, now_dt=DAY) == "priming"
    assert started and not bc.sent


def test_prime_writes_valid_clips(tmp_path, monkeypatch):
    import io
    import wave

    def wav(secs):
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes(b"\0\0" * int(48000 * secs))
        return bio.getvalue()
    b = vh.VRBackchannel(synth=lambda text: wav(3.0 if text == "Mm." else 0.5))
    monkeypatch.setattr(b, "VOICE_DIR", tmp_path)
    monkeypatch.setattr(b, "PHRASES", ("Mm-hm.", "Mm."))
    assert b.prime() == 1 and b.clips[0][0].endswith("bc_0.wav")
    assert not (tmp_path / "bc_1.wav").exists()                           # the 3 s render is not a backchannel


def test_voiced_seconds_ignores_silence():
    x = np.concatenate([_tone(2.0), np.zeros(32000, np.float32), _tone(1.0)])
    assert 2.8 <= vh.voiced_seconds(x) <= 3.1


# ───────────────────────────────────────────── the room mouth's fade (Fish venv only)
FISH_PY = Path.home() / "voice_lab" / "fish" / "venv" / "bin" / "python"

_FADE_SCRIPT = r"""
import sys, threading, time, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import wren_fish_server as f
SR = f.SAMPLE_RATE
class S:
    def __init__(s): s.out = []; s.aborted = False
    def write(s, b): s.out.append(np.array(b)); time.sleep(len(b) / SR)
    def abort(s): s.aborted = True
res = {}
for name, fade in (("hard", None), ("fade", 400.0)):
    s = S(); f._OUT_STREAM = s; f._STOP.clear(); f._FADE.update(left=0, total=0)
    arr = np.full(SR * 3, 0.5, np.float32); ret = {}
    th = threading.Thread(target=lambda: ret.setdefault("r", f._write_stream(s, arr))); th.start()
    time.sleep(0.3); n_before = sum(len(b) for b in s.out); mode = f._request_stop(fade); th.join(5)
    a = np.concatenate(s.out); tail = a[n_before:]
    res[name] = {"mode": mode, "ret": ret.get("r"), "after_ms": 1000 * len(tail) / SR, "aborted": s.aborted,
                 "last": float(abs(a[-1])), "mono": bool(np.all(np.diff(np.abs(tail)) <= 1e-6)) if len(tail) > 1 else True,
                 "stop": f._STOP.is_set()}
# a stale fade never leaks into the next line
f._STOP.clear(); f._FADE.update(left=0, total=0)
s = S(); f._OUT_STREAM = s; f._FADE["last_write"] = 0.0
res["idle"] = {"mode": f._request_stop(400.0)}
print(json.dumps(res))
"""


@pytest.mark.skipif(not FISH_PY.exists(), reason="needs the Fish venv (the mouth's own python)")
def test_room_mouth_fade(tmp_path):
    env = dict(os.environ, IRIS_FISH_ROOT=str(Path.home() / "voice_lab" / "fish"), IRIS_MOUTH_DRYRUN="1",
               IRIS_MOUTH_DRYRUN_DIR=str(tmp_path / "dry"))
    env.pop("FISH_HUSH_FADE_MS", None)
    p = subprocess.run([str(FISH_PY), "-c", _FADE_SCRIPT, str(ROOT / "voice")], env=env, capture_output=True,
                       text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-2000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    hard, fade = res["hard"], res["fade"]
    assert hard["mode"] == "stopping" and hard["aborted"] and hard["after_ms"] <= 60 and hard["ret"] is False
    assert fade["mode"] == "fading" and not fade["aborted"] and fade["ret"] is False and fade["stop"]
    assert 380 <= fade["after_ms"] <= 460, fade          # ~400 ms more, ramped
    assert fade["mono"] and fade["last"] < 0.01           # only ever quieter, ends in silence
    assert res["idle"]["mode"] == "stopping"              # nothing playing: no fade to ride, plain stop


def test_smart_turn_gets_real_scale_audio_even_from_int_scaled_decodes():
    seen = []
    b = vh.VRBackchannel(predict=lambda a: (seen.append(float(np.abs(a).max())), 0.3)[1])
    assert b.endpoint_prob(_tone(1.0, amp=0.5) * 32768.0) == 0.3 and seen[-1] <= 1.0   # decode_opus int scale
    b.endpoint_prob(_tone(1.0, amp=0.5))
    assert abs(seen[-1] - 0.5) < 1e-3                                                   # real scale untouched
