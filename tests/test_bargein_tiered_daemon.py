"""wren_voice_core._cmd_bargein_watch_tiered end to end with a fake mic / VAD / whisper / mouth.

Proves the daemon wiring of the tiers: a backchannel never stops my mouth, real speech pauses
then barges with the usual cut marker, my own echo is resumed, an unverified speaker is deferred
(no /stop, '[barge-defer]'), and the binary watch is untouched unless the host asks for tiers."""
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "voice"))

core = pytest.importorskip("wren_voice_core")
import wren_voice_status  # noqa: E402

LOUD = np.full(512, 0.1, np.float32)
QUIET = np.zeros(512, np.float32)


class _Prob:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class FakeVAD:
    def __call__(self, t, sr):
        return _Prob(0.95 if float(t.abs().mean()) > 0.05 else 0.02)

    def reset_states(self):
        pass


class FakeMic:
    def __init__(self, script):
        self._it = iter(script)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        try:
            c = next(self._it)
        except StopIteration:
            c = QUIET
        return c.reshape(-1, 1), False


class Ctx:
    def __init__(self):
        self.vad_lock = threading.Lock()
        self.vad_model = FakeVAD()
        self.mic_lock = threading.RLock()
        self.bargein_hold = True
        self.bargein_fired = False
        self.server_url = "http://mouth.test"
        self.play_queue = queue.Queue()
        self.play_queue.put("Sentence two.")
        self.now_playing = "Sentence one, half said."
        self.played_chunks = []
        self.speaking = True


@pytest.fixture
def rig(monkeypatch):
    urls = []
    state = {"script": [], "heard": "", "self": None, "verdict": {"status": "skipped"}}

    class _Resp:
        def read(self):
            return b"ok"

    monkeypatch.setattr(core._req, "urlopen", lambda u, timeout=0: (urls.append(u.split("mouth.test")[1]), _Resp())[1])
    monkeypatch.setattr(core, "is_muted", lambda: False)
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    monkeypatch.setattr(core, "_trusted_mic_or_none", lambda ctx: "net")
    monkeypatch.setattr(core._net_audio, "open_mic", lambda sd, sr, idx, bs: FakeMic(state["script"]))
    monkeypatch.setattr(core.wl, "_transcribe_array", lambda a: state["heard"])
    monkeypatch.setattr(core.wl, "append_transcript", lambda t: None)
    monkeypatch.setattr(core, "_self_voice_score", lambda a, bar: state["self"])
    monkeypatch.setattr(core, "_voice_id_verdict", lambda a: state["verdict"])
    monkeypatch.setattr(core, "_voice_id_line", lambda a, v=None: "")
    monkeypatch.setattr(core, "_save_utt_wav", lambda a, tag: None)
    monkeypatch.setattr(core, "set_state", lambda *a, **k: None)
    return urls, state


def speech(ms):
    return [LOUD] * int(ms / 32) + [QUIET] * 25


def test_backchannel_is_an_ack_and_never_stops_the_mouth(rig):
    urls, st = rig
    st["script"], st["heard"] = speech(450), "Mm-hmm."
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    assert out == "[bargein_watch] ack: Mm-hmm."
    assert "/stop" not in urls and urls[0].startswith("/duck?db=-12") and urls[-1].startswith("/duck?db=0")
    assert ctx.play_queue.qsize() == 1 and not ctx.bargein_fired


def test_real_speech_barges_with_cut_marker(rig):
    urls, st = rig
    st["script"], st["heard"] = speech(1600), "no use the other render"
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    assert out.startswith("[barge-in] [cut mid-sentence at: \"Sentence one, half said.\"")
    assert out.endswith("no use the other render")
    assert "/stop" in urls and ctx.bargein_fired


def test_own_echo_pauses_then_resumes_the_reply(rig):
    urls, st = rig
    st["script"], st["heard"], st["self"] = speech(900), "so let me show you", 0.83
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    assert out.startswith("[bargein_watch] self_voice_dropped")
    assert "/stop" in urls
    q = [ctx.play_queue.get_nowait() for _ in range(ctx.play_queue.qsize())]
    assert q == ["Sentence one, half said.", "Sentence two."] and not ctx.bargein_fired


def test_unverified_speaker_is_deferred_not_barged(rig):
    urls, st = rig
    st["script"], st["heard"] = speech(2000), "is this the AI everyone talks about"
    st["verdict"] = {"status": "unknown", "speaker": "unknown", "score": 0.2}
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    # the verdict needs the whole utterance: 2 s of speech paused me, the final verdict resumed me
    assert out.startswith("[barge-defer] [talked over me; I paused, then resumed my reply]")
    assert "/stop" in urls and not ctx.bargein_fired and ctx.play_queue.qsize() == 2


def test_short_unverified_overlap_only_ducks(rig):
    urls, st = rig
    st["script"], st["heard"] = speech(500), "hello there"
    st["verdict"] = {"status": "unknown", "speaker": "unknown", "score": 0.2}
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    assert out.startswith("[barge-defer] [talked over me; my reply kept playing]")
    assert "/stop" not in urls and ctx.play_queue.qsize() == 1


def test_cough_resumes(rig):
    urls, st = rig
    st["script"], st["heard"] = speech(800), ""
    ctx = Ctx()
    out = core._cmd_bargein_watch_tiered(ctx, {"tiers": True}, None)
    assert out == "[bargein_watch] interrupted (unclear)"
    assert ctx.play_queue.qsize() == 2 and not ctx.bargein_fired


def test_binary_watch_is_default(monkeypatch):
    monkeypatch.setattr(core, "_cmd_bargein_watch_impl", lambda ctx, args, live: "BINARY")
    monkeypatch.setattr(core, "_cmd_bargein_watch_tiered", lambda ctx, args, live: "TIERED")
    monkeypatch.setattr(core, "_new_live", lambda *a, **k: None)
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {})
    assert core.cmd_bargein_watch(object(), {}) == "BINARY"
    assert core.cmd_bargein_watch(object(), {"tiers": True}) == "TIERED"
    monkeypatch.setattr(wren_voice_status, "read_control", lambda: {"duplex_tiers": False})
    assert core.cmd_bargein_watch(object(), {"tiers": True}) == "BINARY"
