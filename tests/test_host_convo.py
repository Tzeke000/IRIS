"""Host side of the conversation layer (IRIS_CONVO_LOOKUPS / IRIS_WORD_ASSIST, default OFF)."""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_fake_av = types.ModuleType("avaagent")
_fake_av.load_profile_by_id = lambda person_id: {}  # type: ignore[attr-defined]
sys.modules.setdefault("avaagent", _fake_av)

import iris_body_host as h  # noqa: E402
from brain import iris_lookups as L  # noqa: E402

h._blog = lambda *a, **k: None  # type: ignore[assignment]
ZEKE = {"status": "matched", "speaker": "zeke", "score": 0.95, "threshold": 0.5}


def test_off_by_default(monkeypatch):
    import importlib
    monkeypatch.delenv("IRIS_CONVO_LOOKUPS", raising=False)
    monkeypatch.delenv("IRIS_WORD_ASSIST", raising=False)
    from brain import iris_convo_layer
    importlib.reload(iris_convo_layer)
    m = importlib.reload(h)
    m._blog = lambda *a, **k: None
    assert m.CONVO_LOOKUPS is False and m.WORD_ASSIST is False
    assert m._voice_turn_extras() == ""             # voice prompt byte-identical to before


def test_extras_when_on(monkeypatch):
    monkeypatch.setattr(h, "CONVO_LOOKUPS", True)
    h._convo.OFFERS.add("word offer", "pesto?", "the green sauce, what's it called")
    x = h._voice_turn_extras()
    assert "pesto?" in x and "lookup.py start" in x and "checking, give me a second" in x


def test_note_heard_only_for_verified():
    h._CONVO["last_trusted_voice_ts"] = 0.0
    h._note_heard(json.dumps({"status": "unknown", "score": 0.2}))
    assert h._CONVO["last_trusted_voice_ts"] == 0.0
    h._note_heard(ZEKE)
    assert h._CONVO["last_trusted_voice_ts"] > 0


def test_word_assist_gate(monkeypatch):
    monkeypatch.setattr(h, "BARGEIN_ON", True)
    monkeypatch.setattr(h._convo, "little_brain_parked", lambda: False)
    monkeypatch.setattr(h._convo, "may_speak_now", lambda: (True, ""))
    h._INBOX["turn"] = types.SimpleNamespace(speak_window=False)
    h._CONVO["last_trusted_voice_ts"] = 0.0
    assert h._word_assist_gate()[0] is False          # never heard a verified Zeke recently
    h._note_heard(ZEKE)
    assert h._word_assist_gate() == (True, "")
    h._INBOX["turn"] = types.SimpleNamespace(speak_window=True)
    assert h._word_assist_gate()[0] is False          # I'm talking
    h._INBOX["turn"] = types.SimpleNamespace(speak_window=False)
    monkeypatch.setattr(h, "BARGEIN_ON", False)
    assert "speakers" in h._word_assist_gate()[1]
    monkeypatch.setattr(h, "BARGEIN_ON", True)
    monkeypatch.setattr(h._convo, "little_brain_parked", lambda: True)
    assert "parked" in h._word_assist_gate()[1]
    h._INBOX["turn"] = None


def _run_reader(monkeypatch, tmp_path, partials):
    """Run lookup_reader for a few polls with a fake live partial sequence; return queued items."""
    monkeypatch.setattr(L, "DIR", tmp_path)
    seq = iter(partials)
    monkeypatch.setattr(h, "_his_live_partial", lambda: next(seq, None))

    async def go():
        q = asyncio.PriorityQueue()
        t = asyncio.create_task(h.lookup_reader(q, asyncio.get_running_loop(), poll_s=0.01))
        await asyncio.sleep(0.25)
        t.cancel()
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out
    return asyncio.run(go())


def test_lookup_reader_enqueues_answer_once(monkeypatch, tmp_path):
    j = L.create("when did hubble launch", origin="voice", d=tmp_path)
    L.finish(j["id"], answer="1990.", d=tmp_path)
    got = _run_reader(monkeypatch, tmp_path, [])
    assert len(got) == 1
    prio, _seq, item, _ts = got[0]
    assert item[0] == "lookup" and item[2] == j["id"] and "1990." in item[1]
    assert prio == h._SOURCE_PRIORITY["lookup"] and h._lookup_origin(item[3]) == "voice"


def test_lookup_reader_waits_while_he_talks(monkeypatch, tmp_path):
    j = L.create("q", d=tmp_path)
    L.finish(j["id"], answer="a", d=tmp_path)
    talking = [{"text": "so anyway", "utterance_id": "u"}] * 1000
    assert _run_reader(monkeypatch, tmp_path, talking) == []
    assert L.read(j["id"], tmp_path)["status"] == "done"     # still waiting, not lost


def test_lookup_origin_parse():
    assert h._lookup_origin(json.dumps({"origin": "discord"})) == "discord"
    assert h._lookup_origin("garbage") == "voice" and h._lookup_origin(None) == "voice"
