"""Suite-wide guard: tests never write the REAL state/ of the tree they run in.

Found 10-31: run from ~/IRIS (the live tree), test_host_event_dualwrite's checkpoint delivery appended
"still delivered" / "mid-turn" to the LIVE state/transcript.jsonl as lines Zeke typed on Discord (8 pairs
10-30 → 10-31), and test_chat_reply_speaker's chat_reply nudged the live state/iris_mood.json. Both modules
bind their paths to the repo root at first use, so every test gets them re-bound to its own tmp dir."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_shared_state(tmp_path_factory, monkeypatch):
    base = tmp_path_factory.mktemp("shared_state")
    (base / "state").mkdir()
    from brain import iris_transcript, mood_core
    monkeypatch.setattr(iris_transcript, "_BASE", base)
    monkeypatch.setattr(mood_core, "_BASE", base)
    monkeypatch.setattr(mood_core, "_MOOD_PATH", base / "state" / "iris_mood.json")
    monkeypatch.setattr(mood_core, "_EMOTION_REFERENCE_PATH", base / "state" / "iris_emotion_reference.json")
    # A later configure(<repo root>) (the host's first transcript append, runtime boot code) would re-bind
    # them to the real tree; redirect exactly that call. configure(<a test's own tmp dir>) is honoured.
    for mod in (iris_transcript, mood_core):
        real = mod.configure

        def guarded(base_dir, _real=real):
            return _real(base if Path(base_dir).resolve() == ROOT else base_dir)
        monkeypatch.setattr(mod, "configure", guarded)
    yield base
