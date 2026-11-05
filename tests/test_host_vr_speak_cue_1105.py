"""VR turns don't trigger the room's heads-down cue (11-05, docket I5): `vr_temple speak` IS speaking on purpose, and
a VR turn never plays "Hey Zeke - I've finished…" into the empty room (on 11-04 that cue followed 28 of 28 VR turns and
held the mouth while my next temple line waited: first words 3-7 s)."""
from __future__ import annotations

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


class Blk:
    def __init__(self, name, inp):
        self.name, self.input = name, inp


def test_vr_temple_speak_counts_as_speaking():
    assert h._is_speak_tool(Blk("mcp__iris__iris_tool_call", {"name": "vr_temple", "params": {"action": "speak", "text": "hi"}}))
    assert not h._is_speak_tool(Blk("mcp__iris__iris_tool_call", {"name": "vr_temple", "params": {"action": "state"}}))
    assert h._is_speak_tool(Blk("mcp__iris__voice_speak", {}))
    assert h._is_speak_tool(Blk("mcp__iris__iris_tool_call", {"name": "voice_speak", "params": {}}))


def test_no_heads_down_cue_on_vr_turns():
    n = h.HEADS_DOWN_MIN_TOOLS
    assert not h._heads_down_cue_due("vr", False, False, n + 3, False)
    assert h._heads_down_cue_due("discord", False, False, n, False)          # a heads-down Discord job still pings
    assert not h._heads_down_cue_due("discord", False, True, n, False)
    assert not h._heads_down_cue_due("discord", False, False, n - 1, False)
    assert not h._heads_down_cue_due("discord", False, False, n, True)       # mouth flagged off
