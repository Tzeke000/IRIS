"""decision_catcher: catch answers/decisions into the docket inbox (11-06). Synthetic transcript only.

    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_decision_catcher.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "server" / "decision_catcher.py"
spec = importlib.util.spec_from_file_location("decision_catcher_t", SRC)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

DOCKET = """# Docket

## Decided
- [x] something already settled

## Open
- [ ] Pick a colour for the lighthouse beacon lamp
- [ ] Choose how many ferry boats cross the harbour
- [x] finished item about pancakes

## Later
- [ ] untouched later idea
"""

T0 = 1_800_000_000.0


def turn(i, role, content, modality="discord", dt=60.0):
    return {"id": f"t{i:03d}", "ts": T0 + i * dt, "iso": f"2027-01-15T10:{i:02d}:00-05:00", "role": role,
            "source": "zeke" if role == "user" else "iris", "modality": modality, "content": content}


def setup(tmp_path, turns, docket=DOCKET):
    tr = tmp_path / "transcript.jsonl"
    tr.write_text("\n".join(json.dumps(t) for t in turns) + "\nnot json\n")
    dk = tmp_path / "docket.md"
    dk.write_text(docket)
    cur = tmp_path / "cursor.json"
    args = ["--transcript", str(tr), "--docket", str(dk), "--cursor", str(cur), "--since-hours", "1e9"]
    return tr, dk, cur, args


TURNS = [
    turn(1, "assistant", "Quick one: what colour should the lighthouse beacon lamp be?"),
    turn(2, "user", "Yeah, make it a warm amber, not white."),
    turn(3, "user", "lol"),
    turn(4, "assistant", "Done. Anything else?"),
    turn(5, "user", "thanks"),
    turn(6, "user", "I want five ferry boats crossing the harbour, should be slow ones."),
    turn(7, "user", "the weather is nice today"),
]


def test_open_items_only_unchecked_under_open():
    items = [t for t, _ in dc.open_items(DOCKET)]
    assert items == ["Pick a colour for the lighthouse beacon lamp", "Choose how many ferry boats cross the harbour"]


def test_catches_answers_not_chitchat(tmp_path):
    _, dk, cur, args = setup(tmp_path, TURNS)
    assert dc.main(args) == 0
    out = dk.read_text()
    assert "tid:t002" in out and "tid:t006" in out
    for noise in ("t003", "t005", "t007"):
        assert f"tid:{noise}" not in out
    line2 = next(ln for ln in out.splitlines() if "tid:t002" in ln)
    assert "lighthouse beacon lamp" in line2 and "after I asked" in line2
    line6 = next(ln for ln in out.splitlines() if "tid:t006" in ln)
    assert "ferry boats" in line6.split("probably answers:")[1]


def test_nothing_else_in_docket_changes(tmp_path):
    _, dk, _, args = setup(tmp_path, TURNS)
    dc.main(args)
    out = dk.read_text()
    assert out.startswith(DOCKET)          # appended only
    assert dc.INBOX_HEADING in out


def test_idempotent_by_cursor_and_by_tid(tmp_path):
    tr, dk, cur, args = setup(tmp_path, TURNS)
    dc.main(args)
    first = dk.read_text()
    dc.main(args)                          # cursor: nothing new
    assert dk.read_text() == first
    cur.unlink()                           # cursor lost: the tids already in the docket still stop duplicates
    dc.main(args)
    assert dk.read_text() == first
    assert first.count("tid:t002") == 1


def test_new_turns_append_inside_existing_section(tmp_path):
    docket = DOCKET + "\n" + dc.INBOX_HEADING + "\n- [ ] old candidate `tid:x1`\n\n## Footer\nkeep me\n"
    tr, dk, cur, args = setup(tmp_path, TURNS[:2], docket)
    dc.main(args)
    out = dk.read_text()
    inbox = out.split(dc.INBOX_HEADING)[1].split("## Footer")[0]
    assert "tid:x1" in inbox and "tid:t002" in inbox
    assert out.rstrip().endswith("## Footer\nkeep me")
    assert out.count(dc.INBOX_HEADING) == 1
    assert out.replace(inbox, "") == docket.replace(docket.split(dc.INBOX_HEADING)[1].split("## Footer")[0], "")


def test_dry_touches_nothing(tmp_path, capsys):
    _, dk, cur, args = setup(tmp_path, TURNS)
    dc.main(args + ["--dry"])
    assert dk.read_text() == DOCKET and not cur.exists()
    assert "tid:t002" in capsys.readouterr().out


def test_question_window_and_numbered(tmp_path):
    turns = [turn(1, "assistant", "Two questions: 1) beacon? 2) boats?"),
             turn(2, "user", "1. amber\n2) three boats", dt=60.0),
             turn(400, "user", "ok", dt=60.0)]          # 6.6 h later, no new question: chit-chat
    _, dk, _, args = setup(tmp_path, turns)
    dc.main(args)
    out = dk.read_text()
    assert "tid:t002" in out and "numbered" in out and "tid:t400" not in out


def test_voice_needs_more_evidence(tmp_path):
    turns = [turn(1, "user", "should be fine", modality="voice"),
             turn(2, "user", "should be fine", modality="discord")]
    s_voice = dc.score_turn(turns[0], None, [])[0]
    s_disc = dc.score_turn(turns[1], None, [])[0]
    assert s_voice == s_disc - 1
