"""latency_report.py --transcript: the v2-comparable `reply` stage (Phase 1 exit gate 4).

v2 recorded no stage timestamps, so the one stage both stacks can be measured on is the shared transcript's
user line -> my next line on the same modality. Pairs never cross modalities, a second user line before a
reply waits for the same reply, answers past the cap are not latency, and --split-ts gives before/after."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("latency_report", ROOT / "scripts" / "server" / "latency_report.py")
lr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr)


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n", encoding="utf-8")


def test_pairs_by_modality_and_cap(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write(t, [
        {"ts": 100, "role": "user", "modality": "discord"},
        {"ts": 105, "role": "user", "modality": "voice"},
        {"ts": 110, "role": "user", "modality": "discord"},     # second DM before the reply
        {"ts": 112, "role": "assistant", "modality": "voice"},
        {"ts": 130, "role": "assistant", "modality": "discord"},
        {"ts": 131, "role": "assistant", "modality": "discord"},  # a 2nd reply line pairs with nothing
        {"ts": 200, "role": "user", "modality": "chat"},
        {"ts": 200 + lr.REPLY_CAP_S + 1, "role": "assistant", "modality": "chat"},  # past the cap
    ])
    got = sorted(lr.transcript_replies(t))
    assert got == [("discord", 100.0, 30.0), ("discord", 110.0, 20.0), ("voice", 105.0, 7.0)]


def test_split_before_after(tmp_path):
    t = tmp_path / "transcript.jsonl"
    split = 10 * 86400.0
    _write(t, [
        {"ts": split - 500, "role": "user", "modality": "discord"},
        {"ts": split - 440, "role": "assistant", "modality": "discord"},
        {"ts": split + 50, "role": "user", "modality": "discord"},
        {"ts": split + 60, "role": "assistant", "modality": "discord"},
        {"ts": split - 3 * 86400, "role": "user", "modality": "discord"},   # outside a 1-day window
        {"ts": split - 3 * 86400 + 5, "role": "assistant", "modality": "discord"},
    ])
    rep = lr.transcript_report(t, days=1, split_ts=split)
    assert rep["windows"]["before"]["reply"]["discord"]["p50"] == 60.0
    assert rep["windows"]["before"]["reply"]["discord"]["n"] == 1
    assert rep["windows"]["after"]["reply"]["discord"]["p50"] == 10.0
    assert "no data" not in lr.render_transcript(rep)


def test_missing_transcript_is_no_data(tmp_path, capsys):
    assert lr.main(["--transcript", str(tmp_path / "nope.jsonl"), "--days", "1"]) == 0
    cap = capsys.readouterr()          # (an earlier test in the suite leaves sys.stdout -> stderr)
    assert "no data" in cap.out + cap.err
