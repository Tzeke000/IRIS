"""voice_delivery_check — did what I SAID actually come out of his speakers? (2026-10-08)

Why: on 10-08 a stuck barge-in mute dropped every line I spoke for 14 minutes (16:53-17:07) while voice_speak kept
answering ok — Zeke noticed, nothing else did. Zeke asked for this check ON DEMAND ("not constantly 24/7 … available
for you just in case"; nothing new running on his PC), so it is a tool + one line in the server self-check, never a loop.

Three hops, each read once from logs that already exist:
  1. what I MEANT to say   — state/transcript.jsonl voice lines from me (voice_speak logs them)
  2. what the MOUTH spoke  — journalctl --user -u iris-mouth  "[styletts] reply in …" lines
  3. what the TOWER played — the tower's state/audio_sink.log sessions ("playing from" … "ended"), ONE ssh read
plus the ears daemon's own drop notes ("dropped", "STALE barge-in").
Verdict: ok | warn (some lines never reached the mouth / played with no sink session open) | unknown (a hop unreadable).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout + cp.stderr
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def _journal_times(unit: str, since_min: int, pattern: str) -> list[float]:
    rc, out = _run(["journalctl", "--user", "-u", unit, "--since", f"-{since_min}min", "--no-pager", "-o", "short-unix"])
    if rc != 0:
        return []
    ts = []
    for line in out.splitlines():
        if pattern in line:
            try:
                ts.append(float(line.split(" ", 1)[0]))
            except ValueError:
                pass
    return ts


def _sink_sessions(lines: int = 200) -> tuple[list[tuple[float, float | None]], str | None]:
    """[(start, end-or-None)] from the tower's audio_sink.log (one ssh call)."""
    rc, out = _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "tower",
                    f'powershell -NoProfile -Command "Get-Content D:\\Wren-Companion\\state\\audio_sink.log -Tail {lines}"'],
                   timeout=25)
    if rc != 0:
        return [], f"tower unreachable (rc {rc}): {out[-120:]}"
    pat = re.compile(r"^\[audio_sink (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")
    sessions: list[tuple[float, float | None]] = []
    cur: float | None = None
    for line in out.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()   # tower + server share a timezone
        msg = m.group(2)
        if msg.startswith("playing from"):
            if cur is not None:
                sessions.append((cur, t))
            cur = t
        elif " ended" in msg or " error:" in msg or "preempts" in msg:
            if cur is not None:
                sessions.append((cur, t))
                cur = None
    if cur is not None:
        sessions.append((cur, None))
    return sessions, None


def _voice_delivery_check(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    since_min = max(5, min(24 * 60, int(params.get("since_min") or 30)))
    t0 = time.time() - since_min * 60
    # 1. what I meant to say
    meant: list[float] = []
    try:
        with open(ROOT / "state" / "transcript.jsonl", encoding="utf-8", errors="replace") as fh:
            for line in fh.readlines()[-3000:]:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("role") == "assistant" and r.get("modality") == "voice" and float(r.get("ts") or 0) >= t0:
                    meant.append(float(r["ts"]))
    except Exception:  # noqa: BLE001
        pass
    # 2. what the mouth spoke  ·  daemon drops
    spoke = _journal_times("iris-mouth", since_min, "[styletts] reply in")
    drops = _journal_times("iris-ears", since_min, "dropped") + _journal_times("iris-ears", since_min, "STALE barge-in")
    # 3. what the tower played
    sessions, sink_err = _sink_sessions()

    def covered(t: float) -> bool:
        return any(s - 2 <= t <= ((e if e is not None else time.time()) + 2) for s, e in sessions)

    # a line I meant with NO mouth reply within 60 s after it = never reached the mouth
    unspoken = [t for t in meant if not any(t - 5 <= s <= t + 60 for s in spoke)]
    unplayed = [t for t in spoke if sessions and not covered(t)]
    hhmm = lambda xs: [time.strftime("%H:%M:%S", time.localtime(x)) for x in xs[:8]]   # noqa: E731
    issues = []
    if unspoken:
        issues.append(f"{len(unspoken)} of {len(meant)} voice line(s) never reached the mouth (e.g. {', '.join(hhmm(unspoken))})")
    if unplayed:
        issues.append(f"{len(unplayed)} mouth reply(ies) with NO tower sink session open (e.g. {', '.join(hhmm(unplayed))})")
    if drops:
        issues.append(f"the ears daemon logged {len(drops)} drop/stale-mute note(s)")
    verdict = "warn" if issues else ("unknown" if sink_err and spoke else "ok")
    return {"ok": True, "verdict": verdict, "since_min": since_min,
            "meant": len(meant), "mouth_spoke": len(spoke), "sink_sessions": len(sessions),
            "sink_open_now": bool(sessions and sessions[-1][1] is None), "issues": issues,
            "sink_error": sink_err,
            "note": "on-demand only (Zeke 10-08: no new always-on load on his PC). Silence with nothing meant is not a fault."}


register_tool(
    name="voice_delivery_check",
    description="Did my spoken lines actually play on Zeke's tower? Compares what I meant to say (transcript), what the "
                "mouth spoke (journal) and what the tower's audio sink played (one ssh read). params: since_min (default 30). "
                "On demand only. Tier 1.",
    tier=1,
    handler=_voice_delivery_check,
)
