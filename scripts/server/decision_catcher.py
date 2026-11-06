#!/usr/bin/env python3
"""decision_catcher.py - catch Zeke's DECISIONS/ANSWERS from the shared transcript into the Lumen docket (11-06).

Why: on 11-06 Iris re-asked Zeke three questions he had already answered on Discord - the answers never reached
profiles/vr_temple/lumen_docket.md. This script reads state/transcript.jsonl (every Discord / orb-chat / voice
turn, one JSON object per line: id, ts, iso, role user|assistant, source, modality discord|chat|voice, content),
finds Zeke's turns that LOOK like decisions/answers, and appends CANDIDATE lines to a
`## Decisions inbox (auto, unconfirmed)` section of the docket. Nothing else in the docket is ever edited.
Iris confirms a candidate by moving it into "Decided" by hand (and deleting it from the inbox).

Heuristics (score; a candidate needs >= --min-score, voice +1 because voice transcripts are noisier):
  +2  the previous Iris turn (same channel, <= 6 h before) asked a question ('?')
  +2  opens with an answer word: yes/yeah/yep/no/nope/sure/ok/okay/go for it/do it/sounds good/perfect/correct...
  +1  each decision phrase (cap +3): "go for it", "should be", "I want", "let's", "make it", "keep", "don't",
      "instead", "I'd rather", "prefer", "use the", "default", "switch", "name it", "call it"...
  +2  shares >= 2 keywords with an OPEN docket item (`- [ ]` under "## Open", or any heading containing
      "Needs Zeke"/"Needs-Zeke"/"Open"); the best item is named in the candidate line
  +1  numbered answers (+2 right after a question)
  -4  pure acknowledgement / laughter ("thanks", "lol", "good night"); -2 a bare word with no question before it
Idempotent: a cursor (last processed ts) lives in state/decision_catcher.cursor.json, AND every candidate line
carries its transcript id (`tid:<id>`), so ids already in the docket are never added twice.

usage:
  decision_catcher.py [--dry] [--since-hours 48] [--min-score 4] [--transcript P] [--docket P] [--cursor P]
                      [--reset]   (forget the cursor; next run starts from --since-hours ago)
  --dry  prints the candidates (and what it WOULD append) without touching the docket or the cursor.
Suggested: a step in Iris's self-cron + right after a Discord conversation burst (cheap: ~0.2 s for the whole file).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEF_TRANSCRIPT = ROOT / "state" / "transcript.jsonl"
DEF_DOCKET = ROOT / "profiles" / "vr_temple" / "lumen_docket.md"
DEF_CURSOR = ROOT / "state" / "decision_catcher.cursor.json"
INBOX_HEADING = "## Decisions inbox (auto, unconfirmed)"
QUESTION_WINDOW_S = 6 * 3600

ANSWER_OPENERS = re.compile(
    r"^\s*(yes|yeah|yea|yep|yup|no|nope|nah|sure|ok(ay)?|go for it|do it|sounds good|perfect|correct|exactly|"
    r"definitely|absolutely|of course|both|neither|either|option\s*\d|the (first|second|third|last) one)\b",
    re.I)
DECISION_PHRASES = [
    r"\bgo for it\b", r"\bshould be\b", r"\bi want\b", r"\bi'?d (rather|like|prefer)\b", r"\blet'?s\b",
    r"\bmake (it|them|the)\b", r"\bkeep (it|them|the)\b", r"\bdon'?t\b", r"\binstead\b", r"\bprefer\b",
    r"\buse (it|the|that|this)\b", r"\bdefault\b", r"\bswitch (it|them|on|off)\b", r"\bname (it|him|her)\b",
    r"\bcall (it|him|her)\b", r"\bturn (it|them)? ?(on|off)\b", r"\bnot (a|an|the)\b", r"\bit'?s (fine|ok|good)\b",
    r"\bgo ahead\b", r"\bdo (that|this|both|it)\b", r"\bi (approve|agree|decided?)\b", r"\bplease (do|add|make)\b",
    r"\b(we|you|it|they) (have|need|has|needs) to\b", r"\bmake sure\b", r"\bi meant\b", r"\bwhat i meant\b",
]
DECISION_RE = [re.compile(p, re.I) for p in DECISION_PHRASES]
NUMBERED_RE = re.compile(r"(^|\n)\s*(\d{1,2}[.)]\s|for (number )?\d{1,2}\b)", re.I)
ACK_ONLY_RE = re.compile(      # a bare ok/yes/no is NOT here: right after a question it IS an answer
    r"^\s*(thanks?( you)?|thx|ty|lol|lmao|haha+|nice|cool|good (night|morning)|gn|bye|mm+|hm+|wow|oh)"
    r"[\s.!?,😂🙂👍💛❤️]*$", re.I)
STOP = set("""the a an and or but if then so to of in on at for with from by as is are was were be been being it its this
that these those there here i you he she they we me my your our their his her them us do does did done have has had
not no yes yeah just also very really more most some any all each every can could would should will shall may might
must what which who whom whose when where why how than too into onto over under about after before again once only
own same other such both few nor out up down off now new one two get got make made like want need going gonna
thing things stuff okay ok sure lot well still even much many way back time today tonight later
lumen build built test tests branch merged merge"""  # docket-ubiquitous words match everything
            .split())


def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9'-]{3,}", text.lower()) if w not in STOP}


def open_items(docket_text: str) -> list:
    """Open `- [ ]` items under any heading that is Open / Needs Zeke; returns [(item_text, keywords)]."""
    items, in_open = [], False
    for line in docket_text.splitlines():
        if line.startswith("#"):
            h = line.lower()
            in_open = ("open" in h or "needs zeke" in h or "needs-zeke" in h) and "decisions inbox" not in h
            continue
        m = re.match(r"\s*[-*]\s*\[ \]\s*(.+)", line)
        if in_open and m:
            txt = m.group(1).strip()
            items.append((txt, words(re.sub(r"`[^`]*`|\([^)]*\)", " ", txt)) or words(txt)))
    return items


def best_item(text: str, items: list, min_overlap: int = 2):
    w = words(text)
    best, best_n = None, 0
    for txt, kw in items:
        n = len(w & kw)
        if n > best_n:
            best, best_n = txt, n
    return (best, best_n) if best_n >= min_overlap else (None, best_n)


def score_turn(turn: dict, prev_iris, items: list):
    """Return (score, reasons, prev_question, item) for one Zeke turn."""
    text = (turn.get("content") or "").strip()
    reasons, score = [], 0
    asked = None
    if prev_iris and "?" in (prev_iris.get("content") or "") and \
            float(turn.get("ts", 0)) - float(prev_iris.get("ts", 0)) <= QUESTION_WINDOW_S:
        asked = prev_iris
        score += 2; reasons.append("after-question")
    if ANSWER_OPENERS.match(text):
        score += 2; reasons.append("answer-opener")
    hits = sum(1 for r in DECISION_RE if r.search(text))
    if hits:
        score += min(hits, 3); reasons.append(f"decision-phrases:{hits}")
    if NUMBERED_RE.search(text):
        score += 2 if asked is not None else 1; reasons.append("numbered")
    item, _ = best_item(text, items)
    if item is None and asked is not None:   # a short "yes" answers whatever Iris asked about
        item, _ = best_item(asked.get("content") or "", items, 3)   # Iris's turns are long: stricter
        if item is not None:
            reasons.append("question-matches-open-item")
    if item is not None:
        score += 2; reasons.append("open-item")
    if ACK_ONLY_RE.match(text):
        score -= 4; reasons.append("ack-only")
    elif asked is None and len(words(text)) == 0 and not hits:
        score -= 2; reasons.append("bare-reply")
    if turn.get("modality") == "voice":
        score -= 1; reasons.append("voice-noise")
    return score, reasons, asked, item


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip().replace('"', "'")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def candidate_line(turn, asked, item, reasons) -> str:
    iso = (turn.get("iso") or "")[:16].replace("T", " ")
    parts = [f'- [ ] {iso} ({turn.get("modality", "?")}) Zeke: "{_clip(turn.get("content"), 400)}"']
    if asked is not None:
        parts.append(f'— after I asked: "{_clip(asked.get("content"), 160)}"')
    parts.append(f"— probably answers: {_clip(item, 120)}" if item else "— probably answers: (no open item matched)")
    parts.append(f"`tid:{turn.get('id')}` `{','.join(reasons)}`")
    return " ".join(parts)


def read_turns(path: Path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("content") is not None:
                out.append(d)
    out.sort(key=lambda d: float(d.get("ts") or 0))
    return out


def find_candidates(turns: list, items: list, since_ts: float, min_score: int, known_ids: set):
    last_iris = {}      # modality -> last assistant turn
    cands, max_ts = [], since_ts
    for t in turns:
        ts = float(t.get("ts") or 0)
        mod = t.get("modality") or "?"
        if t.get("role") == "assistant":
            last_iris[mod] = t
            continue
        if t.get("role") != "user" or ts <= since_ts:
            continue
        max_ts = max(max_ts, ts)
        if t.get("source") not in (None, "zeke") or str(t.get("id")) in known_ids:
            continue
        score, reasons, asked, item = score_turn(t, last_iris.get(mod), items)
        if score >= min_score:
            cands.append((t, asked, item, reasons, score))
    return cands, max_ts


def insert_into_docket(docket_text: str, lines: list) -> str:
    """Append lines at the END of the inbox section (creating it at the end of the file); nothing else changes."""
    block = "\n".join(lines) + "\n"
    src = docket_text if (not docket_text or docket_text.endswith("\n")) else docket_text + "\n"
    m = re.search(r"^" + re.escape(INBOX_HEADING) + r"[ \t]*$", src, re.M)
    if m is None:
        sep = "" if (not src or src.endswith("\n\n")) else "\n"
        return (src + sep + INBOX_HEADING + "\n"
                + "*(decision_catcher.py - candidates only; move confirmed ones into Decided, delete from here)*\n"
                + block)
    body_start = m.end() + 1
    nxt = re.search(r"^#{1,2} ", src[body_start:], re.M)
    end = body_start + nxt.start() if nxt else len(src)
    section = src[:end].rstrip("\n") + "\n"
    rest = src[end:]
    return section + block + ("\n" + rest if rest else "")


def atomic_write(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        os.chmod(tmp, os.stat(path).st_mode & 0o777)
    except OSError:
        pass
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--since-hours", type=float, default=48.0, help="start point when there is no cursor")
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--transcript", type=Path, default=DEF_TRANSCRIPT)
    ap.add_argument("--docket", type=Path, default=DEF_DOCKET)
    ap.add_argument("--cursor", type=Path, default=DEF_CURSOR)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args(argv)

    cur = {}
    if a.cursor.exists() and not a.reset:
        try:
            cur = json.loads(a.cursor.read_text())
        except ValueError:
            cur = {}
    since = float(cur.get("last_ts") or (time.time() - a.since_hours * 3600))
    docket = a.docket.read_text(encoding="utf-8") if a.docket.exists() else ""
    known = set(re.findall(r"tid:([A-Za-z0-9_-]+)", docket))
    items = open_items(docket)
    turns = read_turns(a.transcript)
    cands, max_ts = find_candidates(turns, items, since, a.min_score, known)
    lines = [candidate_line(t, q, it, r) for t, q, it, r, _ in cands]
    print(f"DECISIONS since {time.strftime('%Y-%m-%d %H:%M', time.localtime(since))}: "
          f"{len(cands)} candidate(s), {len(items)} open item(s){' [DRY]' if a.dry else ''}")
    for (t, q, it, r, s), ln in zip(cands, lines):
        print(f"  score={s} {ln}")
    if a.dry:
        return 0
    if lines:
        atomic_write(a.docket, insert_into_docket(docket, lines))
    a.cursor.parent.mkdir(parents=True, exist_ok=True)
    a.cursor.touch(exist_ok=True)
    atomic_write(a.cursor, json.dumps({"last_ts": max_ts, "updated": time.time(), "added": len(lines)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
