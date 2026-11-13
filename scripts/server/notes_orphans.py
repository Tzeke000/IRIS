#!/usr/bin/env python3
"""notes_orphans.py — find Zeke's headset/playtest notes that never reached the LUMEN docket (11-13).

Why: 11-13 I found the 11-06 subway-stairs notes had NEVER been docketed (became DK-119 a week late). A note that
isn't on the docket doesn't get built. Notes live in profiles/vr_temple/headset_notes_*.md; the docket is
profiles/vr_temple/lumen_docket.md.

The rule this enforces (going forward): every **BOLD** finding line in a notes file carries its fate at the end:
    "→ DK-119"   (docketed)      "→ skip: <why>"   (deliberately not)      "→ same as DK-87" etc. (any DK id)
Untagged bold lines are ORPHAN CANDIDATES. For each one this prints the closest docket lines by word overlap, so
docketing (or tagging an existing DK id) is quick.

  notes_orphans.py            list orphan candidates in every notes file
  notes_orphans.py --summary  one line per notes file (bold / tagged / orphan counts) — for selfcheck
Exit 1 if any orphan, 0 if none.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path("/home/iris/IRIS/profiles/vr_temple")
STOP = set("the a an and or of to in on at is it its it's this that with for my his he i you as be are was by from "
           "not but so if when then than too very just like into out up down over his her them they there what "
           "should would could does did do has have had all any one two more most less really also still again".split())
TAG = re.compile(r"→\s*(?:skip\b|same as\b|[^\n]*\bDK-\d+)", re.I)


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", s.lower()) if w not in STOP}


def main(argv: list[str]) -> int:
    summary = "--summary" in argv
    docket = [l.strip() for l in (ROOT / "lumen_docket.md").read_text(errors="replace").splitlines() if l.strip()]
    dwords = [(l, words(l)) for l in docket]
    total = 0
    for f in sorted(glob.glob(str(ROOT / "headset_notes_*.md"))):
        lines = Path(f).read_text(errors="replace").splitlines()
        bold = [(i + 1, l) for i, l in enumerate(lines) if l.lstrip().startswith("-") and "**" in l]
        orphans = [(n, l) for n, l in bold if not TAG.search(l)]
        total += len(orphans)
        print(f"{Path(f).name}: {len(bold)} bold findings · {len(bold) - len(orphans)} tagged · {len(orphans)} untagged")
        if summary:
            continue
        for n, l in orphans:
            head = re.search(r"\*\*(.+?)\*\*", l)
            key = head.group(1) if head else l[:80]
            w = words(l)
            best = sorted(((len(w & dw) / (len(w) or 1), dl) for dl, dw in dwords), reverse=True)[:2]
            print(f"  L{n}: {key[:90]}")
            for score, dl in best:
                if score >= 0.15:
                    print(f"      ~{score:.0%} docket: {dl[:110]}")
            if not best or best[0][0] < 0.15:
                print("      (no close docket line — probably an ORPHAN)")
    if not summary:
        print(f"\n{total} untagged finding(s). Docket them (or tag '→ DK-n' / '→ skip: why' on the note line).")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
