#!/usr/bin/env python3
"""list_ids.py — keep the VR open list + LUMEN docket LIVE from what actually merged (Zeke 11-08).

Zeke: "whatever list you look at needs to be live and updated … whenever you commit something it needs to be exactly
what was fixed based off of the list". So:
  * every list line gets a stable ID right after its checkbox: `- [ ] [VR-42] …` (open_list.md) / `- [ ] [DK-7] …` (docket §Open)
  * a commit says what it fixes with a trailer line:  `list: VR-42 DK-7`   (or `list: VR-42 partial` = progress, no tick)
  * `tick` (run by the harness land.sh after a GREEN land) reads the newly merged commits of both repos, ticks each named
    item `- [x] ✅ [VR-42] (merged <sha> <date>)`, and appends `(+progress <sha>)` for `partial`
  * `shipped` (run by deploy_all.sh after a good deploy) stamps ticked-but-unshipped items `(shipped vX)`
  * `untagged` lists merged commits with NO list: trailer (land prints it as a WARN — work that no list knows about)

  list_ids.py assign                       give an ID to every list line that has none (idempotent; never renumbers)
  list_ids.py tick REPO=OLD..NEW [...]     tick from the trailers in those ranges (repo path=git range)
  list_ids.py shipped VERSION              mark merged items as shipped in VERSION
  list_ids.py untagged REPO=OLD..NEW [...] merged non-merge commits without a list: trailer
  list_ids.py show ID                      print the line
Files: profiles/vr_temple/open_list.md, profiles/vr_temple/lumen_docket.md (git-ignored; the edits are the record).
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import os
_D = Path(os.environ.get("LIST_IDS_DIR", ROOT / "profiles/vr_temple"))   # tests point this at a copy
FILES = {"VR": _D / "open_list.md", "DK": _D / "lumen_docket.md"}
ITEM = re.compile(r"^(\s*- \[)([ x~◐])(\]\s*)((?:✅\s*)?\[(VR|DK)-(\d+)\]\s*)?(.*)$")   # a ticked line reads "- [x] ✅ [VR-1] …"
TRAILER = re.compile(r"^\s*list:\s*(.+)$", re.I | re.M)
IDRE = re.compile(r"\b(VR|DK)-(\d+)\b")


def _in_scope(prefix: str, lines: list[str]) -> list[bool]:
    """docket: only §Open..§Decisions inbox counts; open_list: every line."""
    if prefix == "VR":
        return [True] * len(lines)
    on = False; out = []
    for ln in lines:
        if ln.startswith("## "):
            on = ln.startswith("## Open")
        out.append(on)
    return out


def assign() -> None:
    for prefix, f in FILES.items():
        lines = f.read_text().split("\n"); scope = _in_scope(prefix, lines)
        used = [int(m.group(6)) for ln in lines if (m := ITEM.match(ln)) and m.group(6)]
        n = max(used, default=0); added = 0
        for i, ln in enumerate(lines):
            m = ITEM.match(ln)
            if scope[i] and m and not m.group(4):
                n += 1; added += 1
                lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}[{prefix}-{n}] {m.group(7)}"
        f.write_text("\n".join(lines)); print(f"{f.name}: {added} new IDs (last {prefix}-{n})")


def _commits(spec: str) -> list[tuple[str, str, str]]:
    repo, rng = spec.split("=", 1)
    out = subprocess.run(["git", "-C", repo, "log", "--no-merges", "--format=%h%x1f%ad%x1f%B%x1e", "--date=format:%m-%d", rng],
                         capture_output=True, text=True, check=True).stdout
    return [tuple(c.strip("\n").split("\x1f", 2)) for c in out.split("\x1e") if c.strip()]


def _edit(fn) -> None:
    for prefix, f in FILES.items():
        lines = f.read_text().split("\n"); changed = False
        for i, ln in enumerate(lines):
            m = ITEM.match(ln)
            if m and m.group(4):
                new = fn(f"{m.group(5)}-{m.group(6)}", m, ln)
                if new is not None and new != ln:
                    lines[i] = new; changed = True
        if changed:
            f.write_text("\n".join(lines))


def tick(specs: list[str]) -> None:
    done: dict[str, str] = {}; prog: dict[str, list[str]] = {}
    for spec in specs:
        for sha, date, body in _commits(spec):
            trailers = TRAILER.findall(body)
            for t in trailers:
                partial = "partial" in t.lower()
                for p, n in IDRE.findall(t):
                    (prog.setdefault(f"{p}-{n}", []).append(sha) if partial else done.setdefault(f"{p}-{n}", f"{sha} {date}"))
            if not trailers:   # (11-10 listaudit: textui/smallfix/wristmenu named DK-n only in the SUBJECT ⇒ never ticked) — a
                # subject ID is a MENTION, not a claim of done ⇒ record it as PROGRESS only; a tick still needs the trailer
                for p, n in IDRE.findall(body.split("\n", 1)[0]):
                    prog.setdefault(f"{p}-{n}", []).append(sha)
    hit = set(); seen = set()

    def fn(iid, m, ln):
        seen.add(iid)
        if iid in done and m.group(2) != "x":
            hit.add(iid); return f"{m.group(1)}x{m.group(3)}✅ [{iid}] (merged {done[iid]}) {m.group(7)}"
        if iid in prog and not any(s in ln for s in prog[iid]):
            hit.add(iid); return ln + " (+progress " + " ".join(prog[iid]) + ")"
        return None
    _edit(fn)
    named = set(done) | set(prog)
    print(f"list tick: {len(hit)} item(s) updated" + (f" · ⚠ unknown IDs: {' '.join(sorted(named - seen))}" if named - seen else ""))


def shipped(version: str) -> None:
    n = [0]

    def fn(iid, m, ln):
        if m.group(2) == "x" and "(merged " in ln and "(shipped " not in ln:
            n[0] += 1; i = ln.index("(merged "); j = ln.index(")", i) + 1
            return ln[:j] + f" (shipped {version})" + ln[j:]
        return None
    _edit(fn); print(f"list shipped: {n[0]} item(s) stamped {version}")


def untagged(specs: list[str]) -> None:
    bad = [(sha, body.split("\n", 1)[0][:90]) for s in specs for sha, _d, body in _commits(s) if not TRAILER.search(body)]
    for sha, subj in bad:
        hint = "  (names an ID in the subject — counted as progress; add `list: …` to tick it)" if IDRE.search(subj) else ""
        print(f"  ⚠ no list: trailer  {sha} {subj}{hint}")
    print(f"untagged: {len(bad)} commit(s) with no list ID")


def show(iid: str) -> None:
    for f in FILES.values():
        for ln in f.read_text().split("\n"):
            if re.search(rf"\[{re.escape(iid)}\]", ln):
                print(ln); return
    print(f"✗ no {iid}"); sys.exit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    cmd = a[0]
    {"assign": lambda: assign(), "tick": lambda: tick(a[1:]), "shipped": lambda: shipped(a[1]),
     "untagged": lambda: untagged(a[1:]), "show": lambda: show(a[1])}.get(cmd, lambda: (print(__doc__), sys.exit(2)))()
