#!/usr/bin/env python3
"""lumen_map.py — the LIVE half of profiles/vr_temple/lumen_map.md (Zeke 11-11: "you need a live roadmap for LUMEN
so you know what's done"). The map is hand-written per SYSTEM. This script computes the column that goes stale:
what's DEPLOYED, what's ON MAIN but not deployed, and which branches haven't landed.

  lumen_map.py status    deployed version + date + the commit it was built from (both repos), commits on main since
                         then (one line each, with their list: trailers), branches not merged into main
  lumen_map.py check     lint lumen_map.md against reality: a "BRANCH ONLY" line naming a branch that is merged or
                         gone; a backticked path that no longer exists; a map stamped with an older deployed version

READ-ONLY on git: rev-parse / log / for-each-ref / merge-base only. Never checkout, merge, fetch or push, and never
writes anything under /data/projects/vr_iris.

WHERE "DEPLOYED" COMES FROM (no single record names the commit, so this is derived — say so when you quote it):
  * version  = the last line of <vr_iris>/update_store/seq_used.txt  ("SEQ VERSION UTC") — `vr.sh publish` appends it
               when it signs a release; deploy publishes before the temple server, so the last published = live.
  * built at = mtime of <project>/build_version.txt (the deploy's export stamps it; git-ignored).
  * commit   = the main branch's REFLOG at that moment (`main@{time}`), falling back to the last first-parent commit
               before it (marked "approx"). If build/version.txt != the published version, a newer export exists that
               was never published — printed as a warning.
"""
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VR = Path(os.environ.get("LUMEN_VR_ROOT", "/data/projects/vr_iris"))
REPOS = {"project": VR / "project", "harness": VR / "harness"}
IRIS = Path(__file__).resolve().parents[2]
MAP = Path(os.environ.get("LUMEN_MAP", str(IRIS / "profiles/vr_temple/lumen_map.md")))
LIST_RE = re.compile(r"^list:\s*(.+)$", re.I | re.M)


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main_branch(repo):
    for b in ("main", "master"):
        if git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{b}"):
            return b
    return "HEAD"


def trailers(repo, rng):
    out = git(repo, "log", "--no-merges", "--format=%B%x00", rng) or ""
    ids = []
    for m in LIST_RE.finditer(out):
        for tok in m.group(1).split():
            if tok not in ids:
                ids.append(tok)
    return " ".join(ids)


def deployed():
    """-> dict(seq, version, utc, built, built_ver, commits{repo: (sha, how)}) or None"""
    d = {}
    seq = VR / "update_store/seq_used.txt"
    if seq.is_file():
        lines = [ln.split() for ln in seq.read_text().splitlines() if ln.strip()]
        if lines and len(lines[-1]) >= 3:
            d["seq"], d["version"], d["utc"] = lines[-1][:3]
    stamp = REPOS["project"] / "build_version.txt"
    bv = VR / "build/version.txt"
    d["built_ver"] = bv.read_text().strip() if bv.is_file() else None
    if not stamp.is_file():
        return d or None
    t = datetime.fromtimestamp(stamp.stat().st_mtime)
    d["built"] = t.strftime("%Y-%m-%d %H:%M:%S")
    d["commits"] = {}
    for name, repo in REPOS.items():
        b = main_branch(repo)
        sha = git(repo, "rev-parse", "--verify", "--quiet", f"{b}@{{{d['built']}}}")
        how = "reflog"
        if not sha:
            sha = git(repo, "rev-list", "-1", "--first-parent", f"--before={d['built']}", b)
            how = "approx (no reflog)"
        d["commits"][name] = (sha, how)
    return d


def unmerged(repo):
    b = main_branch(repo)
    rows = []
    refs = git(repo, "for-each-ref", "--sort=-committerdate",
               "--format=%(refname:short)%09%(committerdate:short)%09%(subject)", "refs/heads") or ""
    for line in refs.splitlines():
        name, date, subj = (line.split("\t") + ["", ""])[:3]
        if name == b:
            continue
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", name, b]).returncode == 0:
            continue
        ahead = git(repo, "rev-list", "--count", f"{b}..{name}") or "?"
        rows.append((name, date, ahead, subj, trailers(repo, f"{b}..{name}")))
    return rows


def cmd_status():
    d = deployed() or {}
    v = d.get("version", "UNKNOWN")
    print(f"DEPLOYED  v{v}  (release seq {d.get('seq', '?')}, published {d.get('utc', '?')} UTC; "
          f"export stamped {d.get('built', '?')} local)")
    if d.get("built_ver") and d.get("version") and d["built_ver"] != d["version"]:
        print(f"  ⚠ build/version.txt = {d['built_ver']} ≠ published {d['version']} — a newer export was never "
              "published (commit below is the EXPORT's, not the live one)")
    for name, repo in REPOS.items():
        b = main_branch(repo)
        sha, how = d.get("commits", {}).get(name, (None, ""))
        head = git(repo, "rev-parse", "--short", b)
        print(f"\n== {name}: {b} HEAD {head}; deployed from {sha[:7] if sha else '?'} ({how})")
        if not sha:
            continue
        log = git(repo, "log", "--first-parent", "--format=%H%x09%h%x09%cs%x09%s", f"{sha}..{b}") or ""
        rows = [ln.split("\t") for ln in log.splitlines() if ln]
        print(f"  ON MAIN, NOT DEPLOYED: {len(rows)}")
        for full, short, date, subj in rows:
            par = git(repo, "rev-parse", "--verify", "--quiet", f"{full}^2")
            tr = trailers(repo, f"{full}^1..{full}^2") if par else trailers(repo, f"{full}^!")
            print(f"   {short} {date} {subj[:90]}" + (f"   [list: {tr}]" if tr else ""))
        rows = unmerged(repo)
        print(f"  BRANCHES NOT LANDED: {len(rows)}")
        for name_, date, ahead, subj, tr in rows:
            print(f"   {name_:<22} {date} +{ahead:<3} {subj[:70]}" + (f"   [list: {tr}]" if tr else ""))
    return 0


TOK_RE = re.compile(r"`([^`\s]+)`")
BRANCH_RE = re.compile(r"(?<![\w/.-])([a-z][a-z0-9]*(?:-[a-z0-9]+)+)")
DATED = re.compile(r"-1[01]\d\d[a-z]?(?:-|$)")
STATE_RE = re.compile(r"(BRANCH ONLY|ON MAIN|LIVE|PLANNED|UNSURE)")


def branch_only_names(line):
    """branch names that sit in a BRANCH ONLY segment of the line (up to the next state word)"""
    parts = STATE_RE.split(line)
    names = []
    for i in range(1, len(parts) - 1, 2):
        if parts[i] == "BRANCH ONLY":
            names += [b for b in BRANCH_RE.findall(parts[i + 1]) if DATED.search(b)]
    return names
PATH_EXT = (".gd", ".py", ".sh", ".md", ".json", ".tsv", ".txt", ".tscn", ".nsi", ".tres", ".cfg")


def _exists(p):
    try:
        return p.exists()
    except PermissionError:
        return True  # e.g. a secrets file this user can't stat — present-but-private, not "missing"


def path_exists(tok):
    tok = re.sub(r":\d[\d-]*$", "", tok.rstrip(".,;)"))  # file.gd:87-89 → file.gd
    if any(c in tok for c in "<>*${}|") or tok.startswith(("http", "-", "res://", "user://")):
        return True  # a pattern / placeholder / in-game path — not checkable
    if "/" not in tok and not tok.endswith(PATH_EXT):
        return True  # not a path
    p = Path(os.path.expanduser(tok))
    if p.is_absolute():
        return _exists(p)
    bases = [REPOS["project"], REPOS["project"] / "scripts", REPOS["harness"], REPOS["harness"] / "tools", VR, IRIS,
             IRIS / "profiles/vr_temple", IRIS / "scripts/server", Path.home() / ".claude/projects/-home-iris-IRIS/memory"]
    return any(_exists(b / tok) for b in bases)


def cmd_check():
    if not MAP.is_file():
        print(f"✗ no map at {MAP}")
        return 2
    text = MAP.read_text()
    warns = []
    branches = {}
    for name, repo in REPOS.items():
        refs = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads") or ""
        branches[name] = (main_branch(repo), set(refs.split()))
    for i, line in enumerate(text.splitlines(), 1):
        if "BRANCH ONLY" in line:
            for br in branch_only_names(line):
                where = [n for n, (_, s) in branches.items() if br in s]
                if not where:
                    warns.append(f"L{i}: branch {br} is gone in both repos (map says BRANCH ONLY)")
                    continue
                merged = [n for n in where if subprocess.run(
                    ["git", "-C", str(REPOS[n]), "merge-base", "--is-ancestor", br, branches[n][0]]).returncode == 0]
                if merged and len(merged) == len(where):
                    warns.append(f"L{i}: branch {br} is MERGED into main ({'+'.join(merged)}) — map still says BRANCH ONLY")
        for tok in TOK_RE.findall(line):
            if not path_exists(tok):
                warns.append(f"L{i}: path not found: {tok}")
    m = re.search(r"deployed:\s*v?(\d+\.\d+\.\d+\.\d+)", text)
    d = deployed() or {}
    if m and d.get("version") and m.group(1) != d["version"]:
        warns.append(f"map is stamped deployed v{m.group(1)} but LIVE is v{d['version']} — LIVE / ON MAIN columns are stale")
    if not m:
        warns.append("map has no 'deployed: vX' stamp — can't tell if its LIVE column is current")
    for w in warns:
        print("⚠ " + w)
    print(f"{'✓' if not warns else '✗'} lumen_map check: {len(warns)} warning(s) — {MAP}")
    return 1 if warns else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        sys.exit(cmd_status())
    if cmd == "check":
        sys.exit(cmd_check())
    print(__doc__)
    sys.exit(0 if cmd in ("-h", "--help", "help") else 2)
