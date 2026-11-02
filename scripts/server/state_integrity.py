#!/usr/bin/env python3
"""state_integrity.py — find DAMAGED lines in my memory + state files (Zeke's ask, 11-01).

Why: on 11-01 I found 556 NUL bytes glued to the front of one scene-diary line in
state/iris_memory.jsonl — a half-finished write from a power-loss cold boot sometime
09-19..09-24. It made that memory unreadable for ~6 weeks and nothing noticed.

What it checks (read-only unless --fix-nul):
  * every *.jsonl under state/  -> each non-empty line must parse as JSON, no NUL bytes
  * every *.json  under state/  -> whole file must parse, no NUL bytes
  * every *.md in my memory dir + profiles/ -> no NUL bytes (a torn write there is silent too)
Skips files modified in the last few seconds (mid-write) and *.tmp / *.prev / *.bak.

Output: one line per damaged file (path, line numbers, what's wrong) + state/health/integrity.json.
Exit 0 = clean, 1 = damage found.

--fix-nul: for JSONL lines whose ONLY damage is NUL bytes and that parse once the NULs are
stripped, strip them in place (a copy of the file goes to <file>.integrity-bak first).
Anything else is reported, never auto-"fixed" — a truncated line is a lost memory to look at.
"""
import json
import os
import sys
import time
from pathlib import Path

R = Path("/home/iris/IRIS")
MEM = Path.home() / ".claude/projects/-home-iris-IRIS/memory"
OUT = R / "state/health/integrity.json"
SKIP_SUFFIX = (".tmp", ".prev", ".bak", ".integrity-bak", ".partial")
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "chroma",
             "stock_brain",           # Anki's Vector config: JSON-with-comments by design, not damage
             "_quarantine_allnul"}    # all-NUL files already moved aside (11-01)
FRESH_S = 5.0
MAX_BYTES = 200 * 1024 * 1024


def _walk(root: Path, exts):
    if not root.is_dir():
        return
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(SKIP_SUFFIX) or not fn.endswith(exts) or "salvage" in fn:  # salvage = known holding file
                continue
            yield Path(dp) / fn


def check_jsonl(p: Path, fix: bool):
    bad, fixable = [], []
    raw = p.read_bytes()
    lines = raw.split(b"\n")
    for i, b in enumerate(lines, 1):
        if not b.strip(b" \r\t"):
            continue
        has_nul = b"\x00" in b
        try:
            json.loads(b)
            if has_nul:
                bad.append((i, "NUL bytes inside a line that still parses"))
            continue
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
        if has_nul:
            stripped = b.replace(b"\x00", b"")
            try:
                if stripped.strip():
                    json.loads(stripped)
                fixable.append(i)
                bad.append((i, f"{b.count(bytes(1))} NUL bytes — rest of line is intact (strippable)"))
                continue
            except Exception:  # noqa: BLE001
                pass
            bad.append((i, f"{b.count(bytes(1))} NUL bytes + unparseable remainder — torn write, data LOST"))
        elif i == len(lines) and not raw.endswith(b"\n"):
            bad.append((i, f"last line unparseable ({err}) and no trailing newline — truncated write"))
        else:
            bad.append((i, f"unparseable line ({err})"))
    fixed = 0
    if fix and fixable:
        bak = p.with_name(p.name + ".integrity-bak")
        bak.write_bytes(raw)
        for i in fixable:
            lines[i - 1] = lines[i - 1].replace(b"\x00", b"")
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(b"\n".join(lines))
        os.replace(tmp, p)
        fixed = len(fixable)
    return bad, fixed


def check_json(p: Path):
    b = p.read_bytes()
    out = []
    if b"\x00" in b:
        out.append((0, f"{b.count(bytes(1))} NUL bytes"))
    try:
        json.loads(b.replace(b"\x00", b"")) if b.strip(b"\x00 \r\n\t") else None
    except Exception as e:  # noqa: BLE001
        out.append((0, f"does not parse ({type(e).__name__})"))
    return out


def check_md(p: Path):
    b = p.read_bytes()
    if b"\x00" not in b:
        return []
    nl = [i for i, ln in enumerate(b.split(b"\n"), 1) if b"\x00" in ln]
    return [(nl[0], f"{b.count(bytes(1))} NUL bytes on {len(nl)} line(s)")]


def main():
    fix = "--fix-nul" in sys.argv
    t0 = time.time()
    now = time.time()
    findings, n_files, fixed_total = [], 0, 0
    targets = [(p, "jsonl") for p in _walk(R / "state", (".jsonl",))]
    targets += [(p, "json") for p in _walk(R / "state", (".json",))]
    targets += [(p, "md") for p in _walk(MEM, (".md",))]
    targets += [(p, "md") for p in _walk(R / "profiles", (".md",))]
    for p, kind in targets:
        try:
            st = p.stat()
            if now - st.st_mtime < FRESH_S or st.st_size > MAX_BYTES:
                continue
            n_files += 1
            if kind == "jsonl":
                bad, fixed = check_jsonl(p, fix)
                fixed_total += fixed
            elif kind == "json":
                bad = check_json(p)
            else:
                bad = check_md(p)
        except Exception as e:  # noqa: BLE001
            bad = [(0, f"unreadable ({type(e).__name__})")]
        if bad:
            findings.append({"path": str(p), "problems": [{"line": i, "what": w} for i, w in bad[:20]],
                             "n_problems": len(bad)})
    rep = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files_checked": n_files,
           "damaged_files": len(findings), "fixed_nul_lines": fixed_total,
           "took_s": round(time.time() - t0, 2), "findings": findings}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, OUT)
    mark = "✓" if not findings else "⚠"
    print(f"{mark} integrity: {n_files} files checked in {rep['took_s']} s, {len(findings)} damaged"
          + (f", {fixed_total} NUL line(s) stripped" if fixed_total else ""))
    for f in findings[:15]:
        first = f["problems"][0]
        more = f" (+{f['n_problems'] - 1} more)" if f["n_problems"] > 1 else ""
        print(f"   {f['path']}: line {first['line']}: {first['what']}{more}")
    if len(findings) > 15:
        print(f"   ... +{len(findings) - 15} more files — full list in {OUT}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
