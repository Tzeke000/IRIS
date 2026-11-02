#!/usr/bin/env python3
"""scripts/server/incident.py - incident bundles (V3 Phase 4, 2026-11-02).

When something goes wrong, freeze the evidence BEFORE anyone (me included) starts "fixing" it - the
08-25 lesson: my own repair 30 s earlier was the fault. One bundle = one directory:

    state/incidents/<YYYYmmddTHHMMSS>_<slug>/
        manifest.json     reason, subsystem, ts, host, what was collected, sizes, collector errors
        health.json       state/health/now.json (the watchdog's snapshot) as it was
        integrity.json    state/health/integrity.json (torn-write checker)
        journal_<unit>.log  last N lines of iris-stack / iris-mouth / iris-ears / iris-watchdog (+ the subsystem's)
        logs_<name>.log   tails of state/supervise.log + state/server_watchdog.log
        events.json       READ-ONLY view of state/v3/events.sqlite: stats, recent transitions, open rows
        procs.txt         top processes by RSS (pid ppid rss etime cmd)
        system.txt        GPU (nvidia-smi), disk (df), memory (/proc/meminfo), load, uptime
        git.txt           HEAD + dirty files of the repo

    incident.py capture REASON [--subsystem S] [--force] [--journal-lines N]
    incident.py list [-n 20]
    incident.py show ID [--file NAME]

Safety rules:
  * READ-ONLY on the live system: the event store is opened `mode=ro` (never through EventStore, whose
    corruption recovery would MOVE the file); nothing is restarted, nothing is "healed".
  * Secrets masked in EVERY file: the privacy-guard list (config/privacy_guard.local.txt) + its generic
    patterns (MAC, tailnet / private IPs) + token shapes (Discord, bearer, sk-/ghp_, key=value), and every
    VALUE found in ~/.iris_private.env / config/private.local.json (read in memory to mask them; the files
    themselves are never copied). A missing privacy list does not stop a capture - generic masking still
    runs and the manifest says `privacy_list: missing`.
  * Size-capped: each file <= --file-cap (256 KiB, tail kept), the bundle <= --bundle-cap (4 MiB; later
    files are skipped and listed), and only the newest KEEP (100) bundles are kept.
  * Bundles are 0700/0600; written to a .tmp dir and renamed, so `list` never sees half a bundle.
  * `--subsystem S` is rate-limited to ONE bundle per subsystem per hour (state/incidents/.ratelimit.json);
    `--force` overrides. This is what the watchdog calls when a subsystem goes RED.

Env: IRIS_ROOT (repo), IRIS_INCIDENTS_DIR (bundle dir), IRIS_INCIDENT_PRIVATE_FILES (os.pathsep list of
files whose values get masked - tests point it at fakes).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("IRIS_ROOT") or Path(__file__).resolve().parents[2])
HOME = Path.home()
CORE_UNITS = ("iris-stack", "iris-mouth", "iris-ears", "iris-watchdog")
SUBSYSTEM_UNITS = {"ears": ("iris-ears",), "mouth": ("iris-mouth",), "backup": ("iris-backup",),
                   "audio_sink": ("iris-mouth",), "runtime": ("iris-stack",), "sdk_stream": ("iris-stack",)}
RATE_S = 3600.0
KEEP = 100
FILE_CAP = 256 << 10
BUNDLE_CAP = 4 << 20
JOURNAL_LINES = 200
CMD_TIMEOUT_S = 8.0
MIN_SECRET_LEN = 6

TOKEN_PATTERNS = [
    re.compile(r"[MNO][A-Za-z\d_-]{23,27}\.[\w-]{6,7}\.[\w-]{27,40}"),                 # Discord bot token
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),                                  # sk-... API keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),                                      # GitHub tokens
    re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b"),                              # JWT
    re.compile(r"(?i)\b(bearer|authorization:)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b([\w.-]*(?:token|secret|passw(?:or)?d|api[_-]?key|private[_-]?key|cookie)[\w.-]*)"
               r"(\s*[=:]\s*|\"\s*:\s*\")([^\s\"',;]{4,})"),
]


# ── paths ───────────────────────────────────────────────────────────────────
def incidents_dir(root: Path | None = None) -> Path:
    env = os.environ.get("IRIS_INCIDENTS_DIR")
    return Path(env) if env else Path(root or ROOT) / "state" / "incidents"


def fallback_dirs() -> list[Path]:
    """Where a bundle goes when state/ itself can't be written (disk full / read-only) - exactly when an
    incident matters most. /dev/shm is RAM: lost on reboot, but the evidence exists while I look."""
    env = os.environ.get("IRIS_INCIDENTS_FALLBACK")
    if env is not None:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [Path("/dev/shm/iris-incidents")]


def private_files(root: Path) -> list[Path]:
    env = os.environ.get("IRIS_INCIDENT_PRIVATE_FILES")
    if env is not None:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [HOME / ".iris_private.env", root / "config" / "private.local.json"]


# ── masking ─────────────────────────────────────────────────────────────────
SCRIPT_REPO = Path(__file__).resolve().parents[2]


def privacy_list_path(root: Path) -> Path:
    """IRIS_PRIVACY_LIST, else <root>/config/privacy_guard.local.txt, else this script's repo's list
    (a staging root has no list of its own - it must still be masked with the real one)."""
    env = os.environ.get("IRIS_PRIVACY_LIST")
    if env:
        return Path(env)
    own = root / "config" / "privacy_guard.local.txt"
    return own if own.is_file() else SCRIPT_REPO / "config" / "privacy_guard.local.txt"


def _privacy_guard_generic(root: Path) -> list:
    """GENERIC (name, regex) from scripts/privacy_guard.py, imported by path. Never its load_list(),
    which sys.exit()s when the list is missing."""
    try:
        spec = importlib.util.spec_from_file_location("iris_privacy_guard_for_incident",
                                                      root / "scripts" / "privacy_guard.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return [p for _, p in getattr(mod, "GENERIC", [])]
    except Exception:  # noqa: BLE001
        return []


def _private_values(files: list[Path]) -> list[str]:
    vals: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (str, int)) and not isinstance(x, bool):
            s = str(x).strip()
            if len(s) >= MIN_SECRET_LEN:
                vals.add(s)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if f.suffix == ".json":
            try:
                walk(json.loads(text))
                continue
            except Exception:  # noqa: BLE001
                pass
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            v = ln.split("=", 1)[1].strip().strip("'\"")
            if len(v) >= MIN_SECRET_LEN:
                vals.add(v)
    return sorted(vals, key=len, reverse=True)   # longest first: a value inside another still masks whole


class Masker:
    def __init__(self, root: Path):
        self.notes: dict = {}
        self.list_pats: list[re.Pattern] = []
        lst = privacy_list_path(root)
        if lst.is_file():
            for line in lst.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        self.list_pats.append(re.compile(line, re.I))
                    except re.error:
                        self.notes["privacy_list_bad_lines"] = self.notes.get("privacy_list_bad_lines", 0) + 1
            self.notes["privacy_list"] = f"{len(self.list_pats)} patterns"
        else:
            self.notes["privacy_list"] = "missing"
        self.generic = _privacy_guard_generic(SCRIPT_REPO)
        self.values = _private_values(private_files(root))
        self.notes["private_values"] = len(self.values)
        self.hits = 0

    def __call__(self, text: str) -> str:
        n0 = self.hits
        for v in self.values:
            if v in text:
                self.hits += text.count(v)
                text = text.replace(v, "***")
        for p in TOKEN_PATTERNS:
            def sub(m):
                self.hits += 1
                if m.re.groups >= 3:                   # key=value: keep the key, mask the value
                    return m.group(1) + m.group(2) + "***"
                if m.re.groups == 1:
                    return m.group(1) + " ***"
                return "***"
            text = p.sub(sub, text)
        for p in self.generic + self.list_pats:
            text, k = p.subn("***", text)
            self.hits += k
        del n0
        return text


# ── collectors (each returns text; errors are caught by the caller) ─────────
def _run(cmd: list[str], timeout: float = CMD_TIMEOUT_S) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.stdout + (f"\n[stderr] {r.stderr[-2000:]}" if r.returncode and r.stderr else "")
    except FileNotFoundError:
        return f"[{cmd[0]} not installed]"
    except subprocess.TimeoutExpired:
        return f"[{' '.join(cmd[:3])} timed out after {timeout:.0f}s]"


def _tail_file(p: Path, nbytes: int) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read()
        if size > nbytes:
            data = data[data.find(b"\n") + 1:]
        return data.decode("utf-8", "replace")
    except FileNotFoundError:
        return f"[{p.name} not found]"


def collect_json_copy(p: Path) -> str:
    try:
        return json.dumps(json.loads(p.read_text(encoding="utf-8-sig")), indent=1)
    except FileNotFoundError:
        return json.dumps({"_missing": str(p.name)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"_unreadable": repr(e)[:200], "raw_tail": _tail_file(p, 4096)})


def collect_journal(unit: str, lines: int) -> str:
    return _run(["journalctl", "--user", "-u", unit, "-n", str(int(lines)), "--no-pager", "-o", "short-iso"])


def collect_events(root: Path, limit: int = 200) -> str:
    db = Path(os.environ.get("IRIS_EVENTS_DB") or root / "state" / "v3" / "events.sqlite")
    if not db.exists():
        return json.dumps({"_missing": str(db.name)})
    out: dict = {"path": db.name}
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3.0)
    try:
        c.row_factory = sqlite3.Row
        q = lambda sql, *a: [dict(r) for r in c.execute(sql, a).fetchall()]  # noqa: E731
        out["integrity"] = c.execute("PRAGMA quick_check").fetchone()[0]
        out["events_by_status"] = {r["status"]: r["n"] for r in q(
            "SELECT status, count(*) n FROM events GROUP BY status")}
        out["tasks_by_status"] = {r["status"]: r["n"] for r in q(
            "SELECT status, count(*) n FROM tasks GROUP BY status")}
        out["recent_transitions"] = q("SELECT seq, entity, entity_id, from_status, to_status, fence, ts, actor,"
                                      " substr(note,1,200) note FROM transitions ORDER BY seq DESC LIMIT ?", limit)
        out["open_events"] = q("SELECT event_id, source, status, priority, received_ts, owner, lease_until,"
                               " substr(text,1,120) text, substr(note,1,200) note FROM events WHERE status NOT IN"
                               " ('committed','cancelled') ORDER BY received_ts DESC LIMIT 50")
        out["open_tasks"] = q("SELECT task_id, kind, substr(goal,1,120) goal, status, owner, fence, lease_until,"
                              " side_effects, attempts, substr(note,1,200) note FROM tasks WHERE status NOT IN"
                              " ('committed','cancelled') ORDER BY updated_ts DESC LIMIT 50")
    finally:
        c.close()
    return json.dumps(out, indent=1, default=str)


def collect_procs(top: int = 60) -> str:
    rows = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = (d / "status").read_text()
            rss = int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) if "VmRSS" in st else 0
            ppid = int(re.search(r"PPid:\s+(\d+)", st).group(1))
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            oom = (d / "oom_score_adj").read_text().strip()
            rows.append((rss, int(d.name), ppid, oom, cmd[:300] or f"[{st.split()[1]}]"))
        except Exception:  # noqa: BLE001 — processes vanish mid-read
            continue
    rows.sort(reverse=True)
    lines = [f"{'RSS_MiB':>8} {'PID':>7} {'PPID':>7} {'OOM':>5}  CMD"]
    lines += [f"{r[0] / 1024:8.1f} {r[1]:7d} {r[2]:7d} {r[3]:>5}  {r[4]}" for r in rows[:top]]
    lines.append(f"[{len(rows)} processes, total RSS {sum(r[0] for r in rows) / 1024 / 1024:.1f} GiB]")
    return "\n".join(lines)


def collect_system() -> str:
    parts = ["## gpu", _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                             "--format=csv"]),
             _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
             "## disk", _run(["df", "-h", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs", "-x", "overlay"]),
             "## memory"]
    try:
        parts.append("\n".join(Path("/proc/meminfo").read_text().splitlines()[:20]))
    except Exception as e:  # noqa: BLE001
        parts.append(repr(e))
    try:
        parts += ["## load", Path("/proc/loadavg").read_text().strip(),
                  "## uptime_s", Path("/proc/uptime").read_text().split()[0]]
    except Exception:  # noqa: BLE001
        pass
    parts += ["## host", socket.gethostname()]
    return "\n".join(parts)


def collect_git(root: Path) -> str:
    return "\n".join(["## HEAD", _run(["git", "-C", str(root), "log", "-1", "--format=%H %ci %s"]),
                      "## branch", _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]),
                      "## dirty", _run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"])])


# ── capture ─────────────────────────────────────────────────────────────────
def slugify(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return (s[:n].rstrip("-")) or "incident"


def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def rate_limited(subsystem: str, now: float, d: Path, rate_s: float = RATE_S) -> float | None:
    """Seconds until `subsystem` may capture again, or None if it may now."""
    st = _load_json(d / ".ratelimit.json", {})
    if subsystem not in st:
        return None
    wait = float(st[subsystem] or 0.0) + rate_s - now
    return wait if wait > 0 else None


def mark_rate(subsystem: str, now: float, d: Path) -> None:
    p = d / ".ratelimit.json"
    st = _load_json(p, {})
    st[subsystem] = now
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st), encoding="utf-8")
    os.replace(tmp, p)


def prune(d: Path, keep: int = KEEP) -> list[str]:
    bundles = sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))
    gone = []
    for x in bundles[:-keep] if keep > 0 else bundles:
        shutil.rmtree(x, ignore_errors=True)
        gone.append(x.name)
    return gone


def capture(reason: str, *, subsystem: str | None = None, force: bool = False, root: Path | None = None,
            out_dir: Path | None = None, fallbacks: list[Path] | None = None, **kw) -> dict:
    """Write one bundle; returns {"captured": bool, "id", "path", ...}. Never raises for a collector
    failure (it's recorded in the manifest). If the bundle dir itself can't be written (ENOSPC, EROFS,
    EDQUOT, EACCES) the half-written .tmp dir is removed and the bundle goes to the first fallback dir
    that works (manifest `primary_failed`); only if every dir fails does it raise."""
    root = Path(root or ROOT)
    d = Path(out_dir or incidents_dir(root))
    dirs = [d] + [f for f in (fallbacks if fallbacks is not None else fallback_dirs()) if f != d]
    errors: list[str] = []
    for i, target in enumerate(dirs):
        try:
            r = _capture_into(target, reason, subsystem=subsystem, force=force, root=root,
                              primary_failed=errors[0] if errors else None, rate_dir=d, **kw)
            if errors:
                r["primary_failed"] = errors[0]
            return r
        except OSError as e:
            errors.append(f"{target}: {e!r}"[:300])
            if i == len(dirs) - 1:
                raise


def _capture_into(d: Path, reason: str, *, subsystem: str | None = None, force: bool = False,
                  root: Path, primary_failed: str | None = None, rate_dir: Path | None = None,
                  journal_lines: int = JOURNAL_LINES, file_cap: int = FILE_CAP, bundle_cap: int = BUNDLE_CAP,
                  keep: int = KEEP, now: float | None = None, collectors: dict | None = None) -> dict:
    rate_dir = rate_dir or d
    now = time.time() if now is None else float(now)
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    if subsystem and not force:
        wait = rate_limited(subsystem, now, rate_dir) if rate_dir.is_dir() else None
        if wait is not None:
            return {"captured": False, "why": f"rate-limited: {subsystem} already has a bundle this hour "
                                              f"(next allowed in {wait:.0f} s)"}
    stamp = _dt.datetime.fromtimestamp(now).strftime("%Y%m%dT%H%M%S")
    head = reason if not subsystem or slugify(reason).startswith(slugify(subsystem)) else f"{subsystem}-{reason}"
    bid = f"{stamp}_{slugify(head)}"
    final = d / bid
    k = 1
    while final.exists():
        k += 1
        final = d / f"{bid}-{k}"
    bid = final.name
    tmp = d / f".tmp_{bid}"
    tmp.mkdir(mode=0o700)
    try:
        return _fill(tmp, final, d, bid, reason, subsystem, force, root, primary_failed, rate_dir,
                     journal_lines, file_cap, bundle_cap, keep, now, collectors)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)   # never leave half a bundle behind
        raise


def _fill(tmp: Path, final: Path, d: Path, bid: str, reason: str, subsystem, force, root: Path,
          primary_failed, rate_dir: Path, journal_lines: int, file_cap: int, bundle_cap: int, keep: int,
          now: float, collectors) -> dict:
    mask = Masker(root)
    units = list(CORE_UNITS) + [u for u in SUBSYSTEM_UNITS.get(subsystem or "", ()) if u not in CORE_UNITS]
    plan: list[tuple[str, object]] = [
        ("health.json", lambda: collect_json_copy(root / "state" / "health" / "now.json")),
        ("integrity.json", lambda: collect_json_copy(root / "state" / "health" / "integrity.json")),
        ("events.json", lambda: collect_events(root)),
        ("procs.txt", collect_procs),
        ("system.txt", collect_system),
        ("git.txt", lambda: collect_git(root)),
    ]
    plan += [(f"journal_{u}.log", (lambda u=u: collect_journal(u, journal_lines))) for u in units]
    plan += [(f"logs_{n}.log", (lambda p=p: _tail_file(p, file_cap)))
             for n, p in (("supervise", root / "state" / "supervise.log"),
                          ("server_watchdog", root / "state" / "server_watchdog.log"))]
    if collectors:
        plan = [(n, collectors.get(n, fn)) for n, fn in plan] + \
               [(n, fn) for n, fn in collectors.items() if n not in dict(plan)]
    files, errors, skipped, total = {}, {}, [], 0
    for name, fn in plan:
        if total >= bundle_cap:
            skipped.append(name)
            continue
        t0 = time.monotonic()
        try:
            text = fn()  # type: ignore[operator]
        except Exception as e:  # noqa: BLE001
            errors[name] = repr(e)[:300]
            text = f"[collector failed: {e!r}]"
        text = mask(str(text))
        data = text.encode("utf-8", "replace")
        truncated = False
        cap = min(file_cap, max(0, bundle_cap - total))
        if len(data) > cap:
            data = b"[... truncated to the newest %d bytes ...]\n" % cap + data[-cap:]
            truncated = True
        p = tmp / name
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        total += len(data)
        files[name] = {"bytes": len(data), "truncated": truncated, "ms": round((time.monotonic() - t0) * 1000)}
    manifest = {"id": bid, "reason": mask(reason), "subsystem": subsystem, "ts": now,
                "iso": _dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
                "host": socket.gethostname(), "role": os.environ.get("IRIS_ROLE") or "live",
                "files": files, "collector_errors": errors, "skipped_over_cap": skipped,
                "total_bytes": total, "masking": dict(mask.notes, hits=mask.hits), "forced": bool(force),
                "primary_failed": mask(primary_failed) if primary_failed else None}
    fd = os.open(tmp / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    os.rename(tmp, final)
    if subsystem:
        try:
            mark_rate(subsystem, now, rate_dir)
        except OSError:
            if rate_dir != d:
                mark_rate(subsystem, now, d)
    pruned = prune(d, keep)
    return {"captured": True, "id": bid, "path": str(final), "total_bytes": total, "errors": errors,
            "skipped": skipped, "pruned": pruned}


def list_bundles(d: Path) -> list[dict]:
    out = []
    if not d.is_dir():
        return out
    for x in sorted(d.iterdir()):
        if x.is_dir() and not x.name.startswith("."):
            m = _load_json(x / "manifest.json", {"id": x.name, "reason": "[no manifest]"})
            out.append(m)
    return out


def resolve(d: Path, ident: str) -> Path | None:
    if (d / ident).is_dir():
        return d / ident
    hits = [x for x in sorted(d.iterdir()) if x.is_dir() and not x.name.startswith(".") and ident in x.name] \
        if d.is_dir() else []
    return hits[-1] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Iris incident bundles (V3 Phase 4)")
    sub = ap.add_subparsers(dest="op", required=True)
    c = sub.add_parser("capture")
    c.add_argument("reason", nargs="+")
    c.add_argument("--subsystem", default="")
    c.add_argument("--force", action="store_true", help="ignore the 1/subsystem/hour rate limit")
    c.add_argument("--journal-lines", type=int, default=JOURNAL_LINES)
    c.add_argument("--file-cap", type=int, default=FILE_CAP)
    c.add_argument("--bundle-cap", type=int, default=BUNDLE_CAP)
    ls = sub.add_parser("list")
    ls.add_argument("-n", type=int, default=20)
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.add_argument("--file", default="")
    a = ap.parse_args(argv)
    d = incidents_dir()
    if a.op == "capture":
        r = capture(" ".join(a.reason), subsystem=a.subsystem or None, force=a.force,
                    journal_lines=a.journal_lines, file_cap=a.file_cap, bundle_cap=a.bundle_cap)
        print(json.dumps(r, indent=1))
        return 0 if r["captured"] else 75
    if a.op == "list":
        rows = list_bundles(d)[-a.n:]
        for m in rows:
            print(f"{m.get('id')}  sub={m.get('subsystem') or '-'}  {m.get('total_bytes', 0) // 1024}K  "
                  f"err={len(m.get('collector_errors') or {})}  {str(m.get('reason'))[:80]}")
        if not rows:
            print(f"no incident bundles in {d}")
        return 0
    p = resolve(d, a.id)
    if p is None:
        print(f"no bundle matching {a.id!r}", file=sys.stderr)
        return 1
    if a.file:
        f = p / a.file
        if f.parent != p or not f.is_file():
            print(f"{a.file!r} is not a file in {p.name}", file=sys.stderr)
            return 1
        sys.stdout.write(f.read_text(encoding="utf-8", errors="replace"))
        return 0
    print(json.dumps(_load_json(p / "manifest.json", {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
