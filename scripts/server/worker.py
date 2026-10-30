#!/usr/bin/env python3
"""Iris worker launcher + job history (Awareness & Worker Management brief, Project 2).

Every disposable job (render chunk, build, test run, indexing, model download) runs as its
OWN transient systemd user unit inside iris-workers.slice (the shared cap: MemoryHigh 16G /
MemoryMax 20G / swap 2G for ALL workers together). Per job: MemoryMax, RuntimeMaxSec,
OOMScoreAdjust=500, KillMode=control-group (children die with it). The core services
(host, runtime, mouth, ears) are never in that slice, so killing a worker group never
touches me.

While the job runs, this samples the unit's cgroup (memory.current/peak, cpu.stat) and
records the outcome into SQLite (state/workers/jobs.sqlite) so later admission decisions
use MEASURED peaks per kind, not a snapshot.

    worker.py run  --kind render --name city_chunk3 [--mem 8G] [--timeout 3600] -- cmd args...
    worker.py admit --kind render [--mem 8G]      # would a new job fit right now? (exit 0/1)
    worker.py status                              # running worker units
    worker.py history [--kind K] [-n 20]
    worker.py stop NAME                           # stop one worker's whole group

Admission (conservative by design, start at 3 like the brief says):
  * running workers < IRIS_WORKER_CAP (default 3)
  * MemAvailable - RESERVE (6 GiB for the core) >= estimate, where estimate =
    max(requested --mem, 1.25 x p90 of this kind's measured peaks, 1 GiB)
  * slice headroom: slice MemoryMax - slice memory.current >= estimate
  * optional --gpu-mib N: free VRAM - 6 GiB voice reserve >= N
Iris may always run FEWER workers than this allows; the hard cgroup caps are the real wall.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = Path(os.environ.get("IRIS_WORKER_DB", ROOT / "state" / "workers" / "jobs.sqlite"))
SLICE = os.environ.get("IRIS_WORKER_SLICE", "iris-workers.slice")
CAP = int(os.environ.get("IRIS_WORKER_CAP", "3"))
RESERVE = int(os.environ.get("IRIS_WORKER_CORE_RESERVE_MIB", str(6 * 1024))) * 1024 * 1024
GPU_RESERVE_MIB = int(os.environ.get("IRIS_WORKER_GPU_RESERVE_MIB", str(6 * 1024)))
PREFIX = "iris-worker-"
_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_size(s: str | None) -> int:
    if not s:
        return 0
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?\s*", s, re.I)
    if not m:
        raise ValueError(f"bad size {s!r}")
    return int(float(m.group(1)) * _UNITS.get(m.group(2).upper(), 1))


def human(n: int | None) -> str:
    if n is None:
        return "?"
    for u in ("B", "K", "M", "G"):
        if abs(n) < 1024 or u == "G":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return str(n)


# ── SQLite job history ─────────────────────────────────────────────────────────
def db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB), timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY, unit TEXT, name TEXT, kind TEXT, cmd TEXT,
        mem_limit INTEGER, timeout_s INTEGER, start_ts REAL, end_ts REAL,
        outcome TEXT, exit_code INTEGER, peak_mem INTEGER, cpu_usec INTEGER,
        gpu_mib INTEGER, note TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS jobs_kind ON jobs(kind, end_ts)")
    return c


def kind_peaks(kind: str, limit: int = 50) -> list[int]:
    with db() as c:
        rows = c.execute("SELECT peak_mem FROM jobs WHERE kind=? AND peak_mem>0 "
                         "ORDER BY id DESC LIMIT ?", (kind, limit)).fetchall()
    return [r[0] for r in rows]


def p90(xs: list[int]) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


# ── live system probes ─────────────────────────────────────────────────────────
def meminfo_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def user_cgroup_root() -> Path:
    return Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service")


def slice_dir() -> Path:
    # systemd nests slices by dashes: iris-workers.slice lives under iris.slice. Ask systemd.
    cg = unit_cgroup(SLICE)
    if cg:
        return cg
    parts = SLICE[:-len(".slice")].split("-")
    p = user_cgroup_root()
    for i in range(1, len(parts) + 1):
        p = p / ("-".join(parts[:i]) + ".slice")
    return p


def _read_int(p: Path) -> int | None:
    try:
        v = p.read_text().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def unit_cgroup(unit: str) -> Path | None:
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path("/sys/fs/cgroup" + out) if out else None


def cpu_usec(cg: Path) -> int | None:
    try:
        for line in (cg / "cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def running_workers() -> list[str]:
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active,activating",
                              "--no-legend", "--plain", f"{PREFIX}*"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.split()[0] for ln in out.splitlines() if ln.strip()]


# ── admission ──────────────────────────────────────────────────────────────────
def admit(kind: str, mem_req: int, gpu_mib: int = 0) -> tuple[bool, dict]:
    hist = kind_peaks(kind)
    est = max(mem_req, int(1.25 * p90(hist)), 1024 ** 3)
    running = running_workers()
    avail = meminfo_available()
    sl_max = _read_int(slice_dir() / "memory.max")
    sl_cur = _read_int(slice_dir() / "memory.current") or 0
    why = []
    if len(running) >= CAP:
        why.append(f"{len(running)} workers running (cap {CAP})")
    if avail - RESERVE < est:
        why.append(f"MemAvailable {human(avail)} - core reserve {human(RESERVE)} < estimate {human(est)}")
    if sl_max is not None and sl_max - sl_cur < est:
        why.append(f"slice headroom {human(sl_max - sl_cur)} < estimate {human(est)}")
    gfree = None
    if gpu_mib:
        gfree = gpu_free_mib()
        if gfree is None or gfree - GPU_RESERVE_MIB < gpu_mib:
            why.append(f"GPU free {gfree} MiB - voice reserve {GPU_RESERVE_MIB} < {gpu_mib}")
    info = {"ok": not why, "why": why, "estimate": est, "history_n": len(hist), "running": running,
            "mem_available": avail, "slice_max": sl_max, "slice_current": sl_cur, "gpu_free_mib": gfree}
    return (not why), info


# ── run ────────────────────────────────────────────────────────────────────────
def run(a) -> int:
    mem = parse_size(a.mem) if a.mem else 0
    ok, info = admit(a.kind, mem, a.gpu_mib)
    if not ok and not a.force:
        print(json.dumps({"admitted": False, **info}, indent=1))
        return 75  # EX_TEMPFAIL: try later
    mem_limit = mem or info["estimate"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)[:60]
    unit = f"{PREFIX}{safe}-{int(time.time())}.service"
    props = ["-p", f"MemoryMax={mem_limit}", "-p", f"MemorySwapMax={a.swap}", "-p", "OOMScoreAdjust=500",
             "-p", "KillMode=control-group", "-p", f"RuntimeMaxSec={a.timeout}", "-p", f"Nice={a.nice}",
             "-p", "Environment=PYTHONUNBUFFERED=1"]
    cmd = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}", f"--slice={SLICE}",
           f"--working-directory={a.cwd or os.getcwd()}", *props, "--", *a.cmd]
    with db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,cmd,mem_limit,timeout_s,start_ts,outcome) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (unit, a.name, a.kind, json.dumps(a.cmd), mem_limit, a.timeout, time.time(),
                         "running")).lastrowid
    if not a.detach:
        cmd.insert(cmd.index("--quiet"), "--wait")
        cmd.remove("--quiet")  # --wait prints the result summary we parse
        print(json.dumps({"admitted": True, "job": jid, "unit": unit, "mem_limit": human(mem_limit),
                          "estimate_from_history_n": info["history_n"]}), flush=True)
        return watch(jid, unit, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                                 text=True))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        with db() as c:
            c.execute("UPDATE jobs SET end_ts=?, outcome=?, note=? WHERE id=?",
                      (time.time(), "launch_failed", r.stderr[-500:], jid))
        print(r.stderr, file=sys.stderr)
        return r.returncode
    print(json.dumps({"admitted": True, "job": jid, "unit": unit, "mem_limit": human(mem_limit),
                      "estimate_from_history_n": info["history_n"]}))
    with db() as c:
        c.execute("UPDATE jobs SET outcome='detached' WHERE id=?", (jid,))
    return 0


def watch(jid: int, unit: str, proc: subprocess.Popen, poll: float = 1.0) -> int:
    """Sample the unit's cgroup until `systemd-run --wait` returns, then record the outcome.
    Kernel 5.15 has no memory.peak, so the peak is the max of sampled memory.current."""
    peak = 0
    cpu = None
    cg = None
    while proc.poll() is None:
        cg = cg or unit_cgroup(unit)
        if cg:
            cur = _read_int(cg / "memory.peak") or _read_int(cg / "memory.current") or 0
            peak = max(peak, cur)
            cpu = cpu_usec(cg) or cpu
        time.sleep(poll)
    err = proc.stderr.read() if proc.stderr else ""
    m = re.search(r"Finished with result: (\S+)", err)
    result = m.group(1) if m else "unknown"
    m = re.search(r"status=(\d+)", err)
    code = int(m.group(1)) if m else proc.returncode
    m = re.search(r"CPU time consumed: ([\d.]+)(ms|s|min)", err)
    if m:
        cpu = int(float(m.group(1)) * {"ms": 1e3, "s": 1e6, "min": 6e7}[m.group(2)])
    outcome = {"success": "ok", "oom-kill": "oom", "timeout": "timeout", "exit-code": "exit-code",
               "signal": "killed"}.get(result, result)
    if outcome == "oom" and peak == 0:
        # died before the first sample: it needed at least its limit
        with db() as c:
            peak = c.execute("SELECT mem_limit FROM jobs WHERE id=?", (jid,)).fetchone()[0] or 0
    with db() as c:
        c.execute("UPDATE jobs SET end_ts=?, outcome=?, exit_code=?, peak_mem=?, cpu_usec=?, note=? WHERE id=?",
                  (time.time(), outcome, code, peak or None, cpu, err[-400:], jid))
    print(json.dumps({"job": jid, "unit": unit, "outcome": outcome, "exit_code": code,
                      "peak_mem": human(peak or None), "cpu_s": round((cpu or 0) / 1e6, 1)}))
    return 0 if outcome == "ok" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Iris worker launcher + job history")
    sub = ap.add_subparsers(dest="op", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--mem", default="")
    r.add_argument("--swap", default="1G")
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--nice", type=int, default=10)
    r.add_argument("--gpu-mib", type=int, default=0)
    r.add_argument("--cwd", default="")
    r.add_argument("--force", action="store_true", help="skip admission (the cgroup caps still apply)")
    r.add_argument("--detach", action="store_true")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    ad = sub.add_parser("admit")
    ad.add_argument("--kind", required=True)
    ad.add_argument("--mem", default="")
    ad.add_argument("--gpu-mib", type=int, default=0)
    sub.add_parser("status")
    h = sub.add_parser("history")
    h.add_argument("--kind", default="")
    h.add_argument("-n", type=int, default=20)
    s = sub.add_parser("stop")
    s.add_argument("name")
    a = ap.parse_args()

    if a.op == "run":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        if not a.cmd:
            ap.error("run needs a command after --")
        return run(a)
    if a.op == "admit":
        ok, info = admit(a.kind, parse_size(a.mem) if a.mem else 0, a.gpu_mib)
        print(json.dumps(info, indent=1))
        return 0 if ok else 1
    if a.op == "status":
        for u in running_workers():
            cg = unit_cgroup(u)
            cur = _read_int(cg / "memory.current") if cg else None
            print(f"{u}  mem={human(cur)}")
        print(f"slice {SLICE}: current={human(_read_int(slice_dir() / 'memory.current'))} "
              f"max={human(_read_int(slice_dir() / 'memory.max'))}  cap={CAP} workers")
        return 0
    if a.op == "history":
        q = "SELECT id,kind,name,outcome,peak_mem,mem_limit,cpu_usec,start_ts,end_ts FROM jobs"
        args: tuple = ()
        if a.kind:
            q += " WHERE kind=?"
            args = (a.kind,)
        with db() as c:
            for row in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, a.n)):
                jid, kind, name, out, pk, lim, cpu, t0, t1 = row
                dur = f"{(t1 or time.time()) - t0:.0f}s"
                print(f"#{jid} {kind:10} {name:28} {out:10} peak={human(pk)} lim={human(lim)} "
                      f"cpu={(cpu or 0) / 1e6:.0f}s dur={dur}")
        return 0
    if a.op == "stop":
        units = [u for u in running_workers() if a.name in u]
        for u in units:
            subprocess.run(["systemctl", "--user", "stop", u], timeout=30)
        print(json.dumps({"stopped": units}))
        return 0 if units else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
