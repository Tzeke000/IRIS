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
         [--task-id T] [--priority N] [--side-effects none|reversible|external] [--cancel-lower]
         [--cpu-quota 1200%] [--cpu-weight 100]
    worker.py admit --kind render [--mem 8G] [--priority N --cancel-lower]   # exit 0/1 (+ cancel plan)
    worker.py status                              # running worker units
    worker.py history [--kind K] [-n 20]
    worker.py stop NAME                           # stop one worker's whole group
    worker.py sweep                               # expired task leases -> needs_review; lost launchers

Admission (conservative by design, start at 3 like the brief says):
  * running workers < IRIS_WORKER_CAP (default 3)
  * MemAvailable - RESERVE (6 GiB for the core) >= estimate, where estimate =
    max(requested --mem, 1.25 x p90 of this kind's measured peaks, 1 GiB)
  * slice headroom: slice MemoryMax - slice memory.current >= estimate
  * optional --gpu-mib N: free VRAM - 6 GiB voice reserve >= N
Iris may always run FEWER workers than this allows; the hard cgroup caps are the real wall.

V3 Phase 1 step 1.3 (2026-10-30) - every job is a TASK in the event store (brain/iris_events.py):
  * `run --task-id T` claims T (fence F) BEFORE systemd-run; without --task-id a task is auto-created
    (kind worker:<kind>) and claimed. The jobs row stores task_id + fence.
  * watch() heartbeats the claim every lease/4 s; the outcome goes back WITH the fence (complete on ok,
    fail otherwise). A STALE fence (the task was re-claimed meanwhile) is REJECTED by the store - the
    job's measurements are still recorded, its result is not (exit 3). A heartbeat that finds the claim
    gone stops the unit, unless the task is side_effects=external (then it only logs: reconcile by hand).
  * a killed launcher stops heartbeating -> the lease expires -> `sweep` (or anything calling
    store.sweep_expired) moves the task to needs_review. Never silently `running` (checklist P2-T4d).
    --detach jobs have nobody heartbeating, so their lease is timeout + 5 min and they end in
    needs_review unless someone completes them: nobody watched the outcome, so it IS unreviewed.
  * the store fails OPEN: an auto-created task that can't be written just runs untracked (noted);
    an explicit --task-id that can't be claimed is REFUSED (exit 77) - the fence is the point.

Priority + graceful cancel (P2-D3): `--priority N` (LOWER = more urgent, like iris_events; default 5).
`--cancel-lower`: when admission fails, stop strictly-lower-priority running jobs (least urgent first,
then newest first = least work lost) ONLY if a simulation shows that frees enough - otherwise nothing is
touched. Never a job whose side_effects is `external`, never one whose priority/side_effects are unknown
(rows from before 10-30, units not launched by this tool). Every cancel is logged to
state/workers/cancel_log.jsonl + stderr; the job row says `cancelled`; its task is cancelled in the store.

CPU (P2-R4b): per job CPUQuota (default 1200% = 12 of the VM's 36 vCPUs) + CPUWeight; the slice caps ALL
workers at 2400%. These only bite once the cpu controller is delegated to the user manager
(scripts/server/systemd/system/user@.service.d/iris-delegate.conf) - without it systemd ignores them.

GPU (P2-R1c): watch() samples nvidia-smi --query-compute-apps every ~5 s, maps pid -> unit through
/proc/<pid>/cgroup (health_snapshot.unit_of_pid, imported by path - no copy) and stores the job's
peak VRAM in the jobs `gpu_mib` column (0 = sampled, none used; NULL = could not sample).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
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
LEASE_S = float(os.environ.get("IRIS_WORKER_LEASE_S", "120"))
CPU_QUOTA = os.environ.get("IRIS_WORKER_CPU_QUOTA", "1200%")       # per job; the slice caps the sum
CPU_WEIGHT = int(os.environ.get("IRIS_WORKER_CPU_WEIGHT", "100"))  # among siblings inside the slice
GPU_SAMPLE_S = float(os.environ.get("IRIS_WORKER_GPU_SAMPLE_S", "5"))
CANCEL_WAIT_S = float(os.environ.get("IRIS_WORKER_CANCEL_WAIT_S", "30"))
DEFAULT_PRIORITY = 5
SIDE_EFFECTS = ("none", "reversible", "external")
CANCELLABLE = ("none", "reversible")   # the only side_effects classes a lower-priority cancel may stop
EX_TEMPFAIL, EX_NOPERM, EX_REJECTED = 75, 77, 3
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
    have = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
    for col, typ in (("task_id", "TEXT"), ("fence", "INTEGER"), ("priority", "INTEGER"),
                     ("side_effects", "TEXT")):
        if col not in have:  # 10-30 migration: older rows keep NULL = unknown = never auto-cancelled
            try:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:  # a concurrent launcher added it first
                pass
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


def _append_note(jid: int, text: str) -> None:
    with db() as c:
        c.execute("UPDATE jobs SET note=COALESCE(note,'') || ? WHERE id=?", (f" | {text}", jid))


# ── event store (brain/iris_events.py) - fail open ──────────────────────────────
_STORE: object = None


def events_store():
    """The V3 event store, or None if it can't be loaded (worker.py must keep working without it)."""
    global _STORE
    if _STORE is None:
        try:
            if str(ROOT) not in sys.path:
                sys.path.append(str(ROOT))  # append, never insert(0)
            from brain import iris_events
            _STORE = iris_events.get_store()
        except Exception as e:  # noqa: BLE001
            print(f"worker: event store unavailable ({e!r}); running untracked", file=sys.stderr)
            _STORE = False
    return _STORE or None


# ── GPU attribution (reuses health_snapshot.unit_of_pid by import-by-path) ─────
_HS: object = None


def _health_snapshot():
    global _HS
    if _HS is None:
        try:
            spec = importlib.util.spec_from_file_location(
                "iris_health_snapshot_for_worker", Path(__file__).resolve().parent / "health_snapshot.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            _HS = mod
        except Exception:  # noqa: BLE001
            _HS = False
    return _HS or None


def parse_compute_apps(text: str, unit_of) -> dict[str, int]:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` -> {unit: MiB}.
    Rows whose memory isn't a number ("[N/A]") or whose pid has no unit are skipped; several pids of
    one unit are summed."""
    out: dict[str, int] = {}
    for ln in (text or "").splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), int(float(parts[1]))
        except ValueError:
            continue
        u = unit_of(pid)
        if u:
            out[u] = out.get(u, 0) + mib
    return out


def gpu_by_unit() -> dict[str, int] | None:
    """VRAM per systemd unit right now; None if nvidia-smi or the pid->unit helper is unavailable."""
    hs = _health_snapshot()
    if hs is None or not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=6)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return parse_compute_apps(r.stdout, hs.unit_of_pid)


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


def stop_unit(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", unit], timeout=120)


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
            "mem_available": avail, "slice_max": sl_max, "slice_current": sl_cur, "gpu_free_mib": gfree,
            "gpu_need_mib": gpu_mib}
    return (not why), info


# ── priority cancel (P2-D3) ────────────────────────────────────────────────────
def running_jobs_detail() -> list[dict]:
    """Live worker units joined with their jobs rows (priority, side_effects, task) + current mem/VRAM.
    A live unit with no jobs row (launched by hand / another tool) comes back with priority None."""
    live = running_workers()
    if not live:
        return []
    with db() as c:
        rows = {r[0]: r for r in c.execute(
            "SELECT unit, id, name, kind, priority, side_effects, task_id, fence, start_ts FROM jobs "
            "WHERE end_ts IS NULL AND outcome IN ('running','detached')")}
    gpu = gpu_by_unit() or {}
    out = []
    for u in live:
        r = rows.get(u)
        cg = unit_cgroup(u)
        mem = (_read_int(cg / "memory.current") if cg else None) or 0
        d = {"unit": u, "mem": mem, "gpu_mib": gpu.get(u, 0), "job": None, "name": None, "kind": None,
             "priority": None, "side_effects": None, "task_id": None, "fence": None, "start_ts": 0.0}
        if r:
            d.update(job=r[1], name=r[2], kind=r[3], priority=r[4], side_effects=r[5], task_id=r[6],
                     fence=r[7], start_ts=r[8] or 0.0)
        out.append(d)
    return out


def plan_cancellations(info: dict, priority: int, candidates: list[dict]) -> dict:
    """Pure: which strictly-lower-priority jobs to stop so a refused admission would pass.
    Order = least urgent first (highest priority number), then newest first (least work lost).
    Only jobs with a known priority and side_effects in CANCELLABLE are eligible. Returns
    {"fits": bool, "cancel": [...], "skipped": [{unit, why}]}; "cancel" is empty unless it fits."""
    est = info["estimate"]
    running = len(info.get("running") or [])
    avail = info.get("mem_available") or 0
    sl_max, sl_cur = info.get("slice_max"), info.get("slice_current") or 0
    gfree, gneed = info.get("gpu_free_mib"), info.get("gpu_need_mib") or 0

    def fits() -> bool:
        if running >= CAP or avail - RESERVE < est:
            return False
        if sl_max is not None and sl_max - sl_cur < est:
            return False
        if gneed and (gfree is None or gfree - GPU_RESERVE_MIB < gneed):
            return False
        return True

    skipped, eligible = [], []
    for c in candidates:
        if c.get("priority") is None or c.get("side_effects") is None:
            skipped.append({"unit": c["unit"], "why": "unknown priority/side_effects - never auto-cancelled"})
        elif c["side_effects"] not in CANCELLABLE:
            skipped.append({"unit": c["unit"], "why": f"side_effects={c['side_effects']} - never auto-cancelled"})
        elif int(c["priority"]) <= int(priority):
            skipped.append({"unit": c["unit"], "why": f"priority {c['priority']} is not lower than {priority}"})
        else:
            eligible.append(c)
    eligible.sort(key=lambda c: (-int(c["priority"]), -(c.get("start_ts") or 0.0)))
    plan = []
    for c in eligible:
        if fits():
            break
        plan.append(c)
        running -= 1
        avail += c.get("mem") or 0
        sl_cur -= c.get("mem") or 0
        if gfree is not None:
            gfree += c.get("gpu_mib") or 0
    ok = fits()
    return {"fits": ok, "cancel": plan if ok else [], "skipped": skipped}


def cancel_log_path() -> Path:
    return DB.parent / "cancel_log.jsonl"


def cancel_jobs(plan: list[dict], by: dict) -> list[str]:
    """Stop each planned job: log FIRST (jsonl + stderr), mark the row + task cancelled, then stop the unit
    (so the job's own watcher finds a cancelled task and its late result is rejected by the fence)."""
    store = events_store()
    done = []
    for c in plan:
        rec = {"ts": time.time(), "cancelled_unit": c["unit"], "job": c.get("job"), "name": c.get("name"),
               "priority": c.get("priority"), "side_effects": c.get("side_effects"), "task_id": c.get("task_id"),
               "mem": c.get("mem"), "gpu_mib": c.get("gpu_mib"), "by": by}
        try:
            cancel_log_path().parent.mkdir(parents=True, exist_ok=True)
            with cancel_log_path().open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"worker: cancel log write failed: {e!r}", file=sys.stderr)
        print(f"worker: CANCEL {c['unit']} (priority {c.get('priority')}) for {by.get('name')} "
              f"(priority {by.get('priority')})", file=sys.stderr)
        why = f"cancelled for higher-priority {by.get('name')} (p{by.get('priority')})"
        if c.get("job") is not None:
            with db() as cn:
                cn.execute("UPDATE jobs SET outcome='cancelled', note=COALESCE(note,'') || ? WHERE id=?",
                           (f" | {why}", c["job"]))
        if store is not None and c.get("task_id"):
            store.cancel_task(c["task_id"], note=why, actor="worker:cancel-lower")
        stop_unit(c["unit"])
        done.append(c["unit"])
    return done


def admit_with_cancel(a, mem: int) -> tuple[bool, dict]:
    ok, info = admit(a.kind, mem, a.gpu_mib)
    if ok or not getattr(a, "cancel_lower", False):
        return ok, info
    plan = plan_cancellations(info, a.priority, running_jobs_detail())
    info["cancel_plan"] = {"fits": plan["fits"], "cancel": [c["unit"] for c in plan["cancel"]],
                           "skipped": plan["skipped"]}
    if not plan["fits"]:
        info["why"].append("cancelling every eligible lower-priority job would still not fit - none cancelled")
        return False, info
    info["cancelled"] = cancel_jobs(plan["cancel"], {"name": a.name, "kind": a.kind, "priority": a.priority})
    deadline = time.monotonic() + CANCEL_WAIT_S
    while True:  # memory frees as the cgroups die; re-measure, never trust the simulation
        ok, info2 = admit(a.kind, mem, a.gpu_mib)
        if ok or time.monotonic() >= deadline:
            info2["cancelled"], info2["cancel_plan"] = info["cancelled"], info["cancel_plan"]
            return ok, info2
        time.sleep(1.0)


# ── task claim (V3 step 1.3) ────────────────────────────────────────────────────
class Claim:
    def __init__(self, store, task_id: str, fence: int, lease_s: float, side_effects: str):
        self.store, self.task_id, self.fence = store, task_id, fence
        self.lease_s, self.side_effects = lease_s, side_effects
        self.lost = False

    def still_mine(self) -> bool | None:
        """True/False from a real read; None if the store can't answer (fail open)."""
        t = self.store.get_task(self.task_id)
        if t is None:
            return None
        return t.get("status") == "running" and t.get("fence") == self.fence


def precheck_task(store, task_id: str) -> tuple[dict | None, str]:
    if store is None:
        return None, "event store unavailable - an explicit --task-id needs it (the fence is the point)"
    t = store.get_task(task_id)
    if t is None:
        return None, f"task {task_id} not found (or store error: {getattr(store, 'last_error', None)})"
    if t.get("status") != "admitted":
        return None, f"task {task_id} is {t.get('status')} (fence {t.get('fence')}), not claimable"
    return t, ""


# ── run ────────────────────────────────────────────────────────────────────────
RUN_DEFAULTS = {"task_id": "", "priority": DEFAULT_PRIORITY, "side_effects": "none", "cancel_lower": False,
                "lease": LEASE_S, "cpu_quota": CPU_QUOTA, "cpu_weight": CPU_WEIGHT}


def run(a) -> int:
    for k, v in RUN_DEFAULTS.items():  # programmatic callers may pass a namespace without the 10-30 fields
        if not hasattr(a, k):
            setattr(a, k, v)
    mem = parse_size(a.mem) if a.mem else 0
    store = None
    if a.task_id:
        store = events_store()
        task_row, why = precheck_task(store, a.task_id)
        if task_row is None:
            print(json.dumps({"admitted": False, "claimed": False, "why": [why]}, indent=1))
            return EX_NOPERM
        a.side_effects = task_row.get("side_effects") or a.side_effects
    ok, info = admit_with_cancel(a, mem)
    if not ok and not a.force:
        print(json.dumps({"admitted": False, **info}, indent=1))
        return EX_TEMPFAIL
    store = store or events_store()
    mem_limit = mem or info["estimate"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)[:60]
    unit = f"{PREFIX}{safe}-{int(time.time())}.service"

    # claim BEFORE systemd-run
    claim = None
    lease_s = float(a.timeout + 300) if a.detach else float(a.lease)
    note = None
    if store is not None:
        tid = a.task_id
        if not tid:
            r = store.create_task(f"{a.kind}: {a.name}", kind=f"worker:{a.kind}", side_effects=a.side_effects,
                                  note=json.dumps(a.cmd)[:500], actor="worker.py")
            tid = r.id if r else None
        fence = store.claim(tid, f"worker:{unit}", lease_s, actor="worker.py") if tid else None
        if fence is not None:
            claim = Claim(store, tid, fence, lease_s, a.side_effects)
        elif a.task_id:
            print(json.dumps({"admitted": True, "claimed": False,
                              "why": [f"claim of {tid} lost (status now "
                                      f"{(store.get_task(tid) or {}).get('status')})"]}, indent=1))
            return EX_NOPERM
        else:
            note = f"untracked: event store write failed ({store.last_error})"
    else:
        note = "untracked: event store unavailable"

    props = ["-p", f"MemoryMax={mem_limit}", "-p", f"MemorySwapMax={a.swap}", "-p", "OOMScoreAdjust=500",
             "-p", "KillMode=control-group", "-p", f"RuntimeMaxSec={a.timeout}", "-p", f"Nice={a.nice}",
             "-p", f"CPUQuota={a.cpu_quota}", "-p", f"CPUWeight={a.cpu_weight}",
             "-p", "Environment=PYTHONUNBUFFERED=1"]
    if claim:
        # the job can heartbeat/complete itself with these if it wants to (worker.py does it anyway)
        props += [f"--setenv=IRIS_TASK_ID={claim.task_id}", f"--setenv=IRIS_TASK_FENCE={claim.fence}"]
    cmd = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}", f"--slice={SLICE}",
           f"--working-directory={a.cwd or os.getcwd()}", *props, "--", *a.cmd]
    with db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,cmd,mem_limit,timeout_s,start_ts,outcome,task_id,fence,"
                        "priority,side_effects,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (unit, a.name, a.kind, json.dumps(a.cmd), mem_limit, a.timeout, time.time(), "running",
                         claim.task_id if claim else None, claim.fence if claim else None, a.priority,
                         a.side_effects, note)).lastrowid
    hello = {"admitted": True, "job": jid, "unit": unit, "mem_limit": human(mem_limit),
             "estimate_from_history_n": info["history_n"], "priority": a.priority,
             "task_id": claim.task_id if claim else None, "fence": claim.fence if claim else None}
    if info.get("cancelled"):
        hello["cancelled"] = info["cancelled"]
    if not a.detach:
        cmd.insert(cmd.index("--quiet"), "--wait")
        cmd.remove("--quiet")  # --wait prints the result summary we parse
        print(json.dumps(hello), flush=True)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            _launch_failed(jid, claim, repr(e))
            return 1
        return watch(jid, unit, proc, claim=claim)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _launch_failed(jid, claim, r.stderr[-500:])
        print(r.stderr, file=sys.stderr)
        return r.returncode
    print(json.dumps(hello))
    with db() as c:
        c.execute("UPDATE jobs SET outcome='detached' WHERE id=?", (jid,))
    return 0


def _launch_failed(jid: int, claim: Claim | None, err: str) -> None:
    with db() as c:
        c.execute("UPDATE jobs SET end_ts=?, outcome=?, note=COALESCE(note,'') || ? WHERE id=?",
                  (time.time(), "launch_failed", " | " + err, jid))
    if claim:
        claim.store.fail(claim.task_id, claim.fence, note=f"launch failed: {err[-200:]}", actor="worker.py")


def watch(jid: int, unit: str, proc: subprocess.Popen, poll: float = 1.0, claim: Claim | None = None) -> int:
    """Sample the unit's cgroup (+ per-unit VRAM) until `systemd-run --wait` returns, heartbeating the task
    claim, then record the outcome - into the jobs row always, into the task only with the CURRENT fence.
    Kernel 5.15 has no memory.peak, so the peak is the max of sampled memory.current."""
    peak = 0
    cpu = None
    cg = None
    gpu_peak: int | None = None
    next_gpu = 0.0
    hb_every = max(1.0, (claim.lease_s if claim else LEASE_S) / 4.0)
    next_hb = time.monotonic() + hb_every
    while proc.poll() is None:
        cg = cg or unit_cgroup(unit)
        if cg:
            cur = _read_int(cg / "memory.peak") or _read_int(cg / "memory.current") or 0
            peak = max(peak, cur)
            cpu = cpu_usec(cg) or cpu
        now = time.monotonic()
        if now >= next_gpu:
            next_gpu = now + GPU_SAMPLE_S
            g = gpu_by_unit()
            if g is not None:
                gpu_peak = max(gpu_peak or 0, g.get(unit, 0))
        if claim and not claim.lost and now >= next_hb:
            next_hb = now + hb_every
            if not claim.store.heartbeat(claim.task_id, claim.fence) and claim.still_mine() is False:
                claim.lost = True
                msg = f"lost the claim on {claim.task_id} (fence {claim.fence})"
                print(f"worker: {msg}", file=sys.stderr)
                if claim.side_effects == "external":
                    _append_note(jid, msg + " - side_effects=external: left RUNNING, reconcile by hand")
                else:
                    _append_note(jid, msg + " - stopping the unit")
                    stop_unit(unit)
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
    with db() as c:
        row = c.execute("SELECT mem_limit, outcome FROM jobs WHERE id=?", (jid,)).fetchone()
    if outcome == "oom" and peak == 0:
        peak = (row[0] if row else 0) or 0  # died before the first sample: it needed at least its limit
    cancelled = bool(row and row[1] == "cancelled")
    if cancelled:
        outcome = "cancelled"  # a cancel-lower stopped it; the canceller already closed the task
    with db() as c:
        c.execute("UPDATE jobs SET end_ts=?, outcome=?, exit_code=?, peak_mem=?, cpu_usec=?, gpu_mib=?, "
                  "note=COALESCE(note,'') || ? WHERE id=?",
                  (time.time(), outcome, code, peak or None, cpu, gpu_peak, " | " + err[-400:], jid))
    task_result = None
    if claim and not cancelled:
        if outcome == "ok":
            accepted = claim.store.complete(claim.task_id, claim.fence, result_ref=f"job:{jid}",
                                            actor=f"worker:{unit}", note=f"job {jid} ok")
        else:
            accepted = claim.store.fail(claim.task_id, claim.fence, note=f"job {jid} {outcome} exit {code}",
                                        actor=f"worker:{unit}")
        if accepted:
            task_result = "recorded"
        else:
            cur = claim.store.get_task(claim.task_id) or {}
            task_result = "rejected"
            _append_note(jid, f"task result REJECTED: fence {claim.fence} is stale (task now "
                              f"{cur.get('status')} fence {cur.get('fence')})")
    print(json.dumps({"job": jid, "unit": unit, "outcome": outcome, "exit_code": code,
                      "peak_mem": human(peak or None), "cpu_s": round((cpu or 0) / 1e6, 1),
                      "gpu_mib": gpu_peak, "task_id": claim.task_id if claim else None,
                      "fence": claim.fence if claim else None, "task_result": task_result}))
    if task_result == "rejected":
        return EX_REJECTED
    return 0 if outcome == "ok" else 1


# ── sweep (P2-T4d) ─────────────────────────────────────────────────────────────
def sweep() -> dict:
    """Expired task leases -> needs_review (store.sweep_expired), and jobs rows still `running` whose
    unit is gone and whose launcher stopped recording (killed) -> `launcher_lost`."""
    store = events_store()
    out: dict = {"store": store.sweep_expired(actor="worker.py sweep") if store is not None else None}
    live = set(running_workers())
    now = time.time()
    lost = []
    with db() as c:
        for jid, unit, t0 in c.execute("SELECT id, unit, start_ts FROM jobs WHERE outcome='running' "
                                       "AND end_ts IS NULL").fetchall():
            if unit not in live and (t0 or 0) < now - 30:
                c.execute("UPDATE jobs SET outcome='launcher_lost', end_ts=?, note=COALESCE(note,'') || ? "
                          "WHERE id=? AND outcome='running'",
                          (now, " | launcher died before recording the outcome; unit gone", jid))
                lost.append(unit)
    out["launcher_lost"] = lost
    return out


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
    r.add_argument("--task-id", default="", help="claim this event-store task (else one is auto-created)")
    r.add_argument("--priority", type=int, default=DEFAULT_PRIORITY, help="LOWER = more urgent (iris_events)")
    r.add_argument("--side-effects", choices=SIDE_EFFECTS, default="none",
                   help="external = never auto-cancelled / auto-retried (a --task-id's own value wins)")
    r.add_argument("--cancel-lower", action="store_true",
                   help="if refused, stop strictly-lower-priority non-external jobs first (only if that fits)")
    r.add_argument("--lease", type=float, default=LEASE_S, help="task lease seconds (heartbeat every lease/4)")
    r.add_argument("--cpu-quota", default=CPU_QUOTA)
    r.add_argument("--cpu-weight", type=int, default=CPU_WEIGHT)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    ad = sub.add_parser("admit")
    ad.add_argument("--kind", required=True)
    ad.add_argument("--mem", default="")
    ad.add_argument("--gpu-mib", type=int, default=0)
    ad.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
    ad.add_argument("--cancel-lower", action="store_true", help="show (never execute) the cancel plan")
    sub.add_parser("status")
    sub.add_parser("sweep")
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
        if not ok and a.cancel_lower:
            plan = plan_cancellations(info, a.priority, running_jobs_detail())
            info["cancel_plan"] = {"fits": plan["fits"], "would_cancel": [c["unit"] for c in plan["cancel"]],
                                   "skipped": plan["skipped"]}
        print(json.dumps(info, indent=1))
        return 0 if ok else 1
    if a.op == "status":
        for d in running_jobs_detail():
            print(f"{d['unit']}  mem={human(d['mem'])} gpu={d['gpu_mib']}MiB p={d['priority']} "
                  f"fx={d['side_effects']} task={d['task_id']} fence={d['fence']}")
        print(f"slice {SLICE}: current={human(_read_int(slice_dir() / 'memory.current'))} "
              f"max={human(_read_int(slice_dir() / 'memory.max'))}  cap={CAP} workers")
        st = events_store()
        if st is not None:
            print(f"event store: expired task leases={st.stats().get('expired_task_leases')} "
                  f"(run `worker.py sweep` to move them to needs_review)")
        return 0
    if a.op == "sweep":
        print(json.dumps(sweep(), indent=1))
        return 0
    if a.op == "history":
        q = "SELECT id,kind,name,outcome,peak_mem,mem_limit,cpu_usec,start_ts,end_ts,gpu_mib,priority FROM jobs"
        args: tuple = ()
        if a.kind:
            q += " WHERE kind=?"
            args = (a.kind,)
        with db() as c:
            for row in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, a.n)):
                jid, kind, name, out, pk, lim, cpu, t0, t1, g, pr = row
                dur = f"{(t1 or time.time()) - t0:.0f}s"
                print(f"#{jid} {kind:10} {name:28} {out:10} peak={human(pk)} lim={human(lim)} "
                      f"cpu={(cpu or 0) / 1e6:.0f}s gpu={g if g is not None else '?'}MiB p={pr} dur={dur}")
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
