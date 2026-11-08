#!/usr/bin/env python3
"""Iris worker launcher + job history (Awareness & Worker Management brief, Project 2).

Every disposable job (render chunk, build, test run, indexing, model download) runs as its
OWN transient systemd user unit inside iris-workers.slice (the shared cap: MemoryHigh 16G /
MemoryMax 35G (was 20G until 11-04) / swap 2G for ALL workers together). Per job: MemoryMax, RuntimeMaxSec,
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
    worker.py deps T                              # Phase 3: ready | waiting | blocked (+ why)
    worker.py task new --goal G [--kind K] [--after T1,T2] [--key K] [--side-effects ...]
    worker.py verify T --ref ACT_ID               # output_pending -> verified (unblocks dependants)

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
    --detach (opsguard 11-08) starts a small SUPERVISOR unit (iris-wsup-*) that runs the same request in the
    foreground - waits for --after parents, retries admission, watches, records, notifies. Before that,
    a detached unit had nobody heartbeating/recording: rows stayed 'detached' and tasks went needs_review.
  * the store fails OPEN: an auto-created task that can't be written just runs untracked (noted);
    an explicit --task-id that can't be claimed is REFUSED (exit 77) - the fence is the point.

Priority + graceful cancel (P2-D3): `--priority N` (LOWER = more urgent, like iris_events; default 5).
`--cancel-lower`: when admission fails, stop strictly-lower-priority running jobs (least urgent first,
then newest first = least work lost) ONLY if a simulation shows that frees enough - otherwise nothing is
touched. Never a job whose side_effects is `external`, never one whose priority/side_effects are unknown
(rows from before 10-30, units not launched by this tool). Every cancel is logged to
state/workers/cancel_log.jsonl + stderr; the job row says `cancelled`; its task is cancelled in the store.

Task dependencies (V3 Phase 3, 2026-11-02, opt-in): a task may declare `after: [task_ids]`
(`run --after T1,T2`, `task new --after`, or iris_events.create_task(after=...)). It is not claimed - so
not admitted, not launched - until EVERY parent is done-and-verified (store status verified|committed;
a worker's own `ok` only reaches output_pending, so a parent needs `worker.py verify T --ref ACT` or a
store complete(status=verified) first). A parent that failed / was cancelled / is missing, or a cycle,
makes the dependant `failed` with "not started: dependency X is failed" (`settle`, also run by `sweep`
every watchdog minute). `run --wait-deps S` polls up to S seconds; the waiting state lives ONLY in the
event store, so a killed scheduler loses nothing: re-run with `--task-id T` (or the same `--key`) and
it resumes the wait. Exit 75 = still waiting, 65 = a dependency failed. No `after` => unchanged.

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
EX_TEMPFAIL, EX_NOPERM, EX_REJECTED, EX_DEPFAILED = 75, 77, 3, 65
DEP_POLL_S = float(os.environ.get("IRIS_WORKER_DEP_POLL_S", "5"))
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


def supervisors() -> list[str]:
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active,activating",
                              "--no-legend", "--plain", "iris-wsup-*"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.split()[0] for ln in out.splitlines() if ln.strip()]


def stop_unit(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", unit], timeout=120)


# ── admission ──────────────────────────────────────────────────────────────────
def slice_used(d) -> int:
    """What the slice REALLY holds: memory.current minus ALL page cache (active + inactive file), plus shmem
    (tmpfs/shared pages live in "file" but can't be dropped). 10-30: inactive_file alone was subtracted after the
    Sands render's frame cache blocked admission. 11-07: still refused jobs with anon 3.8 G because 17.8 G of
    ACTIVE file cache (Godot imports/exports) counted as used; the kernel reclaims file pages before it OOM-kills
    anything, and a 15-min probe showed the disks idle (util <3 %, iowait 0), so evicting cache is cheap.
    The per-job estimate (1.25 x p90 of measured peaks) is the safety margin. IRIS_SLICE_COUNT_ACTIVE_FILE=1 = old."""
    cur = _read_int(d / "memory.current") or 0
    try:
        st = dict(l.split() for l in (d / "memory.stat").read_text().splitlines())
        if os.environ.get("IRIS_SLICE_COUNT_ACTIVE_FILE") == "1":
            cur -= int(st.get("inactive_file", 0))
        else:
            cur -= int(st.get("inactive_file", 0)) + int(st.get("active_file", 0)) - int(st.get("shmem", 0))
    except Exception:  # noqa: BLE001
        pass
    return max(0, cur)


def admit(kind: str, mem_req: int, gpu_mib: int = 0) -> tuple[bool, dict]:
    hist = kind_peaks(kind)
    est = max(mem_req, int(1.25 * p90(hist)), 1024 ** 3)
    running = running_workers()
    avail = meminfo_available()
    sl_max = _read_int(slice_dir() / "memory.max")
    sl_cur = slice_used(slice_dir())
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
        self.leased = False            # 11-03: file leases taken for this job (released when it ends)

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
                "lease": LEASE_S, "cpu_quota": CPU_QUOTA, "cpu_weight": CPU_WEIGHT, "ledger": False,
                "after": "", "key": "", "wait_deps": 0.0, "after_verified": False, "no_notify": False,
                "out": "", "argv": None, "wait_admit": 0.0}
# opsguard 11-08: how long a queued (--detach) dependant may wait for its parents before it gives up loudly
DEP_WAIT_MAX_S = float(os.environ.get("IRIS_WORKER_DEP_WAIT_MAX_S", str(24 * 3600)))
ADMIT_POLL_S = float(os.environ.get("IRIS_WORKER_ADMIT_POLL_S", "30"))
SUP_ADMIT_WAIT_S = 6 * 3600
SUP_PREFIX = "iris-wsup-"   # supervisor units of --detach jobs (NOT iris-worker-*: they never count as workers)
DONE_OK = ("output_pending", "verified", "committed")   # a parent's job finished OK (a result exists)
DONE_BAD = ("failed", "cancelled", "needs_review")


def resolve_after(ids: list[str]) -> tuple[list[str], list[str]]:
    """--after accepts task ids, `job:N` (worker job id) or `name:NAME` (newest job row with that name)."""
    out, bad = [], []
    for x in ids:
        if x.startswith(("job:", "name:")):
            k, v = x.split(":", 1)
            with db() as c:
                row = (c.execute("SELECT task_id FROM jobs WHERE id=?", (int(v) if v.isdigit() else -1,))
                       if k == "job" else
                       c.execute("SELECT task_id FROM jobs WHERE name=? ORDER BY id DESC LIMIT 1", (v,))).fetchone()
            (out.append(row[0]) if row and row[0] else bad.append(x))
        else:
            out.append(x)
    return out, bad


def parents_done_state(store, parents: list[str]) -> dict:
    """--after default semantics (opsguard 11-08): ready once every parent's job FINISHED OK (output_pending or
    later) - not 'verified', which nobody sets for a worker job, so the 11-07 dependant waited forever."""
    st = {"state": "ready", "waiting_on": [], "failed": [], "reason": None}
    for p in parents:
        t = store.get_task(p)
        s_ = (t or {}).get("status", "missing")
        if s_ in DONE_OK:
            continue
        if t is None or s_ in DONE_BAD:
            st["failed"].append({"task_id": p, "status": s_})
        else:
            st["waiting_on"].append({"task_id": p, "status": s_})
    if st["failed"]:
        f = st["failed"][0]
        st.update(state="blocked", reason=f"dependency {f['task_id']} is {f['status']}")
    elif st["waiting_on"]:
        st.update(state="waiting", reason="waiting on " + ", ".join(f"{w['task_id']} ({w['status']})"
                                                                   for w in st["waiting_on"]))
    return st


def wait_parents_done(store, parents: list[str], wait_s: float, poll: float | None = None,
                      sleep=time.sleep, clock=time.monotonic) -> dict:
    poll = DEP_POLL_S if poll is None else poll
    deadline = clock() + max(0.0, float(wait_s))
    while True:
        st = parents_done_state(store, parents)
        if st["state"] != "waiting" or clock() >= deadline:
            return st
        sleep(min(poll, max(0.05, deadline - clock())))


def supervisor_argv(argv: list[str], task_id: str | None, wait_s: float) -> list[str]:
    """The argv a --detach job's supervisor re-runs: same request, minus --detach / --wait-deps / --task-id,
    plus the task it already owns and a long dependency wait."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--":
            break
        if x == "--detach":
            continue
        if x in ("--wait-deps", "--task-id", "--wait-admit"):
            skip = True
            continue
        if x.startswith(("--wait-deps=", "--task-id=", "--wait-admit=")):
            continue
        out.append(x)
    rest = argv[argv.index("--"):] if "--" in argv else []
    extra = ["--wait-deps", str(int(wait_s)), "--wait-admit", str(SUP_ADMIT_WAIT_S)] + \
        (["--task-id", task_id] if task_id else [])
    return out + extra + rest


def launch_supervisor(a, task_id: str | None) -> int:
    """--detach (opsguard 11-08): instead of a bare unit nobody watches (15 rows stuck 'detached', no
    notify, --after dependants never started), start a small supervisor unit that runs this same request
    in the FOREGROUND: it waits for --after parents, gets admitted, watches, records the outcome, notifies."""
    if not a.argv or "run" not in a.argv:
        print(json.dumps({"admitted": False, "why": ["--detach needs the CLI argv (programmatic callers: run "
                                                     "without detach)"]}), file=sys.stderr)
        return 2
    i = a.argv.index("run")
    sub = supervisor_argv(a.argv[i + 1:], task_id, DEP_WAIT_MAX_S + (a.timeout if a.after else 0))
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)[:60]
    unit = f"{SUP_PREFIX}{safe}-{int(time.time())}.service"
    cmd = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}", "-p", "MemoryMax=512M",
           "-p", f"RuntimeMaxSec={int(DEP_WAIT_MAX_S + a.timeout + 900)}", "-p", "Environment=PYTHONUNBUFFERED=1",
           f"--working-directory={a.cwd or os.getcwd()}", "--", sys.executable, str(Path(__file__).resolve()),
           "run", *sub]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(json.dumps({"admitted": False, "why": [f"supervisor launch failed: {r.stderr[-300:]}"]}), file=sys.stderr)
        return r.returncode or 1
    print(json.dumps({"detached": True, "supervisor": unit, "task_id": task_id, "after": a.after or None,
                      "notify": notify_default(a.kind, a.no_notify),
                      "eta": eta(a.kind, a.name).get("say"),
                      "see": f"journalctl --user -u {unit} -f   |  worker.py history -n 5"}))
    return 0


# ── task dependencies (V3 Phase 3) ──────────────────────────────────────────────
def _split_ids(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        v = v.split(",")
    return [x.strip() for x in v if x and x.strip()]


def wait_for_deps(store, task_id: str, wait_s: float, poll: float | None = None,
                  sleep=time.sleep, clock=time.monotonic) -> dict:
    """Poll the store until `task_id`'s parents are done-and-verified (state=ready), a parent fails
    (settle -> the task is failed, state=blocked), or `wait_s` runs out (state=waiting). Nothing is
    held in this process: kill it at any point and the next caller sees exactly the same state."""
    poll = DEP_POLL_S if poll is None else poll
    deadline = clock() + max(0.0, float(wait_s))
    while True:
        st = store.dependency_state(task_id)
        if st is None:
            return {"state": "unknown", "reason": f"task {task_id} not found / store error "
                                                  f"({getattr(store, 'last_error', None)})"}
        if st["state"] == "blocked":
            store.settle_dependencies(actor="worker.py deps")
            return st
        if st["state"] == "ready" or clock() >= deadline:
            return st
        sleep(min(poll, max(0.05, deadline - clock())))


# ── "delegated != done": the act verification ledger (brain/action_ledger.py) - fail open ──────
# With --ledger the job opens a kind="delegated" action at launch (pending verification), its id
# rides into the unit as IRIS_ACT_ID, and the unit's exit is only NOTED on the row - Iris closes
# it with evidence after checking the result. helper.sh always passes --ledger (2026-11-01).
def _ledger():
    try:
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # append, never insert(0)
        from brain import action_ledger
        return action_ledger
    except Exception as e:  # noqa: BLE001
        print(f"worker: action ledger unavailable ({e!r}); delegated task not recorded", file=sys.stderr)
        return None


def ledger_open(a, unit: str, task_id: str | None) -> str | None:
    al = _ledger()
    if al is None:
        return None
    try:
        row = al.open_delegated(name=a.name, goal=" ".join(a.cmd)[:200], source=f"worker:{a.kind}",
                                ref=f"{unit} task={task_id}", deadline_s=float(a.timeout) + al.DEFAULT_DEADLINE_S)
        return row["id"]
    except Exception as e:  # noqa: BLE001
        print(f"worker: ledger open failed ({e!r}); delegated task not recorded", file=sys.stderr)
        return None


def ledger_exit(act_id: str | None, outcome: str, code=None, detail: str | None = None) -> None:
    if not act_id:
        return
    al = _ledger()
    try:
        if al is not None:
            al.record_delegate_exit(act_id, outcome=outcome, exit_code=code, detail=detail)
    except Exception as e:  # noqa: BLE001
        print(f"worker: ledger exit note failed ({e!r})", file=sys.stderr)


# HELPER LOCKDOWN (V3 spec §6/§13 "no worker or helper gets credentials, root or destructive tools"; the 11-01 doc
# audit's top gap). Works in a USER unit with no sudo: NoNewPrivileges kills sudo/setuid, PrivateUsers gives the
# unprivileged user namespace that InaccessiblePaths needs. Verified 11-02: sudo refused, secrets unreadable,
# repos + /data/projects writable, GPU visible, `claude -p` answers. The helper still needs ~/.claude/.credentials.json
# (its own login) — that one credential is unavoidable.
_SECRET_PATHS = ["~/.iris_private.env", "~/IRIS/config/private.local.json", "~/.claude/channels",
                 "~/.config/iris/secrets", "~/.anki_vector", "~/.config/gh", "~/.git-credentials"]
# Code that would later run OUTSIDE the sandbox if a helper could edit it.
_PERSIST_PATHS = ["~/.config/systemd", "~/.bashrc", "~/.profile", "~/.bash_profile", "~/.claude/settings.json",
                  "~/.claude/settings.local.json", "~/IRIS/.claude/settings.json", "~/IRIS/.claude/settings.local.json",
                  "~/IRIS/scripts/hooks", "~/IRIS/scripts/voice_stop_hook.py"]


def lockdown_props(mode) -> list:
    if mode in (None, "", "off"):
        return []
    paths = list(_SECRET_PATHS) + ([] if mode == "helper-ssh" else ["~/.ssh"])
    # 11-02 ESCAPE FIX (docaudit_1102): a transient `systemd-run --user` unit is NOT sandboxed, so the user manager
    # was a door out (it read ~/.iris_private.env). Deny the user bus + systemd's private socket, and make the files
    # that run code later, outside the sandbox (user units, shell rc, CC settings hooks), read-only.
    run_dir = f"/run/user/{os.getuid()}"
    paths += [f"{run_dir}/bus", f"{run_dir}/systemd"]
    inacc = " ".join("-" + os.path.expanduser(p) for p in paths)   # '-' = ignore if the path doesn't exist
    ro = " ".join("-" + os.path.expanduser(p) for p in _PERSIST_PATHS)
    return ["-p", "NoNewPrivileges=yes", "-p", "PrivateUsers=yes", "-p", f"InaccessiblePaths={inacc}",
            "-p", f"ReadOnlyPaths={ro}", "-p", "UnsetEnvironment=DBUS_SESSION_BUS_ADDRESS",
            "-p", f"Environment=IRIS_LOCKDOWN={mode}"]


def run(a) -> int:
    for k, v in RUN_DEFAULTS.items():  # programmatic callers may pass a namespace without the 10-30 fields
        if not hasattr(a, k):
            setattr(a, k, v)
    mem = parse_size(a.mem) if a.mem else 0
    store = None
    after, bad = resolve_after(_split_ids(a.after))
    if bad:
        print(json.dumps({"admitted": False, "why": [f"--after: no such job {bad}"]}, indent=1))
        return EX_NOPERM
    # default --after = parents FINISHED OK (job semantics); --after-verified = the store's verified edges
    store_after = after if a.after_verified else []
    if (after or a.key) and not a.task_id:
        # Phase 3: a dependant task must exist in the store BEFORE we wait/launch (the wait survives us)
        store = events_store()
        if store is None:
            print(json.dumps({"admitted": False, "claimed": False,
                              "why": ["event store unavailable - --after/--key need it"]}, indent=1))
            return EX_NOPERM
        note_ = json.dumps(a.cmd)[:500] + (f" | after_done={','.join(after)}" if after and not store_after else "")
        r = store.create_task(f"{a.kind}: {a.name}", kind=f"worker:{a.kind}", side_effects=a.side_effects,
                              idempotency_key=a.key or None, after=store_after, note=note_,
                              actor="worker.py")
        if r is None:
            print(json.dumps({"admitted": False, "claimed": False,
                              "why": [f"could not create the task: {store.last_error}"]}, indent=1))
            return EX_NOPERM
        if not r.created and store_after and r.status == "admitted":
            store.add_dependencies(r.id, store_after, actor="worker.py")   # same key, maybe new parents
        a.task_id = r.id
    if a.task_id:
        store = store or events_store()
        task_row, why = precheck_task(store, a.task_id)
        if task_row is None:
            print(json.dumps({"admitted": False, "claimed": False, "why": [why]}, indent=1))
            return EX_NOPERM
        a.side_effects = task_row.get("side_effects") or a.side_effects
        if a.detach:
            # queue it: the supervisor does the waiting (a check-once-and-exit here is what lost 11-07's job)
            return launch_supervisor(a, a.task_id)
        dep = wait_for_deps(store, a.task_id, float(a.wait_deps or 0))
        if dep["state"] == "ready" and after and not store_after:
            dep = wait_parents_done(store, after, float(a.wait_deps or 0))
            if dep["state"] == "blocked":   # the parent failed: this one never runs - say so in the store
                store.cancel_task(a.task_id, note=f"not started: {dep['reason']}")
        if dep["state"] != "ready":
            blocked = dep["state"] in ("blocked", "unknown")
            print(json.dumps({"admitted": False, "claimed": False, "task_id": a.task_id,
                              "deps": dep["state"], "why": [dep.get("reason")],
                              "waiting_on": dep.get("waiting_on", [])}, indent=1))
            print(f"worker: NOT STARTED '{a.name}': {dep.get('reason')}"
                  + ("" if blocked else " - nothing will start it later; use --detach (a supervisor waits and "
                                        "starts it when the parent finishes) or --wait-deps SECONDS"), file=sys.stderr)
            if blocked and notify_default(a.kind, a.no_notify) and float(a.wait_deps or 0) > 0:
                notify_done(a.name, 0, "dep_failed", None, 0, [], "")   # a queued job that will never run
            return EX_DEPFAILED if blocked else EX_TEMPFAIL
    elif a.detach:
        return launch_supervisor(a, None)
    ok, info = admit_with_cancel(a, mem)
    give_up = time.monotonic() + float(getattr(a, "wait_admit", 0) or 0)
    while not ok and not a.force and time.monotonic() < give_up:   # a supervisor queues for admission too
        time.sleep(ADMIT_POLL_S)
        ok, info = admit_with_cancel(a, mem)
    if not ok and not a.force:
        print(json.dumps({"admitted": False, **info}, indent=1))
        if float(getattr(a, "wait_admit", 0) or 0) > 0 and notify_default(a.kind, a.no_notify):
            notify_done(a.name, 0, "never_admitted", None, 0, [], "")
        return EX_TEMPFAIL
    store = store or events_store()
    mem_limit = mem or info["estimate"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)[:60]
    unit = f"{PREFIX}{safe}-{int(time.time())}.service"

    # claim BEFORE systemd-run
    claim = None
    lease_s = float(a.lease)   # --detach never reaches here any more: its supervisor runs us in the foreground
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

    lease_note = None
    globs = [g.strip() for g in (getattr(a, "leases", "") or "").split(",") if g.strip()]
    if claim and globs:
        lr = claim.store.acquire_leases(claim.task_id, globs, fence=claim.fence, owner=f"worker:{unit}") or {}
        if not lr.get("ok"):          # LOG-ONLY (11-03): say so, keep going
            lease_note = f"lease conflict (log-only): {lr.get('reason')} {lr.get('conflicts')}"[:400]
            print(json.dumps({"lease_warning": lease_note}), flush=True)
        else:
            claim.leased = True
    act_id = ledger_open(a, unit, claim.task_id if claim else None) if a.ledger else None
    props = ["-p", f"MemoryMax={mem_limit}", "-p", f"MemorySwapMax={a.swap}", "-p", "OOMScoreAdjust=500",
             "-p", "KillMode=control-group", "-p", f"RuntimeMaxSec={a.timeout}", "-p", f"Nice={a.nice}",
             "-p", f"CPUQuota={a.cpu_quota}", "-p", f"CPUWeight={a.cpu_weight}",
             "-p", "Environment=PYTHONUNBUFFERED=1"]
    if claim:
        # the job can heartbeat/complete itself with these if it wants to (worker.py does it anyway)
        props += [f"--setenv=IRIS_TASK_ID={claim.task_id}", f"--setenv=IRIS_TASK_FENCE={claim.fence}"]
        if globs:
            props += [f"--setenv=IRIS_LEASES={','.join(globs)}"]
    if act_id:
        props += [f"--setenv=IRIS_ACT_ID={act_id}"]
    props += lockdown_props(a.lockdown)
    notify = notify_default(a.kind, a.no_notify)
    out_paths: list[str] = []
    job_cmd = list(a.cmd)
    if notify or a.out:
        # capture output to a file ALWAYS (journal keeps it too); the caller's own redirect target is tailed first
        own = a.out or str(OUT_DIR / f"{unit[:-len('.service')]}.log")
        try:
            Path(own).parent.mkdir(parents=True, exist_ok=True)
            job_cmd = tee_wrap(job_cmd, own)
            out_paths = caller_out_files(a.cmd) + [own]
        except OSError as e:
            note = (note or "") + f" | no output file ({e})"
    cmd = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}", f"--slice={SLICE}",
           f"--working-directory={a.cwd or os.getcwd()}", *props, "--", *job_cmd]
    with db() as c:
        jid = c.execute("INSERT INTO jobs(unit,name,kind,cmd,mem_limit,timeout_s,start_ts,outcome,task_id,fence,"
                        "priority,side_effects,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (unit, a.name, a.kind, json.dumps(a.cmd), mem_limit, a.timeout, time.time(), "running",
                         claim.task_id if claim else None, claim.fence if claim else None, a.priority,
                         a.side_effects, note)).lastrowid
    hello = {"admitted": True, "job": jid, "unit": unit, "mem_limit": human(mem_limit),
             "estimate_from_history_n": info["history_n"], "priority": a.priority,
             "task_id": claim.task_id if claim else None, "fence": claim.fence if claim else None,
             "notify": notify, "out": out_paths[-1] if out_paths else None, "eta": eta(a.kind, a.name).get("say")}
    if act_id:
        hello["act_id"] = act_id
    if info.get("cancelled"):
        hello["cancelled"] = info["cancelled"]
    cmd.insert(cmd.index("--quiet"), "--wait")
    cmd.remove("--quiet")  # --wait prints the result summary we parse
    print(json.dumps(hello), flush=True)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        _launch_failed(jid, claim, repr(e))
        ledger_exit(act_id, "launch_failed", None, repr(e))
        if notify:
            notify_done(a.name, jid, "launch_failed", None, 0, out_paths, a.cwd)
        return 1
    return watch(jid, unit, proc, claim=claim, act_id=act_id,
                 notify={"name": a.name, "paths": out_paths, "cwd": a.cwd} if notify else None)


def _launch_failed(jid: int, claim: Claim | None, err: str) -> None:
    with db() as c:
        c.execute("UPDATE jobs SET end_ts=?, outcome=?, note=COALESCE(note,'') || ? WHERE id=?",
                  (time.time(), "launch_failed", " | " + err, jid))
    if claim:
        claim.store.fail(claim.task_id, claim.fence, note=f"launch failed: {err[-200:]}", actor="worker.py")


def watch(jid: int, unit: str, proc: subprocess.Popen, poll: float = 1.0, claim: Claim | None = None,
          act_id: str | None = None, notify: dict | None = None) -> int:
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
    if claim and getattr(claim, "leased", False):
        try:
            claim.store.release_leases(claim.task_id)
        except Exception:  # noqa: BLE001
            pass
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
    ledger_exit(act_id, outcome, code, f"job {jid}")   # noted only: the action stays open until verified
    if outcome not in ("ok", "cancelled"):
        _wake_on_bad_end(unit, outcome, code, peak)
    if notify:
        with db() as c:
            r0 = c.execute("SELECT start_ts FROM jobs WHERE id=?", (jid,)).fetchone()
        notify_done(notify["name"], jid, outcome, code, time.time() - ((r0 and r0[0]) or time.time()),
                    notify.get("paths") or [], notify.get("cwd") or "")
    print(json.dumps({"job": jid, "unit": unit, "outcome": outcome, "exit_code": code,
                      "peak_mem": human(peak or None), "cpu_s": round((cpu or 0) / 1e6, 1),
                      "gpu_mib": gpu_peak, "task_id": claim.task_id if claim else None,
                      "fence": claim.fence if claim else None, "task_result": task_result}))
    if task_result == "rejected":
        return EX_REJECTED
    return 0 if outcome == "ok" else 1



def _wake_on_bad_end(unit: str, outcome: str, code, peak) -> None:
    """11-04: an OOM/timeout/killed helper used to end SILENTLY - its run.sh dies before submitting helper_done
    (firemvp_1104 timeout, coop5_1104 OOM at 12 G: I only found out by polling). The worker survives its job, so
    it tells me. Helpers only (unit iris-worker-helper-NAME-<ts>); never raises."""
    try:
        m = re.match(r"iris-worker-helper-(.+)-\d+(?:\.service)?$", unit or "")
        if not m:
            return
        name = m.group(1)
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(1, str(root))
        from brain import iris_llm
        iris_llm.submit(
            f"[HELPER ENDED BADLY - automated, not Zeke] helper '{name}' ended with outcome={outcome} "
            f"(exit {code}, peak mem {human(peak or None)}). Its report may be missing or partial: "
            f"state/helpers/{name}.report.md / {name}.full.md / {name}.log. Check what it committed, decide "
            "relaunch / finish myself / drop, close its ledger action, then reply here with one line via llm_reply.",
            kind="helper_done", requester="worker:" + name, deadline_s=24 * 3600, priority=2)
    except Exception as e:  # noqa: BLE001
        print(f"[worker] wake-on-bad-end failed (non-fatal): {e!r}", file=sys.stderr)

# ── completion notification (opsguard 11-08: notify is the DEFAULT) ─────────────
# 11-07/08: land RED at 02:27 and a land abort at step 1 each sat UNSEEN ~1 h - nothing woke me when a
# worker job ended. job_notify.sh fixed it per call, by hand; now every watched job (helpers excepted:
# they have helper_done) submits kind="job_done" with rc + the last lines of its output. --no-notify opts out.
OUT_DIR = Path(os.environ.get("IRIS_WORKER_OUT_DIR", ROOT / "state" / "workers" / "out"))
_REDIRECT_RE = re.compile(r"(?:\|\s*tee(?:\s+-a)?|&?>>?)\s*([^\s|;&<>'\"]+)")


def notify_default(kind: str, no_notify: bool) -> bool:
    return not no_notify and kind != "helper"


def caller_out_files(cmd: list[str]) -> list[str]:
    """Files the caller's own command redirects into (`> f`, `>> f`, `&> f`, `| tee [-a] f`), so the
    notification can tail THOSE when the caller captured output itself (our tee then sees nothing)."""
    out = []
    for m in _REDIRECT_RE.finditer(" ".join(cmd)):
        p = m.group(1)
        if p not in ("/dev/null", "&1", "&2", "1", "2") and not p.startswith("&"):
            out.append(p)
    return out


def tee_wrap(cmd: list[str], out_file: str) -> list[str]:
    """stdout+stderr -> journal AND out_file; the exit code is the command's (pipefail)."""
    return ["bash", "-c", 'o=$1; shift; set -o pipefail; "$@" 2>&1 | tee -a "$o"', "worker-tee", out_file, *cmd]


def tail_text(paths: list[str], cwd: str = "", n: int = 12, limit: int = 1500) -> tuple[str, str]:
    """(path, last lines) of the first path with content - caller files first, then our tee file."""
    for p in paths:
        fp = Path(p) if os.path.isabs(p) else Path(cwd or ".") / p
        try:
            if fp.is_file() and fp.stat().st_size > 0:
                with open(fp, errors="replace") as f:
                    return str(fp), "".join(f.readlines()[-n:])[-limit:]
        except OSError:
            continue
    return (paths[-1] if paths else ""), "(no output captured)"


def _submit_llm(prompt: str, **kw) -> None:  # seam for tests
    sys.path.insert(1, str(ROOT))
    from brain import iris_llm
    iris_llm.submit(prompt, **kw)


def notify_done(name: str, jid: int, outcome: str, code, dur_s: float, paths: list[str], cwd: str = "") -> bool:
    try:
        where, tail = tail_text(paths, cwd)
        _submit_llm(
            f"[JOB DONE - automated, not Zeke] Worker job '{name}' (#{jid}) ended outcome={outcome} exit={code} "
            f"after {dur_s / 60:.0f} min. Output: {where}. Last lines:\n{tail}\n"
            "Read it, act on it (next step / tell Zeke if it matters), then reply here with one line via llm_reply.",
            kind="job_done", requester="job:" + name, deadline_s=24 * 3600, priority=2)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[worker] job_done notify failed (non-fatal): {e!r}", file=sys.stderr)
        return False


# ── ETA from history (opsguard 11-08: two wrong ETAs to Zeke the ship night) ────
def _stem(name: str) -> str:
    """lumen_deploy5 -> lumen_deploy, land_integ4b -> land_integ, integ4_final -> integ4_final."""
    return re.sub(r"_?\d+[a-z]?$", "", name) or name


def eta(kind: str, name: str = "", like: str = "", min_n: int = 2) -> dict:
    """p50/p90 duration of FINISHED jobs (ok or failed - a red land costs as long as a green one) of this
    kind whose name matches, trying the most specific pattern first: --like, the name's stem, its first token."""
    pats = [like] if like else []
    if name:
        st = _stem(name)
        pats += [st + "%", re.split(r"[_\-]", st)[0] + "%"]
    pats.append("%")
    with db() as c:
        for p in pats:
            rows = c.execute("SELECT end_ts - start_ts, outcome FROM jobs WHERE kind=? AND name LIKE ? AND end_ts IS NOT "
                             "NULL AND start_ts IS NOT NULL AND outcome NOT IN ('launch_failed','cancelled','lost',"
                             "'launcher_lost') ORDER BY id DESC LIMIT 30", (kind, p)).fetchall()
            # early aborts (a land dead at step 1 in 20 s) say nothing about how long a real run takes:
            # use the OK runs when there are enough of them, else every finished run
            ok = sorted(r[0] for r in rows if r[0] and r[0] > 0 and r[1] == "ok")
            d = ok if len(ok) >= min_n else sorted(r[0] for r in rows if r[0] and r[0] > 0)
            if len(d) >= min_n or (p == "%" and d):
                med = d[len(d) // 2]
                return {"n": len(d), "like": p, "p50_min": round(med / 60, 1), "p90_min": round(p90(d) / 60, 1),
                        "max_min": round(d[-1] / 60, 1),
                        "say": f"about {med / 60:.0f}-{p90(d) / 60:.0f} min (history: n={len(d)} {kind} jobs like '{p}')"}
    return {"n": 0, "say": f"no history for kind={kind} - say 'unknown, I'll time it'"}


# ── sweep (P2-T4d) ─────────────────────────────────────────────────────────────
_JOURNAL_STATUS = re.compile(r"Main process exited, code=(\w+), status=(\d+)")


def journal_exit(unit: str) -> tuple[str, int | None, float | None] | None:
    """The real end of a unit nobody watched, from the user journal (units are --collect'ed, so
    systemctl has forgotten them): (outcome 'ok'|'exit-code'|'killed'|'timeout'|'oom', code, end_ts) or None.
    end_ts = the journal time of the unit's last line - NOT the sweep time (that would poison eta())."""
    try:
        raw = subprocess.run(["journalctl", "--user", "-u", unit, "--no-pager", "-o", "short-unix", "-n", "60"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    msgs, last_ts = [], None
    for ln in raw.splitlines():
        m = re.match(r"(\d+\.\d+) \S+ [^:]+: (.*)$", ln)
        if m:
            last_ts = float(m.group(1))
            msgs.append(m.group(2))
    r = parse_journal_exit("\n".join(msgs))
    return (r[0], r[1], last_ts) if r else None


def parse_journal_exit(txt: str) -> tuple[str, int | None] | None:
    code = None
    for m in _JOURNAL_STATUS.finditer(txt):
        code = int(m.group(2))
    if "Failed with result 'oom-kill'" in txt or "oom-kill" in txt and "Failed with result" in txt:
        return "oom", code
    if "Failed with result 'timeout'" in txt:
        return "timeout", code
    if "Failed with result 'signal'" in txt:
        return "killed", code
    if "Failed with result 'exit-code'" in txt:
        return "exit-code", code
    if "Deactivated successfully" in txt or "Succeeded." in txt or (code is None and "Consumed " in txt):
        # systemd 249 logs only "Consumed N CPU time" for a clean exit of a --collect'ed unit (seen 11-08)
        return "ok", 0 if code is None else code
    if code is not None:
        return ("ok" if code == 0 else "exit-code"), code
    return None


def sweep() -> dict:
    """Expired task leases -> needs_review (store.sweep_expired), and jobs rows still `running`/`detached`
    whose unit is gone get their REAL end from the journal (opsguard 11-08: 15 detached rows sat 'running'
    on the health board); with no journal evidence a `running` row -> `launcher_lost`, a detached one -> `lost`."""
    store = events_store()
    out: dict = {"store": store.sweep_expired(actor="worker.py sweep") if store is not None else None}
    if store is not None:   # Phase 3: dependants of a failed/cancelled parent never wait forever
        out["deps_failed"] = store.settle_dependencies(actor="worker.py sweep").get("failed", [])
    live = set(running_workers())
    now = time.time()
    lost, reconciled = [], []
    with db() as c:
        rows = c.execute("SELECT id, unit, start_ts, outcome FROM jobs WHERE outcome IN ('running','detached') "
                         "AND end_ts IS NULL").fetchall()
    for jid, unit, t0, was in rows:
        if unit in live or (t0 or 0) >= now - 30:
            continue
        je = journal_exit(unit)
        with db() as c:
            if je:
                c.execute("UPDATE jobs SET outcome=?, exit_code=?, end_ts=?, note=COALESCE(note,'') || ? "
                          "WHERE id=? AND outcome=?", (je[0], je[1], je[2] if len(je) > 2 and je[2] else now,
                                                       f" | reconciled from journal ({was})", jid, was))
                reconciled.append({"unit": unit, "outcome": je[0], "exit_code": je[1]})
            else:
                new = "launcher_lost" if was == "running" else "lost"
                c.execute("UPDATE jobs SET outcome=?, end_ts=?, note=COALESCE(note,'') || ? "
                          "WHERE id=? AND outcome=?",
                          (new, now, " | launcher died before recording the outcome; unit gone" if was == "running"
                           else " | detached, unit gone, no journal record of its end", jid, was))
                lost.append(unit)
    out["launcher_lost"] = lost
    out["reconciled"] = reconciled
    return out


def task_cli(a) -> int:
    store = events_store()
    if store is None:
        print(json.dumps({"ok": False, "why": "event store unavailable"}))
        return EX_NOPERM
    if a.op == "deps":
        st = store.dependency_state(a.task_id)
        print(json.dumps(st if st is not None else {"state": "unknown", "why": "no such task"}, indent=1))
        return 0 if st and st["state"] == "ready" else (EX_TEMPFAIL if st and st["state"] == "waiting"
                                                        else EX_DEPFAILED)
    if a.op == "task":
        r = store.create_task(a.goal, kind=a.kind, side_effects=a.side_effects, idempotency_key=a.key or None,
                              after=_split_ids(a.after), actor="worker.py task")
        print(json.dumps({"ok": r is not None, "task_id": r.id if r else None,
                          "created": r.created if r else None, "why": None if r else store.last_error}))
        return 0 if r else 1
    # verify: the task's own current fence (it must already be output_pending - a result exists)
    t = store.get_task(a.task_id) or {}
    ok = t.get("status") in ("output_pending", "verified", "committed") and store.complete(a.task_id, int(t.get("fence") or 0), status="verified",
                                    verification_ref=a.ref, actor="worker.py verify")
    print(json.dumps({"ok": ok, "task_id": a.task_id, "status": (store.get_task(a.task_id) or {}).get("status")}))
    return 0 if ok else 1


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
    r.add_argument("--ledger", action="store_true",
                   help="record as a DELEGATED action in the act ledger (open = pending verification)")
    r.add_argument("--lockdown", choices=["off", "helper", "helper-ssh"], default="off",
                   help="V3 spec §6/§13 sandbox (11-02): 'helper' = NoNewPrivileges (no sudo/setuid), PrivateUsers, "
                        "secrets + ssh keys + Discord channel creds INACCESSIBLE; 'helper-ssh' = same but keeps ~/.ssh")
    r.add_argument("--after", default="", help="comma-separated task ids / job:N / name:NAME that must FINISH OK "
                   "first (output_pending+); a failed parent => this never runs (loud). See --after-verified")
    r.add_argument("--leases", default="", help="comma-separated path globs this job may write (file leases, "
                   "LOG-ONLY: a conflict is reported, the job still runs); released when it ends")
    r.add_argument("--key", default="", help="idempotency key for the auto-created task (re-run = same task)")
    r.add_argument("--wait-deps", type=float, default=0.0, help="seconds to wait for --after parents (0 = check once; "
                   "--detach queues it: a supervisor waits up to IRIS_WORKER_DEP_WAIT_MAX_S, default 24 h)")
    r.add_argument("--wait-admit", type=float, default=0.0,
                   help="seconds to keep retrying admission (a --detach supervisor uses 6 h)")
    r.add_argument("--after-verified", action="store_true",
                   help="old strict semantics: parents must be VERIFIED (worker.py verify), not just finished ok")
    r.add_argument("--no-notify", action="store_true",
                   help="don't wake Iris (kind job_done) when the job ends - notify is the DEFAULT (helpers excepted)")
    r.add_argument("--out", default="", help="output file (default state/workers/out/<unit>.log when notifying)")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    ad = sub.add_parser("admit")
    ad.add_argument("--kind", required=True)
    ad.add_argument("--mem", default="")
    ad.add_argument("--gpu-mib", type=int, default=0)
    ad.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
    ad.add_argument("--cancel-lower", action="store_true", help="show (never execute) the cancel plan")
    sub.add_parser("status")
    et = sub.add_parser("eta", help="history-based duration (p50/p90) for a job like this - quote THIS, not a guess")
    et.add_argument("--kind", required=True)
    et.add_argument("--name", default="")
    et.add_argument("--like", default="", help="SQL LIKE pattern on the job name (else the name's stem)")
    sub.add_parser("sweep")
    dp = sub.add_parser("deps")
    dp.add_argument("task_id")
    tk = sub.add_parser("task")
    tk.add_argument("action", choices=["new"])
    tk.add_argument("--goal", required=True)
    tk.add_argument("--kind", default="task")
    tk.add_argument("--after", default="")
    tk.add_argument("--key", default="")
    tk.add_argument("--side-effects", choices=SIDE_EFFECTS, default="none")
    vf = sub.add_parser("verify")
    vf.add_argument("task_id")
    vf.add_argument("--ref", required=True, help="act ledger verification id (evidence)")
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
        a.argv = sys.argv[1:]
        return run(a)
    if a.op == "eta":
        print(json.dumps(eta(a.kind, a.name, a.like), indent=1))
        return 0
    if a.op == "admit":
        ok, info = admit(a.kind, parse_size(a.mem) if a.mem else 0, a.gpu_mib)
        if not ok and a.cancel_lower:
            plan = plan_cancellations(info, a.priority, running_jobs_detail())
            info["cancel_plan"] = {"fits": plan["fits"], "would_cancel": [c["unit"] for c in plan["cancel"]],
                                   "skipped": plan["skipped"]}
        print(json.dumps(info, indent=1))
        return 0 if ok else 1
    if a.op == "status":
        sw = sweep()   # opsguard 11-08: rows whose unit is gone get their real end before we list anything
        if sw.get("reconciled") or sw.get("launcher_lost"):
            print(f"(reconciled {len(sw.get('reconciled') or [])} ended job row(s) from the journal, "
                  f"{len(sw.get('launcher_lost') or [])} lost)")
        for u in supervisors():
            print(f"{u}  (supervisor of a --detach job: waiting for deps / watching)")
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
    if a.op in ("deps", "task", "verify"):
        return task_cli(a)
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
        units = [u for u in supervisors() + running_workers() if a.name in u]   # supervisor first: no relaunch
        for u in units:
            subprocess.run(["systemctl", "--user", "stop", u], timeout=30)
        print(json.dumps({"stopped": units}))
        return 0 if units else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
