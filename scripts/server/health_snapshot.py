#!/usr/bin/env python3
"""scripts/server/health_snapshot.py - state/health/now.json, written from OUTSIDE the Claude process tree.

V3 Phase 1 step 1.5 (2026-10-30). Called every minute by iris_watchdog.py (a CPU-only systemd user timer that
keeps running when I'm stalled), and runnable by hand:

    ~/venvs/iris-v100/bin/python scripts/server/health_snapshot.py [--out PATH] [--root REPO] [--print]

One entry per subsystem: {ok, status, last_success_ts, lag_s, last_error, last_error_ts, probe, detail, took_ms}
  status  ok | warn | red | off | parked | n/a
          off    = deliberately off (voice flag, quiet hours, no ~/LIVE) - NOT a fault, never "heal" it
          parked = deliberate eye rest (night park)
          n/a    = nothing to measure yet (e.g. the v3 event store isn't built)
  ok      True iff status == ok; False for warn/red; None for off/parked/n/a
  lag_s   seconds since the subsystem last PROVED itself: for age probes the age of the thing itself (frame,
          backup, stream message), for functional probes now - last_success_ts (0 when it passed just now)
  last_success_ts / last_error carry over from the previous now.json, so a red entry still says when it last worked.

Rule (V3 spec §13): running != healthy. Every entry is a functional probe or an age - a live pid is only ever a
precondition, never the verdict. Each probe runs in its own daemon thread with a timeout; one probe that raises
or hangs turns ITS entry red and every other entry is still written. The file is written atomically
(tmp + fsync + replace + dir fsync), so a reader never sees half a file.

stdlib only (sqlite3, json, socket, urllib) - no chromadb/torch imports, nothing that touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("IRIS_ROOT") or Path(__file__).resolve().parents[2])
HOME = Path.home()
SCHEMA = 1
PROBE_TIMEOUT_S = 8.0          # per probe (each network call inside has its own, shorter, timeout)
TOTAL_BUDGET_S = 20.0          # whole snapshot; the watchdog unit has TimeoutStartSec=90
BOOT_GRACE_S = 600.0           # the body takes minutes after a boot/restart: stack reds read as warn
FRESH_S = 180.0                # readers: a now.json older than this is stale (the timer is every 60 s)
BACKUP_STALE_S = 36 * 3600.0   # mirrors iris_watchdog.BACKUP_STALE_S
DISK_WARN, DISK_RED = 85.0, 95.0
EVENT_WARN_S = 600.0           # an un-committed event older than this = the store is not draining
DONE_STATUSES = ("committed", "failed", "cancelled")


# ── small helpers ────────────────────────────────────────────────────────────
def _load(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _age(p: Path, now: float):
    try:
        return max(0.0, now - p.stat().st_mtime)
    except OSError:
        return None


def _hhmm(s: str, dflt: str) -> str:
    s = str(s or dflt)
    return s if len(s) == 5 and s[2] == ":" else dflt


def quiet_window(root: Path, now: float) -> bool:
    """Same window logic as voice_quiet_hours.sh (state/voice_quiet_hours.json, default 22:30-05:10)."""
    cfg = _load(root / "state" / "voice_quiet_hours.json", {})
    if cfg.get("enabled") is False:
        return False
    start, end = _hhmm(cfg.get("start"), "22:30"), _hhmm(cfg.get("end"), "05:10")
    t = time.strftime("%H:%M", time.localtime(now))
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def voice_off(root: Path) -> bool:
    """True only if state/voice_deliberately_off.json exists AND says {"off": true} - the host's own
    _voice_flag_says_off semantics (fail-open; an existing file with off:false is ON, the 09-05 lesson)."""
    d = _load(root / "state" / "voice_deliberately_off.json", None)
    return isinstance(d, dict) and bool(d.get("off"))


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def _proc_stat(pid: int):
    """(comm, ppid, starttime_ticks) from /proc/<pid>/stat, or None."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        comm = raw[raw.index("(") + 1:raw.rindex(")")]
        rest = raw[raw.rindex(")") + 2:].split()
        return comm, int(rest[1]), int(rest[19])
    except Exception:
        return None


def find_pids(needle: str) -> list[int]:
    """pids whose cmdline contains `needle` (a /proc scan - precondition only, never a health verdict)."""
    out = []
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != me and needle in _proc_cmdline(int(d.name)):
            out.append(int(d.name))
    return sorted(out)


def proc_age_s(pid: int):
    st = _proc_stat(pid)
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
        return up - st[2] / os.sysconf("SC_CLK_TCK") if st else None
    except Exception:
        return None


def unit_of_pid(pid: int) -> str:
    """Last .service/.scope in /proc/<pid>/cgroup (systemd unit attribution), '' if none."""
    try:
        path = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[-1].split(":", 2)[2]
    except Exception:
        return ""
    for part in reversed(path.split("/")):
        if part.endswith((".service", ".scope")):
            return part
    return ""


def unit_active_seconds(unit: str):
    try:
        r = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p",
                            "ActiveEnterTimestampMonotonic"], capture_output=True, text=True, timeout=5)
        kv = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        if kv.get("ActiveState") != "active":
            return None
        return time.monotonic() - int(kv.get("ActiveEnterTimestampMonotonic") or 0) / 1e6
    except Exception:
        return None


def http_get(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read(4 << 20)


def tcp_connect(host: str, port: int, timeout: float = 3.0) -> None:
    with socket.create_connection((host, port), timeout=timeout):
        pass


def ears_ping(port: int = 8770, timeout: float = 2.0) -> dict:
    """The ears daemon's own `ping` command (newline-JSON over TCP) - a functional answer, not a port check."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(b'{"cmd": "ping"}\n')
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0] or b"{}")


def ro_sqlite(p: Path) -> sqlite3.Connection:
    """Read-only connection (never creates the file, never takes a write lock)."""
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
    c.execute("PRAGMA query_only=1")
    return c


def jsonl_tail(p: Path, nbytes: int = 256 << 10) -> list[dict]:
    """Parsed lines from the last `nbytes` of a JSONL file (cheap on multi-MB logs)."""
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            start = max(0, f.tell() - nbytes)
            f.seek(start)
            raw = f.read()
    except OSError:
        return []
    out = []
    for ln in raw.splitlines()[1 if start else 0:]:      # a mid-file seek lands inside a line: drop it
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def last_backup_ok_ts(path: Path):
    """Epoch of the newest 'YYYY-mm-dd HH:MM:SS backup ok' line (same parse as iris_watchdog)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for ln in reversed(lines):
        if " backup ok" in ln:
            try:
                return time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                continue
    return None


def _r(status: str, probe: str, error: str | None = None, age: float | None = None, **detail) -> dict:
    out = {"status": status, "probe": probe, "detail": detail}
    if error:
        out["error"] = error
    if age is not None:
        out["age_s"] = round(float(age), 1)
    return out


def _stack_red(ctx: dict, probe: str, error: str, **detail) -> dict:
    """A failure of something the stack owns: off without ~/LIVE, warn while booting, red otherwise."""
    if not ctx["live"]:
        return _r("off", probe, None, **detail, why="no ~/LIVE (deliberate stop)")
    if ctx["in_grace"]:
        return _r("warn", probe, error + " (booting - inside the grace window)", **detail)
    return _r("red", probe, error, **detail)


# ── probes (each: ctx -> partial entry; may raise; may hang - the runner contains it) ──────────────
def probe_ears(ctx: dict) -> dict:
    root, now = ctx["root"], ctx["now"]
    vc = _load(root / "scratch" / "voice_control.json", {})
    det = {"voice_flag_off": ctx["voice_off"], "quiet_hours": ctx["quiet"], "mic_muted": vc.get("mic_muted")}
    # last heard utterance: the host's body-log "ears" lines (markers start with "[") + the shared transcript's
    # voice lines. Silence != deaf (his mic-mute, nobody talking): informational only, never a verdict.
    ts = [float(e.get("ts_epoch") or 0) for e in jsonl_tail(root / "state" / "iris_body_log.jsonl")
          if e.get("channel") == "ears" and not str(e.get("event", "")).startswith("[")]
    ts += [float(e.get("ts") or 0) for e in jsonl_tail(root / "state" / "transcript.jsonl", 512 << 10)
           if e.get("role") == "user" and e.get("modality") == "voice"]
    det["last_utterance_age_s"] = round(now - max(ts), 1) if ts else None   # None = none in the log tails
    probe = "ears daemon :8770 {'cmd':'ping'} + last heard utterance age (body log)"
    try:
        resp = ears_ping()
        if resp.get("ok") and (resp.get("result") or {}).get("pong"):
            return _r("ok", probe, **det)
        err = f"ping answered without pong: {str(resp)[:120]}"
    except Exception as e:  # noqa: BLE001
        err = f"ping failed: {e!r}"[:200]
    if ctx["voice_off"]:
        return _r("off", probe, None, **det, why="voice_deliberately_off.json says off")
    up = unit_active_seconds("iris-ears")
    det["unit_up_s"] = None if up is None else round(up)
    if up is not None and up < 300:
        return _r("warn", probe, err + " (unit warming)", **det)
    return _r("red", probe, err, **det)


def probe_mouth(ctx: dict) -> dict:
    probe = "mouth :8769 GET /health == 200 'ok'"
    det = {"voice_flag_off": ctx["voice_off"]}
    try:
        code, body = http_get("http://127.0.0.1:8769/health", 3.0)
        if code == 200:
            return _r("ok", probe, **det, body=body[:40].decode("utf-8", "replace").strip())
        err = f"/health -> {code}"
    except Exception as e:  # noqa: BLE001
        err = f"/health failed: {e!r}"[:200]
    if ctx["voice_off"]:
        return _r("off", probe, None, **det, why="voice_deliberately_off.json says off")
    up = unit_active_seconds("iris-mouth")
    det["unit_up_s"] = None if up is None else round(up)
    if up is not None and up < 600:        # Fish warms ~3 min (measured 184 s); same grace as the watchdog
        return _r("warn", probe, err + " (unit warming)", **det)
    return _r("red", probe, err, **det)


def _audio_sink_addr() -> str:
    try:
        for ln in (HOME / ".config" / "iris" / "voice.env").read_text().splitlines():
            if ln.startswith("IRIS_AUDIO_SINK="):
                return ln.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def probe_audio_sink(ctx: dict) -> dict:
    """The tower sink (:8775) is owned by the ORB APP: app closed => voice_speak returns ok but plays NOWHERE.
    Mirrors selfcheck.sh: a WARNING (his app, not my fault), never red. Address from ~/.config/iris/voice.env."""
    probe = "TCP connect to IRIS_AUDIO_SINK (tower orb app, :8775)"
    addr = _audio_sink_addr()
    if not addr or ":" not in addr:
        return _r("warn", probe, "IRIS_AUDIO_SINK not set in ~/.config/iris/voice.env")
    host, _, port = addr.rpartition(":")
    try:
        tcp_connect(host, int(port), 3.0)
        return _r("ok", probe)
    except Exception as e:  # noqa: BLE001
        return _r("warn", probe, "sink unreachable - my voice plays NOWHERE (is the orb app open on the tower?) "
                  + type(e).__name__)


def probe_runtime(ctx: dict) -> dict:
    """The runtime's own loop heartbeat (state/runtime_loop_heartbeat.json, every ~5 s) - a real heartbeat."""
    hb = _load(ctx["root"] / "state" / "runtime_loop_heartbeat.json", {})
    probe = "state/runtime_loop_heartbeat.json ts age (<180 s) + pid is iris_runtime"
    if not hb.get("ts"):
        return _stack_red(ctx, probe, "heartbeat file missing/unreadable")
    age = ctx["now"] - float(hb["ts"])
    pid = int(hb.get("pid") or 0)
    alive = bool(pid) and "iris_runtime" in _proc_cmdline(pid)
    if age > 180 or not alive:
        r = _stack_red(ctx, probe, f"heartbeat {age:.0f}s old, runtime pid {pid} {'alive' if alive else 'gone'}",
                       pid=pid)
        r["age_s"] = round(age, 1)
        return r
    return _r("ok", probe, age=age, pid=pid)


def _live_transcript(root: Path, now: float):
    """Newest top-level session transcript of the main project (the CLI appends one line per stream message;
    helper sub-agents write under <session>/subagents/, so a top-level *.jsonl is the main session)."""
    d = HOME / ".claude" / "projects" / str(root.resolve()).replace("/", "-")
    best = None
    try:
        for p in d.glob("*.jsonl"):
            m = p.stat().st_mtime
            if best is None or m > best[0]:
                best = (m, p.name)
    except OSError:
        return None, None
    return (now - best[0], best[1]) if best else (None, None)


def probe_sdk_stream(ctx: dict) -> dict:
    """PROXY - the body host writes no stream heartbeat file (its _STREAM['last_msg_ts'] is in-memory only).
    Precondition: host + its claude child exist; verdict inputs: state/cognition_held.json (host_hold's
    'received, not read' flag) and the age of the live session transcript the CLI appends per message.
    An idle stream is legitimate, so the age is reported, not judged."""
    root, now = ctx["root"], ctx["now"]
    probe = ("PROXY: host + claude child present, cognition_held.json absent, session-transcript mtime age "
             "(no host stream heartbeat exists yet)")
    hosts = [p for p in find_pids("iris_body_host.py") if "python" in _proc_cmdline(p).split(" ", 1)[0]]
    age, name = _live_transcript(root, now)
    det = {"host_pids": hosts, "transcript": name, "proxy": True}
    if not hosts:
        return _stack_red(ctx, probe, "no iris_body_host.py process", **det)
    kids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            st = _proc_stat(int(d.name))
            if st and st[1] == hosts[0] and st[0] == "claude":
                kids.append(int(d.name))
    det["claude_pids"] = kids
    host_age = proc_age_s(hosts[0])
    det["host_up_s"] = None if host_age is None else round(host_age)
    if len(hosts) > 1:
        return _r("red", probe, f"{len(hosts)} body hosts (ONE-OF-ME broken)", age=age, **det)
    if not kids:
        return _stack_red(ctx, probe, "body host has no claude child", **det)
    held = _load(root / "state" / "cognition_held.json", None)
    if held is not None:
        return _r("red", probe, "cognition HELD (state/cognition_held.json): " + str(held)[:160], age=age, **det)
    return _r("ok", probe, age=age, **det)


def probe_workers(ctx: dict) -> dict:
    root, now = ctx["root"], ctx["now"]
    probe = "iris-workers.slice memory.current vs high/max + running iris-worker-* units + jobs.sqlite stale rows"
    det: dict = {}
    cg = None
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", "iris-workers.slice"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        cg = Path("/sys/fs/cgroup" + out) if out else None
    except Exception as e:  # noqa: BLE001
        det["slice_error"] = repr(e)[:120]

    def rd(name):
        try:
            v = (cg / name).read_text().strip()
            return None if v == "max" else int(v)
        except Exception:
            return None
    cur = rd("memory.current") if cg else None
    high, mx = (rd("memory.high"), rd("memory.max")) if cg else (None, None)
    det.update(slice_mem=cur, slice_high=high, slice_max=mx)
    running = []
    try:
        lu = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active,activating",
                             "--no-legend", "--plain", "iris-worker-*"], capture_output=True, text=True, timeout=5)
        running = [ln.split()[0] for ln in lu.stdout.splitlines() if ln.strip()]
    except Exception as e:  # noqa: BLE001
        det["list_error"] = repr(e)[:120]
    det["running"] = running
    stale = []
    db = root / "state" / "workers" / "jobs.sqlite"
    if db.is_file():
        try:
            with ro_sqlite(db) as c:
                for unit, name, t0 in c.execute(
                        "SELECT unit, name, start_ts FROM jobs WHERE outcome IN ('running','detached') "
                        "AND end_ts IS NULL"):
                    if unit not in running:
                        stale.append({"unit": unit, "name": name, "age_s": round(now - float(t0 or now))})
        except Exception as e:  # noqa: BLE001
            det["jobs_db_error"] = repr(e)[:120]
    det["stale_job_rows"] = stale[:10]
    det["stale_job_rows_n"] = len(stale)
    # v3 task leases (step 1.3 store, if it exists yet)
    ev = root / "state" / "v3" / "events.sqlite"
    if ev.is_file():
        try:
            with ro_sqlite(ev) as c:
                n = c.execute("SELECT count(*) FROM tasks WHERE status IN ('running','assigned','claimed') "
                              "AND lease_until IS NOT NULL AND lease_until < ?", (now,)).fetchone()[0]
                det["expired_task_leases"] = n
        except sqlite3.Error:
            det["expired_task_leases"] = None     # no tasks table yet
    if cur is not None and mx and cur > 0.95 * mx:
        return _r("red", probe, f"worker slice at {cur / mx:.0%} of MemoryMax", **det)
    warns = []
    if cur is not None and high and cur > high:
        warns.append("worker slice above MemoryHigh (throttled)")
    if stale:
        warns.append(f"{len(stale)} job row(s) still 'running' with no live unit (launcher killed?)")
    if det.get("expired_task_leases"):
        warns.append(f"{det['expired_task_leases']} expired task lease(s)")
    if cg is None:
        warns.append("worker slice not loaded")
    return _r("warn" if warns else "ok", probe, "; ".join(warns) or None, **det)


def probe_memory_index(ctx: dict) -> dict:
    """JSONL line count vs the Chroma 'iris_memories' metadata-segment row count - plain sqlite, read-only.
    Known Chroma-only orphans (state/iris_memory.chroma_orphans.jsonl) are expected extras."""
    root, now = ctx["root"], ctx["now"]
    probe = "state/iris_memory.jsonl non-empty lines vs Chroma iris_memories rows (+ known orphans)"
    jl = root / "state" / "iris_memory.jsonl"
    try:
        with open(jl, "rb") as f:
            n_jsonl = sum(1 for ln in f if ln.strip())
    except OSError as e:
        return _r("red", probe, f"memory JSONL unreadable: {e!r}"[:200])
    try:
        with open(root / "state" / "iris_memory.chroma_orphans.jsonl", "rb") as f:
            n_orph = sum(1 for ln in f if ln.strip())
    except OSError:
        n_orph = 0
    det = {"jsonl": n_jsonl, "known_orphans": n_orph}
    db = root / "memory" / "chroma" / "chroma.sqlite3"
    try:
        with ro_sqlite(db) as c:
            row = c.execute(
                "SELECT count(e.id) FROM collections col JOIN segments s ON s.collection = col.id "
                "LEFT JOIN embeddings e ON e.segment_id = s.id "
                "WHERE col.name = 'iris_memories' AND s.scope = 'METADATA'").fetchone()
            n_chroma = int(row[0]) if row else 0
            try:
                q = c.execute("SELECT count(*) FROM embeddings_queue").fetchone()[0]
                det["chroma_queue"] = q
            except sqlite3.Error:
                pass
    except Exception as e:  # noqa: BLE001
        return _r("warn", probe, f"chroma count unreadable (schema change?): {e!r}"[:200], **det)
    det["chroma"] = n_chroma
    missing = n_jsonl + n_orph - n_chroma
    det["missing_from_index"] = missing
    jl_age = _age(jl, now) or 0.0
    if missing > 5 and jl_age > 900:
        return _r("red", probe, f"{missing} memories in the JSONL are not in Chroma (recall can't find them)", **det)
    if missing != 0:
        return _r("warn", probe, f"index differs from JSONL by {missing:+d} (JSONL {n_jsonl}, Chroma {n_chroma}, "
                  f"orphans {n_orph})", **det)
    return _r("ok", probe, **det)


def probe_perception(ctx: dict) -> dict:
    """Runtime's own frame buffer via :5876 /api/v1/camera/live_frame ({ok, age_sec}; ok:false = no frame
    newer than 2 s). Never opens the camera (the runtime holds it exclusively). Night park = parked."""
    root = ctx["root"]
    probe = "GET :5876/api/v1/camera/live_frame -> ok + age_sec (frame_store buffer, <=2 s)"
    pause = (root / ".tmp" / "body_pause.flag").exists()
    eq = _load(root / "state" / "eyes_quiet_hours.json", {})
    servo = (root / "state" / "attention" / "servo_deliberately_off.json").exists()
    det = {"body_pause_flag": pause, "eyes_auto_parked": bool(eq.get("auto_parked_active")),
           "quiet_hours": ctx["quiet"], "servo_flag": servo}
    parked_ok = pause and (ctx["quiet"] or det["eyes_auto_parked"])
    try:
        code, body = http_get("http://127.0.0.1:5876/api/v1/camera/live_frame", 4.0)
        d = json.loads(body) if code == 200 else {}
        if d.get("ok") and d.get("b64"):
            age = float(d.get("age_sec") or 0.0)
            if pause:
                return _r("warn", probe, "live frame while body_pause.flag exists (flag stale?)", age=age, **det)
            return _r("ok", probe, age=age, **det)
        err = f"no live frame (HTTP {code}, ok={d.get('ok')})"
    except Exception as e:  # noqa: BLE001
        err = f"runtime :5876 unreachable: {e!r}"[:200]
        det["listener"] = "down"
    if parked_ok:
        return _r("parked", probe, None, **det, why="night eye rest (quiet hours / eyes_quiet_hours)")
    if pause:
        return _r("warn", probe, err + " - body_pause.flag set OUTSIDE quiet hours: a deliberate park, or the "
                  "stale morning flag (fix: gpu_park action=unpark, not eyes_reload)", **det)
    return _stack_red(ctx, probe, err, **det)


def probe_storage(ctx: dict) -> dict:
    probe = "disk used % for / and /data (warn 85, red 95)"
    det, worst, errs = {}, "ok", []
    for mnt in ("/", "/data"):
        try:
            u = shutil.disk_usage(mnt)
            pct = round(100.0 * u.used / u.total, 1)
            det[mnt] = {"used_pct": pct, "free_gb": round(u.free / 1e9, 1)}
            if pct >= DISK_RED:
                worst = "red"
                errs.append(f"{mnt} {pct}% full")
            elif pct >= DISK_WARN:
                worst = "warn" if worst == "ok" else worst
                errs.append(f"{mnt} {pct}% full")
        except OSError as e:
            worst = "red"
            errs.append(f"{mnt}: {e!r}")
    return _r(worst, probe, "; ".join(errs) or None, **det)


def probe_backup(ctx: dict) -> dict:
    probe = "newest 'backup ok' line in state/backup.log (red > 36 h) + iris-backup.timer enabled"
    ts = last_backup_ok_ts(ctx["root"] / "state" / "backup.log")
    det = {}
    try:
        en = subprocess.run(["systemctl", "--user", "is-enabled", "iris-backup.timer"], capture_output=True,
                            text=True, timeout=5).stdout.strip()
        det["timer"] = en
    except Exception:
        en = ""
    if ts is None:
        return _r("red", probe, "no successful backup on record", **det)
    age = ctx["now"] - ts
    if age > BACKUP_STALE_S:
        return _r("red", probe, f"last good backup {age / 3600:.0f} h ago", age=age, **det)
    if en and en != "enabled":
        return _r("warn", probe, f"iris-backup.timer is {en}", age=age, **det)
    return _r("ok", probe, age=age, **det)


def probe_gpu(ctx: dict) -> dict:
    root = ctx["root"]
    probe = "nvidia-smi --query-compute-apps per-pid VRAM, attributed via /proc/<pid>/cgroup"
    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=6)
    if g.returncode != 0 or not g.stdout.strip():
        return _r("red", probe, "nvidia-smi failed: " + (g.stderr or g.stdout)[-160:].strip())
    used, total, util, temp = (int(float(x)) for x in g.stdout.splitlines()[0].split(","))
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=6)
    worker_names = {}
    db = root / "state" / "workers" / "jobs.sqlite"
    if db.is_file():
        try:
            with ro_sqlite(db) as c:
                worker_names = dict(c.execute("SELECT unit, name FROM jobs WHERE end_ts IS NULL"))
        except sqlite3.Error:
            pass
    apps = []
    for ln in a.stdout.splitlines():
        try:
            pid_s, mib_s = (x.strip() for x in ln.split(","))
            pid = int(pid_s)
        except ValueError:
            continue
        unit = unit_of_pid(pid)
        app = {"pid": pid, "mib": int(float(mib_s)) if mib_s.replace(".", "").isdigit() else None, "unit": unit,
               "cmd": _proc_cmdline(pid)[:80]}
        if unit.startswith("iris-worker-"):
            app["worker_job"] = worker_names.get(unit)
        apps.append(app)
    det = {"used_mib": used, "total_mib": total, "util_pct": util, "temp_c": temp, "apps": apps}
    if total and used > 0.95 * total:
        return _r("warn", probe, f"VRAM {used}/{total} MiB (>95%)", **det)
    if temp >= 85:
        return _r("warn", probe, f"GPU {temp} C", **det)
    return _r("ok", probe, **det)


def probe_events(ctx: dict) -> dict:
    """state/v3/events.sqlite (step 1.1, built in parallel): oldest un-committed age per source + parity of
    discord rows vs state/discord_inbox.json 'unacked'. No-op (n/a) while the store doesn't exist."""
    root, now = ctx["root"], ctx["now"]
    probe = "state/v3/events.sqlite oldest un-committed received_ts per source + discord parity vs discord_inbox.json"
    db = root / "state" / "v3" / "events.sqlite"
    if not db.is_file():
        return _r("n/a", probe, None, why="event store not built yet (state/v3/events.sqlite absent)")
    det: dict = {}
    try:
        with ro_sqlite(db) as c:
            ph = ",".join("?" * len(DONE_STATUSES))
            rows = c.execute(f"SELECT source, count(*), min(received_ts) FROM events WHERE status NOT IN ({ph}) "
                             "GROUP BY source", DONE_STATUSES).fetchall()
            det["uncommitted"] = {s or "?": {"n": n, "oldest_age_s": round(now - float(t), 1) if t else None}
                                  for s, n, t in rows}
            try:
                dids = {str(r[0]) for r in c.execute(
                    f"SELECT channel_msg_id FROM events WHERE source='discord' AND status NOT IN ({ph})",
                    DONE_STATUSES)}
            except sqlite3.Error:
                dids = None
    except sqlite3.Error as e:
        return _r("red", probe, f"event store unreadable: {e!r}"[:200])
    warns = [f"{s}: oldest un-committed {v['oldest_age_s']:.0f}s" for s, v in det["uncommitted"].items()
             if v["oldest_age_s"] is not None and v["oldest_age_s"] > EVENT_WARN_S]
    inbox = _load(root / "state" / "discord_inbox.json", None)
    if isinstance(inbox, dict) and dids is not None:
        jids = {str(i.get("id")) for i in inbox.get("unacked") or [] if isinstance(i, dict)}
        det["parity"] = {"json_only": sorted(jids - dids)[:10], "db_only": sorted(dids - jids)[:10]}
        if jids != dids:
            warns.append(f"discord parity mismatch: {len(jids - dids)} json-only, {len(dids - jids)} db-only")
    oldest = max([v["oldest_age_s"] or 0 for v in det["uncommitted"].values()] or [0])
    return _r("warn" if warns else "ok", probe, "; ".join(warns) or None, age=oldest, **det)


PROBES = {
    "ears": probe_ears,
    "mouth": probe_mouth,
    "audio_sink": probe_audio_sink,
    "runtime": probe_runtime,
    "sdk_stream": probe_sdk_stream,
    "workers": probe_workers,
    "memory_index": probe_memory_index,
    "perception": probe_perception,
    "storage": probe_storage,
    "backup": probe_backup,
    "gpu": probe_gpu,
    "events": probe_events,
}


# ── runner + writer ──────────────────────────────────────────────────────────
def _in_grace(root: Path, now: float) -> bool:
    try:
        if float(Path("/proc/uptime").read_text().split()[0]) < BOOT_GRACE_S:
            return True
    except Exception:
        pass
    if now - float(_load(root / "state" / "planned_restart.json", {}).get("ts") or 0) < BOOT_GRACE_S:
        return True
    if now - float(_load(root / "state" / "server_watchdog.json", {}).get("last_restart_flag_ts") or 0) < BOOT_GRACE_S:
        return True
    for pid in find_pids("iris_body_host.py"):
        a = proc_age_s(pid)
        if a is not None and a < BOOT_GRACE_S:
            return True
    return False


def make_context(root: Path, now: float, **overrides) -> dict:
    ctx = {"root": root, "now": now, "live": (HOME / "LIVE").exists(), "quiet": quiet_window(root, now),
           "voice_off": voice_off(root)}
    if "in_grace" not in overrides:
        ctx["in_grace"] = _in_grace(root, now)
    ctx.update(overrides)
    return ctx


def run_probes(ctx: dict, probes: dict | None = None, timeout: float | None = None,
               budget: float | None = None) -> dict:
    """Every probe in its own DAEMON thread (a hung probe can't block exit). A raise or a timeout makes
    only that entry red."""
    probes = PROBES if probes is None else probes
    timeout = PROBE_TIMEOUT_S if timeout is None else timeout
    budget = TOTAL_BUDGET_S if budget is None else budget
    results: dict = {}
    threads = {}
    t0s = {}

    def call(name, fn):
        try:
            results[name] = fn(ctx)
        except Exception as e:  # noqa: BLE001
            results[name] = _r("red", getattr(fn, "__name__", name), f"probe raised: {e!r}"[:300])
    for name, fn in probes.items():
        t = threading.Thread(target=call, args=(name, fn), daemon=True, name=f"probe-{name}")
        t0s[name] = time.monotonic()
        t.start()
        threads[name] = t
    deadline = time.monotonic() + budget
    for name, t in threads.items():
        t.join(max(0.0, min(timeout - (time.monotonic() - t0s[name]), deadline - time.monotonic())))
        took = round((time.monotonic() - t0s[name]) * 1000)
        if t.is_alive() or name not in results:
            results[name] = _r("red", getattr(probes[name], "__name__", name),
                               f"probe timed out after {took / 1000:.1f}s")
        results[name]["took_ms"] = took
    return {name: results[name] for name in probes}


def finalize(results: dict, prev: dict, now: float) -> dict:
    """Partial probe results + the previous now.json -> the published entries."""
    prev_subs = (prev or {}).get("subsystems") or {}
    out = {}
    for name, r in results.items():
        p = prev_subs.get(name) or {}
        status = r.get("status", "red")
        ok = True if status == "ok" else (None if status in ("off", "parked", "n/a") else False)
        age = r.get("age_s")
        if status == "ok":
            last_ok = (now - age) if age is not None else now
        else:
            last_ok = p.get("last_success_ts")
        if age is not None and status == "ok":
            lag = age
        elif last_ok:
            lag = now - float(last_ok)
        else:
            lag = age        # never succeeded here: the age of the thing, if known
        err = r.get("error")
        out[name] = {
            "ok": ok,
            "status": status,
            "last_success_ts": round(last_ok, 1) if last_ok else None,
            "lag_s": round(lag, 1) if lag is not None else None,
            "last_error": err if err else p.get("last_error"),
            "last_error_ts": round(now, 1) if err else p.get("last_error_ts"),
            "error_now": bool(err),
            "probe": r.get("probe", ""),
            "detail": r.get("detail") or {},
            "took_ms": r.get("took_ms"),
        }
    return out


def write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(doc, indent=1, default=str).encode("utf-8")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def build_snapshot(root: Path | None = None, now: float | None = None, probes: dict | None = None,
                   prev: dict | None = None, **ctx_overrides) -> dict:
    root = Path(root or ROOT)
    now = time.time() if now is None else now
    t0 = time.monotonic()
    ctx = make_context(root, now, **ctx_overrides)
    subs = finalize(run_probes(ctx, probes), prev or {}, now)
    by = {s: sorted(k for k, v in subs.items() if v["status"] == s) for s in ("red", "warn", "off", "parked")}
    return {
        "schema": SCHEMA,
        "ts": round(now, 3),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "writer": "scripts/server/health_snapshot.py",
        "fresh_s": FRESH_S,
        "took_ms": round((time.monotonic() - t0) * 1000),
        "context": {"live": ctx["live"], "quiet_hours": ctx["quiet"], "voice_off": ctx["voice_off"],
                    "in_grace": ctx["in_grace"]},
        "summary": {"ok": not by["red"], **by},
        "subsystems": subs,
    }


def write_snapshot(out: Path | None = None, root: Path | None = None, **kw) -> dict:
    root = Path(root or ROOT)
    out = Path(out or root / "state" / "health" / "now.json")
    doc = build_snapshot(root, prev=_load(out, {}), **kw)
    write_atomic(out, doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--print", action="store_true", help="print the one-line-per-subsystem summary")
    a = ap.parse_args()
    root = Path(a.root) if a.root else ROOT
    doc = write_snapshot(Path(a.out) if a.out else None, root)
    if a.print:
        for k, v in doc["subsystems"].items():
            e = v["last_error"] if v["error_now"] else ""
            lag = '-' if v['lag_s'] is None else f"{v['lag_s']}s"
            print(f"{v['status']:6} {k:13} lag={lag} {e[:110]}")
        print("summary:", json.dumps(doc["summary"]), f"took {doc['took_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
