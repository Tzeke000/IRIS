#!/usr/bin/env python3
"""scripts/server/stack_rollback.py - the automatic v3 -> v2 ROLLBACK decisions (V3 Phase 1 step 1.6, 2026-10-30).

Called ONLY by scripts/server/iris_supervise.sh, and only while ~/.config/iris/active_stack says v3 and the launch
record (state/v3/launched.env) says THIS start really runs v3. With v2 / no file the supervisor never calls it.

Two triggers (phase1_plan.md step 1.6):
  (a) FAILED STARTS: a v3 launcher that exits on its own (not stopped by the restart flag) within
      FAILED_START_S of starting is a failed start; FAIL_COUNT of them within FAIL_WINDOW_S => roll back.
  (b) HEALTH RED: state/health/now.json (scripts/server/health_snapshot.py, every minute from the watchdog) shows
      sdk_stream or events with status "red" CONTINUOUSLY for RED_S, counting only snapshots taken after this
      boot. The snapshot already reads stack faults as "warn" for its own 10-min boot grace, so a v3 stack that
      never comes up rolls back ~RED_S after that grace ends. A missing / stale (> fresh_s) / pre-boot snapshot is
      UNKNOWN: it neither counts toward nor resets the red timer; two red snapshots further apart than fresh_s
      do not make a continuous run.

Pure decision functions take an explicit `now` (fake-clock tests). State: state/v3/rollback_state.json.
This module NEVER writes active_stack and never decides "v3": rolling FORWARD is a deliberate act
(stack_switch.sh v3), never automatic. The supervisor does the write (stack_switch.sh v2 --auto), the restart
(the existing .tmp/restart_cc.flag path) and the DM (`dm` below = iris_watchdog.dm, the watchdog's Discord path).

CLI (exit 0 = keep going, 3 = ROLL BACK with the reason on stdout, 2 = usage, other = error):
  start-exit --start-ts T [--now N] [--rc RC]     a v3 launcher exited on its own
  health --boot-ts T [--now N] [--health PATH]    poll while a v3 stack runs
  reset [--boot-ts T]                             a new v3 start: clear the red tracker (failures are kept)
  dm TEXT                                         DM Zeke ($IRIS_STACK_DM_CMD overrides, for drills/tests)
  counts [--root R]                               memory-intact check: JSONL lines + Chroma rows (read-only)
  status                                          print the rollback state
stdlib only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("IRIS_ROOT") or HERE.parents[1])
WATCH = ("sdk_stream", "events")
ROLLBACK_RC = 3


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return float(default)


def config() -> dict:
    return {
        "failed_start_s": _envf("IRIS_V3_FAILED_START_S", 600.0),   # exited before this = a failed start
        "fail_window_s": _envf("IRIS_V3_FAIL_WINDOW_S", 900.0),     # 15 min
        "fail_count": int(_envf("IRIS_V3_FAIL_COUNT", 2)),
        "red_s": _envf("IRIS_V3_RED_S", 600.0),                     # 10 min continuously red
        "fresh_s": _envf("IRIS_V3_HEALTH_FRESH_S", 180.0),          # = health_snapshot.FRESH_S
    }


def state_path(root: Path = ROOT) -> Path:
    return root / "state" / "v3" / "rollback_state.json"


def load_state(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_state(p: Path, st: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


# ── (a) failed starts ────────────────────────────────────────────────────────
def record_exit(st: dict, start_ts: float, now: float, cfg: dict, rc: int | None = None) -> tuple[bool, str]:
    """A v3 launcher exited WITHOUT the restart flag. Mutates st. Returns (rollback, why)."""
    up = max(0.0, now - float(start_ts))
    fails = [float(t) for t in st.get("failures") or [] if now - float(t) < cfg["fail_window_s"]]
    if up < cfg["failed_start_s"]:
        fails.append(now)
        st["last_failure"] = {"ts": now, "uptime_s": round(up, 1), "rc": rc}
    st["failures"] = fails
    if len(fails) >= cfg["fail_count"]:
        return True, (f"{len(fails)} failed v3 starts within {cfg['fail_window_s'] / 60:.0f} min "
                      f"(last: exited rc={rc} after {up:.0f}s)")
    return False, (f"v3 exit rc={rc} after {up:.0f}s - "
                   + (f"failed start {len(fails)}/{cfg['fail_count']}" if up < cfg["failed_start_s"]
                      else "ran past the boot window, not a failed start"))


# ── (b) health red ───────────────────────────────────────────────────────────
def observe_health(st: dict, doc, boot_ts: float, now: float, cfg: dict) -> tuple[bool, str]:
    """One look at now.json while a v3 stack runs. Mutates st["red"]. Returns (rollback, why)."""
    if st.get("boot_ts") != boot_ts:
        st["boot_ts"] = boot_ts
        st["red"] = {}
    red = st.setdefault("red", {})
    if not isinstance(doc, dict):
        return False, "health unknown (no readable now.json)"
    ts = doc.get("ts")
    if not isinstance(ts, (int, float)):
        return False, "health unknown (now.json has no ts)"
    ts = float(ts)
    if ts < boot_ts:
        return False, "health unknown (now.json predates this boot)"
    if now - ts > cfg["fresh_s"]:
        return False, f"health unknown (now.json is {now - ts:.0f}s old)"
    subs = doc.get("subsystems") or {}
    worst, msgs = 0.0, []
    for name in WATCH:
        ent = subs.get(name) if isinstance(subs, dict) else None
        status = ent.get("status") if isinstance(ent, dict) else None
        if status != "red":
            red.pop(name, None)
            continue
        r = red.get(name)
        if not isinstance(r, dict) or ts - float(r.get("last", 0)) > cfg["fresh_s"]:
            r = {"since": ts, "last": ts}          # first red, or a gap in the evidence: start a new run
        r["last"] = max(float(r["last"]), ts)
        r["error"] = str((ent or {}).get("last_error") or "")[:200]
        red[name] = r
        dur = r["last"] - float(r["since"])
        worst = max(worst, dur)
        msgs.append(f"{name} red {dur:.0f}s ({r['error'][:100] or 'no error text'})")
    if msgs and worst >= cfg["red_s"]:
        return True, f"health red for >= {cfg['red_s'] / 60:.0f} min after boot: " + "; ".join(msgs)
    return False, ("; ".join(msgs) if msgs else "sdk_stream/events not red")


def load_health(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── DM + memory counts (reuse the existing tools, by path - no sys.path edits) ──
def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"iris_stack_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dm(text: str) -> bool:
    cmd = os.environ.get("IRIS_STACK_DM_CMD")
    if cmd:
        r = subprocess.run(shlex.split(cmd) + [text], capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    if (os.environ.get("IRIS_ROLE") or "live").strip().lower() == "staging":
        return False                                  # a staging run never DMs (ONE-OF-ME)
    return bool(_load_sibling("iris_watchdog").dm(text))


def memory_counts(root: Path) -> dict:
    hs = _load_sibling("health_snapshot")
    r = hs.probe_memory_index({"root": root, "now": time.time()})
    d = r.get("detail") or {}
    return {"jsonl": d.get("jsonl"), "chroma": d.get("chroma"), "known_orphans": d.get("known_orphans"),
            "status": r.get("status"), "error": r.get("error")}


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("start-exit")
    a1.add_argument("--start-ts", type=float, required=True)
    a1.add_argument("--now", type=float, default=None)
    a1.add_argument("--rc", type=int, default=None)
    a2 = sub.add_parser("health")
    a2.add_argument("--boot-ts", type=float, required=True)
    a2.add_argument("--now", type=float, default=None)
    a2.add_argument("--health", default="")
    a3 = sub.add_parser("reset")
    a3.add_argument("--boot-ts", type=float, default=None)
    a4 = sub.add_parser("dm")
    a4.add_argument("text")
    a5 = sub.add_parser("counts")
    a5.add_argument("--root", default="")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    cfg = config()
    sp = state_path(ROOT)

    if a.cmd == "start-exit":
        st = load_state(sp)
        now = time.time() if a.now is None else a.now
        roll, why = record_exit(st, a.start_ts, now, cfg, a.rc)
        if roll:
            st["rollback"] = {"ts": now, "why": why, "trigger": "failed_starts"}
            st["failures"] = []
        save_state(sp, st)
        print(why)
        return ROLLBACK_RC if roll else 0
    if a.cmd == "health":
        st = load_state(sp)
        now = time.time() if a.now is None else a.now
        doc = load_health(Path(a.health) if a.health else ROOT / "state" / "health" / "now.json")
        roll, why = observe_health(st, doc, a.boot_ts, now, cfg)
        if roll:
            st["rollback"] = {"ts": now, "why": why, "trigger": "health_red"}
            st["red"] = {}
        save_state(sp, st)
        print(why)
        return ROLLBACK_RC if roll else 0
    if a.cmd == "reset":
        st = load_state(sp)
        st["red"] = {}
        st["boot_ts"] = a.boot_ts
        save_state(sp, st)
        return 0
    if a.cmd == "dm":
        return 0 if dm(a.text) else 1
    if a.cmd == "counts":
        print(json.dumps(memory_counts(Path(a.root) if a.root else ROOT), sort_keys=True))
        return 0
    if a.cmd == "status":
        print(json.dumps({"config": cfg, "state": load_state(sp)}, indent=1, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
