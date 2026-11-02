#!/usr/bin/env python3
"""aging_probe.py - measure whether my long-running processes AGE (leak memory / threads / file handles).

Why (ops research 2026-11-02, Zeke asked about weekly restarts): blind timed restarts cost more than
they fix here (cognition gap, memory-save contract, twin-spawn risk). Restart a part only when a
MEASURED trend says it is aging. This collects the trend:

  every run (iris-aging-probe.timer, 30 min): for each target process, sample RSS, thread count,
  open fds and uptime -> append to state/health/aging.jsonl (capped);
  then, per target and per process LIFETIME (same pid + start time), fit a least-squares slope over
  the last 7 days and write state/health/aging.json:
     verdict  ok      - flat or no trend
              watch   - growing, but < AGING_PCT_WEEK or < MIN_SPAN_H of data
              aging   - growing >= AGING_PCT_WEEK per week over >= MIN_SPAN_H  -> plan a quiet-hours
                        restart under the pre-restart contract (never automatic).

Read-only; never restarts anything. `--print` prints the summary.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "state" / "health" / "aging.jsonl"
OUT = ROOT / "state" / "health" / "aging.json"
LOG_CAP, LOG_KEEP = 8 << 20, 4 << 20
WINDOW_S = 7 * 86400
MIN_SPAN_H = 24.0
AGING_PCT_WEEK = 20.0          # a guess (research §3.4); tune from the baseline once there is one
METRICS = ("rss_kb", "threads", "fds")

# name -> substring that must appear in the process cmdline (first match by largest RSS wins)
TARGETS = {
    "runtime": "IRIS/iris_runtime.py",
    "body_host": "iris_body_host.py",
    "mouth_fish": "voice/wren_fish_server.py",
    "ears_daemon": "voice/wren_voice_daemon.py",
    "wirepod": "wire-pod/chipper/chipper",
    "vector_inhabit": "vector_inhabit_daemon",
    "vr_presence": "--bot --presence",
    "claude_cli": "claude-opus",
}
CLK = os.sysconf("SC_CLK_TCK")
BOOT = None


def boot_time() -> float:
    global BOOT
    if BOOT is None:
        for line in open("/proc/stat"):
            if line.startswith("btime"):
                BOOT = float(line.split()[1])
    return BOOT


def proc_sample(pid: int) -> dict | None:
    try:
        stat = open(f"/proc/{pid}/stat").read()
        fields = stat[stat.rindex(")") + 2:].split()
        start = boot_time() + int(fields[19]) / CLK
        threads = int(fields[17])
        rss_kb = 0
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        try:
            fds = len(os.listdir(f"/proc/{pid}/fd"))
        except PermissionError:
            fds = None
        return {"pid": pid, "start": round(start), "rss_kb": rss_kb, "threads": threads, "fds": fds}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def find_targets() -> dict:
    found: dict = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for name, needle in TARGETS.items():
            if needle in cmd:
                s = proc_sample(int(d))
                if s and (name not in found or s["rss_kb"] > found[name]["rss_kb"]):
                    found[name] = s
    return found


def slope_per_h(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    mx = sum(t for t, _ in pts) / n
    my = sum(v for _, v in pts) / n
    den = sum((t - mx) ** 2 for t, _ in pts)
    return 0.0 if den == 0 else sum((t - mx) * (v - my) for t, v in pts) / den * 3600.0


def analyse(rows: list[dict], now: float) -> dict:
    out: dict = {}
    for name in TARGETS:
        mine = [r for r in rows if r.get("name") == name and now - r["ts"] <= WINDOW_S]
        if not mine:
            continue
        last = mine[-1]
        life = [r for r in mine if r["pid"] == last["pid"] and r["start"] == last["start"]]
        span_h = (life[-1]["ts"] - life[0]["ts"]) / 3600.0
        res = {"pid": last["pid"], "uptime_h": round((now - last["start"]) / 3600.0, 1),
               "samples": len(life), "span_h": round(span_h, 1), "now": {m: last.get(m) for m in METRICS},
               "metrics": {}, "verdict": "ok"}
        for m in METRICS:
            pts = [(r["ts"], float(r[m])) for r in life if r.get(m) is not None]
            if len(pts) < 3:
                continue
            s = slope_per_h(pts)
            base = max(pts[0][1], 1.0)
            pct_week = 100.0 * s * 168.0 / base
            v = "ok"
            if pct_week > 2.0 and s > 0:
                v = "aging" if (pct_week >= AGING_PCT_WEEK and span_h >= MIN_SPAN_H) else "watch"
            res["metrics"][m] = {"per_h": round(s, 3), "pct_per_week": round(pct_week, 1), "verdict": v}
        order = {"ok": 0, "watch": 1, "aging": 2}
        res["verdict"] = max((x["verdict"] for x in res["metrics"].values()), key=order.get, default="ok")
        out[name] = res
    return out


def main() -> int:
    now = time.time()
    found = find_targets()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if "--print" not in sys.argv:
        with LOG.open("a") as f:
            for name, s in found.items():
                f.write(json.dumps({"ts": round(now), "name": name, **s}) + "\n")
        if LOG.stat().st_size > LOG_CAP:
            data = LOG.read_bytes()[-LOG_KEEP:]
            LOG.write_bytes(data[data.index(b"\n") + 1:])
    rows = []
    if LOG.exists():
        for line in LOG.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    summary = analyse(rows, now)
    doc = {"ts": now, "rule": f"aging = >= {AGING_PCT_WEEK}%/week growth over >= {MIN_SPAN_H} h of one "
                              "process lifetime; plan a quiet-hours restart under the pre-restart contract",
           "targets": summary,
           "aging": sorted(k for k, v in summary.items() if v["verdict"] == "aging"),
           "watch": sorted(k for k, v in summary.items() if v["verdict"] == "watch")}
    if "--print" in sys.argv:
        print(json.dumps(doc, indent=1))
        return 0
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(OUT)
    print(f"sampled {len(found)} targets; aging={doc['aging']} watch={doc['watch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
