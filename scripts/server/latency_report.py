#!/usr/bin/env python3
"""scripts/server/latency_report.py - stage latencies from the V3 event store (Phase 1 step 1.4).

Reads state/v3/events.sqlite (brain/iris_events.py; override with --db or IRIS_EVENTS_DB) READ-ONLY and
prints p50/p90 per stage, per local day, per source:

    received->admitted    channel timestamp (events.received_ts) -> admitted into the store
    admitted->assigned    queue wait: admitted -> a turn took it (the first move INTO assigned)
    assigned->committed   the turn's work: assigned -> reply committed
    reply                 received -> committed (what the sender actually waited)
    tool_blocked          time an event waited in `admitted` while a tool call blocked the turn - only
                          where the host recorded it (events meta `tool_blocked_s`); otherwise "not recorded"

plus, for worker tasks (worker.py), per kind: queue (admitted->running) and run (running->done/failed).

    latency_report.py [--days 7] [--json] [--db PATH]

Store incidents (source `store`) are excluded. An event that never reached a stage simply contributes no
sample to it (counted under `open`). A missing / empty / unreadable DB prints "no data" and exits 0.
This doubles as the v2-vs-v3 benchmark the cutover gate needs (Phase 1 exit gate 4). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("received->admitted", "admitted->assigned", "assigned->committed", "reply", "tool_blocked")
TASK_STAGES = ("queue", "run")
_TASK_DONE = ("output_pending", "verified", "committed", "failed")
INCIDENT_SOURCE = "store"


def default_db() -> Path:
    env = os.environ.get("IRIS_EVENTS_DB")
    if env:
        return Path(env)
    try:  # the store's own resolver (honours brain.iris_paths); stdlib-only module
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
        from brain.iris_events import default_db_path
        return Path(default_db_path())
    except Exception:  # noqa: BLE001
        return ROOT / "state" / "v3" / "events.sqlite"


def pct(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no samples."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def summarize(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": _r(pct(xs, 0.5)), "p90": _r(pct(xs, 0.9)), "max": _r(max(xs) if xs else None)}


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def open_ro(path: Path) -> sqlite3.Connection | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        c.execute("SELECT 1 FROM transitions LIMIT 1")
        return c
    except sqlite3.Error:
        return None


_EV_SET = "SELECT event_id FROM events WHERE received_ts>=? AND source<>?"
_TASK_SET = "SELECT task_id FROM tasks WHERE created_ts>=?"


def _first_entries(c: sqlite3.Connection, entity: str, id_set_sql: str, args: tuple) -> dict:
    """{entity_id: {status: first ts it ENTERED that status}} for the ids `id_set_sql` selects."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in c.execute(f"SELECT entity_id, to_status, MIN(ts) AS ts FROM transitions WHERE entity=? AND "
                       f"entity_id IN ({id_set_sql}) GROUP BY entity_id, to_status", (entity, *args)):
        out[r["entity_id"]][r["to_status"]] = r["ts"]
    return out


def build(c: sqlite3.Connection, days: int, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    since = now - days * 86400
    firsts = _first_entries(c, "event", _EV_SET, (since, INCIDENT_SOURCE))
    samples: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # day -> source -> stage
    counts: dict = defaultdict(lambda: defaultdict(lambda: {"events": 0, "open": 0}))
    tool_recorded = False
    rows = c.execute("SELECT event_id, source, received_ts, status, meta_json FROM events "
                     "WHERE received_ts>=? AND source<>?", (since, INCIDENT_SOURCE)).fetchall()
    for r in rows:
        f = firsts.get(r["event_id"], {})
        rec = r["received_ts"]
        day, src = _day(rec), r["source"]
        counts[day][src]["events"] += 1
        if r["status"] not in ("committed", "cancelled"):
            counts[day][src]["open"] += 1
        st = samples[day][src]
        adm, asg, com = f.get("admitted"), f.get("assigned"), f.get("committed")
        if adm is not None:
            st["received->admitted"].append(max(0.0, adm - rec))
        if adm is not None and asg is not None and asg >= adm:
            st["admitted->assigned"].append(asg - adm)
        if asg is not None and com is not None and com >= asg:
            st["assigned->committed"].append(com - asg)
        if com is not None:
            st["reply"].append(max(0.0, com - rec))
        try:
            meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        except ValueError:
            meta = {}
        tb = meta.get("tool_blocked_s") if isinstance(meta, dict) else None
        if isinstance(tb, (int, float)):
            tool_recorded = True
            st["tool_blocked"].append(float(tb))

    tfirst = _first_entries(c, "task", _TASK_SET, (since,))
    tsamples: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # day -> kind -> stage
    for r in c.execute("SELECT task_id, kind, created_ts FROM tasks WHERE created_ts>=?", (since,)):
        f = tfirst.get(r["task_id"], {})
        t0 = f.get("admitted", r["created_ts"])
        run_ts = f.get("running")
        if run_ts is None:
            continue
        day, kind = _day(t0), r["kind"] or "?"
        tsamples[day][kind]["queue"].append(max(0.0, run_ts - t0))
        ends = [f[s] for s in _TASK_DONE if s in f and f[s] >= run_ts]
        if ends:
            tsamples[day][kind]["run"].append(min(ends) - run_ts)

    report = {"db": None, "days": days, "generated_ts": now, "tool_blocked_recorded": tool_recorded,
              "events": {}, "tasks": {}}
    for day in sorted(set(samples) | set(counts)):
        report["events"][day] = {}
        for src in sorted(set(samples[day]) | set(counts[day])):
            ent = dict(counts[day][src])
            ent["stages"] = {s: summarize(samples[day][src].get(s, [])) for s in STAGES
                             if s != "tool_blocked" or samples[day][src].get(s)}
            report["events"][day][src] = ent
    for day in sorted(tsamples):
        report["tasks"][day] = {k: {s: summarize(v.get(s, [])) for s in TASK_STAGES}
                                for k, v in sorted(tsamples[day].items())}
    report["no_data"] = not report["events"] and not report["tasks"]
    return report


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 1000:.0f}ms" if v < 1 else f"{v:.1f}s"


def render_text(rep: dict) -> str:
    if not rep["events"] and not rep["tasks"]:
        return "no data"
    out = [f"latency report - last {rep['days']} day(s) - {rep['db']}"]
    for day, srcs in rep["events"].items():
        out.append(f"\n{day}")
        for src, ent in srcs.items():
            out.append(f"  {src}  ({ent['events']} events, {ent['open']} open)")
            for stage, s in ent["stages"].items():
                out.append(f"    {stage:22} n={s['n']:<5} p50={_fmt(s['p50']):>8}  p90={_fmt(s['p90']):>8}"
                           f"  max={_fmt(s['max']):>8}")
    if not rep["tool_blocked_recorded"]:
        out.append("\n  tool_blocked: not recorded (the host does not write meta.tool_blocked_s yet)")
    if rep["tasks"]:
        out.append("\nworker tasks")
        for day, kinds in rep["tasks"].items():
            for kind, st in kinds.items():
                out.append(f"  {day} {kind:20} queue p50={_fmt(st['queue']['p50'])} p90={_fmt(st['queue']['p90'])}"
                           f"  run p50={_fmt(st['run']['p50'])} p90={_fmt(st['run']['p90'])}  n={st['run']['n']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="p50/p90 stage latencies from the V3 event store")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--db", default="")
    a = ap.parse_args(argv)
    path = Path(a.db) if a.db else default_db()
    c = open_ro(path)
    if c is None:
        print(json.dumps({"db": str(path), "no_data": True}) if a.json else f"no data ({path})")
        return 0
    try:
        rep = build(c, max(1, a.days))
    except sqlite3.Error as e:
        print(json.dumps({"db": str(path), "no_data": True, "error": str(e)}) if a.json
              else f"no data ({path}: {e})")
        return 0
    finally:
        c.close()
    rep["db"] = str(path)
    print(json.dumps(rep, indent=1) if a.json else render_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
