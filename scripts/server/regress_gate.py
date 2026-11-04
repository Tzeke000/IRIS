#!/usr/bin/env python3
"""scripts/server/regress_gate.py - the V3 regression gate suite (V3 plan step 5.1, 2026-11-04).

One command that says PASS/FAIL for "did this stack get worse than the baseline", per check and overall:

    latency   scripts/server/latency_report.py --json over the baseline's window; the gated stages' p50/p90
              (pooled over the window) against the stored baseline, + rel_tol (10 %, the 5.2 spec gate) + abs_tol_s
    health    state/health/now.json is fresh and has no red, AND state/health/history.jsonl (one line per
              snapshot, written by health_snapshot.py) covers the last 24 h with no gap > max_gap_s and no red
    memory    state/iris_memory.jsonl + the Chroma 'iris_memories' collection: every baseline id still present,
              no baseline entry's text changed (new appends are fine). Chroma is opened READ-ONLY (mode=ro +
              query_only, one short query, closed at once) - never written, no lock held past the query
    parity    state/v3/parity_soak.json: still updating, no new persistent mismatch, no new gap, not reset
    pytest    the full suite (tests/ in THIS tree) via junit xml: 0 failed, 0 errors, passed >= baseline

Each check runs in its own thread with its own timeout and FAILS CLOSED with a reason (missing file, missing
baseline, timeout, exception) - one broken check never hides the others. Baselines live in
state/v3/baselines.json (git-ignored with the rest of state/); --record-baseline writes them (merging, so a
--quick record keeps the stored pytest section) and is the ONLY write this script ever makes.

    regress_gate.py [--quick] [--json] [--only latency,memory] [--root LIVE_REPO] [--baselines PATH]
    regress_gate.py --record-baseline [--days 3] [--quick] [--root ...]

--root is where the STATE is read from (default: this repo; from a worktree point it at the live tree).
pytest always runs in the tree this script lives in. --quick skips pytest. Exit 0 = PASS, 1 = FAIL.
Stdlib only (latency_report / pytest run as subprocesses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
DAY = 86400.0

TIMEOUTS = {"latency": 90.0, "health": 30.0, "memory": 60.0, "parity": 15.0, "pytest": 1800.0}
ORDER = ("latency", "health", "memory", "parity", "pytest")

# (source or "*" = all pooled, stage, metrics). The voice stages are V3 plan 2.1's reference stages
# (speech_end->stt_done, decision->first_audio = first_speak->first_audio, end to end); received->admitted and
# admitted->assigned are the store's own machinery. reply / assigned->committed are mostly MY thinking time,
# so they are reported (info) but not gated.
GATED = [
    ["*", "received->admitted", ["p50", "p90"]],
    ["*", "admitted->assigned", ["p50"]],
    ["voice", "speech_end->stt_done", ["p50", "p90"]],
    ["*", "first_speak->first_audio", ["p50", "p90"]],
    ["voice", "speech_end->first_audio", ["p50", "p90"]],
]
INFO = [["discord", "reply"], ["voice", "reply"], ["orb", "reply"]]
LAT_DEFAULTS = {"days": 3, "rel_tol": 0.10, "abs_tol_s": 0.05, "min_n": 20}
HEALTH_DEFAULTS = {"window_s": DAY, "max_gap_s": 900.0, "fresh_s": 180.0}
PARITY_STALE_S = 3600.0


class CheckFail(Exception):
    """Raised inside a check: FAIL with this reason (the fail-closed path)."""


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CheckFail(f"{p} is missing")
    except (OSError, ValueError) as e:
        raise CheckFail(f"{p} unreadable: {e!r}"[:300])


def _need(base: dict, name: str) -> dict:
    sec = (base or {}).get(name)
    if not isinstance(sec, dict):
        raise CheckFail(f"no '{name}' baseline in baselines.json - run --record-baseline")
    return sec


# ── latency ──────────────────────────────────────────────────────────────────
def latency_measure(ctx: dict, days: int) -> dict:
    db = ctx["root"] / "state" / "v3" / "events.sqlite"
    cmd = [sys.executable, str(CODE_ROOT / "scripts" / "server" / "latency_report.py"),
           "--json", "--days", str(int(days)), "--db", str(db)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=ctx["timeouts"]["latency"] - 10)
    except subprocess.TimeoutExpired:
        raise CheckFail("latency_report.py timed out")
    if r.returncode != 0:
        raise CheckFail(f"latency_report.py exit {r.returncode}: {r.stderr.strip()[-200:]}")
    try:
        rep = json.loads(r.stdout)
    except ValueError:
        raise CheckFail(f"latency_report.py printed non-JSON: {r.stdout[:120]!r}")
    if rep.get("no_data"):
        raise CheckFail(f"latency_report.py: no data in {db} ({rep.get('error') or 'empty or missing'})")
    if "overall" not in rep:
        raise CheckFail("latency_report.py has no 'overall' section (older latency_report?)")
    return rep["overall"]


def latency_record(ctx: dict, prev: dict | None) -> dict:
    cfg = {**LAT_DEFAULTS, **{k: v for k, v in (prev or {}).items() if k in LAT_DEFAULTS}}
    if ctx.get("days"):
        cfg["days"] = int(ctx["days"])
    overall = latency_measure(ctx, cfg["days"])
    stages = {}
    for src, stage, _m in GATED + [i + [None] for i in INFO]:
        s = (overall.get(src) or {}).get(stage)
        if s:
            stages[f"{src}|{stage}"] = {k: s.get(k) for k in ("n", "p50", "p90")}
    if not stages:
        raise CheckFail("no gated stage has any samples - nothing to baseline")
    return {**cfg, "gated": GATED, "stages": stages}


def latency_check(ctx: dict, base: dict) -> dict:
    sec = _need(base, "latency")
    cfg = {**LAT_DEFAULTS, **{k: sec[k] for k in LAT_DEFAULTS if k in sec}}
    overall = latency_measure(ctx, cfg["days"])
    bstages = sec.get("stages") or {}
    rows, fails, judged = [], [], 0
    for src, stage, metrics in sec.get("gated") or GATED:
        key = f"{src}|{stage}"
        cur = (overall.get(src) or {}).get(stage) or {}
        b = bstages.get(key) or {}
        n = int(cur.get("n") or 0)
        for m in metrics:
            row = {"stage": key, "metric": m, "n": n, "base": b.get(m), "base_n": b.get("n"), "now": cur.get(m)}
            if b.get(m) is None or int(b.get("n") or 0) < cfg["min_n"]:
                row["verdict"] = "no-baseline"           # too few samples when the baseline was recorded
            elif n < cfg["min_n"]:
                row["verdict"] = "insufficient"          # not enough use in this window to judge
            else:
                judged += 1
                limit = float(b[m]) * (1 + cfg["rel_tol"]) + cfg["abs_tol_s"]
                row["limit"] = round(limit, 3)
                row["verdict"] = "ok" if float(cur[m]) <= limit else "SLOWER"
                if row["verdict"] == "SLOWER":
                    fails.append(f"{key} {m} {cur[m]}s > {row['limit']}s (base {b[m]}s)")
            rows.append(row)
    info = {f"{s}|{st}": (overall.get(s) or {}).get(st) for s, st in INFO}
    detail = {"days": cfg["days"], "rel_tol": cfg["rel_tol"], "abs_tol_s": cfg["abs_tol_s"], "min_n": cfg["min_n"],
              "rows": rows, "info_not_gated": info, "baseline_recorded": sec.get("recorded_iso")}
    if fails:
        return _res(False, "; ".join(fails), detail)
    if judged == 0:
        return _res(False, f"no gated stage has >= {cfg['min_n']} samples in both the baseline and the last "
                           f"{cfg['days']} day(s) - nothing could be compared", detail)
    skipped = [f"{r['stage']} {r['metric']} ({r['verdict']}, n={r['n']})" for r in rows if r["verdict"] not in ("ok",)]
    return _res(True, f"{judged} stage metric(s) within +{cfg['rel_tol']:.0%}+{cfg['abs_tol_s']}s"
                + (f"; not judged: {', '.join(skipped)}" if skipped else ""), detail)


# ── health ───────────────────────────────────────────────────────────────────
def health_check(ctx: dict, base: dict) -> dict:
    cfg = {**HEALTH_DEFAULTS, **((base or {}).get("health") or {})}
    now = ctx["now"]
    hdir = ctx["root"] / "state" / "health"
    doc = _load_json(hdir / "now.json")
    problems = []
    age = now - float(doc.get("ts") or 0)
    if age > cfg["fresh_s"]:
        problems.append(f"now.json is {age:.0f}s old (> {cfg['fresh_s']:.0f}s: is the watchdog timer running?)")
    red_now = sorted(k for k, v in (doc.get("subsystems") or {}).items() if (v or {}).get("status") == "red")
    if red_now:
        problems.append("red now: " + ", ".join(
            f"{k} ({str(doc['subsystems'][k].get('last_error') or '')[:80]})" for k in red_now))
    detail = {"now_age_s": round(age, 1), "red_now": red_now,
              "warn_now": sorted(k for k, v in (doc.get("subsystems") or {}).items()
                                 if (v or {}).get("status") == "warn")}

    hist = hdir / "history.jsonl"
    since = now - cfg["window_s"]
    entries, bad = [], 0
    try:
        with open(hist, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    e = json.loads(ln)
                    ts = float(e["ts"])
                except (ValueError, KeyError, TypeError):
                    bad += 1
                    continue
                if ts >= since - cfg["max_gap_s"]:
                    entries.append((ts, e))
    except FileNotFoundError:
        problems.append(f"no {hist.name} - health_snapshot.py writes it (one line per snapshot) since the 5.1 "
                        f"commit; the 24 h check needs {cfg['window_s'] / 3600:.0f} h of it")
        return _res(False, "; ".join(problems), detail)
    except OSError as e:
        problems.append(f"{hist} unreadable: {e!r}")
        return _res(False, "; ".join(problems), detail)
    entries.sort(key=lambda x: x[0])
    detail.update({"history_lines_in_window": len(entries), "history_bad_lines": bad})
    if not entries:
        problems.append(f"{hist.name} has no snapshot in the last {cfg['window_s'] / 3600:.0f} h")
        return _res(False, "; ".join(problems), detail)
    first = entries[0][0]
    covered_h = (now - first) / 3600
    detail["history_since"] = _iso(first)
    if first > since + cfg["max_gap_s"]:
        problems.append(f"history covers only {covered_h:.1f} h of the {cfg['window_s'] / 3600:.0f} h window "
                        f"(since {_iso(first)})")
    gaps, prev = [], max(first, since)
    for ts, _e in entries + [(now, None)]:
        if ts - prev > cfg["max_gap_s"] and ts > since:
            gaps.append((prev, ts))
        prev = max(prev, ts)
    if gaps:
        detail["gaps"] = [{"from": _iso(a), "to": _iso(b), "s": round(b - a)} for a, b in gaps[:20]]
        problems.append(f"{len(gaps)} gap(s) > {cfg['max_gap_s']:.0f}s in the history (no evidence there), "
                        f"longest {max(b - a for a, b in gaps) / 60:.0f} min")
    reds: dict = {}
    for ts, e in entries:
        if ts < since:
            continue
        for k in e.get("red") or []:
            r = reds.setdefault(k, {"n": 0, "first": ts, "last": ts, "err": None})
            r["n"] += 1
            r["last"] = ts
            r["err"] = r["err"] or (e.get("err") or {}).get(k)
    if reds:
        detail["red_24h"] = {k: {**v, "first": _iso(v["first"]), "last": _iso(v["last"])} for k, v in reds.items()}
        problems.append("red in the window: " + ", ".join(
            f"{k} x{v['n']} ({_iso(v['first'])} .. {_iso(v['last'])}: {str(v['err'] or '')[:60]})"
            for k, v in sorted(reds.items())))
    if problems:
        return _res(False, "; ".join(problems), detail)
    return _res(True, f"no red in {covered_h:.1f} h of history ({len(entries)} snapshots), now.json fresh", detail)


# ── memory ───────────────────────────────────────────────────────────────────
def _text_hash(e: dict) -> str:
    return hashlib.sha1(str(e.get("text", "")).encode("utf-8", "replace")).hexdigest()[:12]


def read_jsonl_ids(p: Path) -> tuple[dict, int]:
    """{id: text hash} for every decodable line (+ count of lines that would not decode / carry no id)."""
    out, bad = {}, 0
    try:
        f = open(p, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise CheckFail(f"{p} is missing")
    with f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
                out[str(e["id"])] = _text_hash(e)
            except (ValueError, KeyError, TypeError):
                bad += 1
    return out, bad


def _logged_ids(root: Path) -> set:
    """ids a deliberate path moved away (iris_memory.delete -> _deleted, memory_decay --commit -> _archive)."""
    ids = set()
    for name in ("iris_memory_deleted.jsonl", "iris_memory_archive.jsonl"):
        try:
            with open(root / "state" / name, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    try:
                        ids.add(str(json.loads(ln)["id"]))
                    except (ValueError, KeyError, TypeError):
                        pass
        except OSError:
            pass
    return ids


def chroma_ids(db: Path, timeout: float = 2.0) -> set:
    """Embedding ids of the 'iris_memories' collection. Read-only URI (never creates/writes the file),
    query_only, a busy timeout of `timeout` s, one query, closed at once - the runtime's writer waits at most
    that long behind this shared lock, and only during the query itself."""
    if not db.is_file():
        raise CheckFail(f"{db} is missing")
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=timeout)
        try:
            c.execute("PRAGMA query_only=1")
            rows = c.execute(
                "SELECT e.embedding_id FROM collections col JOIN segments s ON s.collection = col.id "
                "JOIN embeddings e ON e.segment_id = s.id "
                "WHERE col.name = 'iris_memories' AND s.scope = 'METADATA'").fetchall()
        finally:
            c.close()
    except sqlite3.Error as e:
        raise CheckFail(f"chroma read failed ({db.name}): {e!r}"[:300])
    return {str(r[0]) for r in rows}


def memory_paths(root: Path) -> tuple[Path, Path]:
    return root / "state" / "iris_memory.jsonl", root / "memory" / "chroma" / "chroma.sqlite3"


def memory_record(ctx: dict, _prev) -> dict:
    jl, db = memory_paths(ctx["root"])
    ids, bad = read_jsonl_ids(jl)
    if bad:
        raise CheckFail(f"{bad} undecodable line(s) in {jl.name} - fix (state_integrity.py) before baselining")
    cids = chroma_ids(db)
    return {"jsonl": {"count": len(ids), "ids": ids}, "chroma": {"count": len(cids), "ids": sorted(cids)}}


def memory_check(ctx: dict, base: dict) -> dict:
    sec = _need(base, "memory")
    jl, db = memory_paths(ctx["root"])
    logged = _logged_ids(ctx["root"])
    problems, detail = [], {"baseline_recorded": sec.get("recorded_iso")}

    b_ids = (sec.get("jsonl") or {}).get("ids") or {}
    ids, bad = read_jsonl_ids(jl)
    gone = sorted(set(b_ids) - set(ids))
    changed = sorted(k for k in b_ids if k in ids and ids[k] != b_ids[k])
    detail["jsonl"] = {"baseline": len(b_ids), "now": len(ids), "new": len(set(ids) - set(b_ids)),
                       "gone": gone[:50], "changed": changed[:50], "undecodable_lines": bad}
    if bad:
        problems.append(f"{bad} undecodable line(s) in {jl.name}")
    if gone:
        lg = [g for g in gone if g in logged]
        problems.append(f"{len(gone)} baseline memory id(s) gone from {jl.name}"
                        + (f" ({len(lg)} logged as deliberately deleted/archived - re-record the baseline if "
                           f"that was intended)" if lg else " (NOT in the delete/archive logs)"))
    if changed:
        problems.append(f"{len(changed)} baseline memor(y/ies) had their text changed in place")

    try:
        cids = chroma_ids(db)
    except CheckFail as e:
        problems.append(str(e))
        cids = None
    if cids is not None:
        b_c = set((sec.get("chroma") or {}).get("ids") or [])
        cgone = sorted(b_c - cids)
        detail["chroma"] = {"baseline": len(b_c), "now": len(cids), "new": len(cids - b_c), "gone": cgone[:50]}
        if cgone:
            lg = [g for g in cgone if g in logged]
            problems.append(f"{len(cgone)} baseline id(s) gone from Chroma iris_memories"
                            + (f" ({len(lg)} logged as deliberate)" if lg else ""))
    if problems:
        return _res(False, "; ".join(problems), detail)
    return _res(True, f"JSONL {len(b_ids)} -> {len(ids)} (+{detail['jsonl']['new']}), Chroma "
                      f"{detail['chroma']['baseline']} -> {detail['chroma']['now']} (+{detail['chroma']['new']}); "
                      f"no baseline entry lost or changed", detail)


# ── parity soak ──────────────────────────────────────────────────────────────
PARITY_KEYS = ("since_ts", "samples", "mismatch_samples", "persistent", "gaps")


def parity_read(root: Path) -> dict:
    d = _load_json(root / "state" / "v3" / "parity_soak.json")
    if not isinstance(d, dict) or "since_ts" not in d:
        raise CheckFail("parity_soak.json has no soak record (since_ts)")
    return d


def parity_record(ctx: dict, _prev) -> dict:
    d = parity_read(ctx["root"])
    return {k: d.get(k) for k in PARITY_KEYS}


def parity_check(ctx: dict, base: dict) -> dict:
    sec = _need(base, "parity")
    d = parity_read(ctx["root"])
    now = ctx["now"]
    age = now - float(d.get("last_ts") or 0)
    detail = {"now": {k: d.get(k) for k in PARITY_KEYS}, "baseline": {k: sec.get(k) for k in PARITY_KEYS},
              "last_sample_age_s": round(age, 1),
              "soak_days": round((now - float(d["since_ts"])) / DAY, 2), "last_persistent": d.get("last_persistent")}
    problems = []
    if age > PARITY_STALE_S:
        problems.append(f"last soak sample {age / 60:.0f} min ago (the health probe stopped sampling)")
    if float(d["since_ts"]) != float(sec.get("since_ts") or -1) or int(d.get("samples", 0)) < int(sec.get("samples") or 0):
        problems.append("the soak record was reset since the baseline")
    if int(d.get("persistent", 0)) > int(sec.get("persistent") or 0):
        problems.append(f"{int(d['persistent']) - int(sec.get('persistent') or 0)} new PERSISTENT discord "
                        f"mismatch(es) (last: {d.get('last_persistent')})")
    if int(d.get("gaps", 0)) > int(sec.get("gaps") or 0):
        problems.append(f"{int(d['gaps']) - int(sec.get('gaps') or 0)} new sampling gap(s) > 1 h")
    if problems:
        return _res(False, "; ".join(problems), detail)
    return _res(True, f"{d.get('samples')} samples over {detail['soak_days']} d, persistent {d.get('persistent')}, "
                      f"gaps {d.get('gaps')}, {d.get('mismatch_samples')} brief mismatch sample(s)", detail)


# ── pytest ───────────────────────────────────────────────────────────────────
def pytest_run(ctx: dict) -> dict:
    """One full run per invocation (--record-baseline then the gate reuse it)."""
    if ctx.get("pytest_result") is None:
        ctx["pytest_result"] = _pytest_run(ctx)
    return ctx["pytest_result"]


def _pytest_run(ctx: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="regress_gate_") as td:
        xml = Path(td) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, cwd=str(CODE_ROOT), capture_output=True, text=True,
                               timeout=ctx["timeouts"]["pytest"] - 15)
        except subprocess.TimeoutExpired:
            raise CheckFail(f"pytest timed out after {ctx['timeouts']['pytest'] - 15:.0f}s")
        took = time.monotonic() - t0
        if not xml.is_file():
            raise CheckFail(f"pytest wrote no junit xml (exit {r.returncode}): {(r.stdout + r.stderr).strip()[-300:]}")
        root = ET.parse(xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    tot["passed"] = tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"]
    failed = []
    for s in suites:
        for tc in s.iter("testcase"):
            if tc.find("failure") is not None or tc.find("error") is not None:
                failed.append(f"{tc.get('classname')}::{tc.get('name')}")
    tot.update({"exit": r.returncode, "took_s": round(took, 1), "failed": failed[:50],
                "tail": r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else []})
    return tot


def pytest_record(ctx: dict, _prev) -> dict:
    t = pytest_run(ctx)
    if t["failures"] or t["errors"] or t["exit"] not in (0,):
        raise CheckFail(f"pytest is not green ({t['failures']} failed, {t['errors']} errors, exit {t['exit']}: "
                        f"{', '.join(t['failed'][:5])}) - refusing to baseline a failing suite")
    return {k: t[k] for k in ("tests", "passed", "skipped")}


def pytest_check(ctx: dict, base: dict) -> dict:
    sec = _need(base, "pytest")
    t = pytest_run(ctx)
    problems = []
    if t["failures"] or t["errors"]:
        problems.append(f"{t['failures']} failed, {t['errors']} errors: {', '.join(t['failed'][:8])}")
    elif t["exit"] != 0:
        problems.append(f"pytest exit {t['exit']} ({' '.join(t['tail'])})")
    if t["passed"] < int(sec.get("passed") or 0):
        problems.append(f"{t['passed']} passed < baseline {sec.get('passed')} (tests removed or newly skipped)")
    detail = {**t, "baseline": {k: sec.get(k) for k in ("tests", "passed", "skipped")}}
    if problems:
        return _res(False, "; ".join(problems), detail)
    return _res(True, f"{t['passed']} passed, {t['skipped']} skipped, 0 failed in {t['took_s']}s "
                      f"(baseline {sec.get('passed')} passed)", detail)


# ── runner ───────────────────────────────────────────────────────────────────
CHECKS = {"latency": latency_check, "health": health_check, "memory": memory_check,
          "parity": parity_check, "pytest": pytest_check}
RECORDERS = {"latency": latency_record, "memory": memory_record, "parity": parity_record,
             "pytest": pytest_record}


def _res(ok: bool, reason: str, detail: dict | None = None) -> dict:
    return {"status": "PASS" if ok else "FAIL", "reason": reason, "detail": detail or {}}


def run_contained(fn, args: tuple, timeout: float) -> dict:
    """fn(*args) in a daemon thread; CheckFail/exception/timeout -> a FAIL result (fail closed)."""
    box: dict = {}

    def target():
        try:
            box["r"] = fn(*args)
        except CheckFail as e:
            box["r"] = _res(False, str(e))
        except Exception as e:  # noqa: BLE001 - a broken check is a FAIL, never a crash of the suite
            box["r"] = _res(False, f"check raised {e!r}"[:400])

    t0 = time.monotonic()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    took = round(time.monotonic() - t0, 2)
    if th.is_alive() or "r" not in box:
        return {**_res(False, f"timed out after {timeout:.0f}s"), "took_s": took}
    return {**box["r"], "took_s": took}


def load_baselines(p: Path) -> tuple[dict, str | None]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d, None) if isinstance(d, dict) else ({}, f"{p} is not a JSON object")
    except FileNotFoundError:
        return {}, f"no baselines file at {p} - run --record-baseline"
    except (OSError, ValueError) as e:
        return {}, f"{p} unreadable: {e!r}"


def write_baselines(p: Path, doc: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def _stack(root: Path) -> str | None:
    try:
        for ln in (root / "state" / "v3" / "launched.env").read_text().splitlines():
            if ln.startswith("stack="):
                return ln.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def selected(only: str, quick: bool) -> list[str]:
    names = [n.strip() for n in only.split(",") if n.strip()] if only else list(ORDER)
    bad = [n for n in names if n not in CHECKS]
    if bad:
        raise SystemExit(f"unknown check(s): {', '.join(bad)} (have: {', '.join(ORDER)})")
    return names


def record(ctx: dict, names: list[str], path: Path) -> dict:
    base, _err = load_baselines(path)
    out = {}
    for n in names:
        if n not in RECORDERS or n in ctx["skip"]:
            continue
        box: dict = {}

        def rec(n=n):
            box["v"] = RECORDERS[n](ctx, base.get(n))
            return _res(True, "recorded")
        r = run_contained(rec, (), ctx["timeouts"][n])
        if r["status"] == "PASS":
            base[n] = {**box["v"], "recorded_ts": round(ctx["now"], 1), "recorded_iso": _iso(ctx["now"]),
                       "stack": _stack(ctx["root"])}
        out[n] = r
    base.update({"schema": SCHEMA, "root": str(ctx["root"]), "updated_iso": _iso(ctx["now"]),
                 "writer": "scripts/server/regress_gate.py --record-baseline"})
    base.setdefault("health", dict(HEALTH_DEFAULTS))
    write_baselines(path, base)
    return out


def run_gate(ctx: dict, names: list[str], base: dict, base_err: str | None) -> dict:
    results, threads = {}, {}
    for n in names:                                   # every check in parallel; each bounded by its own timeout
        if n in ctx["skip"]:
            results[n] = {"status": "SKIP", "reason": "--quick", "detail": {}, "took_s": 0}
            continue
        th = threading.Thread(target=lambda n=n: results.__setitem__(
            n, run_contained(CHECKS[n], (ctx, base), ctx["timeouts"][n])), daemon=True)
        th.start()
        threads[n] = th
    for n, th in threads.items():
        th.join(ctx["timeouts"][n] + 5)
        results.setdefault(n, _res(False, "runner lost the check"))
    if base_err:
        for r in results.values():
            if r["status"] == "FAIL" and "baseline" in r["reason"]:
                r["reason"] = f"{r['reason']} ({base_err})"
    ran = [r for r in results.values() if r["status"] != "SKIP"]
    ok = bool(ran) and all(r["status"] == "PASS" for r in ran)
    return {"overall": "PASS" if ok else "FAIL", "quick": "pytest" in ctx["skip"], "ts": round(ctx["now"], 1),
            "iso": _iso(ctx["now"]), "root": str(ctx["root"]), "code_root": str(CODE_ROOT),
            "stack": _stack(ctx["root"]), "checks": {n: results[n] for n in names}}


def render(rep: dict) -> str:
    out = [f"regress gate {rep['iso']}  state={rep['root']}  stack={rep['stack'] or '?'}"
           + ("  (quick: no pytest)" if rep["quick"] else "")]
    for n, r in rep["checks"].items():
        out.append(f"  {r['status']:4}  {n:8} {r['reason']}  [{r.get('took_s', 0)}s]")
        if n == "latency" and r.get("detail", {}).get("rows"):
            for row in r["detail"]["rows"]:
                out.append(f"          {row['stage']:34} {row['metric']} now={row['now']} base={row['base']} "
                           f"n={row['n']} {row['verdict']}")
    out.append(f"OVERALL: {rep['overall']}")
    return "\n".join(out)


def _out(text: str) -> None:
    """stdout, not print(): iris_runtime swaps builtins.print for a stderr writer when it is imported."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V3 regression gate suite (plan step 5.1)")
    ap.add_argument("--root", default=os.environ.get("IRIS_ROOT", ""), help="repo whose state/ to read")
    ap.add_argument("--baselines", default="", help="default <root>/state/v3/baselines.json")
    ap.add_argument("--quick", action="store_true", help="skip the full pytest")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--only", default="", help="comma list of: " + ",".join(ORDER))
    ap.add_argument("--record-baseline", action="store_true", help="write baselines.json, then run the gate")
    ap.add_argument("--days", type=int, default=0, help="with --record-baseline: latency window (default 3)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else CODE_ROOT
    bpath = Path(a.baselines) if a.baselines else root / "state" / "v3" / "baselines.json"
    names = selected(a.only, a.quick)
    ctx = {"root": root, "now": _now(), "timeouts": dict(TIMEOUTS), "days": a.days,
           "skip": {"pytest"} if a.quick else set()}
    recorded = None
    if a.record_baseline:
        recorded = record(ctx, names, bpath)
        ctx["now"] = _now()
    base, err = load_baselines(bpath)
    rep = run_gate(ctx, names, base, err)
    rep["baselines"] = str(bpath)
    if recorded is not None:
        rep["recorded"] = recorded
    if a.json:
        _out(json.dumps(rep, indent=1, default=str))
    else:
        if recorded is not None:
            for n, r in recorded.items():
                _out(f"record {n:8} {r['status']:4} {r['reason']}")
        _out(render(rep))
    if recorded is not None and any(r["status"] != "PASS" for r in recorded.values()):
        return 1
    return 0 if rep["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
