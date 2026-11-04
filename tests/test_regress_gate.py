"""regress_gate.py (V3 plan step 5.1): the regression gate suite, on fake state only.

Covers each check's PASS and its fail-closed paths (missing file, missing baseline, stale, red in the 24 h
history, gaps, lost/changed memories, Chroma opened read-only, parity regressions, slower stages, too few
samples, failing/shrinking pytest), the runner (a hanging or raising check FAILs alone, --quick skips pytest),
--record-baseline (merges, writes only baselines.json), latency_report's pooled `overall`, and the health
history line health_snapshot.py now appends. Nothing touches the live state/.
    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_regress_gate.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

SRV = Path(__file__).resolve().parents[1] / "scripts" / "server"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rg = _load("regress_gate_t", "regress_gate.py")
NOW = 1_800_000_000.0


def mkctx(root: Path, **kw):
    c = {"root": root, "now": NOW, "timeouts": dict(rg.TIMEOUTS), "days": 0, "skip": set()}
    c.update(kw)
    return c


# ── fixtures: a fake live tree ───────────────────────────────────────────────
def write_health(root: Path, *, now_ts=NOW - 30, red_now=(), hist_from=NOW - 86400 - 60, step=60.0,
                 reds_at=None, skip=None):
    h = root / "state" / "health"
    h.mkdir(parents=True, exist_ok=True)
    subs = {"ears": {"status": "ok"}, "mouth": {"status": "ok"}}
    for k in red_now:
        subs[k] = {"status": "red", "last_error": f"{k} broke"}
    (h / "now.json").write_text(json.dumps({"ts": now_ts, "subsystems": subs}))
    lines, t = [], hist_from
    while t <= NOW - 10:
        if not (skip and skip[0] <= t < skip[1]):
            red = (reds_at or {}).get(int(t), [])
            e = {"ts": t, "red": red, "warn": []}
            if red:
                e["err"] = {k: f"{k} was red" for k in red}
            lines.append(json.dumps(e))
        t += step
    (h / "history.jsonl").write_text("\n".join(lines) + "\n")


def make_chroma(db: Path, ids):
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE collections(id TEXT, name TEXT);"
                    "CREATE TABLE segments(id TEXT, collection TEXT, scope TEXT);"
                    "CREATE TABLE embeddings(id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);")
    c.execute("INSERT INTO collections VALUES('c1','iris_memories')")
    c.execute("INSERT INTO collections VALUES('c2','other')")
    c.execute("INSERT INTO segments VALUES('s1','c1','METADATA')")
    c.execute("INSERT INTO segments VALUES('s2','c1','VECTOR')")
    c.execute("INSERT INTO segments VALUES('s3','c2','METADATA')")
    for i in ids:
        c.execute("INSERT INTO embeddings(segment_id, embedding_id) VALUES('s1', ?)", (i,))
    c.execute("INSERT INTO embeddings(segment_id, embedding_id) VALUES('s3', 'not_mine')")
    c.commit()
    c.close()


def write_memory(root: Path, entries, chroma_ids=None):
    jl = root / "state" / "iris_memory.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("".join(json.dumps(e) + "\n" for e in entries))
    db = root / "memory" / "chroma" / "chroma.sqlite3"
    if db.exists():
        db.unlink()
    make_chroma(db, chroma_ids if chroma_ids is not None else [e["id"] for e in entries])


MEMS = [{"id": f"m{i}", "text": f"memory {i}", "ts": i} for i in range(5)]


def write_parity(root: Path, **kw):
    d = {"since_ts": NOW - 3 * 86400, "last_ts": NOW - 60, "samples": 4000, "mismatch_samples": 1,
         "persistent": 0, "gaps": 0, "open": {}, "last_persistent": None}
    d.update(kw)
    p = root / "state" / "v3" / "parity_soak.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


# ── health ───────────────────────────────────────────────────────────────────
def test_health_pass_with_full_clean_history(tmp_path):
    write_health(tmp_path)
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "PASS", r["reason"]


def test_health_fails_closed_without_history(tmp_path):
    write_health(tmp_path)
    (tmp_path / "state" / "health" / "history.jsonl").unlink()
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "FAIL" and "history.jsonl" in r["reason"]


def test_health_short_history_fails(tmp_path):
    write_health(tmp_path, hist_from=NOW - 5 * 3600)
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "FAIL" and "covers only" in r["reason"]


def test_health_red_in_window_named(tmp_path):
    t = int(NOW - 86400 - 60 + 600 * 60)       # a grid point inside the window
    write_health(tmp_path, reds_at={t: ["mouth"]})
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "FAIL" and "mouth x1" in r["reason"] and "mouth was red" in r["reason"]


def test_health_red_older_than_window_ignored(tmp_path):
    write_health(tmp_path, hist_from=NOW - 2 * 86400, reds_at={int(NOW - 2 * 86400): ["mouth"]})
    assert rg.health_check(mkctx(tmp_path), {})["status"] == "PASS"


def test_health_gap_fails(tmp_path):
    write_health(tmp_path, skip=(NOW - 40000, NOW - 38000))
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "FAIL" and "gap" in r["reason"]
    assert r["detail"]["gaps"][0]["s"] >= 2000


def test_health_stale_or_red_now(tmp_path):
    write_health(tmp_path, now_ts=NOW - 3600, red_now=["ears"])
    r = rg.health_check(mkctx(tmp_path), {})
    assert r["status"] == "FAIL" and "old" in r["reason"] and "red now: ears" in r["reason"]


def test_health_missing_now_json(tmp_path):
    r = rg.run_contained(rg.health_check, (mkctx(tmp_path), {}), 5)
    assert r["status"] == "FAIL" and "missing" in r["reason"]


# ── memory ───────────────────────────────────────────────────────────────────
def _mem_base(root):
    return {"memory": rg.memory_record(mkctx(root), None)}


def test_memory_appends_pass(tmp_path):
    write_memory(tmp_path, MEMS)
    base = _mem_base(tmp_path)
    more = MEMS + [{"id": "m9", "text": "new"}]
    write_memory(tmp_path, more)
    r = rg.memory_check(mkctx(tmp_path), base)
    assert r["status"] == "PASS", r["reason"]
    assert r["detail"]["jsonl"]["new"] == 1 and r["detail"]["chroma"]["new"] == 1


def test_memory_lost_and_changed_fail(tmp_path):
    write_memory(tmp_path, MEMS)
    base = _mem_base(tmp_path)
    cur = [dict(e) for e in MEMS[1:]]
    cur[0]["text"] = "rewritten"
    write_memory(tmp_path, cur, chroma_ids=[e["id"] for e in MEMS])
    r = rg.memory_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL"
    assert "1 baseline memory id(s) gone" in r["reason"] and "NOT in the delete" in r["reason"]
    assert "text changed" in r["reason"]
    assert r["detail"]["jsonl"]["gone"] == ["m0"] and r["detail"]["jsonl"]["changed"] == ["m1"]


def test_memory_logged_delete_still_fails_but_says_so(tmp_path):
    write_memory(tmp_path, MEMS)
    base = _mem_base(tmp_path)
    write_memory(tmp_path, MEMS[1:])
    (tmp_path / "state" / "iris_memory_deleted.jsonl").write_text(json.dumps(MEMS[0]) + "\n")
    r = rg.memory_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "logged as deliberately" in r["reason"]


def test_memory_chroma_loss_and_unreadable(tmp_path):
    write_memory(tmp_path, MEMS)
    base = _mem_base(tmp_path)
    write_memory(tmp_path, MEMS, chroma_ids=["m0", "m1"])
    r = rg.memory_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "3 baseline id(s) gone from Chroma" in r["reason"]
    (tmp_path / "memory" / "chroma" / "chroma.sqlite3").unlink()
    r = rg.memory_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "missing" in r["reason"]
    assert r["detail"]["jsonl"]["now"] == 5, "the JSONL half is still reported"


def test_chroma_opened_read_only(tmp_path):
    write_memory(tmp_path, MEMS)
    db = tmp_path / "memory" / "chroma" / "chroma.sqlite3"
    before = (db.read_bytes(), db.stat().st_mtime_ns)
    rg.chroma_ids(db)
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before
    assert sorted(p.name for p in db.parent.iterdir()) == ["chroma.sqlite3"], "no journal/wal left behind"
    with pytest.raises(rg.CheckFail):
        rg.chroma_ids(tmp_path / "nope.sqlite3")
    assert not (tmp_path / "nope.sqlite3").exists(), "mode=ro never creates the file"


def test_chroma_write_lock_fails_closed_not_hangs(tmp_path):
    write_memory(tmp_path, MEMS)
    db = tmp_path / "memory" / "chroma" / "chroma.sqlite3"
    w = sqlite3.connect(db, isolation_level=None)
    w.execute("BEGIN EXCLUSIVE")
    try:
        t0 = time.monotonic()
        with pytest.raises(rg.CheckFail):
            rg.chroma_ids(db, timeout=0.2)
        assert time.monotonic() - t0 < 3
    finally:
        w.execute("ROLLBACK")
        w.close()


def test_memory_record_refuses_torn_jsonl(tmp_path):
    write_memory(tmp_path, MEMS)
    with open(tmp_path / "state" / "iris_memory.jsonl", "a") as f:
        f.write('{"id": "half\n')
    with pytest.raises(rg.CheckFail):
        rg.memory_record(mkctx(tmp_path), None)


# ── parity ───────────────────────────────────────────────────────────────────
def test_parity_pass_and_regressions(tmp_path):
    write_parity(tmp_path)
    base = {"parity": rg.parity_record(mkctx(tmp_path), None)}
    write_parity(tmp_path, samples=4100)
    assert rg.parity_check(mkctx(tmp_path), base)["status"] == "PASS"
    write_parity(tmp_path, samples=4100, persistent=1, last_persistent={"id": "x"})
    r = rg.parity_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "PERSISTENT" in r["reason"]
    write_parity(tmp_path, samples=4100, gaps=1)
    assert "gap" in rg.parity_check(mkctx(tmp_path), base)["reason"]
    write_parity(tmp_path, last_ts=NOW - 7200)
    assert "stopped sampling" in rg.parity_check(mkctx(tmp_path), base)["reason"]
    write_parity(tmp_path, since_ts=NOW - 100, samples=2)
    assert "reset" in rg.parity_check(mkctx(tmp_path), base)["reason"]


def test_parity_without_baseline_fails_closed(tmp_path):
    write_parity(tmp_path)
    r = rg.run_contained(rg.parity_check, (mkctx(tmp_path), {}), 5)
    assert r["status"] == "FAIL" and "--record-baseline" in r["reason"]


# ── latency ──────────────────────────────────────────────────────────────────
def _overall(scale=1.0, n=60):
    s = lambda p50, p90: {"n": n, "p50": round(p50 * scale, 3), "p90": round(p90 * scale, 3)}  # noqa: E731
    return {"*": {"received->admitted": s(0.001, 0.002), "admitted->assigned": s(0.01, 0.8),
                  "first_speak->first_audio": s(1.2, 1.6)},
            "voice": {"speech_end->stt_done": s(1.3, 1.5), "speech_end->first_audio": s(6.8, 9.8),
                      "reply": s(2.6, 26.8)}}


@pytest.fixture()
def fake_lat(monkeypatch):
    box = {"o": _overall()}
    monkeypatch.setattr(rg, "latency_measure", lambda ctx, days: box["o"])
    return box


def test_latency_within_tolerance_passes(tmp_path, fake_lat):
    base = {"latency": rg.latency_record(mkctx(tmp_path), None)}
    fake_lat["o"] = _overall(1.08)
    r = rg.latency_check(mkctx(tmp_path), base)
    assert r["status"] == "PASS", r["reason"]
    assert r["detail"]["info_not_gated"]["voice|reply"]["p50"] == pytest.approx(2.808)


def test_latency_slower_stage_fails(tmp_path, fake_lat):
    base = {"latency": rg.latency_record(mkctx(tmp_path), None)}
    o = _overall()
    o["voice"]["speech_end->stt_done"]["p90"] = 2.5
    fake_lat["o"] = o
    r = rg.latency_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "voice|speech_end->stt_done p90 2.5s" in r["reason"]


def test_latency_insufficient_samples(tmp_path, fake_lat):
    base = {"latency": rg.latency_record(mkctx(tmp_path), None)}
    o = _overall()
    o["voice"]["speech_end->stt_done"]["n"] = 3
    o["voice"]["speech_end->stt_done"]["p50"] = 99.0
    fake_lat["o"] = o
    r = rg.latency_check(mkctx(tmp_path), base)
    assert r["status"] == "PASS" and "insufficient" in r["reason"]
    fake_lat["o"] = _overall(5.0, n=2)
    r = rg.latency_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "nothing could be compared" in r["reason"]


def test_latency_measure_fails_closed_on_no_data(tmp_path):
    with pytest.raises(rg.CheckFail, match="no data"):
        rg.latency_measure(mkctx(tmp_path), 3)


def test_latency_end_to_end_with_real_report(tmp_path):
    """latency_report.py as a subprocess over a real (tmp) event store -> pooled overall -> PASS vs itself."""
    from brain import iris_events as ev
    now = time.time()
    st = ev.EventStore(tmp_path / "state" / "v3" / "events.sqlite", clock=lambda: now)
    for i in range(25):
        se = now - 3600 - i * 4000          # spread over two local days
        st.insert("voice", f"v{i}", status="admitted", received_ts=se + 0.5,
                  meta={"speech_end_ts": se, "stt_done_ts": se + 1.0 + i / 100,
                        "first_speak_ts": se + 3.0, "first_audio_ts": se + 4.2})
    st.close()
    ov = rg.latency_measure(mkctx(tmp_path), 3)
    assert ov["voice"]["speech_end->stt_done"]["n"] == 25
    assert ov["*"]["first_speak->first_audio"]["n"] == 25
    base = {"latency": rg.latency_record(mkctx(tmp_path), None)}
    r = rg.latency_check(mkctx(tmp_path), base)
    assert r["status"] == "PASS", r["reason"]


# ── pytest check (the real suite is never run from here) ─────────────────────
def test_pytest_check_verdicts(tmp_path, monkeypatch):
    res = {"tests": 10, "failures": 0, "errors": 0, "skipped": 1, "passed": 9, "exit": 0, "took_s": 1.0,
           "failed": [], "tail": []}
    monkeypatch.setattr(rg, "_pytest_run", lambda ctx: dict(res))
    base = {"pytest": {"tests": 10, "passed": 9, "skipped": 1}}
    assert rg.pytest_check(mkctx(tmp_path), base)["status"] == "PASS"
    res.update(failures=1, passed=8, exit=1, failed=["tests/x.py::test_y"])
    r = rg.pytest_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "tests/x.py::test_y" in r["reason"]
    with pytest.raises(rg.CheckFail, match="refusing"):
        rg.pytest_record(mkctx(tmp_path), None)
    res.update(failures=0, passed=7, skipped=3, exit=0, failed=[])
    r = rg.pytest_check(mkctx(tmp_path), base)
    assert r["status"] == "FAIL" and "newly skipped" in r["reason"]


def test_pytest_runs_once_per_invocation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rg, "_pytest_run", lambda ctx: calls.append(1) or {"x": 1})
    ctx = mkctx(tmp_path)
    rg.pytest_run(ctx)
    rg.pytest_run(ctx)
    assert len(calls) == 1


# ── runner ───────────────────────────────────────────────────────────────────
def test_one_hanging_and_one_raising_check_dont_hide_others(tmp_path, monkeypatch):
    def hang(ctx, base):
        time.sleep(5)

    def boom(ctx, base):
        raise RuntimeError("exploded")
    monkeypatch.setattr(rg, "CHECKS", {"latency": hang, "health": boom,
                                       "memory": lambda c, b: rg._res(True, "fine"),
                                       "parity": lambda c, b: rg._res(True, "fine"),
                                       "pytest": lambda c, b: rg._res(True, "fine")})
    ctx = mkctx(tmp_path, timeouts={k: 0.3 for k in rg.TIMEOUTS})
    t0 = time.monotonic()
    rep = rg.run_gate(ctx, list(rg.ORDER), {}, None)
    assert time.monotonic() - t0 < 3
    assert rep["checks"]["latency"]["status"] == "FAIL" and "timed out" in rep["checks"]["latency"]["reason"]
    assert rep["checks"]["health"]["status"] == "FAIL" and "exploded" in rep["checks"]["health"]["reason"]
    assert all(rep["checks"][k]["status"] == "PASS" for k in ("memory", "parity", "pytest"))
    assert rep["overall"] == "FAIL"


def test_quick_skips_pytest_and_can_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "CHECKS", {k: (lambda c, b: rg._res(True, "fine")) for k in rg.ORDER}
                        | {"pytest": lambda c, b: pytest.fail("pytest must not run in --quick")})
    rep = rg.run_gate(mkctx(tmp_path, skip={"pytest"}), list(rg.ORDER), {}, None)
    assert rep["checks"]["pytest"]["status"] == "SKIP" and rep["overall"] == "PASS" and rep["quick"]


def _main(argv):
    """main() with stdout captured directly (pytest's capture can miss it after another test swaps sys.stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rg.main(argv)
    return rc, buf.getvalue()


def test_record_then_gate_end_to_end(tmp_path, monkeypatch):
    """main(): --record-baseline --quick on a fake tree writes ONLY baselines.json, keeps a stored pytest section,
    then the gate passes; --json is machine-readable; a later lost memory flips memory (and overall) to FAIL."""
    write_health(tmp_path, hist_from=time.time() - 86400 - 60)
    # health fixture uses NOW; rewrite with real time for main() (which uses time.time())
    h = tmp_path / "state" / "health"
    t = time.time()
    (h / "now.json").write_text(json.dumps({"ts": t, "subsystems": {"ears": {"status": "ok"}}}))
    (h / "history.jsonl").write_text("".join(json.dumps({"ts": t - 86400 - 60 + i * 60, "red": [], "warn": []}) + "\n"
                                             for i in range(1442)))
    write_memory(tmp_path, MEMS)
    write_parity(tmp_path, since_ts=t - 86400, last_ts=t - 30)
    monkeypatch.setattr(rg, "latency_measure", lambda ctx, days: _overall())
    bp = tmp_path / "state" / "v3" / "baselines.json"
    bp.write_text(json.dumps({"pytest": {"tests": 1, "passed": 1, "skipped": 0, "recorded_iso": "earlier"}}))
    snap = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file() and p != bp}
    rc, out = _main(["--root", str(tmp_path), "--record-baseline", "--quick", "--json"])
    rep = json.loads(out)
    assert rc == 0, rep
    assert rep["overall"] == "PASS" and rep["checks"]["pytest"]["status"] == "SKIP"
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file() and p != bp} == snap
    b = json.loads(bp.read_text())
    assert b["pytest"]["recorded_iso"] == "earlier", "a --quick record keeps the stored pytest baseline"
    assert set(b) >= {"latency", "memory", "parity", "pytest", "health", "schema"}
    write_memory(tmp_path, MEMS[1:])
    rc, out = _main(["--root", str(tmp_path), "--quick"])
    assert rc == 1 and "FAIL  memory" in out and "OVERALL: FAIL" in out


def test_missing_baselines_file_fails_every_baselined_check(tmp_path):
    write_parity(tmp_path)
    rc, out = _main(["--root", str(tmp_path), "--only", "parity,memory", "--json"])
    rep = json.loads(out)
    assert rc == 1
    for k in ("parity", "memory"):
        assert rep["checks"][k]["status"] == "FAIL"
    assert "--record-baseline" in rep["checks"]["parity"]["reason"]


def test_unknown_check_name_rejected():
    with pytest.raises(SystemExit):
        rg.selected("latency,bogus", False)


# ── the inputs this gate relies on ───────────────────────────────────────────
def test_latency_report_overall_pools_days_and_sources(tmp_path):
    lr = _load("latency_report_rg", "latency_report.py")
    from brain import iris_events as ev
    now = time.time()
    st = ev.EventStore(tmp_path / "ev.sqlite", clock=lambda: now)
    for i in range(6):
        src = "voice" if i % 2 else "discord"
        st.insert(src, f"e{i}", status="admitted", received_ts=now - 3600 - i * 30000,
                  meta={"first_speak_ts": 10.0, "first_audio_ts": 11.0 + i})
    rep = lr.build(st._conn, days=3, now=now)
    st.close()
    assert len(rep["events"]) >= 2, "spans more than one day"
    assert rep["overall"]["*"]["first_speak->first_audio"]["n"] == 6
    assert rep["overall"]["voice"]["first_speak->first_audio"]["n"] == 3
    assert rep["overall"]["discord"]["first_speak->first_audio"]["n"] == 3


def test_health_snapshot_appends_history_line(tmp_path, monkeypatch):
    hs = _load("health_snapshot_rg", "health_snapshot.py")
    monkeypatch.setattr(hs, "HOME", tmp_path / "home")
    (tmp_path / "home").mkdir()
    (tmp_path / "root" / "state").mkdir(parents=True)
    monkeypatch.setattr(hs, "unit_active_seconds", lambda unit: None)

    def bad(_c):
        return {"status": "red", "probe": "p", "error": "it broke"}
    out = tmp_path / "h" / "now.json"
    hs.write_snapshot(out, tmp_path / "root", probes={"a": lambda c: {"status": "ok", "probe": "p"}, "b": bad},
                      in_grace=False)
    hs.write_snapshot(out, tmp_path / "root", probes={"a": lambda c: {"status": "warn", "probe": "p"}},
                      in_grace=False)
    lines = [json.loads(x) for x in (out.parent / "history.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["red"] == ["b"] and lines[0]["err"] == {"b": "it broke"} and lines[0]["warn"] == []
    assert lines[1]["red"] == [] and lines[1]["warn"] == ["a"] and "err" not in lines[1]


def test_health_history_trims_to_whole_lines(tmp_path, monkeypatch):
    hs = _load("health_snapshot_rg2", "health_snapshot.py")
    monkeypatch.setattr(hs, "HISTORY_CAP", 2000)
    monkeypatch.setattr(hs, "HISTORY_KEEP", 1000)
    p = tmp_path / "history.jsonl"
    for i in range(100):
        hs.append_history(p, {"ts": i, "subsystems": {"x": {"status": "ok"}}})
    assert p.stat().st_size <= 2100
    lines = p.read_text().splitlines()
    assert all(json.loads(x)["ts"] >= 0 for x in lines) and json.loads(lines[-1])["ts"] == 99
    assert not [q for q in tmp_path.iterdir() if q.name != "history.jsonl"], "no trim tmp left"
    os.chmod(tmp_path, 0o500)
    try:
        hs.append_history(tmp_path / "new.jsonl", {"ts": 1})      # unwritable dir: never raises
    finally:
        os.chmod(tmp_path, 0o700)
