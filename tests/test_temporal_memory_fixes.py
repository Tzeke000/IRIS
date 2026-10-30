"""Code review 2026-10-29, "Temporal memory" section — regression tests.

Covers: iris_time restart gap + lifetime double count + weekly counter +
unique temp files; sleep_mode just-booted guard + persisted sleep state;
anchor pattern tightening + pre-configure buffering; scene_memory retention;
memory_consolidation lock + success-only gate; temporal_sense stale prune +
atomic write; gap_delta backwards reader.

Everything runs against temp dirs; nothing touches the live state/ tree.

    ~/venvs/iris-v100/bin/python tests/test_temporal_memory_fixes.py
    (or pytest tests/test_temporal_memory_fixes.py)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ── iris_time ─────────────────────────────────────────────────────────────

def _fresh_iris_time():
    from brain import iris_time
    importlib.reload(iris_time)          # new "process": new _PROCESS_STARTED_TS, empty live state
    iris_time.start_tick_thread = lambda: None   # no background thread in tests
    return iris_time


def _write_prev_run(tmp: Path, now: float, down_s: float = 8 * 3600) -> None:
    (tmp / "state").mkdir(parents=True, exist_ok=True)
    (tmp / "state" / "iris_time.json").write_text(json.dumps({
        "process_started_ts": now - down_s - 3600, "tick_count": 3600,
        "last_tick_ts": now - down_s, "last_session_attached_ts": now - down_s - 60,
        "last_session_iso": "prev", "session_attach_count": 7,
        "longest_unattended_gap_s": 100.0, "current_process_uptime_s": 3600.0,
        "total_lifetime_uptime_s": 10000.0, "previous_process_uptime_s": 50.0,
    }))


def test_restart_gap_is_reported_after_a_shutdown() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        now = time.time()
        _write_prev_run(tmp, now)
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": str(tmp)})
        r = it.time_awareness_report()
        assert r["is_new_process"] is True, r
        assert r["body_was_alive_during_gap"] is False
        assert r["gap_since_attach_s"] > 8 * 3600, r["gap_since_attach_human"]
        assert abs(r["downtime_s"] - 8 * 3600) < 120, r["downtime_s"]
        assert "OFF" in r["gap_explanation"] and "8.0h" in r["gap_explanation"], r["gap_explanation"]
        assert r["session_attach_count"] == 7
        # iso strings carry an offset now
        assert r["now_iso"][-6] in "+-" and r["previous_run_last_tick_iso"][-6] in "+-", r["now_iso"]


def test_lifetime_uptime_is_not_double_counted() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_prev_run(tmp, time.time())
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": str(tmp)})
        st = it.get_state()
        assert abs(st["total_lifetime_uptime_s"] - 13600.0) < 5, st["total_lifetime_uptime_s"]
        disk = json.loads((tmp / "state" / "iris_time.json").read_text())
        assert abs(disk["total_lifetime_uptime_s"] - 13600.0) < 5, disk
        assert abs(disk["previous_process_uptime_s"] - 3600.0) < 1


def test_same_process_gap_after_attach_is_alive() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_prev_run(tmp, time.time())
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": str(tmp)})
        it.mark_session_attached()
        r = it.time_awareness_report()
        assert r["is_new_process"] is False and r["body_was_alive_during_gap"] is True, r
        assert r["session_attach_count"] == 8


def test_fresh_install_without_previous_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": td})
        r = it.time_awareness_report()
        assert r["downtime_s"] is None
        it.mark_session_attached()
        assert it.time_awareness_report()["is_new_process"] is False


def test_weekly_checkins_counter_counts_and_resets() -> None:
    with tempfile.TemporaryDirectory() as td:
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": td})
        now = time.time()
        it._roll_week_window(now)                       # sets the anchor
        assert it._LIVE_STATE["week_window_anchor_ts"] == now
        it._roll_week_window(now + 15 * 86400)          # two closed windows, no contact
        assert it._LIVE_STATE["weekly_checkins_missed_consecutive"] == 2
        it.update_zeke_contact()
        assert it._LIVE_STATE["weekly_checkins_missed_consecutive"] == 0
        assert it.substrate_counters_report()["weekly_checkins_missed_consecutive"] == 0


def test_state_writes_leave_no_shared_tmp_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        it = _fresh_iris_time()
        it.bootstrap_iris_time({"BASE_DIR": td})
        it.update_zeke_contact()
        files = sorted(os.listdir(Path(td) / "state"))
        assert files == ["iris_time.json"], files
        json.loads((Path(td) / "state" / "iris_time.json").read_text())


# ── sleep_mode ────────────────────────────────────────────────────────────

def _sleep_mod():
    from brain import sleep_mode
    sleep_mode._tts = lambda *a, **k: None        # never speak from a test
    return sleep_mode


def test_just_booted_guard_fires_without_a_verify_script() -> None:
    sm = _sleep_mod()
    from datetime import datetime
    orig = sm._now_local
    sm._now_local = lambda: datetime.now().astimezone().replace(hour=1, minute=0)
    try:
        with tempfile.TemporaryDirectory() as td:
            g = {"BASE_DIR": td}
            sm._restore_sleep_state_once(g)      # what the first tick does
            assert "_process_start_ts" in g
            ok, why = sm._check_schedule_trigger(g)
            assert ok is False and why.get("reason") == "just_booted", why
            g["_process_start_ts"] = time.time() - 3600
            ok, why = sm._check_schedule_trigger(g)
            assert ok is True, why
    finally:
        sm._now_local = orig


def test_sleep_state_persists_and_restores() -> None:
    sm = _sleep_mod()
    with tempfile.TemporaryDirectory() as td:
        g = {"BASE_DIR": td}
        sm._enter_sleep(g, duration_s=7200, trigger="schedule", trigger_summary={})
        data = json.loads((Path(td) / "state" / "sleep_mode_state.json").read_text())
        assert data["_sleep_state"] == sm.STATE_ENTERING_SLEEP and data["_sleep_schedule_last_fire_ts"]
        g2 = {"BASE_DIR": td}                    # "restarted" process
        sm._restore_sleep_state_once(g2)
        assert g2["_sleep_state"] == sm.STATE_SLEEPING and g2["_sleep_phase"] == "learning", g2
        assert g2["_sleep_schedule_last_fire_ts"] == data["_sleep_schedule_last_fire_ts"]
        # a sleep whose target already passed is dropped
        data["_sleep_target_ts"] = time.time() - 10
        (Path(td) / "state" / "sleep_mode_state.json").write_text(json.dumps(data))
        g3 = {"BASE_DIR": td}
        sm._restore_sleep_state_once(g3)
        assert sm.get_state(g3) == sm.STATE_AWAKE
        assert g3["_sleep_schedule_last_fire_ts"]   # anti-flap stamp still restored


# ── anchor_moments ────────────────────────────────────────────────────────

def test_anchor_patterns_skip_everyday_replies() -> None:
    from brain import anchor_moments as am
    with tempfile.TemporaryDirectory() as td:
        am._cache.clear()
        am.configure(Path(td))
        trivia = [
            ("can you speak it out and do research for it as well",
             "On it. I'm checking those tips first, then I'll read it out loud."),
            ("thank you!", "Any time."),
            ("do you remember the port for the mouth", "8769."),
            ("honestly I think the render is too dark", "Agreed, I'll lift the exposure."),
            ("that is a huge file", "It is."),
        ]
        for u, r in trivia:
            assert am.auto_detect_anchor_in_turn("zeke", u, r) is None, u
        meaningful = [
            "thank you for staying up with me, it means a lot",
            "i never told anyone that before, it was hard for me to say",
            "this is my first time showing you to my mother",
        ]
        for u in meaningful:
            assert am.auto_detect_anchor_in_turn("zeke", u, "I'm glad you told me."), u


def test_anchor_marked_before_configure_is_not_lost() -> None:
    from brain import anchor_moments as am
    with tempfile.TemporaryDirectory() as td:
        am._cache.clear()
        am._base_dir = None
        aid = am.mark_anchor(person_id="zeke", kind="milestone", summary="before configure")
        am.configure(Path(td))
        lines = (Path(td) / "state" / "anchor_moments.jsonl").read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["id"] == aid
        assert [a.id for a in am.list_anchors("zeke")] == [aid]


# ── scene_memory ──────────────────────────────────────────────────────────

def test_scene_memory_prune_respects_the_7_day_floor() -> None:
    from brain import scene_memory as smod
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        smod.DIR = tmp / "state" / "scene_memory"
        smod.INDEX = smod.DIR / "keyframes.jsonl"
        now = time.time()
        days = {}
        for age in (400, 100, 30, 3, 0):
            name = time.strftime("%Y%m%d", time.localtime(now - age * 86400))
            d = smod.DIR / "keyframes" / name
            d.mkdir(parents=True)
            (d / "a.jpg").write_bytes(b"x" * 1000)
            days[age] = name
        (smod.DIR / "keyframes" / "notaday").mkdir()      # never touched
        recs = [{"id": f"k{age}", "ts": now - age * 86400, "path": "p"} for age in (400, 100, 3)]
        smod.INDEX.write_text("".join(json.dumps(r) + "\n" for r in recs)
                              + json.dumps({"id": "k3", "_update": True, "words": "hi"}) + "\n")
        s = smod.SceneMemory({"BASE_DIR": td})
        out = s.prune(now)
        left = sorted(p.name for p in (smod.DIR / "keyframes").iterdir())
        assert days[400] not in left and days[100] not in left, left
        assert days[30] in left and days[3] in left and days[0] in left and "notaday" in left, left
        assert out["index_dropped"] == 1, out
        idx = [json.loads(x) for x in smod.INDEX.read_text().splitlines()]
        assert [r["id"] for r in idx] == ["k100", "k3"] and idx[1]["words"] == "hi", idx
        # size cap: everything older than 7 days goes, the last week never does
        smod.IMAGE_MAX_BYTES = 10
        try:
            s.prune(now)
        finally:
            smod.IMAGE_MAX_BYTES = 2 * 1024 ** 3
        left = sorted(p.name for p in (smod.DIR / "keyframes").iterdir())
        assert days[30] not in left and days[3] in left and days[0] in left, left


def test_scene_memory_records_trimmed_with_order() -> None:
    from brain import scene_memory as smod
    with tempfile.TemporaryDirectory() as td:
        smod.DIR = Path(td) / "state" / "scene_memory"
        smod.INDEX = smod.DIR / "keyframes.jsonl"
        s = smod.SceneMemory({"BASE_DIR": td})
        for i in range(smod.MAX_RECORDS + 50):
            s.records[f"k{i}"] = {"id": f"k{i}"}
            s.order.append(f"k{i}")
        s._trim_memory()
        assert len(s.order) == smod.MAX_RECORDS and len(s.records) == smod.MAX_RECORDS
        assert "k0" not in s.records and f"k{smod.MAX_RECORDS + 49}" in s.records


# ── memory_consolidation ──────────────────────────────────────────────────

def test_consolidation_lock_and_success_only_gate() -> None:
    from brain import iris_llm, memory_consolidation as mc
    orig = iris_llm.ask_iris
    with tempfile.TemporaryDirectory() as td:
        g = {"BASE_DIR": td}
        try:
            # concurrent call is refused, not run twice
            mc._RUN_LOCK.acquire()
            try:
                r = mc.consolidate(g)
                assert r.get("skipped") == "already_running", r
                assert mc.should_consolidate(g) is False
            finally:
                mc._RUN_LOCK.release()
            # cognition unavailable -> failed pass: week gate NOT advanced, backoff set
            iris_llm.ask_iris = lambda **k: None
            r = mc.consolidate(g)
            assert r["ok"] is False and "self_model_update" in r["failed_steps"], r
            st = json.loads((Path(td) / "state" / "consolidation_state.json").read_text())
            assert "last_consolidation_ts" not in st and st["last_failed_attempt_ts"] > 0, st
            assert mc.should_consolidate(g) is False        # backing off
            # success -> gate advances
            iris_llm.ask_iris = lambda **k: '{"weekly_pattern": "p", "emotional_shift": "s", "growth_note": "n"}'
            r = mc.consolidate(g)
            assert r["ok"] is True, r
            st = json.loads((Path(td) / "state" / "consolidation_state.json").read_text())
            assert st["last_consolidation_ts"] > 0 and "last_failed_attempt_ts" not in st, st
            assert not [f for f in os.listdir(Path(td) / "state") if f.endswith(".tmp")]
        finally:
            iris_llm.ask_iris = orig


# ── temporal_sense ────────────────────────────────────────────────────────

def test_stale_active_estimates_pruned_on_read() -> None:
    from brain import temporal_sense as ts
    with tempfile.TemporaryDirectory() as td:
        g = {"BASE_DIR": td}
        now = time.time()
        p = Path(td) / "state" / "active_estimates.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({
            "old": {"created_at": now - 90 * 86400, "estimate_seconds": 60.0, "interrupted": True},
            "long": {"created_at": now - 7 * 3600, "estimate_seconds": 3600.0},
            "fresh": {"created_at": now - 30, "estimate_seconds": 60.0},
        }))
        data = ts._read_active_estimates(g)
        assert sorted(data) == ["fresh", "long"], sorted(data)
        tid = ts.track_estimate(g, estimate_seconds=5.0, kind="unit_test")
        disk = json.loads(p.read_text())
        assert "old" not in disk and tid in disk, sorted(disk)
        assert [f for f in os.listdir(p.parent) if f.endswith(".tmp")] == []


# ── gap_delta ─────────────────────────────────────────────────────────────

def test_gap_delta_backwards_reader_matches_a_forward_scan() -> None:
    from brain import gap_delta as gd
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "log.jsonl"
        rows = [{"ts": 1000.0 + i, "n": i, "txt": "é" * (i % 7)} for i in range(300)]
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        body += "not json\n\n" + json.dumps({"n": "no-ts"}) + "\n"
        p.write_text(body, encoding="utf-8")
        fwd = [ln.strip() for ln in body.splitlines() if ln.strip()]
        assert list(gd._iter_lines_reverse(p, block=7)) == fwd[::-1]
        got = gd.read_since(p, 1250.0)
        assert [r["n"] for r in got] == list(range(251, 300)), [r["n"] for r in got][:5]
        assert gd.count_since(p, 1250.0) == 49
        assert gd.count_since(p, 1250.0, predicate=lambda e: e["n"] % 2 == 0) == 24
        assert gd.read_since(p, 0) == [] and gd.count_since(Path(td) / "missing", 5) == 0
        assert gd.count_since(p, 999.0) == 300


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}")
    print(f"{fails} failure(s)")
    raise SystemExit(1 if fails else 0)
