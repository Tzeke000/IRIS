"""V3 Phase 1 step 1.6 - the v2/v3 launcher chooser, stack_switch.sh, and the supervisor's automatic rollback.

Covers: chooser fallbacks (missing / v2 / garbled / missing tree / lock-incompatible tree / garbled v3.env -> v2,
valid v3 -> v3 with flags + the shared ~/IRIS lock), the shared ONE-OF-ME lock in both launchers, switch atomicity
+ log + refusals + the pre-restart memory check, the rollback decisions on a fake clock (failed starts, health red,
stale/pre-boot snapshots, no roll-forward), the v2 supervisor loop unchanged in effect vs the pre-1.6 supervisor,
and the full sandbox drill (exit gate 5). Shell logic runs via subprocess with HOME pointed at a tmp dir; nothing
touches the live stack (the real launchers are only ever run TRUNCATED right after their lock).
    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_stack_v3_launcher.py
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRV = REPO / "scripts" / "server"
REAL_SLEEP = shutil.which("sleep")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rb = _load("stack_rollback_t", "stack_rollback.py")


# ── sandbox helpers ──────────────────────────────────────────────────────────
def _env(home: Path, **extra) -> dict:
    env = {"HOME": str(home), "PATH": f"{home}/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
           "USER": os.environ.get("USER", "iris")}
    env.update({k: str(v) for k, v in extra.items()})
    return env


STUB = """#!/usr/bin/env bash
{{ echo "tree={tree} entry=$(basename "$0") stack=${{IRIS_STACK:-}} lock=${{IRIS_LAUNCHER_LOCK:-}} start_id=${{IRIS_STACK_START_ID:-}} flag=${{IRIS_V3_TEST_FLAG:-}}"; }} >> "$HOME/ran.log"
"""


def make_tree(home: Path, name: str, lock_aware: bool = True) -> Path:
    t = home / name
    (t / "scripts" / "server").mkdir(parents=True, exist_ok=True)
    (t / "state").mkdir(exist_ok=True)
    for entry in ("start_iris_v2.sh", "start_iris_v2_fable.sh", "start_iris.sh"):
        p = t / entry
        p.write_text(STUB.format(tree=name))
        p.chmod(0o755)
    for f in ("start_iris_linux.sh", "start_iris_cli_linux.sh"):
        src = (SRV / f).read_text()
        if not lock_aware:
            src = src.replace("IRIS_LAUNCHER_LOCK", "OLD_LOCK_VAR")
        (t / "scripts" / "server" / f).write_text(src)
    return t


@pytest.fixture()
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".config" / "iris").mkdir(parents=True)
    (h / "bin").mkdir()
    make_tree(h, "IRIS")
    return h


def launch(home: Path, mode="opus", **extra):
    r = subprocess.run(["bash", str(SRV / "stack_launch.sh"), mode], env=_env(home, **extra),
                       capture_output=True, text=True, timeout=30)
    ran = (home / "ran.log").read_text() if (home / "ran.log").exists() else ""
    log = home / "IRIS" / "state" / "v3" / "stack_launch.log"
    return r, ran, (log.read_text() if log.exists() else "")


def set_stack(home: Path, content: bytes | str):
    p = home / ".config" / "iris" / "active_stack"
    p.write_bytes(content if isinstance(content, bytes) else content.encode())


# ── chooser ──────────────────────────────────────────────────────────────────
def test_missing_active_stack_runs_v2_and_logs(home):
    make_tree(home, "IRIS-v3")
    r, ran, log = launch(home)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tree=IRIS entry=start_iris_v2.sh stack=v2" in ran
    assert f"lock={home}/IRIS/.tmp/launcher.lock" in ran
    assert "no active_stack file" in log


def test_v2_runs_v2(home):
    make_tree(home, "IRIS-v3")
    set_stack(home, "v2\n")
    _, ran, log = launch(home)
    assert "tree=IRIS " in ran and "stack=v2" in ran and "active_stack=v2" in log


@pytest.mark.parametrize("content", [b"v4\n", b"v2\nv3\n", b"", b"\x00\xffv3", b"V3", b"v3 please"])
def test_garbled_active_stack_falls_back_to_v2(home, content):
    make_tree(home, "IRIS-v3")
    set_stack(home, content)
    _, ran, log = launch(home)
    assert "tree=IRIS " in ran and "stack=v2" in ran, (content, ran)
    assert "garbled active_stack" in log


def test_v3_with_missing_tree_falls_back_to_v2(home):
    set_stack(home, "v3\n")
    _, ran, log = launch(home)
    assert "tree=IRIS " in ran and "stack=v2" in ran
    assert "FALLBACK" in log and "IRIS-v3 is missing" in log


def test_v3_tree_without_shared_lock_is_refused(home):
    make_tree(home, "IRIS-v3", lock_aware=False)
    set_stack(home, "v3")
    _, ran, log = launch(home)
    assert "tree=IRIS " in ran and "FALLBACK" in log and "IRIS_LAUNCHER_LOCK" in log


def test_v3_with_garbled_flags_falls_back_to_v2(home):
    make_tree(home, "IRIS-v3")
    set_stack(home, "v3")
    (home / ".config" / "iris" / "v3.env").write_text("IRIS_V3_EVENTS=1\nthis is not an assignment\n")
    _, ran, log = launch(home)
    assert "tree=IRIS " in ran and "garbled" in log and "line 2" in log


def test_v3_valid_runs_v3_with_flags_shared_lock_and_launch_record(home):
    make_tree(home, "IRIS-v3")
    set_stack(home, "  v3 \n")
    (home / ".config" / "iris" / "v3.env").write_text(
        "# v3 flags\n\nIRIS_V3_TEST_FLAG=\"on\"\nexport IRIS_V3_TASKS=1\n")
    r, ran, log = launch(home, IRIS_STACK_START_ID="123-4-1")
    assert r.returncode == 0
    assert "tree=IRIS-v3 entry=start_iris_v2.sh stack=v3" in ran
    assert f"lock={home}/IRIS/.tmp/launcher.lock" in ran          # the v2 tree's lock, not ~/IRIS-v3/.tmp
    assert "flag=on" in ran and "start_id=123-4-1" in ran
    rec = dict(ln.split("=", 1) for ln in (home / "IRIS/state/v3/launched.env").read_text().splitlines())
    assert rec["stack"] == "v3" and rec["wanted"] == "v3" and rec["start_id"] == "123-4-1"
    assert rec["root"] == str(home / "IRIS-v3")


@pytest.mark.parametrize("mode,entry", [("fable", "start_iris_v2_fable.sh"), ("cli", "start_iris.sh")])
def test_modes_map_to_the_same_entries_as_the_gate(home, mode, entry):
    make_tree(home, "IRIS-v3")
    set_stack(home, "v3")
    _, ran, _ = launch(home, mode=mode)
    assert f"tree=IRIS-v3 entry={entry}" in ran


def test_unknown_mode_is_refused(home):
    r, ran, _ = launch(home, mode="sonnet")
    assert r.returncode == 2 and ran == ""


def test_fallback_record_says_wanted_v3_ran_v2(home):
    set_stack(home, "v3")
    launch(home, IRIS_STACK_START_ID="9-9-9")
    rec = dict(ln.split("=", 1) for ln in (home / "IRIS/state/v3/launched.env").read_text().splitlines())
    assert rec["stack"] == "v2" and rec["wanted"] == "v3" and rec["start_id"] == "9-9-9"


def test_gate_template_runs_the_chooser_with_one_line():
    src = (REPO / "scripts" / "zorin_launcher" / "iris_start.sh").read_text()
    lines = [ln for ln in src.splitlines() if "stack_launch.sh" in ln and not ln.lstrip().startswith("#")]
    assert lines == ['if [ -x "$HOME/IRIS/scripts/server/stack_launch.sh" ]; then '
                     '"$HOME/IRIS/scripts/server/stack_launch.sh" "$MODE"; else "$SCRIPT"; fi']


# ── the shared ONE-OF-ME lock (launchers TRUNCATED right after the flock - never the sweeps) ──
def _truncated_launcher(home: Path, tree: str, name: str) -> Path:
    src = (SRV / name).read_text().splitlines()
    cut = next(i for i, ln in enumerate(src) if "flock -n 9" in ln)
    body = "\n".join(src[:cut + 1]) + '\necho "LOCK ACQUIRED"\nexit 0\n'
    assert "pkill" not in body and "fuser" not in body and "iris_body_host.py" not in body   # never the sweeps
    p = home / tree / "scripts" / "server" / name
    p.write_text(body)
    return p


@pytest.mark.parametrize("name", ["start_iris_linux.sh", "start_iris_cli_linux.sh"])
@pytest.mark.parametrize("tree", ["IRIS", "IRIS-v3"])
def test_both_trees_take_the_same_lock(home, name, tree):
    make_tree(home, "IRIS-v3")
    p = _truncated_launcher(home, tree, name)
    lock = home / "IRIS" / ".tmp" / "launcher.lock"
    env = _env(home, IRIS_START_GATE="test")
    r = subprocess.run(["bash", str(p), "opus"], env=env, capture_output=True, text=True, timeout=20)
    assert "LOCK ACQUIRED" in r.stdout, r.stdout + r.stderr
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = subprocess.run(["bash", str(p), "opus"], env=env, capture_output=True, text=True, timeout=20)
    assert r.returncode == 10 and "LOCK ACQUIRED" not in r.stdout, r.stdout


# ── stack_switch.sh ──────────────────────────────────────────────────────────
def switch(home: Path, *args, **extra):
    return subprocess.run(["bash", str(SRV / "stack_switch.sh"), *args], env=_env(home, **extra),
                          capture_output=True, text=True, timeout=30)


def _stack(home):
    p = home / ".config" / "iris" / "active_stack"
    return p.read_text() if p.exists() else None


def test_switch_status_with_nothing_says_v2(home):
    r = switch(home, "status")
    assert r.returncode == 0 and "next start runs: v2 (no file = v2)" in r.stdout and "MISSING" in r.stdout


def test_switch_v2_writes_logs_reminds_and_never_restarts(home):
    r = switch(home, "v2", "testing the switch")
    assert r.returncode == 0, r.stdout
    assert _stack(home) == "v2\n"
    assert "PRE-RESTART CONTRACT" in r.stdout and "cold-wake" in r.stdout and "NOT restarting" in r.stdout
    log = (home / "IRIS/state/v3/stack_switch.log").read_text()
    assert re.search(r"<missing> -> v2 by=\S+.* restart=0 reason=testing the switch", log)
    assert not (home / "IRIS/.tmp/restart_cc.flag").exists()


def test_switch_refusals_leave_the_file_alone(home):
    switch(home, "v2", "base")
    assert switch(home, "v2").returncode == 2                         # no reason
    assert switch(home, "v3", "no tree").returncode == 4              # ~/IRIS-v3 missing
    make_tree(home, "IRIS-v3", lock_aware=False)
    assert switch(home, "v3", "old launcher").returncode == 4         # no shared lock
    assert switch(home, "v3", "forward", "--auto").returncode == 2    # --auto never rolls forward
    assert switch(home, "v4", "x").returncode == 2
    assert _stack(home) == "v2\n"


def test_switch_restart_needs_a_fresh_memory_note(home):
    make_tree(home, "IRIS-v3")
    mem = home / "mem"
    mem.mkdir()
    old = mem / "handoff_old.md"
    old.write_text("x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    flag = home / "IRIS/.tmp/restart_cc.flag"
    r = switch(home, "v3", "go", "--restart", IRIS_MEMORY_DIR=mem)
    assert r.returncode == 6 and _stack(home) is None and not flag.exists()
    (mem / "handoff_now.md").write_text("saved")
    r = switch(home, "v3", "go", "--restart", IRIS_MEMORY_DIR=mem)
    assert r.returncode == 0 and _stack(home) == "v3\n"
    assert flag.read_text().startswith("stack_switch: <missing> -> v3: go")
    flag.unlink()
    (mem / "handoff_now.md").unlink()
    r = switch(home, "v2", "emergency", "--restart", "--force", IRIS_MEMORY_DIR=mem)
    assert r.returncode == 0 and _stack(home) == "v2\n" and flag.exists()


def test_switch_is_atomic_under_concurrency(home):
    make_tree(home, "IRIS-v3")
    target = home / ".config" / "iris" / "active_stack"
    switch(home, "v2", "seed")
    procs = [subprocess.Popen(["bash", str(SRV / "stack_switch.sh"), "v3" if i % 2 else "v2", f"race {i}"],
                              env=_env(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i in range(24)]
    seen = set()
    while any(p.poll() is None for p in procs):
        try:
            seen.add(target.read_bytes())
        except FileNotFoundError:
            seen.add(b"<missing>")
    assert all(p.wait(timeout=60) == 0 for p in procs)
    seen.add(target.read_bytes())
    assert seen <= {b"v2\n", b"v3\n"}, seen                          # a reader never sees half a write
    log = (home / "IRIS/state/v3/stack_switch.log").read_text().splitlines()
    assert len(log) == 25 and all(" -> v" in ln for ln in log)
    assert not list((home / ".config" / "iris").glob(".active_stack.*.tmp"))


# ── rollback decisions on a fake clock ───────────────────────────────────────
CFG = {"failed_start_s": 600.0, "fail_window_s": 900.0, "fail_count": 2, "red_s": 600.0, "fresh_s": 180.0}


def test_two_failed_starts_within_15_min_roll_back():
    st: dict = {}
    roll, why = rb.record_exit(st, start_ts=1000, now=1040, cfg=CFG, rc=1)
    assert not roll and "failed start 1/2" in why
    roll, why = rb.record_exit(st, start_ts=1100, now=1130, cfg=CFG, rc=1)
    assert roll and "2 failed v3 starts" in why


def test_failed_starts_further_apart_than_the_window_do_not():
    st: dict = {}
    assert not rb.record_exit(st, 1000, 1040, CFG, 1)[0]
    assert not rb.record_exit(st, 1940, 1960, CFG, 1)[0]              # 920 s after the first failure


def test_an_exit_after_the_boot_window_is_not_a_failed_start():
    st: dict = {}
    assert not rb.record_exit(st, 1000, 1040, CFG, 1)[0]
    roll, why = rb.record_exit(st, 1100, 1100 + 700, CFG, 0)          # ran 700 s > failed_start_s
    assert not roll and "not a failed start" in why and len(st["failures"]) == 1


def _doc(ts, sdk="ok", events="n/a"):
    return {"ts": ts, "subsystems": {"sdk_stream": {"status": sdk, "last_error": "no host" if sdk == "red" else None},
                                     "events": {"status": events}}}


def test_health_red_for_10_min_after_boot_rolls_back():
    st: dict = {}
    boot = 10_000.0
    for t in range(0, 600, 60):                                       # red 0..540 s: not yet
        roll, _ = rb.observe_health(st, _doc(boot + 600 + t, "red"), boot, boot + 600 + t + 5, CFG)
        assert not roll
    roll, why = rb.observe_health(st, _doc(boot + 1200, "red"), boot, boot + 1205, CFG)
    assert roll and "sdk_stream red 600s" in why


def test_events_red_counts_too_and_warn_does_not():
    st: dict = {}
    boot = 0.0
    for t in range(0, 660, 60):
        roll, _ = rb.observe_health(st, _doc(100 + t, "warn", "warn"), boot, 100 + t, CFG)
        assert not roll
    st = {}
    rb.observe_health(st, _doc(100, "ok", "red"), boot, 100, CFG)
    for t in range(160, 760, 60):
        roll, why = rb.observe_health(st, _doc(t, "ok", "red"), boot, t, CFG)
    assert roll and "events red" in why


def test_green_in_between_resets_the_red_run():
    st: dict = {}
    for t in range(0, 540, 60):
        rb.observe_health(st, _doc(t, "red"), 0, t, CFG)
    rb.observe_health(st, _doc(540, "ok"), 0, 540, CFG)
    for t in range(600, 1140, 60):
        roll, _ = rb.observe_health(st, _doc(t, "red"), 0, t, CFG)
        assert not roll


def test_stale_missing_and_pre_boot_snapshots_are_unknown():
    st: dict = {}
    boot = 5000.0
    assert not rb.observe_health(st, None, boot, boot + 10, CFG)[0]
    assert not rb.observe_health(st, {"no": "ts"}, boot, boot + 10, CFG)[0]
    roll, why = rb.observe_health(st, _doc(boot - 1, "red"), boot, boot + 900, CFG)
    assert not roll and "predates" in why and st["red"] == {}
    rb.observe_health(st, _doc(boot + 10, "red"), boot, boot + 10, CFG)
    roll, why = rb.observe_health(st, _doc(boot + 10, "red"), boot, boot + 900, CFG)  # watchdog stopped writing
    assert not roll and "old" in why and "sdk_stream" in st["red"]            # unknown: no count, no reset


def test_a_gap_in_the_evidence_restarts_the_red_run():
    st: dict = {}
    rb.observe_health(st, _doc(100, "red"), 0, 100, CFG)
    roll, _ = rb.observe_health(st, _doc(800, "red"), 0, 800, CFG)          # 700 s since the last red seen
    assert not roll and st["red"]["sdk_stream"]["since"] == 800


def test_a_new_boot_clears_the_red_tracker():
    st: dict = {}
    for t in range(0, 540, 60):
        rb.observe_health(st, _doc(t, "red"), 0, t, CFG)
    roll, _ = rb.observe_health(st, _doc(1000, "red"), 990, 1000, CFG)
    assert not roll and st["red"]["sdk_stream"]["since"] == 1000


def _cli(tmp_path, *args, **env):
    e = dict(os.environ, IRIS_ROOT=str(tmp_path), **{k: str(v) for k, v in env.items()})
    return subprocess.run([sys.executable, str(SRV / "stack_rollback.py"), *args], env=e,
                          capture_output=True, text=True, timeout=30)


def test_cli_start_exit_and_health_exit_codes(tmp_path):
    assert _cli(tmp_path, "start-exit", "--start-ts", "1000", "--now", "1010", "--rc", "1").returncode == 0
    r = _cli(tmp_path, "start-exit", "--start-ts", "1050", "--now", "1060", "--rc", "1")
    assert r.returncode == 3 and "2 failed v3 starts" in r.stdout
    st = json.loads((tmp_path / "state/v3/rollback_state.json").read_text())
    assert st["rollback"]["trigger"] == "failed_starts" and st["failures"] == []
    h = tmp_path / "now.json"
    for t in (2000, 2300, 2600):
        h.write_text(json.dumps(_doc(t, "red")))
        r = _cli(tmp_path, "health", "--boot-ts", "1900", "--now", str(t + 1), "--health", str(h),
                 IRIS_V3_RED_S=600, IRIS_V3_HEALTH_FRESH_S=400)
    assert r.returncode == 3 and "health red" in r.stdout


def test_cli_dm_uses_the_override_and_staging_never_dms(tmp_path):
    out = tmp_path / "dm.txt"
    stub = tmp_path / "dm.sh"
    stub.write_text(f'#!/bin/sh\nprintf "%s" "$1" > {out}\n')
    stub.chmod(0o755)
    assert _cli(tmp_path, "dm", "hello", IRIS_STACK_DM_CMD=stub).returncode == 0 and out.read_text() == "hello"
    e = {k: v for k, v in os.environ.items() if k != "IRIS_STACK_DM_CMD"}
    r = subprocess.run([sys.executable, str(SRV / "stack_rollback.py"), "dm", "x"],
                       env=dict(e, IRIS_ROOT=str(tmp_path), IRIS_ROLE="staging"), capture_output=True, timeout=30)
    assert r.returncode == 1


def test_nothing_automatic_ever_writes_v3():
    """No roll-forward: the decision module never touches active_stack, the supervisor only ever asks for v2."""
    code = re.sub(r'""".*?"""', "", (SRV / "stack_rollback.py").read_text(), flags=re.S)
    code = "\n".join(ln.split("#")[0] for ln in code.splitlines())
    assert "active_stack" not in code and "stack_switch" not in code
    sup = (SRV / "iris_supervise.sh").read_text()
    calls = re.findall(r'"\$V3_SWITCH"\s+(\S+)', sup)
    assert calls == ["v2"], calls
    writes = re.findall(r"printf '(v\d)\\n' > \"\$STACK_FILE", sup)
    assert writes == ["v2"], writes


# ── supervisor: v2 / missing active_stack == the pre-1.6 supervisor, in effect ──
FAKE_GATE = """#!/usr/bin/env bash
N=$(( $(cat "$HOME/gate_n" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$HOME/gate_n"
echo "start $N id=${IRIS_STACK_START_ID:-none}" >> "$HOME/gate.log"
case "$N" in
  1) exit 1 ;;
  2) exec "$REAL_SLEEP_BIN" 1000 ;;
  3) exit 0 ;;
  *) exit 5 ;;
esac
"""


def _old_supervisor() -> str | None:
    """iris_supervise.sh as it was right before step 1.6 (the parent of the commit that added the v3 block)."""
    try:
        shas = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", "V3 Phase 1 step 1.6", "--",
                               "scripts/server/iris_supervise.sh"], cwd=REPO, capture_output=True, text=True,
                              timeout=30).stdout.split()
        sha = shas[0] if shas else ""
        if sha:
            r = subprocess.run(["git", "show", f"{sha}^:scripts/server/iris_supervise.sh"], cwd=REPO,
                               capture_output=True, text=True, timeout=30)
        else:   # not committed yet: HEAD still has the old one
            r = subprocess.run(["git", "show", "HEAD:scripts/server/iris_supervise.sh"], cwd=REPO,
                               capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and "V3 Phase 1 step 1.6" not in r.stdout:
            return r.stdout
    except Exception:
        pass
    return None


def _run_supervisor(tmp: Path, script_text: str, stack: bytes | None) -> tuple[list[str], str, Path]:
    home = tmp / "home"
    root = home / "IRIS"
    (root / "scripts" / "server").mkdir(parents=True)
    (home / "bin").mkdir()
    (home / ".config" / "iris").mkdir(parents=True)
    if stack is not None:
        (home / ".config" / "iris" / "active_stack").write_bytes(stack)
    sup = root / "scripts" / "server" / "iris_supervise.sh"
    sup.write_text(script_text)
    for f in ("stack_rollback.py", "stack_switch.sh"):
        shutil.copy(SRV / f, root / "scripts" / "server" / f)
    gate = home / "iris_start.sh"
    gate.write_text(FAKE_GATE)
    gate.chmod(0o755)
    fs = home / "bin" / "sleep"
    fs.write_text(f'#!/bin/sh\nexec {REAL_SLEEP} "$(awk -v s="${{1:-1}}" \'BEGIN{{printf "%.3f", s*0.02}}\')"\n')
    fs.chmod(0o755)
    env = _env(home, IRIS_PYTHON=sys.executable, REAL_SLEEP_BIN=REAL_SLEEP, IRIS_STACK_DM_CMD="/bin/false")
    p = subprocess.Popen(["bash", str(sup), "opus"], env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    glog = home / "gate.log"
    end = time.time() + 30
    while time.time() < end and "start 2" not in (glog.read_text() if glog.exists() else ""):
        time.sleep(0.05)
    time.sleep(0.3)
    (root / ".tmp" / "restart_cc.flag").write_text("test restart\n")
    p.wait(timeout=60)
    lines = (root / "state" / "supervise.log").read_text().splitlines()
    norm = [re.sub(r"^\[[^\]]+\] ", "", ln).replace(str(home), "~") for ln in lines]
    norm = [re.sub(r"boot=.*", "boot=X", ln) for ln in norm]
    return norm, glog.read_text(), root


@pytest.mark.parametrize("stack", [None, b"v2\n", b"garbled\n"])
def test_supervisor_with_v2_or_no_choice_behaves_like_the_pre_v3_supervisor(tmp_path, stack):
    old = _old_supervisor()
    if old is None:
        pytest.skip("pre-1.6 supervisor not reachable in git history")
    want, want_gate, _ = _run_supervisor(tmp_path / "old", old, None)
    got, got_gate, root = _run_supervisor(tmp_path / "new", (SRV / "iris_supervise.sh").read_text(), stack)
    assert got == want
    assert got_gate == want_gate and "id=none" in got_gate and "start 4" in got_gate
    assert not (root / "state" / "v3").exists()                       # the v2 loop never touches v3 state
    assert any("restart flag seen (test restart )" in ln for ln in got)
    assert got[-2:] == want[-2:] and "supervisor END" in got[-1]


# ── exit gate 5: the full sandbox drill ──────────────────────────────────────
def test_rollback_drill_passes_in_a_sandbox():
    r = subprocess.run(["bash", str(SRV / "stack_rollback_drill.sh")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "DRILL PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    assert "FAIL " not in r.stdout
