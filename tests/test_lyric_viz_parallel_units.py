"""lyric_viz_parallel: workers run as capped units in iris-workers.slice (2026-10-29 code review HIGH).

Runs a FAKE lyric_viz.sh (prep / chunk / streaming assemble) through the real driver, so the whole
unit path is exercised - systemd-run --wait --pipe, exit codes, logs, the stop-on-failure path - with
no whisper, no GPU and ~no CPU. Unit-mode checks skip when user systemd is not reachable.

    ~/venvs/iris-v100/bin/python -m pytest -q tests/test_lyric_viz_parallel_units.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "server" / "lyric_viz_parallel.py"


def _load():
    spec = importlib.util.spec_from_file_location("lyric_viz_parallel_t", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lvp = _load()
UNITS = lvp.user_systemd_ok()

FAKE = textwrap.dedent(r"""
    #!/usr/bin/env bash
    # fake lyric_viz.sh: --prep-only prints PREP; --frame-range writes a chunk; --assemble waits for the
    # chunks' .ok files and writes a real tiny video with the right frame count.
    set -eu
    out=""; fr=""; man=""; prep=0; asm=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --out) out=$2; shift;;
        --frame-range) fr=$2; shift;;
        --assemble) man=$2; asm=1; shift;;
        --prep-only) prep=1;;
      esac; shift
    done
    echo "oom=$(cat /proc/self/oom_score_adj) omp=${OMP_NUM_THREADS:-unset} cg=$(cat /proc/self/cgroup)"
    if [ $prep = 1 ]; then echo 'PREP {"f0": 0, "f1": 12, "fps": 30, "W": 16, "H": 16}'; exit 0; fi
    if [ $asm = 1 ]; then
      n=$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$man")
      for _ in $(seq 1 200); do
        k=$(ls "$(dirname "$man")"/*.mkv.ok 2>/dev/null | wc -l)
        [ "$k" -ge "$n" ] && break
        [ -e "$man.abort" ] && exit 9
        sleep 0.2
      done
      ffmpeg -loglevel error -y -f lavfi -i color=c=black:s=16x16:r=30 -frames:v 12 "$out"
      exit 0
    fi
    a=${fr%%:*}
    if [ -e "$(dirname "$out")/fail_at_$a" ]; then echo "chunk boom" >&2; exit 5; fi
    echo chunk > "$out"
""").lstrip()


def test_unit_argv_shape():
    argv = lvp.unit_argv("lvp-1-c000", ["/bin/true", "x"], {"OMP_NUM_THREADS": "1", "SECRET": "no"},
                         mem_max="6G", max_sec=3600, nice=15)
    assert argv[:2] == ["systemd-run", "--user"]
    for want in ("--wait", "--pipe", "--collect", "--unit=lvp-1-c000", "--slice=iris-workers.slice",
                 "MemoryMax=6G", "RuntimeMaxSec=3600", "OOMScoreAdjust=500", "KillMode=control-group",
                 "Nice=15", "OMP_NUM_THREADS=1"):
        assert want in argv, want
    assert "SECRET=no" not in argv          # only the allow-listed env crosses into the unit
    assert argv[-2:] == ["/bin/true", "x"]


def test_default_workers_is_3():
    src = SRC.read_text()
    assert 'add_argument("--workers", type=int, default=3' in src


@pytest.mark.skipif(not UNITS, reason="no user systemd here")
def test_unit_runs_capped_and_returns_exit_code(tmp_path):
    log = tmp_path / "u.log"
    unit = f"t-lvp-probe-{os.getpid()}"
    argv = lvp.unit_argv(unit, ["sh", "-c", "cat /proc/self/oom_score_adj; echo $OMP_NUM_THREADS; "
                                            "cat /proc/self/cgroup; exit 3"],
                         {"OMP_NUM_THREADS": "1"}, mem_max="1G", max_sec=60, nice=15)
    with open(log, "w") as lf:
        rc = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            timeout=60).returncode
    out = log.read_text()
    assert rc == 3, out
    assert out.splitlines()[0] == "500" and out.splitlines()[1] == "1", out
    assert "iris-workers.slice" in out, out


@pytest.mark.skipif(not UNITS, reason="no user systemd here")
def test_stop_unit_kills_the_work_not_just_the_waiter():
    unit = f"t-lvp-stop-{os.getpid()}"
    p = subprocess.Popen(lvp.unit_argv(unit, ["sleep", "120"], {}, mem_max="1G", max_sec=300, nice=15),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True,
                          text=True).stdout.strip() == "active":
            break
        time.sleep(0.1)
    lvp.stop_unit(unit)
    p.wait(timeout=30)
    st = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True).stdout.strip()
    assert st in ("inactive", "failed", "unknown", ""), st


def _run_driver(tmp_path, extra_env=None, extra_args=(), fail_at=None):
    fake = tmp_path / "fake_lyric_viz.sh"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    drv = tmp_path / "drv.py"
    # run the real driver with RUNNER pointed at the fake (no repo sys.path games: load by file path)
    drv.write_text(textwrap.dedent(f"""
        import importlib.util, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("lvp", {str(SRC)!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.RUNNER = Path({str(fake)!r})
        sys.argv = ["lvp"] + sys.argv[1:]
        sys.exit(m.main())
    """))
    out = tmp_path / "out.mp4"
    if fail_at is not None:          # a marker file, not env: a unit does not inherit our environment
        (tmp_path / ".out.parts").mkdir(exist_ok=True)
        (tmp_path / ".out.parts" / f"fail_at_{fail_at}").write_text("x")
    env = dict(os.environ, **(extra_env or {}))
    r = subprocess.run([sys.executable, str(drv), "--chunks", "3", *extra_args, "--out", str(out)],
                       capture_output=True, text=True, timeout=180, env=env)
    return r, out


@pytest.mark.skipif(not UNITS, reason="no user systemd here")
def test_driver_end_to_end_in_units(tmp_path):
    r, out = _run_driver(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.is_file()
    logs = "".join(p.read_text() for p in (tmp_path / ".out.parts").glob("chunk_*.log"))
    assert "oom=500" in logs and "iris-workers.slice" in logs and "omp=1" in logs, logs
    left = subprocess.run(["systemctl", "--user", "list-units", "--no-legend", "--plain", "lvp-*"],
                          capture_output=True, text=True).stdout
    assert "lvp-" not in left, left


@pytest.mark.skipif(not UNITS, reason="no user systemd here")
def test_driver_chunk_failure_stops_everything(tmp_path):
    r, out = _run_driver(tmp_path, fail_at=4)
    assert r.returncode == 3, r.stdout + r.stderr
    assert not out.exists()
    time.sleep(1)
    left = subprocess.run(["systemctl", "--user", "list-units", "--no-legend", "--plain", "lvp-*"],
                          capture_output=True, text=True).stdout
    assert "lvp-" not in left, left


def test_driver_no_units_fallback(tmp_path):
    r, out = _run_driver(tmp_path, extra_args=("--no-units",))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WITHOUT systemd caps" in r.stderr
