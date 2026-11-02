#!/usr/bin/env python3
"""scripts/server/fault_inject.py - sandbox fault injection (V3 Phase 4, 2026-11-02).

Each scenario breaks ONE thing inside a throw-away STAGING root (a fresh /tmp/iris-fi-* dir, IRIS_ROLE=staging,
CUDA hidden, HF offline) and asserts the system DEGRADES and RECORDS AN INCIDENT instead of crashing:

    disk_full     a private 1 MiB tmpfs mounted over <staging>/state inside an unprivileged user+mount
                  namespace (`unshare -rm`: no sudo, invisible to every other process), filled to ENOSPC.
                  -> the event store fails OPEN (insert returns None, nothing raises) and RECOVERS when space
                     returns; health_snapshot's storage probe goes red; the watchdog's write_health contains the
                     ENOSPC; the incident bundle can't go to the full state/ so it lands in the fallback dir with
                     `primary_failed` and no half-written .tmp dir is left behind.
    model_load    the ears' whisper model pointed at a path that doesn't exist (+ the smart-turn model).
                  -> the daemon's _warm_whisper logs FAILED and returns (no crash); _rich_health reports
                     smartturn degraded; incident recorded.
    gpu_pressure  a fake `nvidia-smi` on PATH reports the V100 at 31.3/32.5 GiB.
                  -> worker.py admission DEFERS a GPU job (exit 1, "GPU free ... < N"); health_snapshot's gpu
                     probe says warn (>95%). warn is not red, so the watchdog does NOT auto-bundle: the harness
                     records the incident explicitly (`incident.py capture` - the operator step).
    mic_gone      the ears' network mic (IRIS_MIC_SOURCE) pointed at a port nobody listens on, and at a source
                  that dies mid-stream. -> _capture_utterance raises a typed ConnectionError fast (no hang); the
                  daemon's real _handle_connection turns it into {"ok": false} and KEEPS serving (a ping on the
                  same connection still pongs); probe_ears against a dead staging port is red -> incident.

    fault_inject.py list
    fault_inject.py run SCENARIO|all [--keep] [--json]

NEVER the live stack: refuses to run as a child unless IRIS_ROLE=staging and the staging root is a fresh temp
dir outside the repo; never binds 8769/8770/5876; never touches the real GPU (CUDA_VISIBLE_DEVICES=""); voice
status writes are redirected into the staging root. The two acceptance tests that need the REAL stack (a real
CUDA hog next to a speaking mouth; unplugging the real headset) are documented procedures in
docs/V3_FAULT_INJECTION.md, not code here.
Exit 0 = every scenario passed its asserts; 1 = a scenario failed (degraded wrongly / crashed / no incident).
"""
from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("IRIS_FI_REPO") or Path(__file__).resolve().parents[2])
HERE = Path(__file__).resolve().parent
PY = sys.executable
SCENARIOS = ("disk_full", "model_load", "gpu_pressure", "mic_gone")
CHILD_TIMEOUT_S = 180


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── safety ──────────────────────────────────────────────────────────────────
def assert_staging(staging: Path) -> None:
    if os.environ.get("IRIS_ROLE") != "staging":
        raise SystemExit("fault_inject: refusing - IRIS_ROLE is not 'staging'")
    s = staging.resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    if tmp not in s.parents or not s.name.startswith("iris-fi-"):
        raise SystemExit(f"fault_inject: refusing - {s} is not a fresh iris-fi-* temp dir")
    if ROOT.resolve() in s.parents or s == ROOT.resolve():
        raise SystemExit("fault_inject: refusing - staging root is inside the repo")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── shared: the real watchdog -> incident pipeline, pointed at the staging root ──
def no_journals() -> dict:
    inc = _load(HERE / "incident.py", "fi_incident_cols")
    return {f"journal_{u}.log": (lambda u=u: f"[staging: {u} journal not collected]") for u in inc.CORE_UNITS}


def record_incident(staging: Path, doc: dict, *, out_dir: Path | None = None,
                    fallbacks: list[Path] | None = None) -> dict:
    """doc (a health snapshot) -> iris_watchdog.capture_red_incidents -> incident.capture in-process."""
    wd = _load(HERE / "iris_watchdog.py", "fi_watchdog")
    wd.LOG = staging / "watchdog.log"                      # never the live state/server_watchdog.log
    inc = _load(HERE / "incident.py", "fi_incident")
    results: list = []

    def spawn(k, reason):
        r = inc.capture(reason, subsystem=k, root=staging, out_dir=out_dir or staging / "state" / "incidents",
                        fallbacks=fallbacks if fallbacks is not None else [staging / "fallback"],
                        collectors=no_journals())
        results.append(r)
        return r["captured"]
    started = wd.capture_red_incidents(doc, {}, time.time(), spawn=spawn)
    return {"started": started, "bundles": results}


def red_doc(sub: str, error: str, probe: str) -> dict:
    return {"subsystems": {sub: {"status": "red", "ok": False, "last_error": error, "probe": probe}}}


# ── scenarios (run in the child) ────────────────────────────────────────────
def sc_disk_full(staging: Path) -> dict:
    if os.environ.get("IRIS_FI_IN_NS") != "1":
        raise SystemExit("disk_full must run inside `unshare -rm` (the parent does that)")
    mnt = staging / "state"
    mnt.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mount", "-t", "tmpfs", "-o", "size=1m,mode=0700", "iris-fi", str(mnt)], check=True)
    sys.path.append(str(ROOT))
    from brain import iris_events
    db = mnt / "v3" / "events.sqlite"
    store = iris_events.EventStore(db, synchronous="FULL")
    before = store.insert("fi", "before the disk filled", status="admitted")
    ballast = mnt / "ballast.bin"
    with open(ballast, "wb") as f:
        try:
            while True:
                f.write(b"\0" * 65536)
                f.flush()
        except OSError as e:
            filled = errno.errorcode.get(e.errno, str(e.errno))
    out: dict = {"filled": filled, "insert_before": bool(before)}
    rows = [store.insert("fi", "x" * 2000, status="admitted") for _ in range(20)]
    out["insert_while_full_ok"] = sum(1 for r in rows if r)
    out["store_last_error"] = store.last_error
    out["store_raised"] = False                            # we got here: nothing propagated
    ballast.unlink()
    after = store.insert("fi", "after space came back", status="admitted")
    out["insert_after_recovery"] = bool(after)
    out["integrity_after"] = store.integrity_check()
    store.close()
    # fill again for the health + incident steps
    with open(ballast, "wb") as f:
        try:
            while True:
                f.write(b"\0" * 65536)
                f.flush()
        except OSError:
            pass
    hs = _load(HERE / "health_snapshot.py", "fi_hs")
    doc = hs.build_snapshot(staging, probes={"storage": hs.probe_storage}, disk_mounts=(str(mnt),),
                            in_grace=False)
    out["storage_status"] = doc["subsystems"]["storage"]["status"]
    out["storage_error"] = doc["subsystems"]["storage"]["last_error"]
    wd = _load(HERE / "iris_watchdog.py", "fi_watchdog_h")
    wd.LOG, wd.ROOT, wd.HEALTH_OUT = staging / "watchdog.log", staging, mnt / "health" / "now.json"
    out["write_health_contained"] = wd.write_health(time.time()) is None
    rec = record_incident(staging, doc, out_dir=mnt / "incidents", fallbacks=[staging / "fallback"])
    b = (rec["bundles"] or [{}])[0]
    out["incident"] = b.get("path")
    out["incident_primary_failed"] = b.get("primary_failed")
    out["half_bundles_left"] = sorted(p.name for p in (mnt / "incidents").glob(".tmp_*")) \
        if (mnt / "incidents").is_dir() else []
    out["passed"] = bool(
        out["insert_before"] and out["insert_while_full_ok"] < 20 and out["insert_after_recovery"]
        and out["integrity_after"] == "ok" and out["storage_status"] == "red" and out["write_health_contained"]
        and out["incident"] and str(staging / "fallback") in out["incident"] and out["incident_primary_failed"]
        and not out["half_bundles_left"])
    return out


def sc_model_load(staging: Path) -> dict:
    sys.path.append(str(ROOT / "voice"))
    import io
    import wren_listen as wl
    import wren_voice_status as vs
    vs.STATUS_FILE, vs.CONTROL_FILE = staging / "voice_status.json", staging / "voice_control.json"
    import wren_voice_daemon as dmn
    core = dmn.core
    missing = staging / "models" / "whisper-does-not-exist"
    wl.WHISPER_SIZE, wl._WHISPER = str(missing), None
    err = io.StringIO()
    old_err, sys.stderr = sys.stderr, err
    try:
        dmn._warm_whisper(None)                            # the daemon's real warm path
        raised = None
    except BaseException as e:  # noqa: BLE001 — SystemExit counts as a crash too
        raised = repr(e)
    finally:
        sys.stderr = old_err
    out: dict = {"warm_raised": raised, "warm_log": err.getvalue().strip()[-300:], "whisper_loaded": wl._WHISPER
                 is not None}
    try:
        core._smartturn._MODEL_PATH = staging / "models" / "smartturn-missing.onnx"
    except Exception as e:  # noqa: BLE001
        out["smartturn_patch"] = repr(e)
    out["rich_health_degraded"] = core._rich_health(None)
    doc = red_doc("ears_models", f"whisper load failed ({out['warm_log'][-160:]}); degraded: "
                                 f"{out['rich_health_degraded']}",
                  "fault_inject - NO live probe sees this: ears ping still pongs with no whisper (gap)")
    rec = record_incident(staging, doc)
    out["incident"] = (rec["bundles"] or [{}])[0].get("path")
    out["passed"] = bool(raised is None and "FAILED" in out["warm_log"] and not out["whisper_loaded"]
                         and "smartturn" in out["rich_health_degraded"] and out["incident"])
    return out


FAKE_SMI = """#!/bin/sh
case "$*" in
  *memory.free*) echo "1200" ;;
  *memory.used,memory.total*) echo "31300, 32510, 100, 71" ;;
  *query-compute-apps*) echo "" ;;
  *) echo "Tesla V100-PCIE-32GB, 31300 MiB, 32510 MiB, 100 %, 71" ;;
esac
"""


def sc_gpu_pressure(staging: Path) -> dict:
    bindir = staging / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    smi = bindir / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ.get('PATH', '')}",
               IRIS_WORKER_DB=str(staging / "jobs.sqlite"), IRIS_EVENTS_DB=str(staging / "events.sqlite"))
    r = subprocess.run([PY, str(HERE / "worker.py"), "admit", "--kind", "fi_gpu", "--gpu-mib", "4096"],
                       env=env, capture_output=True, text=True, timeout=60)
    try:
        info = json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        info = {"raw": r.stdout[-300:], "stderr": r.stderr[-300:]}
    out: dict = {"admit_exit": r.returncode, "admit_why": info.get("why"), "gpu_free_mib": info.get("gpu_free_mib")}
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = env["PATH"]
    try:
        hs = _load(HERE / "health_snapshot.py", "fi_hs_gpu")
        doc = hs.build_snapshot(staging, probes={"gpu": hs.probe_gpu}, in_grace=False)
    finally:
        os.environ["PATH"] = old_path
    g = doc["subsystems"]["gpu"]
    out["gpu_status"], out["gpu_error"] = g["status"], g["last_error"]
    rec = record_incident(staging, doc)                    # warn: the watchdog must NOT auto-bundle
    out["watchdog_autocaptured"] = rec["started"]
    inc = _load(HERE / "incident.py", "fi_inc_gpu")
    b = inc.capture(f"GPU pressure: {g['last_error']}", subsystem="gpu", root=staging,
                    out_dir=staging / "state" / "incidents", fallbacks=[], collectors=no_journals())
    out["incident"] = b.get("path")
    out["passed"] = bool(r.returncode == 1 and any("GPU free" in w for w in (out["admit_why"] or []))
                         and out["gpu_status"] == "warn" and out["watchdog_autocaptured"] == []
                         and out["incident"])
    return out


def sc_mic_gone(staging: Path) -> dict:
    dead = free_port()                                     # bound then released: nobody listens here
    os.environ["IRIS_MIC_SOURCE"] = f"127.0.0.1:{dead}"
    sys.path.append(str(ROOT / "voice"))
    import wren_voice_status as vs
    vs.STATUS_FILE, vs.CONTROL_FILE = staging / "voice_status.json", staging / "voice_control.json"
    import wren_voice_daemon as dmn
    core = dmn.core
    core._net_audio.MIC_SOURCE = os.environ["IRIS_MIC_SOURCE"]
    ctx = dmn.Ctx()
    out: dict = {}
    t0 = time.monotonic()
    try:
        core._capture_utterance(ctx, 2.0, 5.0)
        out["refused"] = "no error (WRONG: captured from a mic that doesn't exist)"
    except Exception as e:  # noqa: BLE001
        out["refused"] = type(e).__name__
    out["refused_in_s"] = round(time.monotonic() - t0, 2)

    # a source that streams 0.3 s of silence then dies mid-stream
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.recv(4096)
        c.sendall(b"\0" * 4 * 4800)
        time.sleep(0.2)
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    core._net_audio.MIC_SOURCE = f"127.0.0.1:{srv.getsockname()[1]}"
    try:
        core._capture_utterance(ctx, 3.0, 5.0)
        out["midstream"] = "no error"
    except Exception as e:  # noqa: BLE001
        out["midstream"] = type(e).__name__
    srv.close()
    core._net_audio.MIC_SOURCE = f"127.0.0.1:{dead}"

    # the daemon's real connection handler: the mic error becomes {"ok": false}, the connection keeps serving
    core.cmd_listen = lambda c, a: core._capture_utterance(c, 2.0, 5.0)   # the capture is the fault under test
    a, b = socket.socketpair()
    th = threading.Thread(target=dmn._handle_connection, args=(b, None, ctx), daemon=True)
    th.start()
    a.sendall(b'{"cmd": "listen"}\n{"cmd": "ping"}\n')
    a.settimeout(20)
    buf = b""
    while buf.count(b"\n") < 2:
        chunk = a.recv(65536)
        if not chunk:
            break
        buf += chunk
    a.close()
    resp = [json.loads(x) for x in buf.split(b"\n") if x.strip()]
    out["daemon_listen"] = resp[0] if resp else None
    out["daemon_ping_after"] = resp[1] if len(resp) > 1 else None
    hs = _load(HERE / "health_snapshot.py", "fi_hs_mic")
    doc = hs.build_snapshot(staging, probes={"ears": hs.probe_ears}, ears_port=dead,
                            ears_unit="iris-fault-inject-no-such-unit", in_grace=False, voice_off=False)
    e = doc["subsystems"]["ears"]
    out["ears_status"], out["ears_error"] = e["status"], e["last_error"]
    rec = record_incident(staging, doc)
    out["incident"] = (rec["bundles"] or [{}])[0].get("path")
    out["passed"] = bool(
        "Connection" in out["refused"] and out["refused_in_s"] < 8
        and "Connection" in out["midstream"]
        and out["daemon_listen"] and out["daemon_listen"].get("ok") is False
        and (out["daemon_ping_after"] or {}).get("result", {}).get("pong") is True
        and out["ears_status"] == "red" and out["incident"])
    return out


CHILD = {"disk_full": sc_disk_full, "model_load": sc_model_load, "gpu_pressure": sc_gpu_pressure,
         "mic_gone": sc_mic_gone}


# ── parent ──────────────────────────────────────────────────────────────────
def run_scenario(name: str, keep: bool = False) -> dict:
    staging = Path(tempfile.mkdtemp(prefix="iris-fi-"))
    env = dict(os.environ, IRIS_ROLE="staging", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", IRIS_EVENTS_DB=str(staging / "events.sqlite"),
               IRIS_WORKER_DB=str(staging / "jobs.sqlite"), IRIS_INCIDENTS_DIR=str(staging / "state" / "incidents"),
               IRIS_INCIDENTS_FALLBACK=str(staging / "fallback"), IRIS_INCIDENT_CAPTURE="1",
               IRIS_FI_REPO=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    env.pop("IRIS_MIC_SOURCE", None)
    cmd = [PY, str(Path(__file__).resolve()), "_child", name, str(staging)]
    if name == "disk_full":
        if not shutil.which("unshare"):
            return {"scenario": name, "passed": False, "skipped": "unshare not installed"}
        env["IRIS_FI_IN_NS"] = "1"
        cmd = ["unshare", "-rm", *cmd]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S, cwd=str(staging))
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{\"fi_result\"")]
        res = json.loads(lines[-1])["fi_result"] if lines else {"passed": False}
        res["crashed"] = r.returncode != 0 or not lines
        if res["crashed"]:
            res["passed"] = False
            res["stderr_tail"] = r.stderr[-1200:]
    except subprocess.TimeoutExpired:
        res = {"passed": False, "crashed": True, "why": f"child timed out after {CHILD_TIMEOUT_S}s"}
    res.update(scenario=name, took_s=round(time.monotonic() - t0, 1), staging=str(staging))
    if not keep:
        shutil.rmtree(staging, ignore_errors=True)
        res["staging"] += " (removed)"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Iris sandbox fault injection (V3 Phase 4)")
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("scenario", choices=SCENARIOS + ("all",))
    r.add_argument("--keep", action="store_true", help="keep the staging dir (bundles) for inspection")
    r.add_argument("--json", action="store_true")
    ch = sub.add_parser("_child")
    ch.add_argument("scenario", choices=SCENARIOS)
    ch.add_argument("staging")
    a = ap.parse_args(argv)
    if a.op == "list":
        for s in SCENARIOS:
            print(s)
        return 0
    if a.op == "_child":
        staging = Path(a.staging)
        assert_staging(staging)
        res = CHILD[a.scenario](staging)
        print(json.dumps({"fi_result": res}, default=str), flush=True)
        return 0
    names = SCENARIOS if a.scenario == "all" else (a.scenario,)
    results = [run_scenario(n, keep=a.keep) for n in names]
    if a.json:
        print(json.dumps(results, indent=1, default=str))
    else:
        for res in results:
            print(f"{'PASS' if res.get('passed') else 'FAIL'}  {res['scenario']:13} {res.get('took_s')}s"
                  f"{'  CRASHED' if res.get('crashed') else ''}  incident={res.get('incident')}")
    return 0 if all(r.get("passed") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
