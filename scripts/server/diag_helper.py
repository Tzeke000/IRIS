#!/usr/bin/env python3
"""scripts/server/diag_helper.py - the dormant diagnostic helper's WAKE GATE (V3 plan Phase 4 step 4.2, 2026-11-04).

Before the (not yet chosen - step 4.3 is Zeke's) small diagnostic model may be loaded, `wake` asks whether this is
a moment the GPU and the house can spare. It loads NOTHING itself; it only answers, and logs the answer.

    check            source                                           a "no" means
    park             state/little_brain/pilot_deliberately_off.json   cpu_only  (parsed: {"off": true} = parked;
                                                                                 absent or off:false = not parked;
                                                                                 unreadable = parked, fail-closed)
    quiet_hours      state/voice_quiet_hours.json window (22:30-05:10 defer      (the house is asleep)
                     default) OR state/voice_deliberately_off.json
                     with off != false
    speaking         scratch/voice_status.json state=speaking, fresh   defer      (never compete with my own mouth)
                     (< SPEAK_STALE_S) OR scratch/voice_amplitude.json
                     written < AMP_FRESH_S ago with amp > 0
    ram              /proc/meminfo MemAvailable >= need + RAM_RESERVE  defer
    gpu              gpu_admit.admit(model + KV) - its 1 GiB reserve   cpu_only  (fail-CLOSED here: no nvidia-smi =
                                                                                 cpu_only, unlike the ears' fail-open)

    verdict: any defer -> "defer" (exit 4, with retry_after_s) ; else any cpu_only -> "cpu_only" (exit 3: BM25
             retrieval via diag_corpus.py still runs, no model) ; else "gpu" (exit 0). Every check is evaluated
             every time so the log shows the whole picture, and every decision is appended to
             state/health/diag_wake.jsonl.

    diag_helper.py wake [--model-mib N] [--kv-mib N] [--ram-mib N] [--json]

The model sizes default to the PROPOSED budget from the plan (<= ~4 GB weights + 1 GiB KV) until 4.3 picks a
model; pass the real numbers then. Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("IRIS_ROOT") or Path(__file__).resolve().parents[2])

MODEL_MIB_DEFAULT = 4096     # PROPOSED (plan 4.3: "<= ~4 GB VRAM"), replace once Zeke picks the model
KV_MIB_DEFAULT = 1024
RAM_MIB_DEFAULT = 2048       # what the helper process itself may take in RAM (weights are mmapped by Ollama too)
RAM_RESERVE_MIB = 4096       # never leave the box with less than this available
SPEAK_STALE_S = 120.0        # a "speaking" status older than this is a crashed mouth's leftover, not speech
AMP_FRESH_S = 3.0
RETRY = {"speaking": 30, "ram": 300, "quiet_hours": 1800}

DEFER, CPU = "defer", "cpu_only"


def _paths(root: Path) -> dict:
    return {"park": root / "state" / "little_brain" / "pilot_deliberately_off.json",
            "quiet_cfg": root / "state" / "voice_quiet_hours.json",
            "voice_off": root / "state" / "voice_deliberately_off.json",
            "status": root / "scratch" / "voice_status.json",
            "amp": root / "scratch" / "voice_amplitude.json",
            "log": root / "state" / "health" / "diag_wake.jsonl"}


def _json(p: Path):
    """(exists, parsed-or-None)."""
    if not p.is_file():
        return False, None
    try:
        return True, json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, None


# ── the checks: each returns (pass, detail) ─────────────────────────────────
def check_park(p: dict) -> tuple[bool, str]:
    exists, d = _json(p["park"])
    if not exists:
        return True, "no park flag"
    if not isinstance(d, dict):
        return False, "park flag unreadable - treated as parked (fail-closed)"
    if d.get("off") is False:
        return True, "park flag present with off:false"
    return False, f"little brain parked ({str(d.get('reason', ''))[:80]})"


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def check_quiet(p: dict, now: _dt.datetime) -> tuple[bool, str]:
    exists, d = _json(p["voice_off"])
    if exists and not (isinstance(d, dict) and d.get("off") is False):
        return False, "voice_deliberately_off.json is set"
    cfg = {"enabled": True, "start": "22:30", "end": "05:10"}
    _, c = _json(p["quiet_cfg"])
    if isinstance(c, dict):
        cfg.update({k: c[k] for k in ("enabled", "start", "end") if k in c})
    if not cfg["enabled"]:
        return True, "quiet hours disabled"
    try:
        start, end = _hm(cfg["start"]), _hm(cfg["end"])
    except (ValueError, AttributeError):
        start, end = _hm("22:30"), _hm("05:10")
    t = now.hour * 60 + now.minute
    inside = (start <= t or t < end) if start > end else (start <= t < end)
    if inside:
        return False, f"inside quiet hours {cfg['start']}-{cfg['end']} (now {now:%H:%M})"
    return True, f"outside quiet hours {cfg['start']}-{cfg['end']}"


def check_speaking(p: dict, now_ts: float) -> tuple[bool, str]:
    _, st = _json(p["status"])
    if isinstance(st, dict) and st.get("state") == "speaking":
        age = now_ts - float(st.get("ts") or 0)
        if age < SPEAK_STALE_S:
            return False, f"mouth is speaking (status {age:.0f}s old)"
    _, a = _json(p["amp"])          # {"amp": 0-1, "ts": ...} from wren_voice_status.set_amplitude
    if isinstance(a, dict):
        try:
            age, amp = now_ts - float(a.get("ts") or 0), float(a.get("amp") or 0)
        except (TypeError, ValueError):
            age, amp = 1e9, 0.0
        if age < AMP_FRESH_S and amp > 0:
            return False, f"audio playing (amplitude {amp:.2f}, {age:.1f}s ago)"
    return True, "not speaking"


def read_mem_available_mib() -> int | None:
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def check_ram(need_mib: int, mem_fn) -> tuple[bool, str]:
    avail = mem_fn()
    if avail is None:
        return False, "MemAvailable unreadable (fail-closed)"
    if avail < need_mib + RAM_RESERVE_MIB:
        return False, f"RAM available {avail} MiB < {need_mib} + {RAM_RESERVE_MIB} reserve"
    return True, f"RAM available {avail} MiB"


def check_gpu(need_mib: int, admit_fn) -> tuple[bool, str, dict]:
    ok, info = admit_fn(need_mib, who="diag_helper")
    if str(info.get("reason", "")).startswith("no nvidia-smi"):
        return False, "no nvidia-smi - can't verify VRAM, CPU only (fail-closed)", info
    if not ok:
        return False, (f"gpu_admit refused {need_mib} MiB (free {info.get('free_mib')}, "
                       f"reserve {info.get('reserve_mib')})"), info
    return True, f"gpu_admit ok: {need_mib} MiB of {info.get('free_mib')} free", info


# ── the gate ────────────────────────────────────────────────────────────────
def _default_admit(need_mib: int, who: str):
    sys.path.insert(0, str(HERE))
    from gpu_admit import admit  # type: ignore
    return admit(need_mib, who=who)


def wake(model_mib: int = MODEL_MIB_DEFAULT, kv_mib: int = KV_MIB_DEFAULT, ram_mib: int = RAM_MIB_DEFAULT, *,
         root: Path | None = None, now: _dt.datetime | None = None, admit_fn=None, mem_fn=None,
         log: bool = True) -> dict:
    p = _paths(root or ROOT)
    now = now or _dt.datetime.now()
    now_ts = now.timestamp()
    need = model_mib + kv_mib
    checks, refusals = {}, []

    def note(name: str, ok: bool, detail: str, on_fail: str) -> None:
        checks[name] = {"ok": ok, "detail": detail}
        if not ok:
            refusals.append({"check": name, "action": on_fail, "detail": detail})

    note("park", *check_park(p), CPU)
    note("quiet_hours", *check_quiet(p, now), DEFER)
    note("speaking", *check_speaking(p, now_ts), DEFER)
    note("ram", *check_ram(ram_mib, mem_fn or read_mem_available_mib), DEFER)
    g_ok, g_detail, g_info = check_gpu(need, admit_fn or _default_admit)
    note("gpu", g_ok, g_detail, CPU)
    checks["gpu"]["admit"] = {k: g_info.get(k) for k in ("free_mib", "total_mib", "reserve_mib", "ok", "reason")
                              if k in g_info}

    defers = [r for r in refusals if r["action"] == DEFER]
    verdict = DEFER if defers else (CPU if refusals else "gpu")
    rec = {"ts": now_ts, "iso": now.isoformat(timespec="seconds"), "verdict": verdict,
           "need": {"model_mib": model_mib, "kv_mib": kv_mib, "gpu_mib": need, "ram_mib": ram_mib},
           "refusals": refusals, "checks": checks}
    if defers:
        rec["retry_after_s"] = max(RETRY.get(r["check"], 300) for r in defers)
    if log:
        try:
            p["log"].parent.mkdir(parents=True, exist_ok=True)
            with p["log"].open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            rec["log_error"] = True
    return rec


EXIT = {"gpu": 0, CPU: 3, DEFER: 4}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dormant diagnostic helper (V3 4.2: wake gate only)")
    sub = ap.add_subparsers(dest="op", required=True)
    w = sub.add_parser("wake", help="may the diagnostic model load now? (loads nothing)")
    w.add_argument("--model-mib", type=int, default=MODEL_MIB_DEFAULT)
    w.add_argument("--kv-mib", type=int, default=KV_MIB_DEFAULT)
    w.add_argument("--ram-mib", type=int, default=RAM_MIB_DEFAULT)
    w.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rec = wake(a.model_mib, a.kv_mib, a.ram_mib)
    if a.json:
        print(json.dumps(rec, indent=1))
    else:
        print(f"verdict: {rec['verdict']}" + (f" (retry in {rec['retry_after_s']}s)" if "retry_after_s" in rec else ""))
        for name, c in rec["checks"].items():
            print(f"  {'ok ' if c['ok'] else 'NO '} {name:12} {c['detail']}")
    return EXIT[rec["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
