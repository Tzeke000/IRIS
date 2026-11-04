#!/usr/bin/env python3
"""scripts/server/gpu_admit.py — load-time GPU admission for the one V100 (11-03, V3 spec / doc audit "GPU checks").

Before a model loads, ask: is there NEED MiB free plus a reserve, so this load can't squeeze the mouth/ears
that are already resident? Usable two ways:
  CLI   gpu_admit.py NEED_MIB [--who NAME] [--wait S]   exit 0 = admitted, 1 = refused, 2 = no GPU/nvidia-smi
        (as a unit's ExecStartPre: a refused load fails the START loudly instead of OOM-ing a neighbour)
        gpu_admit.py NEED_MIB --core [--max-wait S] [--exec -- CMD ...]   CORE units (mouth, ears; V3 3.8, 11-04)
        never fail: a refusal logs WHO holds the GPU, backs off (5 s doubling, cap 60 s) and retries; past
        --max-wait (default 20 min) it starts ANYWAY, logged "forced" — a core voice is never kept down by its
        own gate. Always exit 0; with --exec it replaces itself with CMD (a launcher pre-step, so the unit is
        "active" while it waits and `systemctl start` never blocks on it).
  py    from gpu_admit import admit; ok, info = admit(1200, who="whisper")
Every decision is appended to state/health/gpu_admission.jsonl. Fail-OPEN when nvidia-smi itself is missing
(exit 2 / ok=True with reason) so a broken probe never blocks the ears from starting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = Path(os.environ.get("IRIS_GPU_ADMIT_LOG") or ROOT / "state" / "health" / "gpu_admission.jsonl")
RESERVE_MIB = 1024          # always leave this much free for activations / spikes of what's already resident
CORE_MAX_WAIT_S = 1200.0    # a core unit waits at most this long for room, then starts anyway (logged "forced")
CORE_BACKOFF = (5.0, 60.0)  # first retry after 5 s, doubling, capped at 60 s


def free_mib() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6).stdout.strip().splitlines()[0]
        used, total = (int(float(x)) for x in out.split(","))
        return total - used, total
    except Exception:
        return None


def gpu_holders(limit: int = 5) -> list[dict]:
    """Who holds the GPU right now (biggest first) — named in every core refusal so the log says WHY."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return []
    rows = []
    for ln in out.strip().splitlines():
        try:
            pid, mib = (int(float(x)) for x in ln.split(","))
        except ValueError:
            continue
        name = "?"
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            scripts = [a for a in argv if a.endswith(b".py")]
            name = Path((scripts or argv)[0].decode(errors="replace")).name or "?"
        except OSError:
            pass
        rows.append({"pid": pid, "mib": mib, "name": name})
    return sorted(rows, key=lambda r: -r["mib"])[:limit]


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)     # stderr -> the unit's journal


def _log(rec: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def admit(need_mib: int, who: str = "?", wait_s: float = 0.0, reserve_mib: int = RESERVE_MIB) -> tuple[bool, dict]:
    deadline = time.time() + max(0.0, wait_s)
    while True:
        fm = free_mib()
        if fm is None:
            info = {"ts": time.time(), "who": who, "need_mib": need_mib, "ok": True, "reason": "no nvidia-smi (fail-open)"}
            _log(info)
            return True, info
        free, total = fm
        ok = free >= need_mib + reserve_mib
        if ok or time.time() >= deadline:
            info = {"ts": time.time(), "who": who, "need_mib": need_mib, "reserve_mib": reserve_mib,
                    "free_mib": free, "total_mib": total, "ok": ok}
            _log(info)
            return ok, info
        time.sleep(2.0)


def admit_core(need_mib: int, who: str = "?", max_wait_s: float = CORE_MAX_WAIT_S,
               reserve_mib: int = RESERVE_MIB, backoff: tuple[float, float] = CORE_BACKOFF,
               sleep=time.sleep, clock=time.monotonic) -> dict:
    """Admission for a CORE unit: wait + retry with bounded backoff, log every refusal, never refuse for good.
    Returns the final record; its "ok" is always True ("forced" / "reason" say how it got there)."""
    if os.environ.get("IRIS_GPU_ADMIT", "1") == "0":
        info = {"ts": time.time(), "who": who, "need_mib": need_mib, "ok": True, "core": True,
                "reason": "IRIS_GPU_ADMIT=0 (gate disabled)"}
        _log(info); _say(f"gpu_admit[{who}]: gate disabled by IRIS_GPU_ADMIT=0 - starting")
        return info
    t0 = clock()
    delay, cap = backoff
    attempt = 0
    while True:
        attempt += 1
        fm = free_mib()
        waited = round(clock() - t0, 1)
        if fm is None:
            info = {"ts": time.time(), "who": who, "need_mib": need_mib, "ok": True, "core": True,
                    "attempt": attempt, "waited_s": waited, "reason": "no nvidia-smi (fail-open)"}
            _log(info)
            _say(f"gpu_admit[{who}]: WARNING nvidia-smi missing or failing - cannot check VRAM, starting anyway (fail-open)")
            return info
        free, total = fm
        base = {"ts": time.time(), "who": who, "need_mib": need_mib, "reserve_mib": reserve_mib, "free_mib": free,
                "total_mib": total, "core": True, "attempt": attempt, "waited_s": waited}
        if free >= need_mib + reserve_mib:
            info = {**base, "ok": True}
            _log(info)
            _say(f"gpu_admit[{who}]: admitted - need {need_mib}+{reserve_mib} MiB, free {free}/{total} MiB"
                 + (f" (after {attempt - 1} refusals, {waited:.0f} s)" if attempt > 1 else ""))
            return info
        holders = gpu_holders()
        if waited >= max_wait_s:
            info = {**base, "ok": True, "forced": True, "holders": holders,
                    "reason": f"still short after {waited:.0f} s - starting anyway (core unit, may OOM)"}
            _log(info)
            _say(f"gpu_admit[{who}]: FORCED START after {waited:.0f} s - need {need_mib}+{reserve_mib} MiB, "
                 f"only {free} MiB free; GPU held by {_fmt(holders)}. This load may hit CUDA OOM.")
            return info
        wait = min(delay, cap, max(0.0, max_wait_s - waited))
        _log({**base, "ok": False, "retry_in_s": wait, "holders": holders})
        _say(f"gpu_admit[{who}]: REFUSED #{attempt} - need {need_mib}+{reserve_mib} MiB, free {free}/{total} MiB; "
             f"GPU held by {_fmt(holders)}; retry in {wait:.0f} s (forced start in {max(0.0, max_wait_s - waited):.0f} s)")
        sleep(wait)
        delay = min(delay * 2, cap)


def _fmt(holders: list[dict]) -> str:
    return ", ".join(f"{h['name']}[{h['pid']}] {h['mib']} MiB" for h in holders) or "unknown"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("need_mib", type=int)
    ap.add_argument("--who", default="cli")
    ap.add_argument("--wait", type=float, default=0.0, help="keep re-checking up to S seconds")
    ap.add_argument("--reserve", type=int, default=RESERVE_MIB)
    ap.add_argument("--core", action="store_true", help="core unit: wait+retry with backoff, never refuse (exit 0)")
    ap.add_argument("--max-wait", type=float, default=CORE_MAX_WAIT_S, help="--core: start anyway after S seconds")
    ap.add_argument("--backoff", type=float, nargs=2, default=CORE_BACKOFF, metavar=("FIRST", "CAP"))
    cmd: list[str] = []
    if "--exec" in argv:                        # everything after --exec [--] is the command to become
        i = argv.index("--exec")
        argv, cmd = argv[:i], argv[i + 1:]
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        if not cmd:
            ap.error("--exec needs a command")
    a = ap.parse_args(argv)
    if cmd and not a.core:
        ap.error("--exec needs --core")
    if a.core:
        info = admit_core(a.need_mib, a.who, a.max_wait, a.reserve, tuple(a.backoff))
        print(json.dumps(info), flush=True)
        if cmd:
            os.execvp(cmd[0], cmd)
        return 0
    ok, info = admit(a.need_mib, a.who, a.wait, a.reserve)
    print(json.dumps(info))
    if info.get("reason", "").startswith("no nvidia-smi"):
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
