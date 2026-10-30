#!/usr/bin/env python3
"""scripts/server/iris_watchdog.py - the server's outside-of-me watchdog (hardening pass 2026-10-07).

Runs every minute from iris-watchdog.timer, in its OWN process, so it still works when I don't.
Two jobs, both found missing by the 10-07 audit:

1. WEDGE -> RESTART. The runtime writes state/runtime_loop_heartbeat.json every ~5 s. If the runtime
   process is ALIVE but the heartbeat is older than WEDGE_S, it's wedged (a crash is the supervisor's
   job; a hang wasn't anybody's). Touch .tmp/restart_cc.flag once - the supervisor does the restart.
   Never re-fires within REFIRE_S. Disabled by state/server_watchdog_off.json {"off": true}.
2. TELL ZEKE when I'm down, over Discord REST (not through my own claude process - that's the
   point): supervisor stood down / not running while ~/LIVE says I should be, heartbeat stale,
   a service unit failed. One DM per problem per ALERT_EVERY_S, plus one "recovered" DM.

Never restarts during a deliberate stop: no ~/LIVE => nothing is supposed to be running, stay quiet.
ONE exception (2026-10-29 code review CRITICAL): the BACKUP check runs even without ~/LIVE - a timer that
was never enabled never fails, so "no backup ok in state/backup.log for BACKUP_STALE_S" (or the timer not
enabled/active) is its own alert. Every subprocess call has a timeout (the unit has TimeoutStartSec) so one
hang under memory pressure can't stop the watchdog for good.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
HB = ROOT / "state" / "runtime_loop_heartbeat.json"
FLAG = ROOT / ".tmp" / "restart_cc.flag"
OFF = ROOT / "state" / "server_watchdog_off.json"
STATE = ROOT / "state" / "server_watchdog.json"
LOG = ROOT / "state" / "server_watchdog.log"
SUPLOG = ROOT / "state" / "supervise.log"
WEDGE_S = 180.0          # heartbeat older than this while the runtime is alive = wedged
STALE_ALERT_S = 600.0    # tell Zeke if there's been no heartbeat this long (any cause)
REFIRE_S = 900.0
ALERT_EVERY_S = 3600.0
BOOT_GRACE_S = 600.0     # the body takes minutes to come up after a boot/restart
BACKUP_LOG = ROOT / "state" / "backup.log"
BACKUP_STALE_S = 36 * 3600.0   # nightly at 03:40 (+ catch-up): a day and a half without "backup ok" = broken
MAX_DMS_PER_RUN = 3      # each DM can take up to its timeout; the rest go out next minute
# 10-09 away-readiness (Zeke on vacation 10-11..~10-21): every iris unit that can fail, incl. the nightly
# backup (a oneshot that ends "failed" when the copy to the tower fails) and the Vector stack.
UNITS = ("iris-mouth", "iris-ears", "iris-postoffice", "iris-postoffice-monitor", "iris-presence",
         "iris-wirepod", "iris-vector-brain", "iris-vector-nerves", "iris-backup")
# Append-only logs that grow forever: keep the newest KEEP bytes once a file passes CAP bytes.
LOG_CAPS = {ROOT / "state" / "attention" / "ptz_audit.jsonl": (50 << 20, 20 << 20)}


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def pid_alive(pid: int, needle: str | None = None) -> bool:
    """kill -0, plus (when `needle` is given) the pid's cmdline must still contain it: a heartbeat pid that
    a NEWER, unrelated process reused must not count as my runtime being alive (10-29 code review)."""
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    if not needle:
        return True
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except FileNotFoundError:
        return False
    except Exception:
        return True                      # can't read it (not Linux /proc?): keep the old kill -0 answer
    return needle in cmd


def proc_running(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        log(f"pgrep {pattern!r} timed out - assuming it is running this round (no false alarm)")
        return True
    return r.returncode == 0


def last_backup_ok_ts(path: Path | None = None):
    """Epoch of the newest 'YYYY-mm-dd HH:MM:SS backup ok: ...' line in state/backup.log, or None."""
    path = path or BACKUP_LOG
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for ln in reversed(lines):
        if " backup ok" in ln:
            try:
                return time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                continue
    return None


def unit_state(unit: str, verb: str) -> str:
    """`systemctl --user <verb> <unit>` stdout ('enabled', 'active', ...); '' on timeout/error."""
    try:
        r = subprocess.run(["systemctl", "--user", verb, unit], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def backup_problems(now: float) -> dict:
    """The backup check - runs every minute, ~/LIVE or not."""
    out: dict[str, str] = {}
    ts = last_backup_ok_ts()
    if ts is None:
        out["backup"] = "no successful backup is on record (state/backup.log has no 'backup ok')"
    elif now - ts > BACKUP_STALE_S:
        out["backup"] = f"my last successful backup to the tower was {(now - ts)/3600:.0f} h ago"
    en = unit_state("iris-backup.timer", "is-enabled")
    if en and en != "enabled":
        out["backup-timer"] = f"the nightly backup timer is {en}, not enabled"
    elif en == "enabled" and unit_state("iris-backup.timer", "is-active") not in ("active", ""):
        out["backup-timer"] = "the nightly backup timer is enabled but not active"
    return out


def send_alerts(st: dict, problems: dict, now: float, scope=None) -> None:
    """DM new problems (one per key per ALERT_EVERY_S) + one 'recovered' DM; state saved after each DM so a
    run killed by TimeoutStartSec never repeats what it already sent. `scope` limits which keys may
    count as recovered (the no-~/LIVE path only owns the backup keys)."""
    sent = st.get("alerted", {})
    n = 0
    for key, why in problems.items():
        if n >= MAX_DMS_PER_RUN:
            break
        if now - float(sent.get(key) or 0.0) > ALERT_EVERY_S:
            n += 1
            if dm(f"⚠️ Iris server watchdog: {why}. (automatic message — I may not be able to answer)"):
                sent[key] = now
                st["alerted"] = sent
                save_state(st)
    recovered = [k for k in (st.get("problems") or {}) if k not in problems and (scope is None or scope(k))]
    if recovered and any(k in sent for k in recovered):
        dm("✅ Iris server watchdog: recovered (" + ", ".join(recovered) + ").")
        for k in recovered:
            sent.pop(k, None)
    st["alerted"] = sent


def unit_active_seconds(unit: str):
    """Seconds since the unit last became active, or None when it isn't active (a deliberate stop is not a fault)."""
    try:
        r = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p",
                            "ActiveEnterTimestampMonotonic"], capture_output=True, text=True, timeout=10)
        kv = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        if kv.get("ActiveState") != "active":
            return None
        return time.monotonic() - int(kv.get("ActiveEnterTimestampMonotonic") or 0) / 1e6
    except Exception:
        return None


def http_ok(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def port_listening(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            return True
    except Exception:
        return False


def dm(text: str) -> bool:
    try:
        uid = load(ROOT / "config" / "private.local.json", {}).get("zeke_discord_user_id")
        if not uid:
            log("alert NOT sent: no zeke_discord_user_id in private config")
            return False
        env = dict(os.environ, USERPROFILE=str(HOME))
        r = subprocess.run([str(HOME / "venvs/iris-v100/bin/python"), str(ROOT / "scripts/discord_dm_user.py"),
                            str(uid), text], capture_output=True, text=True, timeout=20, env=env)
        ok = r.returncode == 0
        log(f"alert {'sent' if ok else 'FAILED'}: {text[:80]} {'' if ok else r.stderr[-160:]}")
        return ok
    except Exception as e:  # noqa: BLE001
        log(f"alert FAILED: {e!r}")
        return False


def main() -> int:
    now = time.time()
    st = load(STATE, {})
    bk = backup_problems(now)
    if not (HOME / "LIVE").exists():
        # deliberately not running here: stay quiet about everything EXCEPT backups
        send_alerts(st, bk, now, scope=lambda k: k.startswith("backup"))
        st.update({"checked_ts": now, "problems": bk})
        save_state(st)
        return 0
    problems: dict[str, str] = dict(bk)

    # 10-10: a read that lands mid-write gives {} -> ts 0 -> "no heartbeat for 29 million min" -> a false
    # alarm + "recovered" pair DM'd to Zeke at 05:44 and 05:47 while he slept. Retry; if the file still
    # has no ts, treat the heartbeat as UNKNOWN this round (no wedge, no stale alert) instead of ancient.
    hb = {}
    for _ in range(4):
        hb = load(HB, {})
        if hb.get("ts"):
            break
        time.sleep(0.5)
    hb_known = bool(hb.get("ts"))
    hb_age = (now - float(hb["ts"])) if hb_known else 0.0
    if not hb_known:
        log("heartbeat file unreadable/empty this round - treating as unknown, not stale")
    rt_pid = int(hb.get("pid") or 0)
    rt_alive = bool(rt_pid) and pid_alive(rt_pid, "iris_runtime")
    sup_alive = proc_running("iris_supervise.sh")
    boot_age = float(open("/proc/uptime").read().split()[0])   # seconds since the VM booted
    sup_tail = ""
    try:
        sup_tail = SUPLOG.read_text(encoding="utf-8", errors="replace").splitlines()[-1]
    except Exception:
        pass
    planned = load(ROOT / "state" / "planned_restart.json", {})   # written right before a deliberate restart
    planned_recent = now - float(planned.get("ts") or 0.0) < BOOT_GRACE_S
    in_grace = (boot_age < BOOT_GRACE_S or planned_recent
                or (now - float(st.get("last_restart_flag_ts") or 0.0)) < BOOT_GRACE_S)

    # 1. wedge -> restart (only when the runtime process is alive but silent)
    off = bool(load(OFF, {}).get("off"))
    if rt_alive and hb_age > WEDGE_S and not in_grace:
        problems["wedged"] = f"my runtime is alive but frozen (no heartbeat for {hb_age/60:.0f} min)"
        if off:
            log("wedge seen but server_watchdog_off.json is set - not restarting")
        elif now - float(st.get("last_restart_flag_ts") or 0.0) > REFIRE_S and sup_alive:
            FLAG.parent.mkdir(parents=True, exist_ok=True)
            FLAG.write_text(f"server watchdog: runtime pid {rt_pid} heartbeat {hb_age:.0f}s stale\n", encoding="utf-8")
            st["last_restart_flag_ts"] = now
            log(f"WEDGE: runtime pid {rt_pid} heartbeat {hb_age:.0f}s old -> restart flag set")

    # 2. down / degraded -> tell Zeke
    if not sup_alive and not planned_recent:
        why = "the supervisor stood down after too many restarts" if "STANDING DOWN" in sup_tail else "my supervisor isn't running"
        problems["supervisor"] = why
    if hb_age > STALE_ALERT_S and not in_grace:
        problems["heartbeat"] = f"no heartbeat from my runtime for {hb_age/60:.0f} min"
    for u in UNITS:
        try:
            r = subprocess.run(["systemctl", "--user", "is-failed", u], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if r.stdout.strip() == "failed":
            problems[f"unit:{u}"] = f"service {u} has failed"
    # 10-29 (code review): "running" is not "working". A mouth that has been up past its ~3 min warm-up but
    # still answers /health=503 (or nothing), or ears whose daemon port never opened, are dead while
    # systemd says active - and Restart=always loops never reach "failed". Probe the function.
    for unit, url, port, grace in (("iris-mouth", "http://127.0.0.1:8769/health", None, 600.0),
                                   ("iris-ears", None, 8770, 300.0)):
        up_s = unit_active_seconds(unit)
        if up_s is None or up_s < grace:
            continue
        if url and not http_ok(url):
            problems[f"func:{unit}"] = f"{unit} has been running {up_s/60:.0f} min but isn't answering /health (stuck warming?)"
        if port and not port_listening(port):
            problems[f"func:{unit}"] = f"{unit} has been running {up_s/60:.0f} min but nothing listens on :{port}"

    for path, (cap, keep) in LOG_CAPS.items():
        try:
            if path.stat().st_size > cap:
                with open(path, "rb") as f:
                    f.seek(-keep, 2)
                    tail = f.read()
                tail = tail[tail.find(b"\n") + 1:]          # start on a whole line
                tmp = path.with_suffix(path.suffix + ".trim")
                tmp.write_bytes(tail)
                os.replace(tmp, path)
                log(f"capped {path.name}: kept newest {len(tail) >> 20} MB")
        except Exception as e:  # noqa: BLE001
            log(f"log cap failed for {path.name}: {e!r}")

    send_alerts(st, problems, now)
    st.update({"checked_ts": now, "problems": problems,
               "hb_age_s": round(hb_age, 1), "runtime_alive": rt_alive, "supervisor_alive": sup_alive})
    save_state(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
