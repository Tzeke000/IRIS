"""app_status_board — ONE honest JSON for the app's main page (2026-10-07, Zeke).

Zeke: put where I'm running, the model, CPU, RAM, GPU, SSD (space + read/write
speed) and network along the top, the server/tower link stuff bottom-right —
"and then lets make sure it all good and cant false say things are good or vise
versa". The old Server-tab rows did exactly that: on the server backend the
Windows-only checks (schtasks / PowerShell) raised on Linux and showed the
desktop bridge + wake-on-LAN as broken while both were fine on the tower.

HONESTY RULES (the point of this module):
  * Every link status is a REAL round-trip, not a config read: the desktop
    bridge is pinged through itself, the tower is reached over SSH, the
    post-office port is connected to.
  * ok is tri-state: True (checked, works) / False (checked, broken) / None
    (not checked, or the last check is STALE). The UI shows None as grey —
    never green. A result older than 3x its interval is reported as None.
  * Numbers carry their own sample time; the UI can grey a stale bar.
  * Nothing is defaulted: a reading that failed is null, not 0.

Samplers run in daemon threads started on the first request (never at import).
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_STARTED = False
_SAMPLE: dict[str, Any] = {}
_LINKS: dict[str, dict[str, Any]] = {}
_MODEL_CACHE = {"ts": 0.0, "value": None}

_IS_WIN = sys.platform == "win32"
_NOWIN = 0x08000000 if _IS_WIN else 0
_TOWER_ALIAS = os.environ.get("IRIS_TOWER_SSH", "tower")   # ~/.ssh/config alias, no address in code
_TOWER_REPO = r"D:\Wren-Companion"


# ── small helpers ────────────────────────────────────────────────────────────
def _run(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=_NOWIN)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    except Exception as e:  # noqa: BLE001
        return 125, repr(e)[:160]


def _ssh_tower(remote: str, timeout: float = 20.0) -> tuple[int, str]:
    return _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 _TOWER_ALIAS, remote], timeout)


def _default_iface() -> str | None:
    try:
        with open("/proc/net/route", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) > 1 and parts[1] == "00000000":
                    return parts[0]
    except Exception:
        pass
    return None


def _root_block_device() -> str | None:
    """'sda' for a / on /dev/sda1 — the disk whose I/O counters we report."""
    try:
        import psutil
        for p in psutil.disk_partitions(all=False):
            if p.mountpoint == "/":
                dev = os.path.basename(p.device)
                base = dev.rstrip("0123456789")
                if base.endswith("p") and base[:-1].startswith("nvme"):
                    base = base[:-1]
                return base or dev
    except Exception:
        pass
    return None


def live_model() -> str | None:
    """The model the LIVE claude process was started with (its --model flag) —
    the truth, not the launcher's pin. Falls back to IRIS_MODEL, then None."""
    now = time.time()
    if _MODEL_CACHE["value"] and now - _MODEL_CACHE["ts"] < 30.0:
        return _MODEL_CACHE["value"]
    value = None
    try:
        import psutil
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name not in ("claude", "claude.exe"):
                    continue
                cmd = proc.info.get("cmdline") or []
                if "--model" in cmd:
                    i = cmd.index("--model")
                    if i + 1 < len(cmd):
                        value = str(cmd[i + 1]).strip()
                        break
            except Exception:
                continue
    except Exception:
        pass
    if not value:
        value = (os.environ.get("IRIS_MODEL") or "").strip() or None
    _MODEL_CACHE.update({"ts": now, "value": value})
    return value


# ── 1 Hz machine sampler ─────────────────────────────────────────────────────
def _gpu_read() -> list[dict[str, Any]] | None:
    rc, out = _run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
                    "memory.total,temperature.gpu", "--format=csv,noheader,nounits"], 5.0)
    if rc != 0:
        return None
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        try:
            gpus.append({"name": p[0], "util_pct": float(p[1]),
                         "mem_used_mb": float(p[2]), "mem_total_mb": float(p[3]),
                         "temp_c": float(p[4])})
        except ValueError:
            continue
    return gpus or None


def _sampler() -> None:
    import psutil
    disk_dev = _root_block_device()
    iface = _default_iface()
    prev_io = prev_net = None
    prev_t = time.time()
    psutil.cpu_percent(interval=None)            # prime
    n = 0
    while True:
        time.sleep(1.0)
        now = time.time()
        dt = max(1e-3, now - prev_t)
        s: dict[str, Any] = {"ts": now}
        try:
            per = psutil.cpu_percent(interval=None, percpu=True)
            s["cpu"] = {"percent": round(sum(per) / len(per), 1) if per else None,
                        "cores": len(per), "per_core": [round(x) for x in per]}
            try:
                temps = psutil.sensors_temperatures() or {}
                cands = temps.get("coretemp") or temps.get("k10temp") or []
                s["cpu"]["temp_c"] = round(max(t.current for t in cands), 1) if cands else None
            except Exception:
                s["cpu"]["temp_c"] = None
        except Exception:
            s["cpu"] = None
        try:
            vm = psutil.virtual_memory()
            s["ram"] = {"used_gb": round((vm.total - vm.available) / 2**30, 1),
                        "total_gb": round(vm.total / 2**30, 1), "percent": vm.percent}
        except Exception:
            s["ram"] = None
        try:
            path = "C:\\" if _IS_WIN else "/"
            du = psutil.disk_usage(path)
            disk = {"path": path, "device": disk_dev,
                    "used_gb": round(du.used / 2**30, 1),
                    "total_gb": round(du.total / 2**30, 1), "percent": du.percent,
                    "read_mb_s": None, "write_mb_s": None}
            io = psutil.disk_io_counters(perdisk=True) or {}
            cur = io.get(disk_dev) if disk_dev else None
            if cur is not None and prev_io is not None:
                disk["read_mb_s"] = round((cur.read_bytes - prev_io.read_bytes) / dt / 2**20, 2)
                disk["write_mb_s"] = round((cur.write_bytes - prev_io.write_bytes) / dt / 2**20, 2)
            prev_io = cur
            s["disk"] = disk
        except Exception:
            s["disk"] = None
        try:
            nio = psutil.net_io_counters(pernic=True) or {}
            cur = nio.get(iface) if iface else None
            net = {"iface": iface, "rx_mbps": None, "tx_mbps": None}
            if cur is not None and prev_net is not None:
                net["rx_mbps"] = round((cur.bytes_recv - prev_net.bytes_recv) * 8 / dt / 1e6, 2)
                net["tx_mbps"] = round((cur.bytes_sent - prev_net.bytes_sent) * 8 / dt / 1e6, 2)
            prev_net = cur
            s["net"] = net
        except Exception:
            s["net"] = None
        if n % 2 == 0:
            s["gpu"] = _gpu_read()
            s["gpu_ts"] = now
        else:
            with _LOCK:
                s["gpu"] = _SAMPLE.get("gpu")
                s["gpu_ts"] = _SAMPLE.get("gpu_ts")
        n += 1
        prev_t = now
        with _LOCK:
            _SAMPLE.clear()
            _SAMPLE.update(s)


# ── the server's two physical CPUs (Proxmox host, read-only forced-command key) ──
# Zeke 10-07: "the servers 2 CPUs please". The VM sees one virtual socket, so per-socket
# usage + package temps come from the HOST over SSH with ~/.ssh/id_ed25519_pve_stats, whose
# authorized_keys line FORCES a command that only prints /proc/stat, the cpu->package map and
# hwmon temps (scratch/pve_stats_key_line.txt). The host address comes from
# config/private.local.json at runtime. No key / no answer => host_cpus None + the reason.
_PVE_KEY = os.path.expanduser("~/.ssh/id_ed25519_pve_stats")
_HOST_CPUS: dict[str, Any] = {"value": None, "error": "not checked yet", "ts": 0.0}


def _pve_host() -> str | None:
    try:
        import json as _json
        from pathlib import Path as _P
        d = _json.loads((_P(__file__).resolve().parents[1] / "config" / "private.local.json")
                        .read_text(encoding="utf-8"))
        return str(d.get("proxmox_host") or "") or None
    except Exception:
        return None


def _parse_host(out: str):
    stat: dict[int, list[int]] = {}
    pkg: dict[int, int] = {}
    hw_name: dict[str, str] = {}
    labels: dict[str, str] = {}
    temps: dict[str, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("cpu") and not line.startswith("cpu "):
            parts = line.split()
            try:
                stat[int(parts[0][3:])] = [int(x) for x in parts[1:9]]
            except ValueError:
                continue
        elif "/topology/physical_package_id:" in line:
            path, _, val = line.rpartition(":")
            try:
                pkg[int(path.split("/cpu/cpu")[1].split("/")[0])] = int(val)
            except (ValueError, IndexError):
                continue
        elif line.startswith("/sys/class/hwmon/"):
            path, _, val = line.partition(":")
            hw = path.split("/")[4]
            leaf = path.rsplit("/", 1)[-1]
            if leaf == "name":
                hw_name[hw] = val.strip()
            elif leaf.endswith("_label"):
                labels[f"{hw}/{leaf[:-6]}"] = val.strip()
            elif leaf.endswith("_input"):
                try:
                    temps[f"{hw}/{leaf[:-6]}"] = int(val) / 1000.0
                except ValueError:
                    pass
    sock_temp: dict[int, float] = {}
    for key, lab in labels.items():
        hw = key.split("/")[0]
        if hw_name.get(hw) == "coretemp" and lab.lower().startswith("package id") and key in temps:
            try:
                sock_temp[int(lab.split()[-1])] = temps[key]
            except ValueError:
                pass
    return stat, pkg, sock_temp


def _host_cpu_loop() -> None:
    host = _pve_host()
    prev: dict[int, list[int]] | None = None
    while True:
        if not host or not os.path.isfile(_PVE_KEY):
            _HOST_CPUS.update({"value": None, "error": "no Proxmox stats key / host configured",
                               "ts": time.time()})
            time.sleep(30.0)
            host = _pve_host()
            continue
        rc, out = _run(["ssh", "-i", _PVE_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                        "-o", "IdentitiesOnly=yes",
                        "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/iris-pve-%r@%h",
                        "-o", "ControlPersist=120", f"root@{host}", "stats"], 10.0)
        if rc != 0 or "cpu0 " not in out:
            msg = "the key isn't installed on the Proxmox host yet" if "Permission denied" in out \
                else f"no answer (rc {rc}): {out[-100:]}"
            _HOST_CPUS.update({"value": None, "error": msg, "ts": time.time()})
            prev = None
            time.sleep(15.0)
            continue
        stat, pkg, sock_temp = _parse_host(out)
        if prev is not None and pkg:
            agg: dict[int, list[float]] = {}
            for cpu, cur in stat.items():
                old = prev.get(cpu)
                if old is None or cpu not in pkg:
                    continue
                d = [c - o for c, o in zip(cur, old)]
                total = sum(d)
                idle = d[3] + d[4]
                if total > 0:
                    agg.setdefault(pkg[cpu], []).append(100.0 * (total - idle) / total)
            sockets = sorted(set(pkg.values()))
            _HOST_CPUS.update({
                "value": [{"socket": s, "threads": sum(1 for v in pkg.values() if v == s),
                           "percent": round(sum(agg[s]) / len(agg[s]), 1) if agg.get(s) else None,
                           "temp_c": sock_temp.get(s)} for s in sockets],
                "error": None, "ts": time.time()})
        prev = stat
        time.sleep(2.0)


# ── link checks: real round-trips, tri-state ─────────────────────────────────
def _check_tower_ssh() -> tuple[bool | None, str]:
    rc, out = _ssh_tower("echo iris-ok", timeout=12.0)
    if rc == 0 and "iris-ok" in out:
        return True, "SSH answered"
    return False, f"SSH failed (rc {rc}): {out[-120:]}"


def _check_desktop_bridge() -> tuple[bool | None, str]:
    rc, out = _ssh_tower(
        rf"{_TOWER_REPO}\.venv\Scripts\python.exe {_TOWER_REPO}\scripts\desktop_bridge.py submit ping {{}}",
        timeout=40.0)
    if rc == 0 and '"ok": true' in out:
        return True, "ping went through his desktop session"
    if '"ok": false' in out:
        return False, f"bridge answered not-ok: {out[-140:]}"
    return False, f"no answer (rc {rc}): {out[-140:]}"


def _check_wol() -> tuple[bool | None, str]:
    ps = ("$a = Get-NetAdapter -Physical | Where-Object { $_.Status -eq 'Up' -and $_.MediaType -eq '802.3' }"
          " | Select-Object -First 1; if ($a) { $a.Name + '|' + (Get-NetAdapterPowerManagement -Name $a.Name)"
          ".WakeOnMagicPacket } else { 'NO-WIRED-UP' }")
    rc, out = _ssh_tower(f'powershell -NoProfile -Command "{ps}"', timeout=40.0)
    line = out.strip().splitlines()[-1] if out.strip() else ""
    if rc != 0 or not line:
        return None, f"could not read it (rc {rc}): {out[-120:]}"
    if line == "NO-WIRED-UP":
        return False, "no wired adapter is up (magic packets need the wired card)"
    name, _, val = line.partition("|")
    if val.strip() == "Enabled":
        return True, f"enabled on {name} (setting checked; a wake from OFF has not been tested)"
    return False, f"{name}: WakeOnMagicPacket = {val.strip() or '?'}"


def _check_post_office() -> tuple[bool | None, str]:
    url = os.environ.get("IRIS_POSTOFFICE_URL", "")
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        host, port = u.hostname, u.port or 80
    except Exception:
        host, port = None, None
    if not host:
        return None, "IRIS_POSTOFFICE_URL not set"
    try:
        with socket.create_connection((host, int(port)), timeout=3.0):
            return True, f"port {port} accepted a connection"
    except Exception as e:  # noqa: BLE001
        return False, f"port {port}: {e.__class__.__name__}"


def _check_unit(unit: str):
    def _f() -> tuple[bool | None, str]:
        rc, out = _run(["systemctl", "--user", "is-active", unit], 5.0)
        state = out.strip().splitlines()[-1] if out.strip() else "?"
        return (state == "active"), f"{unit}: {state}"
    return _f


_CHECKS: dict[str, dict[str, Any]] = {}
if not _IS_WIN:
    _CHECKS = {
        "tower_ssh":      {"label": "Server reaches the tower", "fn": _check_tower_ssh, "every": 30.0},
        "desktop_bridge": {"label": "Desktop bridge", "fn": _check_desktop_bridge, "every": 60.0},
        "wake_on_lan":    {"label": "Wake-on-LAN (tower)", "fn": _check_wol, "every": 600.0},
        "post_office":    {"label": "Post-office (letters)", "fn": _check_post_office, "every": 30.0},
        "mouth":          {"label": "Mouth service", "fn": _check_unit("iris-mouth"), "every": 15.0},
        "ears":           {"label": "Ears service", "fn": _check_unit("iris-ears"), "every": 15.0},
    }


def _link_loop(key: str, spec: dict[str, Any]) -> None:
    while True:
        t0 = time.time()
        try:
            ok, detail = spec["fn"]()
        except Exception as e:  # noqa: BLE001
            ok, detail = None, f"check crashed: {e!r}"[:160]
        with _LOCK:
            _LINKS[key] = {"label": spec["label"], "ok": ok, "detail": detail,
                           "checked_ts": time.time(), "took_s": round(time.time() - t0, 2),
                           "every_s": spec["every"]}
        time.sleep(spec["every"])


def _ensure_started() -> None:
    global _STARTED
    if _STARTED:
        return
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(target=_sampler, name="status_board_sampler", daemon=True).start()
    if not _IS_WIN:
        threading.Thread(target=_host_cpu_loop, name="status_board_host_cpus", daemon=True).start()
    for k, spec in _CHECKS.items():
        threading.Thread(target=_link_loop, args=(k, spec), name=f"status_board_{k}",
                         daemon=True).start()


def board(g: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_started()
    now = time.time()
    with _LOCK:
        sample = dict(_SAMPLE)
        links = {k: dict(v) for k, v in _LINKS.items()}
    # never report a stale check as a verdict
    for k, spec in _CHECKS.items():
        v = links.get(k)
        if v is None:
            links[k] = {"label": spec["label"], "ok": None, "detail": "first check still running",
                        "checked_ts": None, "every_s": spec["every"]}
            continue
        age = now - float(v.get("checked_ts") or 0.0)
        v["age_s"] = round(age, 1)
        if age > 3 * float(spec["every"]) + 45.0:
            v["ok"] = None
            v["detail"] = f"STALE — last check {age:.0f}s ago ({v.get('detail')})"
    sample_age = now - float(sample.get("ts") or 0.0) if sample else None
    fps = None
    if g is not None:
        try:
            fps = g.get("_capture_fps_measured")
        except Exception:
            fps = None
    return {
        "ok": True, "ts": now,
        "host": {"role": "tower" if _IS_WIN else "server", "hostname": socket.gethostname()},
        "model": live_model(),
        "sample_age_s": None if sample_age is None else round(sample_age, 1),
        "cpu": sample.get("cpu"), "ram": sample.get("ram"), "gpu": sample.get("gpu"),
        "disk": sample.get("disk"), "net": sample.get("net"),
        "host_cpus": _HOST_CPUS["value"] if now - float(_HOST_CPUS["ts"] or 0.0) < 30.0 else None,
        "host_cpus_error": _HOST_CPUS["error"],
        "camera_fps": fps,
        "links": links,
    }
