#!/usr/bin/env python3
"""scripts/server/vr_players.py — "how many players?" for the VR temple (Zeke 11-03 docket: downloads / whitelisted / online).

  downloads    GitHub release asset download_count for every Setup .exe (LUMEN-Setup.exe; older VRIris-Setup.exe still counted) (token ~/.config/iris/secrets/github_friends_token)
  whitelisted  invites on the temple (server.cfg [invites] names) + approved tunnel peers on wg-gate (wg show peers)
  online       (`vr_players.py online [--json]` = only this; exit 3 = UNKNOWN) players the temple welcomed and hasn't seen leave since its last start (journal '[net] welcome' / 'left')
  perf         (11-04) pull the opt-in performance reports (perf_reports.jsonl [+ .1]) off the temple into
               state/vr_temple/perf/ and print harness/tools/perf_table.py's "how it runs on what" table
Read-only everywhere. Prints JSON (--json) or a short human summary. Never prints keys, IPs or invite codes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HOME = Path.home()
REPO = "Tzeke000/vr-iris-friends"


def _ssh(target: list[str], cmd: str, timeout: float = 20) -> str:
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *target, cmd],
                              capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return ""


def downloads() -> dict:
    try:
        tok = (HOME / ".config/iris/secrets/github_friends_token").read_text().strip()
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases",
                                     headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"})
        rel = json.load(urllib.request.urlopen(req, timeout=15))
        per = {r["tag_name"]: sum(a["download_count"] for a in r.get("assets", []) if a["name"].endswith(".exe")) for r in rel}
        return {"total": sum(per.values()), "per_release": per}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def temple_host() -> str:
    try:
        for ln in (HOME / ".iris_private.env").read_text().splitlines():
            if ln.startswith("TEMPLE_HOST="):
                return ln.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def whitelisted(th: str) -> dict:
    out: dict = {}
    cfg = _ssh(["-i", str(HOME / ".ssh/id_ed25519_temple"), f"root@{th}"],
               f'sed -n "/^\\[invites\\]/,/^\\[/p" "{_UD_SH}/server.cfg"') if th else ""
    names = set()
    for ln in cfg.splitlines():
        for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', ln):
            names.add(m.group(1))
    out["invites"] = sorted(names) if names else None
    peers = _ssh(["wg-gate"], "wg show wg0 peers | wc -l").strip()
    out["tunnel_peers"] = int(peers) if peers.isdigit() else None
    return out


def online(th: str) -> dict:
    if not th:
        return {"error": "no TEMPLE_HOST"}
    log = _ssh(["-i", str(HOME / ".ssh/id_ed25519_temple"), f"root@{th}"],
               "journalctl -u vr-temple -b --no-pager -o cat --since \"$(systemctl show vr-temple -p ActiveEnterTimestamp --value)\" "
               "| grep -E '^\\[net\\] (welcome|.* left$)'; echo __VRP_OK__", timeout=30)
    return parse_online(log)


def parse_online(log: str) -> dict:
    """opsguard 11-08: an ssh that FAILED used to read as an empty journal = "0 online" - the one answer a deploy gate
    must never invent. The command now ends with a marker; no marker = unknown (an error, never a count)."""
    if "__VRP_OK__" not in log:
        return {"error": "temple unreachable (ssh failed) - roster UNKNOWN"}
    here: dict[str, str] = {}
    for ln in log.splitlines():
        m = re.match(r"\[net\] welcome \d+ = (.+?) \((\w+)\)", ln)
        if m:
            here[m.group(1)] = m.group(2)
            continue
        m = re.match(r"\[net\] (.+) left$", ln)
        if m:
            here.pop(m.group(1), None)
    humans = {n: k for n, k in here.items() if k != "bot"}
    return {"count": len(humans), "players": humans, "bots": sorted(n for n, k in here.items() if k == "bot")}


# LUMEN rename 11-07: the server copies "VR Iris" -> "LUMEN" on its first renamed start; read LUMEN when it exists
TEMPLE_USERDATA_BASE = "/srv/vrtemple/home/.local/share/godot/app_userdata"
TEMPLE_USERDATA = TEMPLE_USERDATA_BASE + "/VR Iris"   # legacy constant (pre-rename)
_UD_SH = f'$( [ -d "{TEMPLE_USERDATA_BASE}/LUMEN" ] && echo "{TEMPLE_USERDATA_BASE}/LUMEN" || echo "{TEMPLE_USERDATA_BASE}/VR Iris" )'
PERF_TABLE = Path("/data/projects/vr_iris/harness/tools/perf_table.py")


def perf(th: str, out_dir: Path | None = None) -> dict:
    """Copy the temple's perf_reports*.jsonl here (read-only on the temple) and render the table."""
    out_dir = out_dir or Path(__file__).resolve().parents[2] / "state" / "vr_temple" / "perf"
    if not th:
        return {"error": "no TEMPLE_HOST"}
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for name in ("perf_reports.jsonl", "perf_reports.1.jsonl"):
        body = _ssh(["-i", str(HOME / ".ssh/id_ed25519_temple"), f"root@{th}"],
                    f'cat "{_UD_SH}/{name}" 2>/dev/null', timeout=30)
        if body.strip():
            (out_dir / name).write_text(body)
            got.append(str(out_dir / name))
    if not got:
        return {"files": [], "reports": 0, "table": "(no reports on the temple yet — nobody opted in, or perf-1104 isn't deployed)"}
    n = sum(1 for f in got for line in open(f) if line.strip())
    table = ""
    if PERF_TABLE.is_file():
        r = subprocess.run([sys.executable, "-I", str(PERF_TABLE), *got], capture_output=True, text=True, timeout=60)
        table = r.stdout.strip() or r.stderr.strip()[:300]
    return {"files": got, "reports": n, "table": table}


def main(argv: list[str]) -> int:
    th = temple_host()
    if argv[:1] == ["online"]:   # opsguard 11-08: just the roster, fast (vr.sh safe's fallback when my VR body is off)
        on = online(th)
        print(json.dumps(on) if "--json" in argv else
              f"online now: {on.get('count', '?')} {on.get('players', on.get('error', ''))}")
        return 0 if "count" in on else 3
    if argv[:1] == ["perf"]:
        d = perf(th)
        print(json.dumps(d, indent=1) if "--json" in argv else
              f"perf reports: {d.get('reports', '?')} ({', '.join(d.get('files', [])) or d.get('error', 'none')})\n{d.get('table', '')}")
        return 0
    d = {"downloads": downloads(), "whitelisted": whitelisted(th), "online": online(th)}
    if "--json" in argv:
        print(json.dumps(d, indent=1))
        return 0
    dl, wl, on = d["downloads"], d["whitelisted"], d["online"]
    print(f"downloads: {dl.get('total', '?')} {dl.get('per_release', dl.get('error', ''))}")
    print(f"whitelisted: invites {len(wl['invites']) if wl.get('invites') is not None else '?'}"
          f"{' (' + ', '.join(wl['invites']) + ')' if wl.get('invites') else ''} · tunnel peers {wl.get('tunnel_peers', '?')}")
    print(f"online now: {on.get('count', '?')} {on.get('players', on.get('error', ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
