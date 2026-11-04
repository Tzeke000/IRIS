#!/usr/bin/env python3
"""scripts/server/vr_players.py — "how many players?" for the VR temple (Zeke 11-03 docket: downloads / whitelisted / online).

  downloads    GitHub release asset download_count for every VRIris-Setup.exe (token ~/.config/iris/secrets/github_friends_token)
  whitelisted  invites on the temple (server.cfg [invites] names) + approved tunnel peers on wg-gate (wg show peers)
  online       players the temple welcomed and hasn't seen leave since its last start (journal '[net] welcome' / 'left')
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
               'sed -n "/^\\[invites\\]/,/^\\[/p" "/srv/vrtemple/home/.local/share/godot/app_userdata/VR Iris/server.cfg"') if th else ""
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
               "| grep -E '^\\[net\\] (welcome|.* left$)'", timeout=30)
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


def main(argv: list[str]) -> int:
    th = temple_host()
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
