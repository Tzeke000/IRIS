"""vr_temple — ME in the VR temple (the Mineflayer-style body Zeke asked for, 2026-10-23).

My body there is a headless game client on iris-home (unit `iris-vr-presence`, user systemd) that
joins the temple server (Proxmox CT 110) with my own admin invite and listens on a LOCAL control
socket localhost:7790. This tool talks to that socket.

Actions:
  status                         is my body running + a state snapshot
  join | leave                   start / stop my body (the unit)
  state                          SEE: me, people (name, where, distance, facing me, avatar), props near, stage
  events                         FEEL: touches on my eye / hands since the last call
  move  to=[x,y,z]               glide there (I float)
  look  at=[x,y,z] | (none)      look at a point / stop
  hand  side=L|R to=[x,y,z] | rest=true
  grab  side=L|R prop=<id>       (hand must be within 0.6 m)   release side=L|R [throw=[vx,vy,vz]]
  speak text="…"                 SAY IT ALOUD in the world: my own mouth (:8769 /synth) renders it, my body streams it
                                 positionally (everyone near me hears it from my eye; it glows while I talk)
  hush                           stop mid-line
  color hex="#rrggbb" minutes=N  my glow (it follows my mood by itself; this overrides for N min)
  build op=spawn|edit|delete|copy|glue|split|unstroke|clear_strokes data={…}
                                 live building, e.g. op=spawn data={kind:"plank", at:[x,y,z], color:"#c8955a", material:"wood"}
                                 edit data={id, c?, m?, scale?:[x,y,z], fixed?}; glue data={a, b}. state → builds_near
  draw  points=[[x,y,z],…] color="#…" width=0.012   a stroke (air, or on a surface)
  say   text="…"                 speech bubble only (bubbles are OFF by default for players — Zeke: no captions)
  action name=stage_on|stage_off|drop_test|reset_cans      kick name=<player>
Coordinates: metres, temple centre = (0,0,0), +Y up, spawn ≈ (0,0,9) facing −Z.
"""
from __future__ import annotations

import json
import socket
import subprocess
from typing import Any

from tools.tool_registry import register_tool

_ADDR = ("localhost", 7790)
_UNIT = "iris-vr-presence.service"


def _send(req: dict[str, Any], timeout: float = 4.0) -> dict[str, Any]:
    with socket.create_connection(_ADDR, timeout=timeout) as s:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode() or "{}")


_VOICE_DIR = "/tmp/iris_vr_voice"


def _speak(text: str) -> dict[str, Any]:
    import os
    import time
    import urllib.request
    text = text.strip()[:400]
    if not text:
        return {"ok": False, "error": "no text"}
    os.makedirs(_VOICE_DIR, exist_ok=True)
    for f in os.listdir(_VOICE_DIR):                       # keep the folder small
        fp = os.path.join(_VOICE_DIR, f)
        if time.time() - os.path.getmtime(fp) > 600:
            os.remove(fp)
    req = urllib.request.Request("http://localhost:8769/synth", data=json.dumps({"text": text, "rate": 48000}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        wav = r.read()
    path = os.path.join(_VOICE_DIR, f"line_{int(time.time() * 1000)}.wav")
    with open(path, "wb") as f:
        f.write(wav)
    out = _send({"cmd": "voice", "path": path})
    out["wav_bytes"] = len(wav)
    return out


def _unit(cmd: str) -> str:
    r = subprocess.run(["systemctl", "--user", cmd, _UNIT], capture_output=True, text=True, timeout=20)
    return (r.stdout or r.stderr).strip()


def _vr_temple(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    p = dict(params or {})
    action = str(p.pop("action", "status")).strip().lower()
    try:
        if action in ("join", "leave"):
            _unit("start" if action == "join" else "stop")
            return {"ok": True, "unit": _UNIT, "active": _unit("is-active")}
        if action == "status":
            active = _unit("is-active")
            out: dict[str, Any] = {"ok": True, "unit_active": active}
            if active == "active":
                try:
                    out["state"] = _send({"cmd": "state"})
                except OSError as e:
                    out["socket_error"] = repr(e)
            return out
        if action == "speak":
            return _speak(str(p.get("text", "")))
        if action not in ("state", "events", "move", "look", "hand", "grab", "release", "say", "action", "kick", "omniscient", "hush", "color", "build", "draw"):
            return {"ok": False, "error": f"unknown action {action!r}", "help": __doc__}
        req = {"cmd": action, **p}
        return _send(req)
    except OSError as e:
        return {"ok": False, "error": f"my temple body isn't reachable ({e!r}) — vr_temple action=join first"}


register_tool(
    "vr_temple",
    "MY BODY in the VR temple (Zeke 10-23). actions: status | join | leave | state (SEE: people/props/stage) | "
    "events (FEEL: touches) | move to=[x,y,z] (I float) | look at=[x,y,z] | hand side=L|R to=[x,y,z]|rest=true | "
    "grab side prop=<id> | release side [throw] | speak text (ALOUD, my voice, positional) | hush | say text (bubble only) | action name=stage_on|stage_off|drop_test|reset_cans|make_day|make_night | kick name | omniscient on=true|false (see everything vs field of view).",
    1,
    _vr_temple,
)
