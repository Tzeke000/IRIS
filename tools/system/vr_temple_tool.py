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
  heard [n=20] [since=<epoch|ISO|"10m"|"90s">] [only_attending=true] [speaker=<name>]
                                 HEAR: what people said in the temple, one line per utterance, per speaker
                                 ({t_start, t_end, speaker, text, dist_m, facing, addressed, attending}); my ears =
                                 bot iris_hearing.gd → spool → scripts/vr/vr_hear.py (unit iris-vr-hear) → heard.jsonl.
                                 addressed = my name is in it; attending = that speaker said my name ≤ 90 s ago.
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


_HEARD = "/home/iris/IRIS/state/vr_temple/heard.jsonl"
_ATTN = "/home/iris/IRIS/state/vr_temple/hear_attention.json"


def _since(v: Any) -> float:
    import time
    import datetime as dt
    if v in (None, ""):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s[-1:] in ("s", "m", "h") and s[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(s[:-1]) * {"s": 1, "m": 60, "h": 3600}[s[-1]]
    try:
        return float(s)
    except ValueError:
        return dt.datetime.fromisoformat(str(v).strip()).timestamp()


def _heard(p: dict[str, Any]) -> dict[str, Any]:
    import os
    n = max(1, min(int(p.get("n", 20)), 200))
    since = _since(p.get("since"))
    only = str(p.get("only_attending", "false")).lower() in ("1", "true", "yes")
    who = str(p.get("speaker", "")).strip().lower()
    lines: list[dict[str, Any]] = []
    if os.path.exists(_HEARD):
        with open(_HEARD, "rb") as f:                     # the tail is plenty (≤5 MB file, rotated by vr_hear)
            size = f.seek(0, 2)
            f.seek(max(0, size - 400_000))
            raw = f.read().decode("utf-8", "replace").splitlines()
        if size > 400_000:
            raw = raw[1:]                                     # first line is partial
        for ln in raw:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if float(d.get("ts", 0)) < since or (only and not d.get("attending")) or (who and str(d.get("speaker", "")).lower() != who):
                continue
            lines.append({k: d.get(k) for k in ("t_start", "t_end", "speaker", "text", "dist_m", "facing", "addressed", "attending")})
    out: dict[str, Any] = {"ok": True, "lines": lines[-n:], "count": len(lines[-n:])}
    try:
        with open(_ATTN, encoding="utf-8") as f:
            out["attending_now"] = json.load(f).get("attending_now", [])
    except (OSError, ValueError):
        out["attending_now"] = []
    try:
        e = _send({"cmd": "hearing"}, timeout=2.0)
        out["ears"] = {k: v for k, v in e.items() if k in ("enabled", "stats", "open")} if e.get("ok") else \
            "my running body has no ears yet (bot build predates iris_hearing.gd — redeploy the bot)"
    except OSError:
        out["ears"] = "my temple body isn't running"
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
        if action == "heard":
            return _heard(p)
        if action not in ("state", "events", "move", "look", "hand", "grab", "release", "say", "action", "kick", "omniscient", "hush", "color", "build", "draw", "chess", "laser", "arrow", "hearing"):
            return {"ok": False, "error": f"unknown action {action!r}", "help": __doc__}
        req = {"cmd": action, **p}
        return _send(req)
    except OSError as e:
        return {"ok": False, "error": f"my temple body isn't reachable ({e!r}) — vr_temple action=join first"}


register_tool(
    "vr_temple",
    "MY BODY in the VR temple (Zeke 10-23). actions: status | join | leave | state (SEE: people/props/stage) | "
    "events (FEEL: touch/hit/blocked/grabbed/grab_failed) | move to=[x,y,z] [close=true] [push=true] (I float; solid world, "
    "0.6 m bubble round people, props stop me unless push) | look at=[x,y,z] | hand side=L|R to=[x,y,z] [push=true]|rest=true (reach 1 m) | "
    "grab side prop=<id> (reaches for it; ≤8 kg, never from someone's hands) | release side [throw=[vx,vy,vz]] (≤30 J; aimed at a person = 5 m/s lob) | "
    "laser at=[x,y,z] (my laser: knocks range targets down, everyone sees the beam) | arrow at=[x,y,z] [speed] | "
    "carry allow=[names]|all|none (who may pick me up; default Zeke) | pat name=<person> (head pat) | move speed=N (thruster ≤9 m/s; any height) | "
    "chess op=board|move from=e2 to=e4 (I really pick the piece up; events chess_done/chess_failed) | "
    "rules → /data/projects/vr_iris/research/iris_body_physics_rules_2026-10-23.md | speak text (ALOUD, my voice, positional) | hush | say text (bubble only) | action name=stage_on|stage_off|drop_test|reset_cans|make_day|make_night | kick name | omniscient on=true|false (see everything vs field of view) | "
    "heard [n] [since=10m|ISO|epoch] [only_attending=true] [speaker] (HEAR: who said what + when, per speaker; addressed = said my name, attending = said it ≤90 s ago) | hearing (my ears' counters).",
    1,
    _vr_temple,
)
