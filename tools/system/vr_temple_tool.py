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
  follow name=<player> [dist=3] [height=0] [look=true]
                                 FOLLOW MODE (Zeke 10-29): I keep `dist` m behind them ALONG their path, tracking them
                                 from the network (out of my sight is fine), predicting from their velocity, flying my
                                 thruster up to 24 m/s to keep up with web-swinging; I round corners the way they went
                                 (never through a building) and never fly past them. follow on=false stops; follow (no
                                 name) = status {mode direct|trail|hold|stuck|lost|gone, gap_m, his_speed_mps, ...}.
                                 events: follow_lost / follow_found / follow_teleport. An explicit move ends it; an
                                 explicit look takes my gaze back.
  comm  to=[names] | "all" | "none"
                                 who hears me IN-EAR when they're out of earshot (quiet, non-spatial; near me it's my
                                 normal voice). Default all; each player can also turn it off / set its volume in
                                 Settings > Audio. (No args = who's on it now.)
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


def _sentences(text: str) -> list[str]:
    """split into sentences; fold pieces under 12 chars into the next so a 'Yes!' doesn't get its own synth call"""
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < 12:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text]


def _mouth_ready() -> Any:
    """NEVER WAIT ON A COLD MOUTH (Zeke 10-29: "ignore the warm-up and keep working"). True when Fish answers
    /health=ok; otherwise a short reason string, returned in ~2 s instead of a 30 s synth timeout. If the mouth
    unit is simply DOWN (quiet hours stop it, so does a crash) it is started here - the temple is not the room,
    so the room's quiet-hours flag does not silence my VR voice. Warm-up is ~3 min; say it in text meanwhile."""
    import subprocess
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8769/health", timeout=2) as r:
            if r.status == 200:
                return True
    except urllib.error.HTTPError as e:
        if e.code == 503:
            return "mouth is warming up (~3 min after a start); try again shortly"
    except Exception:
        pass
    try:
        active = subprocess.run(["systemctl", "--user", "is-active", "iris-mouth.service"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        if active not in ("active", "activating"):
            subprocess.run(["systemctl", "--user", "start", "--no-block", "iris-mouth.service"],
                           capture_output=True, timeout=5)
            return "mouth was down (" + (active or "unknown") + ") - started it; warm in ~3 min"
    except Exception as e:
        return "mouth unreachable and could not check the unit: " + repr(e)
    return "mouth unit is up but not answering yet (still binding) - try again shortly"


def _speak(text: str) -> dict[str, Any]:
    """SAY IT, SENTENCE BY SENTENCE (2026-10-29, Zeke: "speed up your mouth — when you can finally start saying your
    first words"): the mouth renders the FIRST sentence and my body starts playing it at once; the rest are rendered
    while it plays and queue behind it (Voice.speak_wav appends to the playing feed). Time-to-first-word = one
    sentence's synth, not the whole line's."""
    import os
    import time
    import urllib.request
    text = text.strip()[:400]
    if not text:
        return {"ok": False, "error": "no text"}
    ready = _mouth_ready()
    if ready is not True:
        return {"ok": False, "warming": True, "error": ready}
    os.makedirs(_VOICE_DIR, exist_ok=True)
    for f in os.listdir(_VOICE_DIR):                       # keep the folder small
        fp = os.path.join(_VOICE_DIR, f)
        if f.startswith("line_") and time.time() - os.path.getmtime(fp) > 600:
            os.remove(fp)
    t0 = time.time()
    first_s = None
    total = 0.0
    nbytes = 0
    play_end = t0
    sents = _sentences(text)
    out: dict[str, Any] = {"ok": False}

    def _stop(by: dict[str, Any], k: int) -> dict[str, Any]:
        _send({"cmd": "hush"})
        return {"ok": True, "interrupted": True, "by": by.get("speaker"), "dist_m": by.get("dist_m"),
                "sentences_sent": k, "of": len(sents), "first_word_s": first_s,
                "note": "someone talked over me - I stopped (barge-in). Their words arrive as a heard line; "
                        "answer THAT, don't resume the old line unless they ask."}

    for k, sent in enumerate(sents):
        b = _barge_since(t0)
        if b:
            return _stop(b, k)
        req = urllib.request.Request("http://localhost:8769/synth", data=json.dumps({"text": sent, "rate": 48000}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            wav = r.read()
        path = os.path.join(_VOICE_DIR, f"line_{int(time.time() * 1000)}_{k}.wav")
        with open(path, "wb") as f:
            f.write(wav)
        out = _send({"cmd": "voice", "path": path})
        if not out.get("ok"):
            return out
        if first_s is None:
            first_s = round(time.time() - t0, 2)
        secs = float(out.get("seconds", 0.0))
        play_end = max(play_end, time.time()) + secs
        total += secs
        nbytes += len(wav)
    # FULL-DUPLEX (10-29): keep listening until my line has finished PLAYING - a barge-in during the
    # last sentence still stops me. Bounded so a bad duration can't hold the tool call.
    while time.time() < min(play_end, t0 + 90.0):
        b = _barge_since(t0)
        if b:
            return _stop(b, len(sents))
        time.sleep(0.2)
    out["seconds"] = round(total, 2)
    out["wav_bytes"] = nbytes
    out["first_word_s"] = first_s
    out["sentences"] = len(sents)
    return out


_ACTIVITY = "/home/iris/IRIS/state/vr_temple/voice_activity.json"
BARGE_MAX_DIST_M = 6.0


def _barge_since(t0: float) -> dict[str, Any] | None:
    """Someone (within BARGE_MAX_DIST_M) started an audible phrase AFTER I began speaking. Written by
    vr_hear the moment a clip lands. A phrase that began before I spoke is their trailing words, not a barge."""
    import os
    try:
        if not os.path.exists(_ACTIVITY) or os.path.getmtime(_ACTIVITY) < t0 - 2.0:
            return None
        with open(_ACTIVITY, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    for who, a in (d or {}).items():
        try:
            if float(a.get("t_start", 0)) >= t0 + 0.3:
                dist = a.get("dist_m")
                if dist is None or float(dist) <= BARGE_MAX_DIST_M:
                    return {"speaker": who, **a}
        except Exception:
            continue
    return None


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
        if action not in ("state", "events", "move", "look", "hand", "grab", "release", "say", "action", "kick", "omniscient", "hush", "color", "build", "draw", "chess", "laser", "arrow", "hearing", "gun", "power", "follow", "comm"):
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
    "gun op=deploy|stow|status|targets|aim|fire [target=range:N | at=[x,y,z]] [compensate=true] [adjust={up_deg,right_deg}] "
    "(MY LASER PISTOL out of the top of my eye, fires where I LOOK; the muzzle is 0.30 m above my centre so looking straight at "
    "a point lands HIGH — each aim/fire reply gives the error (high_m/right_m/angle) + to_hit angles to correct; compensate=true angles "
    "for me; targets = the range plates; never fires with a person within 0.7 m of the beam. SEMI-AUTO ≤5 shots/s; the beam is light "
    "(hitscan, no drop) that spreads + fades and is gone past 69 m (energy_at_hit 0.82 at 10 m); each shot +5.7 % HEAT — spamming "
    "overheats on the 18th shot ⇒ locked 10 s venting; ≤2.5 shots/s never overheats; every reply carries heat{heat_pct, overheated, "
    "shots_before_overheat|vent_s_left, ready_in_s} — pace yourself; everyone sees the gauge + steam) | "
    "power [pick=web|''] (superpowers: mine + who has what) | "
    "follow name=<player> [dist=3] (FOLLOW MODE: stay dist m behind them along their path, out of sight too, thruster ≤24 m/s; "
    "on=false stops; no name = status) | comm to=[names]|all|none (who hears me in-ear when out of earshot; default all) | "
    "carry allow=[names]|all|none (who may pick me up; default Zeke) | pat name=<person> (head pat) | move speed=N (thruster ≤9 m/s; any height) | "
    "chess op=board|move from=e2 to=e4 (I really pick the piece up; events chess_done/chess_failed) | "
    "rules → /data/projects/vr_iris/research/iris_body_physics_rules_2026-10-23.md | speak text (ALOUD, my voice, positional) | hush | say text (bubble only) | action name=stage_on|stage_off|drop_test|reset_cans|make_day|make_night | kick name | omniscient on=true|false (see everything vs field of view) | "
    "heard [n] [since=10m|ISO|epoch] [only_attending=true] [speaker] (HEAR: who said what + when, per speaker; addressed = said my name, attending = said it ≤90 s ago) | hearing (my ears' counters).",
    1,
    _vr_temple,
)
