# SELF_ASSESSMENT: I drive the app's GUIDE overlay — my eye + cable-arms above every tab: point at, press, switch
# tabs, gesture, and speak each line in time with the motion (a spoken tutorial of the app).
"""app_guide — script the in-app guide overlay (2026-10-08, Zeke: tutorial of the app for his mother;
"pointing should match up with what you're saying. Disney animation like").

Writes state/app_guide.json; the app polls GET /api/v1/app/guide, runs the steps, and reports progress
(state/app_guide_progress.json) and its live layout (state/app_ui_map.json, every element tagged
data-iris="<id>").

actions:
  ui_map                      what the app says is on screen: [{id, label, kind, tab, visible}] + active tab
  (every action takes client?: a TEST channel — only app pages opened with ?guide_client=<name> listen;
   omit it to play on the real app windows)
  run   {steps: [...], silent?} start a script (silent = captions only, never speak — tests / nobody home). Each step (all keys optional, run in this order):
          tab:   "<tab id>"         switch the panel tab first (the arm presses the tab button)
          move:  "home"|"center"|"<element id>"|{x,y}   where the eye floats to (fractions of the window)
          point: "<element id>"     an arm reaches to it and holds while the line is spoken
          press: "<element id>"     an arm taps it, and the app clicks it AT the contact frame
          say:   "<one sentence>"   spoken (voice-off flag honoured) + shown as a caption
          brain: spin|overview|focus:<node id>   (Brain tab: spin the memory graph / fly to a node and point at it)
          close: console|camera|panel|ssh|proxmox  close that window / the camera view / the panel
          via: ssh|console|proxmox_guide   with press: open THIS (a closable window) at contact instead of clicking
          gesture: wave|nod|shake|curious
          twist: -1..1              ribbon twist on the pointing arm
          hold_ms: int              extra dwell after the line (default ~400)
          minimize: true            after the line, minimize the app (the widget appears, like normal)
        WIDGET steps (client "widget" — the floating eye on the desktop; a tour's `then` hands off to it):
          travel: {x,y}|"home"      glide the widget across the screen (fractions of the primary monitor)
          look:   {x,y}|null        point my gaze (-1..1)
          desk:   {do: move|circle|click|type|keys|scroll|open|snap|place|close_new, x,y,w,h (fractions), ms, r, text, combo,
                   amount (scroll: + = down), app (chrome|notepad|calculator|explorer|url), arg (https url), pattern, button, double}
          head:   {pan_deg, tilt_deg}|"home"   turn my REAL head (the PTZ camera)
          camera: show|hide         open my live camera view beside the widget eye
          image:  "<file in state/guide_media>"|"hide"   show a picture beside the eye
          wait_ms: int
  tour  {name, silent?}       play a saved tour from config/guide_tours/<name>.json (e.g. "basics"); a tour file may
                              carry "then": {"client": "widget", "steps": [...]}
  tours                       list saved tours
  stop                        end the current script (eye returns home)
  status                      the app's last progress report
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.tool_registry import register_tool

ROOT = Path(__file__).resolve().parents[2]
ST = ROOT / "state"
STEP_KEYS = {"via", "brain", "tab", "move", "point", "press", "close", "say", "gesture", "twist", "hold_ms", "emotion",
             # 10-08 "what I can do" tour: main window hands off to the WIDGET, which drives the desktop
             "minimize", "travel", "desk", "head", "camera", "image", "wait_ms", "look"}
DESK_DO = {"move", "click", "type", "keys", "scroll", "open", "snap", "close_new", "circle", "place"}
GESTURES = {"wave", "nod", "shake", "curious"}


def _read(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _clean_steps(steps: Any) -> tuple[list[dict[str, Any]], list[str]]:
    out, errs = [], []
    if not isinstance(steps, list) or not steps:
        return [], ["steps must be a non-empty list"]
    for i, s in enumerate(steps[:80]):
        if not isinstance(s, dict):
            errs.append(f"step {i}: not an object")
            continue
        bad = set(s) - STEP_KEYS
        if bad:
            errs.append(f"step {i}: unknown keys {sorted(bad)}")
        c: dict[str, Any] = {}
        for k in ("tab", "point", "press", "emotion"):
            if s.get(k):
                c[k] = str(s[k])[:80]
        if s.get("say"):
            c["say"] = str(s["say"])[:400]
        if s.get("via"):
            if s["via"] not in ("ssh", "console", "proxmox_guide"):
                errs.append(f"step {i}: via must be ssh|console|proxmox_guide")
            else:
                c["via"] = s["via"]
        if s.get("brain"):
            b = str(s["brain"])[:80]
            if b in ("spin", "overview") or b.startswith("focus:"):
                c["brain"] = b
            else:
                errs.append(f"step {i}: brain must be spin|overview|focus:<node id>")
        if s.get("close"):
            if s["close"] not in ("console", "camera", "panel", "ssh", "proxmox"):
                errs.append(f"step {i}: close must be console|camera|panel|ssh|proxmox")
            else:
                c["close"] = s["close"]
        if s.get("gesture"):
            if s["gesture"] not in GESTURES:
                errs.append(f"step {i}: gesture must be one of {sorted(GESTURES)}")
            else:
                c["gesture"] = s["gesture"]
        if "move" in s:
            mv = s["move"]
            if isinstance(mv, dict):
                c["move"] = {"x": max(0.0, min(1.0, float(mv.get("x", 0.5)))),
                             "y": max(0.0, min(1.0, float(mv.get("y", 0.5))))}
            else:
                c["move"] = str(mv)[:80]
        if "twist" in s:
            c["twist"] = max(-1.0, min(1.0, float(s["twist"])))
        if "hold_ms" in s:
            c["hold_ms"] = max(0, min(15000, int(s["hold_ms"])))
        if s.get("minimize"):
            c["minimize"] = True
        if "wait_ms" in s:
            c["wait_ms"] = max(0, min(20000, int(s["wait_ms"])))
        for k in ("travel", "look"):
            if k in s:
                v = s[k]
                if isinstance(v, dict):
                    lo = -1.0 if k == "look" else 0.0
                    c[k] = {"x": max(lo, min(1.0, float(v.get("x", 0.5)))), "y": max(lo, min(1.0, float(v.get("y", 0.5))))}
                elif k == "travel" and v in ("home", "center"):
                    c[k] = v
                elif k == "look" and v is None:
                    c[k] = None
                else:
                    errs.append(f"step {i}: {k} must be {{x,y}}" + (" or 'home'" if k == "travel" else " or null"))
        if "desk" in s:
            d = s["desk"]
            if not isinstance(d, dict) or d.get("do") not in DESK_DO:
                errs.append(f"step {i}: desk must be {{do: {'|'.join(sorted(DESK_DO))}, ...}}")
            else:
                dc: dict[str, Any] = {"do": d["do"]}
                for k in ("x", "y", "w", "h"):
                    if k in d:
                        dc[k] = max(0.0, min(1.0, float(d[k])))
                for k, lim in (("ms", 8000), ("amount", 30), ("r", 600)):
                    if k in d:
                        dc[k] = max(-lim, min(lim, int(d[k])))
                for k in ("text", "combo", "app", "arg", "pattern", "button"):
                    if d.get(k):
                        dc[k] = str(d[k])[:300]
                if d.get("double"):
                    dc["double"] = True
                c["desk"] = dc
        if "head" in s:
            h = s["head"]
            if h == "home":
                c["head"] = "home"
            elif isinstance(h, dict):
                c["head"] = {"pan_deg": max(-150.0, min(150.0, float(h.get("pan_deg", 0)))),
                             "tilt_deg": max(-60.0, min(90.0, float(h.get("tilt_deg", 10))))}
            else:
                errs.append(f"step {i}: head must be {{pan_deg, tilt_deg}} or 'home'")
        if "camera" in s:
            if s["camera"] in ("show", "hide"):
                c["camera"] = s["camera"]
            else:
                errs.append(f"step {i}: camera must be show|hide")
        if "image" in s:
            c["image"] = str(s["image"])[:80]
        if c:
            out.append(c)
    return out, errs


def _app_guide(params: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action") or "status")
    if action == "ui_map":
        m = _read(ST / "app_ui_map.json", None)
        if not m:
            return {"ok": False, "error": "the app hasn't reported a layout yet (is the new app build open?)"}
        age = time.time() - float(m.get("ts") or 0)
        items = [{k: it.get(k) for k in ("id", "label", "kind", "tab", "visible")} for it in m.get("items", [])]
        return {"ok": True, "age_s": round(age, 1), "tab": m.get("tab"), "window": m.get("window"),
                "n": len(items), "items": items}
    if action == "tours":
        d = ROOT / "config" / "guide_tours"
        return {"ok": True, "tours": sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []}
    if action == "tour":
        name = str(params.get("name") or "basics")
        if not name.replace("_", "").replace("-", "").isalnum():
            return {"ok": False, "error": "bad tour name"}
        t = _read(ROOT / "config" / "guide_tours" / f"{name}.json", None)
        if not t:
            return {"ok": False, "error": f"no tour {name!r} (action=tours lists them)"}
        params = {**params, "steps": t.get("steps"), "then": t.get("then"), "action": "run"}
        action = "run"
    client = "".join(c for c in str(params.get("client") or "") if c.isalnum())[:20]
    gfile = ST / (f"app_guide.{client}.json" if client else "app_guide.json")
    if action == "run":
        steps, errs = _clean_steps(params.get("steps"))
        if errs and not steps:
            return {"ok": False, "errors": errs}
        then = None
        th = params.get("then")
        if isinstance(th, dict):   # the next part, handed to another channel when this one finishes (widget)
            tsteps, terrs = _clean_steps(th.get("steps"))
            errs += [f"then: {e}" for e in terrs]
            if tsteps:
                then = {"client": "".join(c for c in str(th.get("client") or "") if c.isalnum())[:20], "steps": tsteps}
        cur = _read(gfile, {"seq": 0})
        seq = int(cur.get("seq") or 0) + 1
        _write(gfile, {"seq": seq, "issued_ts": time.time(), "steps": steps,
                       "silent": bool(params.get("silent")), **({"then": then} if then else {})})
        return {"ok": True, "seq": seq, "n_steps": len(steps), "then_steps": len(then["steps"]) if then else 0, "warnings": errs,
                "note": "the app picks this up within ~0.5 s; check action=status for progress"}
    if action == "stop":
        cur = _read(gfile, {"seq": 0})
        seq = int(cur.get("seq") or 0) + 1
        _write(gfile, {"seq": seq, "issued_ts": time.time(), "steps": [], "stop": True})
        return {"ok": True, "seq": seq}
    if action == "status":
        cur = _read(gfile, {"seq": 0, "steps": []})
        prog = _read(ST / "app_guide_progress.json", None)
        hist = []
        try:
            lines = (ST / "app_guide_progress.jsonl").read_text(encoding="utf-8").splitlines()[-400:]
            hist = [json.loads(x) for x in lines if x.strip()]
            hist = [h for h in hist if h.get("seq") == cur.get("seq") and h.get("status") not in ("step",)]
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "script_seq": cur.get("seq"), "n_steps": len(cur.get("steps") or []), "events": hist[-30:],
                "progress": prog, "progress_age_s": round(time.time() - float(prog.get("ts") or 0), 1) if prog else None}
    return {"ok": False, "error": f"unknown action {action!r} (ui_map|run|stop|status)"}


register_tool(
    name="app_guide",
    description="Drive the app's guide overlay (my eye + cable-arms above every tab): action=ui_map (what's on "
                "screen, element ids) | run {steps:[{tab,move,point,press,say,gesture,twist,hold_ms}]} | stop | "
                "status | tour {name} | tours. The app speaks each 'say' and lands the arm on 'point' in time with the line. Tier 1.",
    tier=1,
    handler=_app_guide,
)
