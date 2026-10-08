"""App GUIDE routes (2026-10-08, Zeke: "give a tutorial of the app to my mother… your eye layer has to sit on
top of even the tab… point at items and explain them… know where things are in the app mechanically and be
able to control the app… pointing should match up with what you're saying. Disney animation like.")

Installed by brain.app_extra_routes.install() (boot AND live via the `app_routes` tool), never by reloading
brain.orb_http. The app polls the script; I write it with the `app_guide` tool.

  GET  /api/v1/app/guide                      current guide script {seq, steps[], issued_ts} (seq 0 = none)
  POST /api/v1/app/guide/progress {seq, index, status, note?}   the app reports where it is in the script
  POST /api/v1/app/guide/ui_map   {tab, viewport, items[]}      the app reports its live layout (ids + rects)
  POST /api/v1/app/guide/say      {text, emotion?}              speak one sentence (the app times it to the
                                                                arm motion); honours the voice-off flag
  POST /api/v1/app/guide/hush     {}                            cut the line I'm saying short (someone clicked)
  POST /api/v1/app/guide/head     {pan_deg, tilt_deg}           turn my real head (PTZ) to an absolute bearing
  GET  /api/v1/app/guide/media/<name>                           a picture from state/guide_media/ for the widget

Same origin policy as the Jarvis routes: state-changing POSTs refuse a browser Origin that isn't the app's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from fastapi import Request
except Exception:  # noqa: BLE001
    Request = Any  # type: ignore[assignment,misc]

MAX_MAP_ITEMS = 400
MAX_TEXT = 400


def _read(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _voice_off(root: Path) -> bool:
    d = _read(root / "state" / "voice_deliberately_off.json", None)
    return bool(isinstance(d, dict) and d.get("off"))


def install(app: Any, g: dict[str, Any], root: Path, existing: set) -> list[str]:
    from fastapi.responses import JSONResponse
    from brain.app_jarvis_routes import origin_ok, rate_ok

    st = root / "state"
    added: list[str] = []

    def deny() -> JSONResponse:
        return JSONResponse({"ok": False, "error": "origin not allowed"}, status_code=403)

    def _client_file(client: str) -> str:
        c = "".join(ch for ch in client if ch.isalnum())[:20]
        return "app_guide.json" if not c else f"app_guide.{c}.json"

    def guide_get(client: str = "") -> dict[str, Any]:
        # channels: the live app polls the default; my test pages poll ?client=<name> so tests never play on
        # Zeke's real app window (10-08: a test tour pressed "Open my console" on whatever copy was open).
        # The widget polls ?client=widget.
        d = _read(st / _client_file(client), {"seq": 0, "steps": []})
        d = {k: v for k, v in d.items() if k not in ("then", "chained")} if isinstance(d, dict) else {"seq": 0, "steps": []}
        return {"ok": True, **d}

    async def guide_progress(request: Request):
        if not origin_ok(request.headers.get("origin")):
            return deny()
        try:
            b = await request.json()
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "bad json"}
        rec = {"seq": int(b.get("seq") or 0), "index": int(b.get("index") or 0),
               "status": str(b.get("status") or "")[:40], "note": str(b.get("note") or "")[:300],
               "ts": time.time()}
        _write(st / "app_guide_progress.json", rec)
        # Hand-off (10-08 "what I can do" tour): a script may carry `then` = {client, steps} — when it finishes,
        # the next part goes to that channel (the main window minimizes itself, the widget carries on).
        if rec["status"] == "done":
            try:
                src = _client_file(str(b.get("client") or ""))
                cur = _read(st / src, {})
                nxt = cur.get("then") if isinstance(cur, dict) else None
                if isinstance(nxt, dict) and int(cur.get("seq") or -1) == rec["seq"] and not cur.get("chained"):
                    cur["chained"] = True
                    _write(st / src, cur)
                    dst = _client_file(str(nxt.get("client") or ""))
                    prev = _read(st / dst, {"seq": 0})
                    _write(st / dst, {"seq": int(prev.get("seq") or 0) + 1, "steps": nxt.get("steps") or [],
                                      "silent": bool(cur.get("silent")), "issued_ts": time.time(),
                                      **({"then": nxt["then"]} if isinstance(nxt.get("then"), dict) else {})})
            except Exception:  # noqa: BLE001
                pass
        try:  # history, so a quick 'missing' between steps isn't lost (trimmed by the tool)
            with open(st / "app_guide_progress.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True}

    async def guide_ui_map(request: Request):
        if not origin_ok(request.headers.get("origin")):
            return deny()
        if not rate_ok("guide_ui_map", 30, 60):
            return {"ok": False, "error": "slow down"}
        try:
            b = await request.json()
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "bad json"}
        items = []
        for it in (b.get("items") or [])[:MAX_MAP_ITEMS]:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            r = it.get("rect") or {}
            items.append({"id": str(it["id"])[:80], "label": str(it.get("label") or "")[:120],
                          "kind": str(it.get("kind") or "")[:20], "tab": str(it.get("tab") or "")[:40],
                          "visible": bool(it.get("visible", True)),
                          "rect": {k: round(float(r.get(k) or 0.0), 1) for k in ("x", "y", "w", "h")}})
        vp = b.get("viewport") or {}
        _write(st / "app_ui_map.json", {"ts": time.time(), "tab": str(b.get("tab") or "")[:40],
                                        "window": str(b.get("window") or "main")[:20],
                                        "viewport": {"w": float(vp.get("w") or 0), "h": float(vp.get("h") or 0)},
                                        "items": items})
        return {"ok": True, "n": len(items)}

    async def guide_say(request: Request):
        if not origin_ok(request.headers.get("origin")):
            return deny()
        if not rate_ok("guide_say", 40, 60):
            return {"ok": False, "error": "slow down"}
        try:
            b = await request.json()
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "bad json"}
        text = str(b.get("text") or "").strip()[:MAX_TEXT]
        if not text:
            return {"ok": False, "error": "empty"}
        if _voice_off(root):
            return {"ok": True, "spoken": False, "why": "voice deliberately off (quiet hours / Zeke's word)"}
        fn = g.get("voice_speak") if isinstance(g, dict) else None
        if not callable(fn):  # the runtime runs as __main__; @mcp.tool() returns the plain function
            import sys as _sys
            fn = getattr(_sys.modules.get("__main__"), "voice_speak", None)
            fn = getattr(fn, "fn", fn)
        if not callable(fn):
            return {"ok": False, "spoken": False, "error": "voice_speak not available in this process"}
        try:
            import asyncio
            r = await asyncio.to_thread(fn, text, str(b.get("emotion") or "neutral"), 0.5)
            return {"ok": True, "spoken": bool(r.get("ok")) if isinstance(r, dict) else True, "engine":
                    (r or {}).get("engine") if isinstance(r, dict) else None}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "spoken": False, "error": repr(e)[:200]}

    async def guide_hush(request: Request):
        """Cut my current line short (10-08, Zeke: if someone clicks during the tour, stop for a sec, say
        'please don't click', then carry on). Barge-in on the mouth: GET :8769/stop aborts the in-flight line."""
        if not origin_ok(request.headers.get("origin")):
            return deny()
        if not rate_ok("guide_hush", 30, 60):
            return {"ok": False, "error": "slow down"}
        import os as _os
        import urllib.request as _url
        host = _os.environ.get("IRIS_VOICE_HOST", "127.0.0.1")
        port = int(_os.environ.get("WREN_VOICE_PORT", "8769"))
        try:
            import asyncio
            def _stop() -> str:
                with _url.urlopen(f"http://{host}:{port}/stop", timeout=2) as r:
                    return r.read().decode("utf-8", "replace")[:40]
            return {"ok": True, "mouth": await asyncio.to_thread(_stop)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e)[:160]}

    async def guide_head(request: Request):
        """Turn my real head (the PTZ camera) for the tour: {pan_deg, tilt_deg} absolute bearing, or {home:true}."""
        if not origin_ok(request.headers.get("origin")):
            return deny()
        if not rate_ok("guide_head", 60, 60):
            return {"ok": False, "error": "slow down"}
        try:
            b = await request.json()
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "bad json"}
        try:
            pan = max(-150.0, min(150.0, float(b.get("pan_deg", 0.0))))
            tilt = max(-60.0, min(90.0, float(b.get("tilt_deg", 10.0))))
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "pan_deg / tilt_deg must be numbers"}
        try:
            import asyncio
            from tools.system.room_map_tool import _look
            home = abs(pan) < 0.5 and abs(tilt - 10.0) < 0.5
            if not home:
                await asyncio.to_thread(_hold_tracking)          # tracking OFF while I move my head on purpose
            r = await asyncio.to_thread(_look, g, {"pan_deg": pan, "tilt_deg": tilt})
            if home:
                _release_tracking()                              # the demo's over: tracking back on
            return {"ok": bool(r.get("ok")), "pan_deg": pan, "tilt_deg": tilt, "error": r.get("error"),
                    "tracking": "held" if not home else "resumed"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e)[:200]}

    # ---- tracking hold for the head demo (Zeke 10-08: "you need to turn the camera tracking off") --------------
    def _hold_tracking() -> None:
        import threading
        from tools.system.attention_smooth_tool import _attention_smooth
        sm = g.get("_attention_smooth") or {}
        running = bool(sm.get("thread") and sm["thread"].is_alive())
        if running:
            tgt = (g.get("_attention_state_obj") or {}).get("target")
            g["_guide_tracking_was"] = tgt or "zeke"
            _attention_smooth({"action": "stop", "home": False}, g)
        g["_guide_head_hold_until"] = time.time() + 90.0       # also blocks the sentry re-engaging
        old = g.get("_guide_head_timer")
        if old:
            old.cancel()
        t = threading.Timer(92.0, _release_tracking)            # never leave tracking off if the tour dies
        t.daemon = True
        g["_guide_head_timer"] = t
        t.start()

    def _release_tracking() -> None:
        g["_guide_head_hold_until"] = 0
        old = g.pop("_guide_head_timer", None)
        if old:
            old.cancel()
        tgt = g.pop("_guide_tracking_was", None)
        if tgt:
            try:
                from tools.system.attention_smooth_tool import _attention_smooth
                _attention_smooth({"action": "start", "target": tgt}, g)
            except Exception:  # noqa: BLE001
                pass

    def guide_media(name: str):
        """Pictures the widget shows during a tour (state/guide_media/<name>, png/jpg only)."""
        from fastapi.responses import FileResponse, JSONResponse as _J
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "._-")[:80]
        p = st / "guide_media" / safe
        if not safe or safe != name or p.suffix.lower() not in (".png", ".jpg", ".jpeg") or not p.is_file():
            return _J({"ok": False, "error": "no such picture"}, status_code=404)
        return FileResponse(str(p))

    # This module owns these paths: drop older copies so a live re-install picks up edited handlers.
    mine = {"/api/v1/app/attention", "/api/v1/app/guide", "/api/v1/app/guide/progress", "/api/v1/app/guide/ui_map", "/api/v1/app/guide/say",
            "/api/v1/app/guide/hush", "/api/v1/app/guide/head", "/api/v1/app/guide/media/{name}"}
    try:
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in mine]
        existing = {e for e in existing if e.split(" ", 1)[-1] not in mine}
    except Exception:  # noqa: BLE001
        pass
    def attention() -> dict[str, Any]:
        """Where the person I'm following is, for the app eye's GAZE (the whole ball turns toward them)."""
        d = _read(st / "attention" / "attention_state.json", {})
        b = d.get("bearing") or {}
        off = d.get("offset") or {}
        return {"ok": True, "status": d.get("status"), "target": d.get("target_label"),
                "pan_deg": b.get("pan_deg"), "tilt_deg": b.get("tilt_deg"),
                "dx": off.get("dx"), "dy": off.get("dy"), "age_s": round(time.time() - float(d.get("ts") or 0), 2)}

    for method, path, fn in (("GET", "/api/v1/app/attention", attention),
                             ("GET", "/api/v1/app/guide", guide_get),
                             ("POST", "/api/v1/app/guide/progress", guide_progress),
                             ("POST", "/api/v1/app/guide/ui_map", guide_ui_map),
                             ("POST", "/api/v1/app/guide/say", guide_say),
                             ("POST", "/api/v1/app/guide/hush", guide_hush),
                             ("POST", "/api/v1/app/guide/head", guide_head),
                             ("GET", "/api/v1/app/guide/media/{name}", guide_media)):
        if f"{method} {path}" in existing:
            continue
        app.add_api_route(path, fn, methods=[method])
        added.append(f"{method} {path}")
    return added
