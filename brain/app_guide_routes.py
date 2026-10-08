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

    def guide_get(client: str = "") -> dict[str, Any]:
        # channels: the live app polls the default; my test pages poll ?client=<name> so tests never play on
        # Zeke's real app window (10-08: a test tour pressed "Open my console" on whatever copy was open)
        name = "app_guide.json" if not client else f"app_guide.{''.join(c for c in client if c.isalnum())[:20]}.json"
        d = _read(st / name, {"seq": 0, "steps": []})
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

    # This module owns these paths: drop older copies so a live re-install picks up edited handlers.
    mine = {"/api/v1/app/guide", "/api/v1/app/guide/progress", "/api/v1/app/guide/ui_map", "/api/v1/app/guide/say"}
    try:
        app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in mine]
        existing = {e for e in existing if e.split(" ", 1)[-1] not in mine}
    except Exception:  # noqa: BLE001
        pass
    for method, path, fn in (("GET", "/api/v1/app/guide", guide_get),
                             ("POST", "/api/v1/app/guide/progress", guide_progress),
                             ("POST", "/api/v1/app/guide/ui_map", guide_ui_map),
                             ("POST", "/api/v1/app/guide/say", guide_say)):
        if f"{method} {path}" in existing:
            continue
        app.add_api_route(path, fn, methods=[method])
        added.append(f"{method} {path}")
    return added
