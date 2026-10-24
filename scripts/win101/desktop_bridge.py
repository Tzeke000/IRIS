"""Iris desktop bridge for Windows VM 101 — runs IN the interactive session (Task Scheduler, at logon of Iris).
Lets the server see the screen and drive mouse/keyboard (Ableton tests, plugin GUI checks).
Auth: header X-Iris-Token must match C:\\iris_dev\\bridge_token.txt. Listens on :8790 (firewall: server IP only).
GET  /health                     -> {"ok":true,"size":[w,h]}
GET  /screenshot?max=900&q=70    -> JPEG, downscaled so the long side <= max
POST /click   {"x","y","button":"left","clicks":1}     (x,y in REAL screen pixels)
POST /move    {"x","y"}  POST /drag {"x","y","x2","y2","button"}
POST /type    {"text"}   POST /key {"keys":["ctrl","s"]}   POST /scroll {"x","y","amount","mods":["ctrl"]}
  (scroll amount is in Windows wheel units: one notch = 120)
"""
import io, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import mss, pyautogui
from PIL import Image

pyautogui.FAILSAFE = False
TOKEN = open(r"C:\iris_dev\bridge_token.txt").read().strip()


class H(BaseHTTPRequestHandler):
    def _auth(self):
        if self.headers.get("X-Iris-Token") != TOKEN:
            self.send_response(403); self.end_headers(); return False
        return True

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if not self._auth(): return
        try:
            self._get()
        except Exception as e:   # e.g. secure desktop (UAC prompt) blocks capture -> say so instead of dropping the socket
            import traceback; self._json({"ok": False, "error": repr(e), "trace": traceback.format_exc()[-600:]}, 500)

    def _get(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/health":
            return self._json({"ok": True, "size": list(pyautogui.size())})
        if u.path == "/screenshot":
            mx = int(q.get("max", ["900"])[0]); quality = int(q.get("q", ["70"])[0])
            with mss.mss() as s:
                shot = s.grab(s.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.rgb); img.thumbnail((mx, mx))
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality); b = buf.getvalue()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Screen-Size", f"{shot.size[0]}x{shot.size[1]}")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._auth(): return
        n = int(self.headers.get("Content-Length", "0")); d = json.loads(self.rfile.read(n) or b"{}")
        p = urlparse(self.path).path
        try:
            if p == "/click": pyautogui.click(d["x"], d["y"], clicks=int(d.get("clicks", 1)), button=d.get("button", "left"))
            elif p == "/move": pyautogui.moveTo(d["x"], d["y"], duration=float(d.get("duration", 0.1)))
            elif p == "/drag":
                pyautogui.moveTo(d["x"], d["y"]); pyautogui.dragTo(d["x2"], d["y2"], duration=float(d.get("duration", 0.4)), button=d.get("button", "left"))
            elif p == "/type": pyautogui.write(d["text"], interval=float(d.get("interval", 0.02)))
            elif p == "/key": pyautogui.hotkey(*d["keys"])
            elif p == "/scroll":
                pyautogui.moveTo(d["x"], d["y"])
                with pyautogui.hold(list(d.get("mods", []))): pyautogui.scroll(int(d["amount"]))
            else: return self._json({"error": "not found"}, 404)
            self._json({"ok": True})
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8790), H).serve_forever()
