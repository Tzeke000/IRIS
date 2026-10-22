"""CLI for the win101 desktop bridge. usage: vmctl.py STEP [STEP...]
STEP: shot:PATH[:MAX] | click:X,Y[,N] | rclick:X,Y | key:k1+k2 | type:TEXT | drag:X,Y,X2,Y2 | wait:SECONDS | scroll:X,Y,AMT
Coordinates are REAL screen pixels (1920x1080)."""
import json, os, sys, time, urllib.request

env = dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.iris_private.env")) if "=" in l and not l.startswith("#"))
HOST = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "config", "private.local.json")))["win101_host"]
B = f"http://{HOST}:8790"; H = {"X-Iris-Token": env["WIN101_BRIDGE_TOKEN"]}

def post(p, d):
    r = urllib.request.Request(B + p, data=json.dumps(d).encode(), headers={**H, "Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(r, timeout=60).read().decode()

for step in sys.argv[1:]:
    k, _, a = step.partition(":")
    if k == "shot":
        path, _, mx = a.partition(":")
        r = urllib.request.Request(f"{B}/screenshot?max={mx or 900}&q=70", headers=H)
        open(path, "wb").write(urllib.request.urlopen(r, timeout=60).read()); print("shot", path)
    elif k in ("click", "rclick"):
        v = [int(x) for x in a.split(",")]
        post("/click", {"x": v[0], "y": v[1], "clicks": v[2] if len(v) > 2 else 1, "button": "right" if k == "rclick" else "left"})
    elif k == "key": post("/key", {"keys": a.split("+")})
    elif k == "type": post("/type", {"text": a})
    elif k == "drag":
        v = [int(x) for x in a.split(",")]; post("/drag", {"x": v[0], "y": v[1], "x2": v[2], "y2": v[3], "duration": 1.0})
    elif k == "scroll":
        v = [int(x) for x in a.split(",")]; post("/scroll", {"x": v[0], "y": v[1], "amount": v[2]})
    elif k == "wait": time.sleep(float(a))
    time.sleep(0.4)
