"""Plan-view preview of an OSM fetch, in scene coordinates (same mapping as city_osm.py). PIL venv, run with -I."""
import json, math, sys
from PIL import Image, ImageDraw
src, out = sys.argv[1], sys.argv[2]
LAT0, LON0, OFF_X, OFF_Y, ROT = (float(x) for x in sys.argv[3:8])
j = json.load(open(src))
kx = 111320.0 * math.cos(math.radians(LAT0)); ky = 110574.0
c, s = math.cos(math.radians(ROT)), math.sin(math.radians(ROT))
def xy(p):
    x, y = (p["lon"] - LON0) * kx, (p["lat"] - LAT0) * ky
    return (c * x - s * y + OFF_X, s * x + c * y + OFF_Y)
W = 900; S = 2800.0; ox, oy = -1400, -800
im = Image.new("RGB", (W, W), (10, 10, 14)); d = ImageDraw.Draw(im)
tp = lambda q: ((q[0] - ox) / S * W, W - (q[1] - oy) / S * W)
for e in j["elements"]:
    t = e.get("tags", {})
    if "highway" in t and "geometry" in e:
        d.line([tp(xy(p)) for p in e["geometry"]], fill=(60, 60, 70), width=1)
for e in j["elements"]:
    t = e.get("tags", {})
    if "building" in t and "geometry" in e:
        h = None
        try: h = float(str(t.get("height", "")).split()[0])
        except Exception: pass
        if h is None and "building:levels" in t:
            try: h = float(t["building:levels"]) * 3.1
            except Exception: pass
        col = (90, 90, 90) if h is None else (int(min(255, 40 + h * 1.6)), int(min(255, 255 - h)), 80)
        d.polygon([tp(xy(p)) for p in e["geometry"]], fill=col)
for (x, y, cc) in [(0, 0, (255, 255, 255)), (80, 520, (255, 0, 255)), (40, 212, (0, 255, 255)), (30, 440, (255, 255, 0))]:
    X, Y = tp((x, y)); d.ellipse([X - 5, Y - 5, X + 5, Y + 5], outline=cc, width=2)
im.save(out, quality=80)
