"""Fetch OSM buildings (+ heights/levels) and main roads for a bbox via the Overpass API (plain python3 -I).
Data © OpenStreetMap contributors, ODbL 1.0 — credit required in the video description.
usage: osm_fetch.py SOUTH WEST NORTH EAST OUTDIR"""
import json, os, sys, urllib.parse, urllib.request
s, w, n, e = (float(x) for x in sys.argv[1:5]); out = sys.argv[5]
q = f"""[out:json][timeout:300][maxsize:536870912];
(way["building"]({s},{w},{n},{e});
 way["building:part"]({s},{w},{n},{e});
 way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service)$"]({s},{w},{n},{e});
 way["natural"="water"]({s},{w},{n},{e}); way["leisure"="park"]({s},{w},{n},{e}););
out body geom;"""
import time
URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
for url in URLS * 2:
    try:
        req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "iris-scene/1.0"})
        data = urllib.request.urlopen(req, timeout=240).read()
        break
    except Exception as ex:
        print("fail", url, ex); time.sleep(5)
else:
    raise SystemExit("overpass unreachable")
j = json.loads(data)
os.makedirs(out, exist_ok=False)
json.dump(j, open(os.path.join(out, "osm.json"), "w"))
json.dump({"source": "OpenStreetMap via Overpass API", "bbox": [s, w, n, e], "licence": "ODbL 1.0 — © OpenStreetMap contributors",
           "url": "https://www.openstreetmap.org/copyright"}, open(os.path.join(out, "SOURCE.json"), "w"))
els = j["elements"]
b = [x for x in els if "building" in x.get("tags", {}) or "building:part" in x.get("tags", {})]
hh = [x for x in b if "height" in x["tags"] or "building:levels" in x["tags"]]
r = [x for x in els if "highway" in x.get("tags", {})]
print(f"elements {len(els)} buildings {len(b)} with height/levels {len(hh)} roads {len(r)} bytes {len(data)}")
