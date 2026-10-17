"""Sketchfab search, licence-filtered to CC0 + CC-BY, with an optional thumbnail contact sheet.
Replaces sf_search.py ×4, sf_thumbs.py ×2, sf_details.py, sf_vet.py.

    PY=~/venvs/iris-v100/bin/python       # PIL needed only for --thumbs
    $PY -I search.py "medieval stone well" ["gothic arch" ...] [--max-faces 300000] [--count 24] [--thumbs DIR] [--json OUT]

Prints one line per model: uid | name | author | licence | faces | likes | query. Results whose licence is not
CC0/CC-BY are dropped (licence.gate), even if the API filter let them through. --thumbs writes DIR/NN.jpg +
DIR/index.txt + DIR/sheet.jpg (labelled NN, safe to Read). Then: fetch.py sketchfab SLUG UID --root ...
No token needed (search is public).
"""
import argparse
import json
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch  # noqa: E402
import licence  # noqa: E402

API_LICENCE_SLUGS = {"CC0": "cc0", "CC-BY": "by"}


def search(queries, max_faces=None, count=24):
    seen = {}
    for q in queries:
        for can in licence.ALLOWED_LICENCES:
            params = dict(type="models", q=q, downloadable="true", license=API_LICENCE_SLUGS[can], sort_by="-likeCount",
                          count=count)
            if max_faces:
                params["max_face_count"] = max_faces
            try:
                r = fetch.get_json("https://api.sketchfab.com/v3/search?" + urllib.parse.urlencode(params))
            except Exception as e:  # noqa: BLE001
                print(f"[search] ERR {q!r}: {e}", file=sys.stderr)
                continue
            for m in r.get("results", []):
                lic = m.get("license")
                label = lic.get("label") if isinstance(lic, dict) else (lic or can)
                ok, c, _ = licence.gate(label)
                if not ok or m["uid"] in seen:
                    continue
                ims = sorted((m.get("thumbnails") or {}).get("images", []), key=lambda x: x.get("width", 0))
                th = next((x for x in ims if x.get("width", 0) >= 400), ims[-1] if ims else None)
                seen[m["uid"]] = {"uid": m["uid"], "name": m.get("name", ""), "author": (m.get("user") or {}).get("username"),
                                  "licence": c, "faces": m.get("faceCount"), "likes": m.get("likeCount"), "query": q,
                                  "url": m.get("viewerUrl"), "thumb": th["url"] if th else None}
    return list(seen.values())


def thumbs(results, out):
    os.makedirs(out, exist_ok=True)
    paths, idx = [], []
    for i, r in enumerate(results):
        if not r["thumb"]:
            continue
        p = os.path.join(out, f"{i:02d}.jpg")
        fetch.write_bytes(p, fetch.GET(r["thumb"]))
        paths.append(p)
        idx.append(f"{i:02d} {r['uid']} {r['licence']} {r['name'][:50]}")
    with open(os.path.join(out, "index.txt"), "w") as f:
        f.write("\n".join(idx) + "\n")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "qa"))
        import sheet
        s = sheet.make_sheet(paths, os.path.join(out, "sheet.jpg"), cols=4, labels="name")   # NN = result number
        print(f"[search] sheet {s['path']} ({s['bytes'] // 1024} KB)")
    except ImportError:
        print("[search] PIL not available — no sheet (run with ~/venvs/iris-v100/bin/python)")
    return paths


def _main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--max-faces", type=int)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--thumbs")
    ap.add_argument("--json")
    a = ap.parse_intermixed_args(argv)
    res = search(a.queries, a.max_faces, a.count)
    for r in res:
        print(f"{r['uid']} | {r['name'][:50]} | {r['author']} | {r['licence']} | f={r['faces']} | likes={r['likes']} | q={r['query']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    if a.thumbs:
        thumbs(res, a.thumbs)
    return res


if __name__ == "__main__":
    _main(sys.argv[1:])
