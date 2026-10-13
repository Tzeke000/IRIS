"""Download a Sketchfab model (licence-gated) into a NEW dir. usage: sf_download.py SLUG UID
Token from env SKETCHFAB_TOKEN. Prefers GLB; else glTF zip with safe extraction (basenames, allowed exts only)."""
import json, os, sys, urllib.request, zipfile, datetime, io
slug, uid = sys.argv[1], sys.argv[2]
FORCE = sys.argv[3] if len(sys.argv) > 3 else None  # optional: "gltf" to fetch the zip (often full-res textures)
ROOT = os.environ.get("SF_ROOT", "/data/iris_assets/bemyone")
OK = {"CC Attribution", "CC0 Public Domain"}
ALLOWED = {".gltf", ".glb", ".bin", ".png", ".jpg", ".jpeg", ".webp"}
tok = os.environ["SKETCHFAB_TOKEN"]
def get(url, auth=False):
    req = urllib.request.Request(url, headers={"Authorization": f"Token {tok}"} if auth else {})
    return urllib.request.urlopen(req, timeout=120)
m = json.load(get(f"https://api.sketchfab.com/v3/models/{uid}"))
lic = m["license"]["label"] if m.get("license") else None
if lic not in OK:
    sys.exit(f"REFUSED: licence {lic!r}")
d = os.path.join(ROOT, slug)
if os.path.exists(d):
    sys.exit(f"REFUSED: {d} exists")
os.makedirs(d)
dl = json.load(get(f"https://api.sketchfab.com/v3/models/{uid}/download", auth=True))
fmt = FORCE if FORCE in dl else ("glb" if "glb" in dl else "gltf")
info = dl[fmt]
print(f"{slug}: {fmt} {info.get('size', 0) / 1e6:.1f} MB")
data = get(info["url"]).read()
files = []
if fmt == "glb":
    p = os.path.join(d, "model.glb"); open(p, "wb").write(data); files.append("model.glb")
else:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for zi in z.infolist():
            if zi.is_dir(): continue
            name = zi.filename.replace("\\", "/")
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALLOWED: print("  skip", name); continue
            rel = os.path.normpath(name)
            if rel.startswith("..") or os.path.isabs(rel): print("  skip unsafe", name); continue
            out = os.path.join(d, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out): continue
            open(out, "wb").write(z.read(zi)); files.append(rel)
meta = dict(slug=slug, uid=uid, name=m["name"], author=m["user"]["username"], author_name=m["user"].get("displayName"),
            author_url=m["user"].get("profileUrl"), license=lic, license_url=m["license"].get("url"),
            url=m.get("viewerUrl"), faces=m.get("faceCount"), verts=m.get("vertexCount"), format=fmt,
            downloaded=datetime.date.today().isoformat(), files=files)
json.dump(meta, open(os.path.join(d, "SOURCE.json"), "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("name", "author", "license", "url")}))
