"""Poly Haven CC0 fetch (plain python3 -I). Each asset into its own NEW dir; md5-checked; safe relative paths only.
  ph_fetch.py model <id> <res> <outdir>      glTF + its textures
  ph_fetch.py hdri  <id> <res> <outdir>
  ph_fetch.py tex   <id> <res> <outdir>      Diffuse/Rough/nor_gl/arm jpgs"""
import hashlib, json, os, sys, urllib.request
kind, aid, res, out = sys.argv[1:5]
UA = {"User-Agent": "iris-fetch/1.0"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=180).read()
os.makedirs(out, exist_ok=False)
def save(url, md5, rel):
    rel = os.path.normpath(rel)
    if rel.startswith("..") or os.path.isabs(rel) or os.path.splitext(rel)[1].lower() not in {".gltf", ".bin", ".jpg", ".png", ".hdr", ".exr"}:
        raise SystemExit("unsafe " + rel)
    data = get(url)
    if md5 and hashlib.md5(data).hexdigest() != md5: raise SystemExit("md5 mismatch " + url)
    p = os.path.join(out, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data); print(rel, len(data))
files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
if kind == "model":
    g = files["gltf"][res]["gltf"]; save(g["url"], g.get("md5"), os.path.basename(g["url"]))
    for rel, f in g.get("include", {}).items(): save(f["url"], f.get("md5"), rel)
elif kind == "hdri":
    f = files["hdri"][res]["hdr"]; save(f["url"], f["md5"], os.path.basename(f["url"]))
elif kind == "tex":
    for m in ("Diffuse", "Rough", "nor_gl", "arm"):
        if m in files and res in files[m]: f = files[m][res]["jpg"]; save(f["url"], f.get("md5"), os.path.basename(f["url"]))
json.dump({"source": f"https://polyhaven.com/a/{aid}", "id": aid, "res": res, "licence": "CC0 1.0"}, open(os.path.join(out, "SOURCE.json"), "w"))
