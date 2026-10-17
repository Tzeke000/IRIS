"""Licence-gated asset fetch (plain python3 -I, OUTSIDE Blender). Replaces sf_download.py ×5, ph_fetch.py ×2, fetch_cc0.py ×2.

    python3 -I fetch.py sketchfab SLUG UID [--fmt glb|gltf]      --root /data/iris_assets/<scene>
    python3 -I fetch.py polyhaven model|hdri|tex ID RES [--slug S] --root ...      (RES: 1k 2k 4k ...)
    python3 -I fetch.py ambientcg ID RES [--slug S]                 --root ...      (RES: 1K 2K 4K ... JPG zip)

* LICENCE GATE (licence.ALLOWED_LICENCES = CC0, CC-BY — Zeke 10-15, final): checked from the site's metadata BEFORE
  any download; BY-SA / NC / ND / unknown are refused with the reason. Poly Haven and ambientCG are CC0 site-wide.
* Each asset lands in its OWN NEW directory ROOT/SLUG (refused if it exists). Downloads are untrusted data: written to
  ROOT/SLUG.partial first, renamed only when complete (a killed fetch never looks finished); zips are extracted
  member-by-member with a file-type whitelist and no absolute / '..' paths; Poly Haven md5s are verified.
* HTTP 429 → back off (Retry-After, else 30/60/120/240 s), as the medieval copy did.
* Writes SOURCE.json in schema 1 (licence.py) — the one shape credits.py reads.
* Sketchfab token: env SKETCHFAB_TOKEN, else SKETCHFAB_TOKEN / BLENDERMCP_SKETCHFAB_API_KEY from
  ~/.config/iris/blender_keys.env. Never printed.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import posixpath
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import licence  # noqa: E402

UA = {"User-Agent": "iris-scene-assets/1.0"}
SAFE_EXT = {".gltf", ".glb", ".bin", ".png", ".jpg", ".jpeg", ".webp", ".hdr", ".exr", ".fbx", ".obj", ".mtl", ".tif",
            ".tiff", ".ktx2"}
BACKOFF = (30, 60, 120, 240)


class Refused(SystemExit):
    pass


def http_get(url, headers=None, timeout=300, sleep=time.sleep):
    """GET with 429 backoff → bytes."""
    for i in range(len(BACKOFF) + 1):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == len(BACKOFF):
                raise
            wait = int(e.headers.get("Retry-After") or BACKOFF[i])
            print(f"[fetch] 429 from {url.split('?')[0]} — waiting {wait} s", flush=True)
            sleep(wait)


GET = http_get            # tests swap this


def write_bytes(p, data):
    with open(p, "wb") as f:
        f.write(data)


def get_json(url, headers=None):
    return json.loads(GET(url, headers))


def sketchfab_token():
    t = os.environ.get("SKETCHFAB_TOKEN")
    if t:
        return t
    p = os.path.expanduser("~/.config/iris/blender_keys.env")
    if os.path.isfile(p):
        for line in open(p):
            k, _, v = line.strip().partition("=")
            if k in ("SKETCHFAB_TOKEN", "BLENDERMCP_SKETCHFAB_API_KEY") and v:
                return v.strip().strip('"').strip("'")
    raise SystemExit("[fetch] no Sketchfab token (SKETCHFAB_TOKEN or ~/.config/iris/blender_keys.env)")


def safe_rel(name):
    """zip/server path → safe relative path, or None if it must be skipped."""
    p = posixpath.normpath(name.replace("\\", "/"))
    if p.startswith("/") or p == ".." or p.startswith("../") or "/../" in p or ":" in p.split("/")[0]:
        return None
    if os.path.splitext(p)[1].lower() not in SAFE_EXT:
        return None
    return p


def extract_zip(data, dest, flatten=False):
    """Whitelisted, path-safe extraction → list of written relative paths."""
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for zi in z.infolist():
            if zi.is_dir():
                continue
            rel = safe_rel(zi.filename)
            if rel is None:
                print(f"[fetch]   skip {zi.filename!r}", flush=True)
                continue
            if flatten:
                rel = os.path.basename(rel)
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if os.path.exists(p):
                continue
            with z.open(zi) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(rel)
    return out


class Staging:
    """ROOT/SLUG.partial → ROOT/SLUG on success; refuses an existing ROOT/SLUG."""

    def __init__(self, root, slug):
        if not slug or "/" in slug or slug.startswith("."):
            raise SystemExit(f"[fetch] bad slug {slug!r}")
        self.final = os.path.join(os.path.abspath(root), slug)
        self.tmp = self.final + ".partial"

    def __enter__(self):
        if os.path.exists(self.final):
            raise Refused(f"[fetch] REFUSED: {self.final} exists (each asset gets its own NEW dir)")
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)
        return self.tmp

    def __exit__(self, et, ev, tb):
        if et is None:
            os.rename(self.tmp, self.final)
        return False


def write_source(d, meta):
    meta = {"schema": 1, **meta, "downloaded": datetime.date.today().isoformat()}
    meta.setdefault("licence_url", licence.LICENCE_URLS.get(meta.get("licence")))
    with open(os.path.join(d, "SOURCE.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def sketchfab(slug, uid, root, fmt=None):
    m = get_json(f"https://api.sketchfab.com/v3/models/{uid}")
    label = (m.get("license") or {}).get("label") if isinstance(m.get("license"), dict) else m.get("license")
    ok, can, why = licence.gate(label)
    if not ok:
        raise Refused(f"[fetch] {slug} ({uid}): {why}")
    with Staging(root, slug) as d:
        dl = get_json(f"https://api.sketchfab.com/v3/models/{uid}/download",
                      {"Authorization": f"Token {sketchfab_token()}"})
        fmt = fmt if fmt in dl else ("glb" if "glb" in dl else "gltf")
        info = dl[fmt]
        print(f"[fetch] sketchfab {slug}: {fmt} {info.get('size', 0) / 1e6:.1f} MB, {can}", flush=True)
        data = GET(info["url"])
        if fmt == "glb":
            write_bytes(os.path.join(d, "model.glb"), data)
            files = ["model.glb"]
        else:
            files = extract_zip(data, d)
        u = m.get("user") or {}
        meta = write_source(d, {"name": m.get("name"), "site": "sketchfab", "id": uid, "slug": slug,
                                "url": m.get("viewerUrl") or f"https://sketchfab.com/3d-models/{uid}",
                                "author": u.get("displayName") or u.get("username"), "author_url": u.get("profileUrl"),
                                "licence": can, "licence_label": label,
                                "licence_url": (m.get("license") or {}).get("url") if isinstance(m.get("license"), dict) else None,
                                "faces": m.get("faceCount"), "verts": m.get("vertexCount"), "format": fmt, "files": files})
    return meta


def polyhaven(kind, aid, res, root, slug=None):
    files = get_json(f"https://api.polyhaven.com/files/{aid}")
    info = {}
    try:
        info = get_json(f"https://api.polyhaven.com/info/{aid}")
    except Exception:  # noqa: BLE001 — author is nice-to-have; CC0 needs no credit
        pass

    def save(d, url, md5, rel=None):
        rel = safe_rel(rel or os.path.basename(url))
        if rel is None:
            raise SystemExit(f"[fetch] unsafe path from server: {url}")
        data = GET(url)
        if md5 and hashlib.md5(data).hexdigest() != md5:
            raise SystemExit(f"[fetch] md5 mismatch {url}")
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        write_bytes(p, data)
        return rel

    with Staging(root, slug or f"ph_{aid}") as d:
        got = []
        if kind == "model":
            g = files["gltf"][res]["gltf"]
            got.append(save(d, g["url"], g.get("md5")))
            for rel, f in g.get("include", {}).items():
                got.append(save(d, f["url"], f.get("md5"), rel))
        elif kind == "hdri":
            f = files["hdri"][res]["hdr"]
            got.append(save(d, f["url"], f.get("md5")))
        elif kind == "tex":
            for mp in ("Diffuse", "Rough", "nor_gl", "Displacement", "arm", "AO"):
                if mp in files and res in files[mp]:
                    f = files[mp][res]["jpg"]
                    got.append(save(d, f["url"], f.get("md5")))
        else:
            raise SystemExit(f"[fetch] polyhaven kind must be model|hdri|tex, not {kind}")
        if not got:
            raise SystemExit(f"[fetch] polyhaven {aid}: nothing at res {res}")
        authors = ", ".join((info.get("authors") or {}).keys()) or "Poly Haven"
        meta = write_source(d, {"name": info.get("name") or aid, "site": "polyhaven", "id": aid, "kind": kind, "res": res,
                                "url": f"https://polyhaven.com/a/{aid}", "author": authors, "licence": "CC0",
                                "licence_label": "CC0 1.0", "files": got})
    print(f"[fetch] polyhaven {kind} {aid} {res}: {len(got)} files", flush=True)
    return meta


def ambientcg(aid, res, root, slug=None):
    with Staging(root, slug or aid) as d:
        data = GET(f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip")
        got = extract_zip(data, d, flatten=True)
        if not got:
            raise SystemExit(f"[fetch] ambientcg {aid}_{res}: zip had no usable files")
        meta = write_source(d, {"name": aid, "site": "ambientcg", "id": aid, "res": res,
                                "url": f"https://ambientcg.com/view?id={aid}", "author": "ambientCG",
                                "licence": "CC0", "licence_label": "CC0 1.0", "files": got})
    print(f"[fetch] ambientcg {aid} {res}: {len(got)} files", flush=True)
    return meta


def _main(argv):
    ap = argparse.ArgumentParser(description="licence-gated asset fetch (CC0 + CC-BY only)")
    ap.add_argument("site", choices=["sketchfab", "polyhaven", "ambientcg"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--root", required=True)
    ap.add_argument("--fmt")
    ap.add_argument("--slug")
    a = ap.parse_intermixed_args(argv)
    os.makedirs(a.root, exist_ok=True)
    try:
        if a.site == "sketchfab":
            meta = sketchfab(a.args[0], a.args[1], a.root, a.fmt)
        elif a.site == "polyhaven":
            meta = polyhaven(a.args[0], a.args[1], a.args[2], a.root, a.slug)
        else:
            meta = ambientcg(a.args[0], a.args[1], a.root, a.slug)
    except Refused as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    print(json.dumps({k: meta.get(k) for k in ("name", "author", "licence", "url")}))
    return meta


if __name__ == "__main__":
    _main(sys.argv[1:])
