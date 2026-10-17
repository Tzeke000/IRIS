"""Licence gate + ONE SOURCE.json schema (pure python, stdlib). Used by fetch.py, search.py and credits.py.

POLICY (Zeke, final, 2026-10-15 — the videos go to TikTok/YouTube):
    ALLOWED_LICENCES = CC0 + CC-BY ONLY. CC-BY-SA, anything NC (non-commercial), anything ND (no-derivatives) and
    anything unrecognised are REFUSED with a message saying why. Every CC-BY asset MUST appear in the video-description
    credits block (credits.py fails if one lacks an author or URL).
    Before this, the gate depended on which sf_download.py copy ran (arcade2/police3d/medieval accepted BY-SA).
MAP/TERRAIN DATA (not assets, never fetched by lib/assets): OpenStreetMap (ODbL) and Copernicus DEM are recognised so
    credits.py can print their mandatory attribution lines ("© OpenStreetMap contributors", the Copernicus notice).

SOURCE.json schema 1 (what fetch.py writes; normalize() reads it AND the 4 legacy shapes found under /data/iris_assets):
    {"schema": 1, "name", "site", "id", "url", "author", "author_url", "licence" (canonical), "licence_label" (as
     published), "licence_url", "attribution" (data credits), "downloaded", "files", "note"}
"""
import json
import os
import re

ALLOWED_LICENCES = ("CC0", "CC-BY")
DATA_LICENCES = ("ODbL", "Copernicus")          # map/terrain data: allowed WITH their attribution line
LICENCE_URLS = {"CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
                "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
                "ODbL": "https://www.openstreetmap.org/copyright"}
WHY_REFUSED = {
    "CC-BY-SA": "ShareAlike would bind the whole video to CC-BY-SA — not allowed (Zeke 10-15: CC0 + CC-BY only)",
    "NC": "NonCommercial — the videos go to TikTok/YouTube, which can be commercial use",
    "ND": "NoDerivatives — we re-light, crop and modify every asset",
    "unknown": "licence not recognised as CC0 or CC-BY — check it by hand, don't guess",
}


def canonical(label):
    """Published licence label/slug → 'CC0' | 'CC-BY' | 'CC-BY-SA' | 'CC-BY-NC[-SA|-ND]' | 'CC-BY-ND' | 'ODbL' |
    'Copernicus' | 'unknown'."""
    if not label:
        return "unknown"
    s = str(label).strip()
    low = s.lower()
    if "odbl" in low or "openstreetmap" in low:
        return "ODbL"
    if "copernicus" in low:
        return "Copernicus"
    if re.search(r"\bcc0\b|public domain|publicdomain/zero", low):
        return "CC0"
    nc = bool(re.search(r"non-?commercial|\bnc\b|by-nc", low))
    nd = bool(re.search(r"no-?deriv|\bnd\b|-nd\b", low))
    sa = bool(re.search(r"share-?alike|\bsa\b|-sa\b", low))
    by = bool(re.search(r"attribution|\bby\b|cc[- ]by|licenses/by", low))
    if not by and not (nc or nd or sa):
        return "unknown"
    out = "CC-BY"
    if nc:
        out += "-NC"
    if sa:
        out += "-SA"
    if nd:
        out += "-ND"
    return out


def gate(label, data_ok=False):
    """(ok, canonical, reason). data_ok=True also passes ODbL/Copernicus (credits.py); fetch.py never passes it."""
    c = canonical(label)
    if c in ALLOWED_LICENCES or (data_ok and c in DATA_LICENCES):
        return True, c, ""
    if c == "CC-BY-SA":
        why = WHY_REFUSED["CC-BY-SA"]
    elif "NC" in c:
        why = WHY_REFUSED["NC"]
    elif "ND" in c:
        why = WHY_REFUSED["ND"]
    elif c in DATA_LICENCES:
        why = f"{c} is map/terrain data, not an asset licence — fetch it with the city/terrain tools"
    else:
        why = WHY_REFUSED["unknown"]
    return False, c, f"REFUSED licence {label!r} ({c}): {why}. Allowed: {', '.join(ALLOWED_LICENCES)}."


def _site(d, url):
    blob = " ".join(str(x) for x in (d.get("kind"), d.get("site"), url, d.get("source"))).lower().replace(" ", "")
    for key in ("sketchfab", "polyhaven", "ambientcg", "cgheven", "openstreetmap", "copernicus", "jangafx", "quaternius",
                "kenney"):
        if key in blob:
            return "osm" if key == "openstreetmap" else key
    return "other"


def normalize(d, path=None):
    """Any SOURCE.json shape → schema 1 dict (never raises on missing keys; unknowns become None)."""
    if d.get("schema") == 1:
        return dict(d)
    label = d.get("license") or d.get("licence")
    src = d.get("source")
    url = d.get("url") or (src if isinstance(src, str) and src.startswith("http") else None) or d.get("download_url")
    site = _site(d, url)
    aid = d.get("id") or d.get("uid") or d.get("slug")
    if not url and site == "ambientcg" and aid:
        url = f"https://ambientcg.com/view?id={aid}"
    if not url and site == "polyhaven" and aid:
        url = f"https://polyhaven.com/a/{aid}"
    if site == "ambientcg" and not aid and url and "id=" in url:
        aid = url.split("id=")[-1]
    author = d.get("author_name") or d.get("author")
    if not author and site in ("polyhaven", "ambientcg"):
        author = {"polyhaven": "Poly Haven", "ambientcg": "ambientCG"}[site]
    c = canonical(label)
    name = d.get("name") or aid or (os.path.basename(os.path.dirname(path)) if path else None)
    return {"schema": 1, "name": name, "site": site, "id": aid, "url": url, "author": author,
            "author_url": d.get("author_url"), "licence": c, "licence_label": label,
            "licence_url": d.get("license_url") or d.get("licence_url") or LICENCE_URLS.get(c),
            "attribution": d.get("attribution") or ("© OpenStreetMap contributors" if c == "ODbL" else None),
            "downloaded": d.get("downloaded"), "files": d.get("files"),
            "note": d.get("license_note") or d.get("note"), "path": path}


def load(path):
    with open(path) as f:
        return normalize(json.load(f), os.path.abspath(path))


def find_source(file_path, stop="/data/iris_assets"):
    """Nearest SOURCE.json at or above file_path's directory (not above `stop`). None if there is none."""
    d = os.path.abspath(file_path if os.path.isdir(file_path) else os.path.dirname(file_path))
    stop = os.path.abspath(stop)
    while True:
        p = os.path.join(d, "SOURCE.json")
        if os.path.isfile(p):
            return p
        if d in (stop, "/") or not d.startswith(stop.rstrip("/") + "/") and d != stop:
            return None
        d = os.path.dirname(d)
