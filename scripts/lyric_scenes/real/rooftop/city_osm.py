"""Real street layout + building footprints from OpenStreetMap (Mong Kok, Kowloon) for the Be My One city.

Data © OpenStreetMap contributors (ODbL 1.0) — credit in the video description. Fetched with osm_fetch.py
into /data/iris_assets/bemyone/osm_hk_mongkok/osm.json. Pure parsing + numpy; Blender objects are built by
build_scene.py from what these functions return.
"""
import json
import math

import numpy as np

LAT0, LON0 = 22.318, 114.170      # OSM bbox centre
OFF = (0.0, 500.0)                 # scene = local metres + OFF  (home roof at scene 0,0 ≈ Nathan Road, Jordan/Yau Ma Tei)
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110574.0
BBOX_SCENE = (-1030.0, -605.0, 1030.0, 1605.0)     # area the OSM data covers (procedural filler only outside it)

LOW_TYPES = {"garage", "garages", "roof", "shed", "kiosk", "construction", "carport", "hut", "toilets", "service",
             "train_station", "transportation", "bridge", "grandstand", "ruins"}
MAIN_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
RES_ROADS = {"residential", "unclassified"}


def to_scene(lat, lon):
    return ((lon - LON0) * KX + OFF[0], (lat - LAT0) * KY + OFF[1])


def _num(v):
    try:
        return float(str(v).replace("m", " ").split()[0])
    except Exception:
        return None


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load(paths, rng):
    """one or several Overpass JSON files (tiles overlap → de-duplicated by way id)"""
    if isinstance(paths, str):
        paths = [paths]
    seen = set()
    els = []
    for path in paths:
        for e in json.load(open(path))["elements"]:
            if e.get("id") in seen:
                continue
            seen.add(e.get("id"))
            els.append(e)
    buildings, roads = [], []
    for e in els:
        t = e.get("tags", {})
        g = e.get("geometry")
        if not g or e.get("type") != "way":
            continue
        pts = np.array([to_scene(p["lat"], p["lon"]) for p in g], dtype=np.float64)
        if "building" in t:
            if t.get("building") == "roof" or len(pts) < 4:
                continue
            if np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            keep = np.ones(len(pts), bool)
            keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.3
            pts = pts[keep]
            if len(pts) < 3:
                continue
            a = poly_area(pts)
            if abs(a) < 35.0:
                continue
            if a < 0:
                pts = pts[::-1]
            h = _num(t.get("height"))
            if h is None and "building:levels" in t:
                lv = _num(t["building:levels"])
                h = lv * 3.15 + 1.5 if lv else None
            if h is None:
                if t.get("building") in LOW_TYPES:
                    h = rng.uniform(5, 14)
                else:   # untagged Kowloon tenement / tower: area-dependent, plausible for Mong Kok
                    A = abs(a)
                    h = rng.uniform(18, 42) if A < 150 else (rng.uniform(26, 68) if A < 700 else rng.uniform(32, 95))
            buildings.append(dict(poly=pts, h=float(min(h, 260.0)), area=abs(a), kind=t.get("building"), id=e.get("id")))
        elif "highway" in t:
            cls = t["highway"]
            if len(pts) >= 2:
                roads.append(dict(pts=pts, cls=cls, oneway=t.get("oneway") == "yes", id=e.get("id"),
                                  lanes=_num(t.get("lanes")) or (3 if cls in MAIN_ROADS else 1)))
    return buildings, roads


def prism_mesh(buildings, attrs):
    """Extrude footprints (CCW) into closed-top prisms; optional inset crown tier. Returns verts, faces, per-face attrs."""
    V, F, FA = [], [], {k: [] for k in attrs[0]} if attrs else {}
    nv = 0

    def add_prism(p, z0, z1, a):
        nonlocal nv
        n = len(p)
        V.extend([(x, y, z0) for x, y in p])
        V.extend([(x, y, z1) for x, y in p])
        for i in range(n):
            j_ = (i + 1) % n
            F.append((nv + i, nv + j_, nv + n + j_, nv + n + i))
        F.append(tuple(nv + n + i for i in range(n)))
        for k, v in a.items():
            FA[k].extend([v] * (n + 1))
        nv += 2 * n

    for b, a in zip(buildings, attrs):
        p, h, z0 = b["poly"], b["h"], b.get("z0", 0.0)
        if b.get("crown"):
            z1 = h * b["crown"]
            add_prism(p, z0, z0 + z1, a)
            c = p.mean(0)
            add_prism(c + (p - c) * 0.78, z0 + z1, z0 + h, a)
        else:
            add_prism(p, z0, z0 + h, a)
    return V, F, FA


def resample(pts, step):
    """points every `step` metres along a polyline (+ unit tangents)"""
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    if cum[-1] < step:
        return np.zeros((0, 2)), np.zeros((0, 2))
    s = np.arange(step * 0.5, cum[-1], step)
    idx = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
    f = (s - cum[idx]) / np.maximum(L[idx], 1e-6)
    P = pts[idx] + seg[idx] * f[:, None]
    Tn = seg[idx] / np.maximum(L[idx], 1e-6)[:, None]
    return P, Tn


def length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def rot_boxes(cx, cy, w, d, yaw, z0, h):
    """N rotated boxes (closed top, open bottom) → verts, faces (numpy)"""
    n = len(cx)
    c, s_ = np.cos(yaw), np.sin(yaw)
    lx = np.array([-1, 1, 1, -1]) * 0.5
    ly = np.array([-1, -1, 1, 1]) * 0.5
    X = cx[:, None] + (lx[None, :] * w[:, None]) * c[:, None] - (ly[None, :] * d[:, None]) * s_[:, None]
    Y = cy[:, None] + (lx[None, :] * w[:, None]) * s_[:, None] + (ly[None, :] * d[:, None]) * c[:, None]
    Zb = np.repeat(z0[:, None], 4, 1)
    Zt = Zb + h[:, None]
    V = np.concatenate([np.stack([X, Y, Zb], -1), np.stack([X, Y, Zt], -1)], 1).reshape(-1, 3)
    base = (np.arange(n) * 8)[:, None, None]
    F = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])[None] + base
    return V, F.reshape(-1, 4)
