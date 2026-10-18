"""Lights, look-dev still cameras (named CAM_*, NOT final shots) and the render look (EEVEE, colour, 5.x compositor)."""
import math

import numpy as np

import common as C
from common import X
from lib import camera as CM
from lib import look as L


def lights(coll):
    md = C.dir_to((0, 0, 0), C.MOON_POS)
    moon = X.light("MoonKey", "SUN", tuple(md * 50), target=(0, 0, 0), energy=1.6, color=(1.0, 0.62, 0.95), size=0.02,
                   coll=coll)
    moon.data.angle = math.radians(2.0)
    fill = X.light("CoolFill", "SUN", (0, -60, 20), target=(0, 0, 0), energy=0.22, color=(0.45, 0.4, 1.0), coll=coll,
                   shadow=False)
    # the moon's bounce off the water onto the ruins' faces (cover: ruins faintly lit from below/front)
    bounce = X.light("WaterBounce", "AREA", (4, 8, -1.0), target=(-4, 0, 3), energy=900.0, color=(0.6, 0.3, 1.0),
                     size=30.0, coll=coll, shadow=False)
    return dict(moon=moon, fill=fill, bounce=bounce)


def _toward(cam, yaw_deg, pitch_deg, dist=60.0):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return tuple(np.asarray(cam) + dist * np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)]))


def cameras(coll, ghosts):
    g = {o.name: o for o in ghosts}
    big = g["Ghost_Big"]
    gh = np.array(big.location)
    yaw = big.rotation_euler.z
    face = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    head = gh + np.array([0, 0, 0.68 * big.scale[0]])
    s4 = np.array(g["Ghost_S4"].location) + np.array([0, 0, 0.6 * g["Ghost_S4"].scale[0]])
    cx, cy, _ = C.CASTLE_POS
    cams = {
        # the cover: ledge + big ghost left, moon upper centre-right, castle + waterfall centre-right, arches in water
        "a_hero": dict(loc=tuple(C.HERO_CAM), tgt=_toward(C.HERO_CAM, 8.0, 12.5), lens=24, fstop=5.6,
                       focus=float(np.linalg.norm(head - C.HERO_CAM))),
        # moon close: long lens, clouds crossing its lower-left
        "b_moon": dict(loc=tuple(C.HERO_CAM), tgt=tuple(C.MOON_POS + np.array([-120.0, 0, -60.0])), lens=85),
        # ruins + water at eye level: arches standing in the glowing water, the moon path between them
        "c_ruins_water": dict(loc=(2.2, 3.0, 0.55), tgt=(9.0, 30.0, 3.6), lens=26, fstop=8.0, focus=18.0),
        # castle on the cliff, waterfall, misty mountains
        "d_castle": dict(loc=(4.0, -6.0, 3.0), tgt=(cx - 8.0, cy, 36.0), lens=70),
        # big ghost close-up: face 3/4, moon + clouds soft behind
        "e_ghost_close": dict(loc=tuple(head + face * 3.4 + np.array([0.9, 0, 0.15])), tgt=tuple(head + np.array([0, 0, -0.1])),
                              lens=50, fstop=2.8),
        # a small ghost drifting among the arches
        "f_ghost_small": dict(loc=tuple(s4 + np.array([0.5, -3.4, -0.3])), tgt=tuple(s4), lens=45, fstop=2.8),
        # low foreground foliage, looking up through it at the ledge, the ghost and the moon
        "g_foliage_low": dict(loc=(-3.6, -7.2, 1.25), tgt=_toward((-3.6, -7.2, 1.25), 9.0, 19.0), lens=20, fstop=4.0,
                              focus=10.5),
        # high wide establishing (layout check)
        "h_wide": dict(loc=(-24.0, -34.0, 22.0), tgt=(20.0, 120.0, 14.0), lens=22),
    }
    obs = {}
    for nm, c in cams.items():
        kw = {}
        if c.get("fstop"):
            kw["fstop"] = c["fstop"]
            if c.get("focus"):
                kw["focus"] = c["focus"]
        obs[nm] = CM.still_cam(f"CAM_{nm}", c["loc"], c["tgt"], c["lens"], coll=coll, clip=(0.05, 9000.0), **kw)
    return obs


def look(sc, samples=32):
    L.eevee_defaults(sc, samples=samples, res=(1920, 1080), fps=C.FPS, mb=True, shutter=0.5, raytrace=True,
                     rt_scale="2", fast_gi=True, view="Standard", look="None", exposure=-0.15)
    L.eevee_volumes(sc, start=1.0, end=1400.0, tile="8", samples=48, distribution=0.85, shadows=False)
    sc.eevee.ray_tracing_options.screen_trace_quality = 0.5
    P = L.post_chain(sc, glare=[dict(type="Bloom", threshold=0.75, strength=0.55, size=0.9, quality="Medium"),
                                dict(type="Fog Glow", threshold=2.5, strength=0.25, size=0.6, quality="Medium")],
                     lens=(-0.004, 0.006), vignette=0.78)
    return P
