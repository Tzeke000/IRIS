"""Lights, the ONE camera (v2 "live picture": fixed on the cover framing, bass-driven vibration baked as keys — Zeke
10-19 overrides the never-react-to-music rule for this scene only), music keys on lamps + glare, and the render look
(EEVEE, colour, 5.x compositor). v1's look-dev still cameras are gone: stills are frames of the one camera."""
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
    bounce = X.light("WaterBounce", "AREA", (4, 8, -1.0), target=(-4, 0, 3), energy=520.0, color=(0.45, 0.35, 1.0),
                     size=30.0, coll=coll, shadow=False)
    return dict(moon=moon, fill=fill, bounce=bounce)


def live_camera(coll, n, SH):
    """CAM_live: C.CAM_LOC / CAM_YAW / CAM_PITCH / 24 mm, + music.shake() (loc in CAMERA axes: x right, y up,
    z forward; rot = pitch, yaw, roll rad). Baked through lib/camera (one shot, no cuts). Returns (camera, arrays)."""
    import bpy
    f, r, u = C.cam_basis()
    cd = bpy.data.cameras.new("CAM_live")
    cam = bpy.data.objects.new("CAM_live", cd)
    coll.objects.link(cam)
    cd.clip_start, cd.clip_end = 0.05, 9000.0
    loc, rot = SH["loc"], SH["rot"]

    def fn(i, _u, _n):
        p = C.CAM_LOC + r * loc[i, 0] + u * loc[i, 1] + f * loc[i, 2]
        d = f + r * math.tan(rot[i, 1]) + u * math.tan(rot[i, 0])
        return dict(cam=tuple(p), tgt=tuple(p + 100.0 * d / np.linalg.norm(d)), lens=C.CAM_LENS, fstop=C.CAM_FSTOP,
                    focus=C.CAM_FOCUS, roll=float(rot[i, 2]))
    A = CM.bake_shots(cam, [CM.Shot(1, "live", fn)], n, markers=True, clip=(0.05, 9000.0), blades=6)
    return cam, A


def music_keys(sc, M, lamps, post):
    """lamp energies + glare strength follow the music (lamps can't read the CTRL shader group)"""
    out = {}
    for ob, base, key in lamps:
        X.bake(ob.data, "energy", base * np.asarray(M[key]))
        out[ob.name] = key
    X.bake(post["tree"], 'nodes["Glare0"].inputs["Strength"].default_value', np.asarray(M["bloom"]))
    out["Glare0.Strength"] = "bloom"
    return out


def look(sc, samples=32):
    L.eevee_defaults(sc, samples=samples, res=(1920, 1080), fps=C.FPS, mb=True, shutter=0.5, raytrace=True,
                     rt_scale="2", fast_gi=True, view="Standard", look="None", exposure=-0.15)
    L.eevee_volumes(sc, start=1.0, end=1400.0, tile="8", samples=48, distribution=0.85, shadows=False)
    sc.eevee.ray_tracing_options.screen_trace_quality = 0.5
    P = L.post_chain(sc, glare=[dict(type="Bloom", threshold=0.75, strength=0.55, size=0.9, quality="Medium"),
                                dict(type="Fog Glow", threshold=2.5, strength=0.25, size=0.6, quality="Medium")],
                     lens=(-0.004, 0.006), vignette=0.78)
    return P
