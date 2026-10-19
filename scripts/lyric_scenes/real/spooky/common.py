"""Spooky v2 — shared layout, palette, the CTRL group and small helpers used by every stage module.

World frame: +Y = away from the hero camera (into the cover), +X = right, +Z = up, water surface z = 0.
The hero camera reproduces the cover (ruined ledge + big ghost on the left, moon upper centre-right, castle cliff with a
waterfall centre-right in the distance, arches standing in glowing water).
"""
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(REPO, "scripts/lyric_scenes"))

from lib import bpyx as X  # noqa: E402
from lib.bpyx import NB  # noqa: E402

ASSETS = "/data/iris_assets/spooky_futurebass"
NATURE = "/data/iris_assets/library/quaternius/stylized-nature-megakit/glTF"
SONG_MAP = ASSETS + "/analysis/song_map.json"
FPS = 30

# ---------------------------------------------------------------- the ONE camera (v2 "live picture") + layout
# v2 is a single fixed camera on the cover composition. Everything far away is placed by SCREEN ANCHORS from this
# camera (ndc x 0 left → 1 right, y 0 top → 1 bottom, 16:9), measured off the cover (spooky_v2_cover.safe.jpg):
# big ghost centre (0.31, 0.47), horizon 0.71, moon centre (0.60, 0.285) ≈ 0.46 of the frame high, castle base on its
# cliff at (0.765, 0.60), small ghosts upper-left / right of the moon / far right.
CAM_LOC = np.array([0.56, -12.12, 1.46])
CAM_YAW, CAM_PITCH, CAM_LENS = 4.0, 10.05, 24.0          # deg (0 = +Y, positive = toward +X), deg up, mm (36 mm sensor)
CAM_FSTOP, CAM_FOCUS = 2.2, 13.0                          # soft foreground foliage, sharp ghost (13 m)
HERO_CAM = CAM_LOC


def cam_basis(yaw=CAM_YAW, pitch=CAM_PITCH):
    """(forward, right, up) unit vectors of the camera"""
    y, p = math.radians(yaw), math.radians(pitch)
    f = np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def ndc_dir(nx, ny, lens=CAM_LENS):
    """unit world direction through screen point (nx, ny)"""
    f, r, u = cam_basis()
    tx = 18.0 / lens
    ty = tx * 9.0 / 16.0
    d = f + (nx - 0.5) * 2 * tx * r + (0.5 - ny) * 2 * ty * u
    return d / np.linalg.norm(d)


def at_range(nx, ny, horiz):
    """the point on the screen ray (nx, ny) at HORIZONTAL distance `horiz` from the camera"""
    d = ndc_dir(nx, ny)
    return CAM_LOC + d * horiz / math.hypot(d[0], d[1])


def to_ndc(p):
    """world point → (nx, ny, depth)"""
    f, r, u = cam_basis()
    v = np.asarray(p, float) - CAM_LOC
    z = v @ f
    tx = 18.0 / CAM_LENS
    ty = tx * 9.0 / 16.0
    return 0.5 + 0.5 * (v @ r / z) / tx, 0.5 - 0.5 * (v @ u / z) / ty, z


MOON_DIST = 3300.0
MOON_POS = CAM_LOC + ndc_dir(0.60, 0.285) * MOON_DIST
MOON_R = MOON_DIST * math.tan(math.radians(10.6))        # ≈ 21° across: 0.46 of the frame height (cover 0.43)
CASTLE_TOP = at_range(0.765, 0.600, 430.0)               # where the castle's foot sits (cliff lip)
CASTLE_CLIFF_H = float(CASTLE_TOP[2])
CASTLE_POS = np.array([CASTLE_TOP[0], CASTLE_TOP[1], 0.0])  # cliff island foot centre
FALL_DX, FALL_DY = -23.0, -27.0                          # waterfall column relative to the island centre (west face)
FALL_X = float(CASTLE_POS[0] + FALL_DX)
LEDGE_TOP = 2.55
BIG_GHOST = dict(name="Ghost_Big", loc=(-2.25, 0.85, LEDGE_TOP - 0.05), height=3.1, yaw=35.0, phase=0.0, trail=(-1, 0.6, 0.15))
# small ghosts by anchor: (name, ndc x, ndc y, horizontal distance m, height m)
SMALL_GHOST_ANCHORS = {"Ghost_S1": (0.215, 0.255, 27.0, 1.25), "Ghost_S2": (0.885, 0.415, 62.0, 1.9),
                       "Ghost_S3": (0.955, 0.575, 150.0, 3.4), "Ghost_S4": (0.575, 0.565, 36.0, 1.0)}

# ---------------------------------------------------------------- palette (linear RGB)
# v2: re-measured off the cover (sRGB samples → linear): the dark sky is BLUE-violet (top-left 0.03/0.05/0.14), the sky
# beside the moon 0.15/0.14/0.42, clouds 0.16/0.14/0.39, horizon + mountains 0.10–0.15/0.13–0.18/0.33–0.50. v1's
# magenta-lavender veil came from the moon glow term (cos^18 ≈ 0.2 across the whole frame) — fixed in sky.world().
SKY_ZENITH = (0.0018, 0.0035, 0.016)
SKY_MID = (0.006, 0.008, 0.050)
SKY_HORIZON = (0.016, 0.020, 0.105)
SKY_MOONSIDE = (0.030, 0.022, 0.150)
RIM_VIOLET = (0.26, 0.20, 0.85)                       # ruin / stone edge light (was magenta 0.5/0.25/0.9)
HAZE_FAR = (0.030, 0.036, 0.170)                      # aerial perspective colour (ruins, ridges)
MOON_PINK = (1.00, 0.42, 0.82)
MOON_LILAC = (0.52, 0.30, 0.95)
HAZE = (0.10, 0.10, 0.45)
STONE_DARK = (0.010, 0.007, 0.028)
STONE_LIT = (0.11, 0.065, 0.24)
IVY = (0.012, 0.010, 0.030)
GHOST_WHITE = (0.93, 0.90, 1.00)
GHOST_GLOW = (0.70, 0.35, 1.00)
PETAL_PINK = (1.00, 0.25, 0.62)
FIREFLY = (1.00, 0.55, 0.88)
WATER_DEEP = (0.012, 0.006, 0.050)

# ---------------------------------------------------------------- CTRL (music-reactive channels, baked keys)
# Two groups with the same idea: CTRL (shader node group — every material reads it) and CTRL_GN (geometry node group —
# every analytic GN tree reads it). Each output is a Value node whose default_value is keyed once per frame from
# music.channels(); the CLOCKS (g_clock, drift_t, wind_t, ff_t) are integrated rates, so "faster in the drops" never
# makes anything jump. T is plain seconds (two LINEAR keys, linear extrapolation). v2: the CAMERA reacts too (bass
# shake, Zeke 10-19, this scene only) — but through its own baked keys in cams.py, not through CTRL.
CTRL_KEYS = {
    "T": 0.0,              # seconds — shader animation clock
    "glow": 1.0,           # global emissive multiplier (moon halo, water glow, fireflies, mist, waterfall)
    "ghost_glow": 1.0,     # ghost rim / self glow ↔ vocal chops + mid (+ the ghost light, baked on the lamp)
    "petal_density": 0.7,  # fraction of petal instances visible (soft hash gate per instance: constant count)
    "petal_glow": 1.0,     # petal brightness ↔ onsets + drop bursts
    "water_shimmer": 1.0,  # moon glitter path on the water ↔ hi-hats (high band)
    "moon_glow": 1.0,      # moon surface + halo ↔ bass
    "sky_glow": 1.0,       # cloud rim light, ridges, shooting stars
    "stars": 1.0,          # star twinkle amplitude ↔ hi-hats
    "firefly": 1.0,        # firefly brightness ↔ onsets + energy
    "ff_density": 1.0,     # fraction of fireflies lit (soft gate)
}
GN_KEYS = {
    "g_clock": 0.0,        # ghost float/sway clock (s-equiv; runs ~1.8× in the drops)
    "g_sway": 1.0,         # sway / hem ripple / wisp amplitude ↔ section energy
    "g_kick": 0.0,         # spring response to kicks (squash, overshoot, settle)
    "g_kick_late": 0.0,    # the same, 3 frames later (the hem follows the crown)
    "drift_t": 0.0,        # small ghosts' drift clock (faster in drops)
    "wind_t": 0.0,         # petal wind clock (bursts on drops/onsets)
    "ff_t": 0.0,           # firefly wander clock
}
_CTRL = {}


def _ctrl_group(name, keys, tree_type):
    import bpy
    g = bpy.data.node_groups.new(name, tree_type)
    nb = NB(g)
    go = nb.n("NodeGroupOutput")
    for k, v in keys.items():
        g.interface.new_socket(k, in_out="OUTPUT", socket_type="NodeSocketFloat")
        nb.link(nb.val(k, v), go.inputs[k])
    return g


def ctrl_tree():
    if "tree" in _CTRL:
        return _CTRL["tree"]
    g = _ctrl_group("CTRL", CTRL_KEYS, "ShaderNodeTree")
    fc = X.bake(g, 'nodes["T"].outputs[0].default_value', [0.0, 1.0 / FPS], frames=[1, 2], thin=False)
    fc.extrapolation = "LINEAR"
    for k, v in CTRL_KEYS.items():
        if k != "T":
            X.bake(g, f'nodes["{k}"].outputs[0].default_value', v)      # one key now; bake_ctrl() replaces it
    _CTRL["tree"] = g
    return g


def ctrl_gn_tree():
    if "gn" in _CTRL:
        return _CTRL["gn"]
    g = _ctrl_group("CTRL_GN", GN_KEYS, "GeometryNodeTree")
    for k, v in GN_KEYS.items():
        X.bake(g, f'nodes["{k}"].outputs[0].default_value', v)
    _CTRL["gn"] = g
    return g


def ctrl(nb):
    """CTRL group node in nb's (shader) tree → its outputs (dict-like)."""
    gn = nb.n("ShaderNodeGroup")
    gn.node_tree = ctrl_tree()
    return gn.outputs


def ctrl_gn(nb):
    """CTRL_GN group node in nb's (geometry) tree → its outputs."""
    gn = nb.n("GeometryNodeGroup")
    gn.node_tree = ctrl_gn_tree()
    return gn.outputs


def bake_ctrl(M):
    """music.channels() → one LINEAR key per frame on every CTRL / CTRL_GN channel (except T). Returns {name: n keys}."""
    out = {}
    for tree, keys in ((ctrl_tree(), CTRL_KEYS), (ctrl_gn_tree(), GN_KEYS)):
        for k in keys:
            if k == "T":
                continue
            v = np.asarray(M[k], np.float64)
            if not np.isfinite(v).all():
                raise ValueError(f"CTRL channel {k} not finite")
            fc = X.bake(tree, f'nodes["{k}"].outputs[0].default_value', v)
            out[k] = len(fc.keyframe_points)
    return out


# ---------------------------------------------------------------- helpers

def set_mod_input(mod, name, value):
    """NODES modifier input by interface NAME (5.2 RNA properties API; dict form on older builds)."""
    ident = None
    for it in mod.node_group.interface.items_tree:
        if getattr(it, "in_out", None) == "INPUT" and it.name == name:
            ident = it.identifier
            break
    if ident is None:
        raise KeyError(f"{mod.node_group.name} has no input {name!r}")
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, ident).value = value
    else:
        mod[ident] = value


def facing(nb, blend=0.5):
    """Layer Weight 'Facing' (0 = facing the camera, 1 = grazing) — rim light term"""
    lw = nb.n("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = blend
    return lw.outputs["Facing"]


def emission(nb, color, strength):
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    nb.set(e.inputs["Strength"], strength)
    return e.outputs[0]


def add_shader(nb, a, b):
    s = nb.n("ShaderNodeAddShader")
    nb.link(a, s.inputs[0])
    nb.link(b, s.inputs[1])
    return s.outputs[0]


def mix_shader(nb, fac, a, b):
    s = nb.n("ShaderNodeMixShader")
    nb.set(s.inputs[0], fac)
    nb.link(a, s.inputs[1])
    nb.link(b, s.inputs[2])
    return s.outputs[0]


def transparent(nb):
    return nb.n("ShaderNodeBsdfTransparent").outputs[0]


def noise(nb, vec=None, scale=5.0, detail=2.0, rough=0.5, dims="3D", w=None, distortion=0.0):
    t = nb.n("ShaderNodeTexNoise", noise_dimensions=dims)
    nb.set(t.inputs["Scale"], scale)
    nb.set(t.inputs["Detail"], detail)
    nb.set(t.inputs["Roughness"], rough)
    nb.set(t.inputs["Distortion"], distortion)
    if vec is not None:
        nb.set(t.inputs["Vector"], vec)
    if w is not None:
        nb.set(t.inputs["W"], w)
    return t


def ramp(nb, fac, stops):
    """ColorRamp with stops [(pos, (r,g,b)), …] → color socket"""
    cr = nb.n("ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    while len(els) > len(stops):
        els.remove(els[-1])
    while len(els) < len(stops):
        els.new(0.5)
    for el, (p, c) in zip(els, stops):
        el.position = p
        el.color = (*c, 1.0) if len(c) == 3 else c
    nb.set(cr.inputs["Fac"], fac)
    return cr.outputs["Color"]


def blended(m, shadow=False):
    """transparent/emissive cards: blended, no shadow, show backfaces"""
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    try:
        m.use_transparent_shadow = True
    except AttributeError:
        pass
    return m


def hide_shadow(ob):
    """object visible to camera but casts no shadow (cards, sky things)"""
    ob.visible_shadow = False
    return ob


def dir_to(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return d / np.linalg.norm(d)


def billboard_rot(pos, cam=HERO_CAM):
    """Euler for a plane whose +Z normal faces `cam` (local +Y up-ish)"""
    from mathutils import Vector
    n = Vector(tuple(np.asarray(cam, float) - np.asarray(pos, float))).normalized()
    return n.to_track_quat("Z", "Y").to_euler()


def deg(a):
    return math.radians(a)
