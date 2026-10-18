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

# ---------------------------------------------------------------- layout (metres)
HERO_CAM = np.array([0.6, -9.0, 2.35])
MOON_POS = np.array([888.6, 2896.0, 1291.7])         # 17° right, 23° up, 3300 m from the hero camera
MOON_R = 762.0                                         # ≈ 26° across from the hero camera: the cover's "huge" moon
CASTLE_POS = np.array([80.0, 195.0, 0.0])              # cliff island foot centre
CASTLE_CLIFF_H = 50.0
FALL_X = 57.0                                          # waterfall column x (cliff's west face)
LEDGE_TOP = 2.55
BIG_GHOST = dict(name="Ghost_Big", loc=(-2.25, 0.85, LEDGE_TOP - 0.05), height=3.1, yaw=35.0, phase=0.0, trail=(-1, 0.6, 0.15))

# ---------------------------------------------------------------- palette (linear RGB)
SKY_ZENITH = (0.004, 0.002, 0.022)
SKY_MID = (0.022, 0.008, 0.085)
SKY_HORIZON = (0.200, 0.050, 0.300)
MOON_PINK = (1.00, 0.42, 0.82)
MOON_LILAC = (0.52, 0.30, 0.95)
HAZE = (0.28, 0.12, 0.58)
STONE_DARK = (0.010, 0.007, 0.028)
STONE_LIT = (0.11, 0.065, 0.24)
IVY = (0.012, 0.010, 0.030)
GHOST_WHITE = (0.93, 0.90, 1.00)
GHOST_GLOW = (0.70, 0.35, 1.00)
PETAL_PINK = (1.00, 0.25, 0.62)
FIREFLY = (1.00, 0.55, 0.88)
WATER_DEEP = (0.012, 0.006, 0.050)

# ---------------------------------------------------------------- CTRL (one shader group, every material reads it)
# Music-reactive later = one curve per output. The CAMERA never reads it (house rule).
CTRL_KEYS = {
    "T": 0.0,              # seconds (two LINEAR keys, linear extrapolation) — shader animation clock
    "glow": 1.0,           # global emissive multiplier (moon halo, water glow, fireflies)
    "ghost_glow": 1.0,     # ghost rim / self glow (+ the ghost lights, baked separately — lights can't read shaders)
    "petal_density": 0.7,  # fraction of petal instances visible (0..1, hash-gated per instance: constant count)
    "water_shimmer": 1.0,  # moon glitter path on the water
    "moon_glow": 1.0,      # moon surface + halo
    "sky_glow": 1.0,       # stars, cloud rim light, shooting stars
    "firefly": 1.0,        # firefly brightness
}
_CTRL = {}


def ctrl_tree():
    import bpy
    if "tree" in _CTRL:
        return _CTRL["tree"]
    g = bpy.data.node_groups.new("CTRL", "ShaderNodeTree")
    nb = NB(g)
    go = nb.n("NodeGroupOutput")
    for k, v in CTRL_KEYS.items():
        g.interface.new_socket(k, in_out="OUTPUT", socket_type="NodeSocketFloat")
        s = nb.val(k, v)
        nb.link(s, go.inputs[k])
    for k, v in CTRL_KEYS.items():
        if k == "T":
            fc = X.bake(g, 'nodes["T"].outputs[0].default_value', [0.0, 1.0 / FPS], frames=[1, 2], thin=False)
            fc.extrapolation = "LINEAR"
        else:
            X.bake(g, f'nodes["{k}"].outputs[0].default_value', v)      # one key now; the music pass replaces it
    _CTRL["tree"] = g
    return g


def ctrl(nb):
    """CTRL group node in nb's tree → its outputs (dict-like)."""
    gn = nb.n("ShaderNodeGroup")
    gn.node_tree = ctrl_tree()
    return gn.outputs


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
