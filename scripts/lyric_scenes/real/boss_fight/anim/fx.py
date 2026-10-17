"""Blocking-pass FX helpers: placeholder meshes (emissive, shadowless), keyed paths, visibility windows.
Every FX object is registered in S.fx_bodies (name → dict(kind, owner)) so the audit knows what may touch what."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from anim import common as C
from lib import bpyx as X


# --------------------------------------------------------------------------------------------- materials
_MATS = {}


def mat_emit(name, color, strength):
    if name not in _MATS:
        m = X.emit_mat(name, color, strength)
        m.diffuse_color = (*color, 1.0)              # workbench animatic draws the FX in their glow colour
        _MATS[name] = m
    return _MATS[name]


def mat_glass(name, color, strength, alpha=0.35):
    """emission mixed with transparency (shield bubble / aura)."""
    if name in _MATS:
        return _MATS[name]
    m, nb, out = X.new_mat(name)
    m.diffuse_color = (*color, alpha if alpha > 0.15 else 0.35)
    e = nb.n("ShaderNodeEmission")
    nb.set(e.inputs["Color"], color)
    nb.set(e.inputs["Strength"], strength)
    t = nb.n("ShaderNodeBsdfTransparent")
    lw = nb.n("ShaderNodeLayerWeight")
    nb.set(lw.inputs["Blend"], 0.35)
    mix = nb.n("ShaderNodeMixShader")
    f = nb.m("MAXIMUM", lw.outputs["Facing"], alpha)
    nb.link(f, mix.inputs[0])
    nb.link(t.outputs[0], mix.inputs[1])
    nb.link(e.outputs[0], mix.inputs[2])
    nb.link(mix.outputs[0], out.inputs["Surface"])
    try:
        m.surface_render_method = "BLENDED"
    except Exception:  # noqa: BLE001
        pass
    _MATS[name] = m
    return m


def mat_solid(name, color, rough=0.6, metal=0.0):
    if name not in _MATS:
        _MATS[name] = X.simple_mat(name, color, rough=rough, metal=metal)
        _MATS[name].diffuse_color = (*color, 1.0)
    return _MATS[name]


# --------------------------------------------------------------------------------------------- meshes
def _bm_obj(name, bm, coll, mat=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.visible_shadow = False
    return ob


def ico(name, r, coll, mat=None, subdiv=2):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    return _bm_obj(name, bm, coll, mat)


def cone(name, r1, r2, depth, coll, mat=None, segs=12, base_at_origin=True):
    """cone along +Z (base at origin when base_at_origin)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r1, radius2=r2, depth=depth)
    if base_at_origin:
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, depth / 2))
    return _bm_obj(name, bm, coll, mat)


def box(name, size, coll, mat=None, origin=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, verts=bm.verts, vec=origin)
    return _bm_obj(name, bm, coll, mat)


def torus(name, R, r, coll, mat=None):
    bm = bmesh.new()
    segs, rs = 32, 8
    verts = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        for j in range(rs):
            b = 2 * math.pi * j / rs
            verts.append(bm.verts.new(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b))))
    for i in range(segs):
        for j in range(rs):
            a0 = verts[i * rs + j]
            a1 = verts[((i + 1) % segs) * rs + j]
            a2 = verts[((i + 1) % segs) * rs + (j + 1) % rs]
            a3 = verts[i * rs + (j + 1) % rs]
            bm.faces.new((a0, a1, a2, a3))
    return _bm_obj(name, bm, coll, mat)


def rock(name, r, coll, mat=None, seed=0):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
    rng = np.random.default_rng(seed)
    for v in bm.verts:
        v.co *= 0.75 + 0.5 * rng.random()
    return _bm_obj(name, bm, coll, mat)


# --------------------------------------------------------------------------------------------- keys
def key_path(ob, frames, pos, rot_euler=None, scale=None):
    fr = np.asarray(frames, np.float64)
    C.bake_obj(ob, "location", fr, np.asarray(pos))
    if rot_euler is not None:
        C.bake_obj(ob, "rotation_euler", fr, np.asarray(rot_euler))
    if scale is not None:
        sc = np.asarray(scale, np.float64)
        if sc.ndim == 1:
            sc = np.repeat(sc[:, None], 3, 1)
        C.bake_obj(ob, "scale", fr, sc)


def window(ob, a, b, viewport=True):
    """visible on frames a..b inclusive, hidden elsewhere"""
    C.key_visibility(ob, [(1, True), (a, False), (b + 1, True)], viewport=viewport)


def look_euler(direction, up=(0, 0, 1), axis="Z"):
    """euler that points the object's +axis along direction"""
    d = Vector(direction).normalized()
    q = d.to_track_quat(axis, "Y" if axis == "Z" else "Z")
    return q.to_euler()


def flash(S, name, p, f, size=0.8, color=(1.0, 0.75, 0.35), strength=60.0, light=1500.0):
    """impact package: 1-frame emissive spark + 2-frame point-light spike at p on frame f"""
    ob = ico(f"FX_flash_{name}", size * 0.45, S.col.fx, mat_emit(f"M_flash_{color}", color, strength), subdiv=2)
    ob.location = Vector(p)
    window(ob, f, f)
    ld = bpy.data.lights.new(f"FXL_{name}", "POINT")
    ld.energy = light
    ld.color = color
    ld.shadow_soft_size = 0.5
    lo = bpy.data.objects.new(f"FXL_{name}", ld)
    S.col.fx.objects.link(lo)
    lo.location = Vector(p)
    window(lo, f, f + 2)
    return ob


def ring(S, name, p, f, r_end, dur=14, color=(1.0, 0.8, 0.5), strength=12.0):
    """ground shockwave ring: scales 0.05 → r_end over dur frames, then hidden"""
    ob = torus(f"FX_ring_{name}", 1.0, 0.06, S.col.fx, mat_emit(f"M_ring_{color}", color, strength))
    fr = np.arange(f, f + dur + 1)
    t = (fr - f) / dur
    s = 0.05 + (r_end - 0.05) * (1 - (1 - t) ** 2)
    pos = np.repeat(np.asarray(p, np.float64)[None], len(fr), 0)
    key_path(ob, fr, pos, scale=np.c_[s, s, np.full_like(s, 1.0 + 3 * t)])
    window(ob, f, f + dur)
    return ob


def path_arc(p0, p1, n, h=0.0):
    t = np.linspace(0, 1, n)[:, None]
    p = p0 + (p1 - p0) * t
    p[:, 2] += h * 4 * t[:, 0] * (1 - t[:, 0])
    return p


def surface_hit(bvh, p_from, p_to, radius=0.0):
    """first surface point on the segment p_from→p_to (or None)"""
    d = Vector(p_to) - Vector(p_from)
    L = d.length
    hit = bvh.ray_cast(Vector(p_from), d.normalized(), L + 2.0)
    if hit[0] is None:
        return None
    return np.array(hit[0])


def attach(ob, arm, bone, loc_world=(0.0, 0.0, 0.0)):
    """bone-parent ob at the bone's tail, cancelling the armature's object scale (glTF rigs are cm-scaled)."""
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = bone
    s = arm.matrix_world.to_scale()[0]
    ob.scale = (1.0 / s,) * 3
    ob.location = tuple(v / s for v in loc_world)
    return ob
