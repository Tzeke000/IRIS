"""Render the arena preview stills from arena_v1.blend (per-still visibility tweaks are applied in memory only; the
.blend on disk is never modified). Launched by render.sh (under scripts/server/render.sh + the GPU lock):

    blender -b arena_v1.blend --python-exit-code 1 --python render_stills.py -- --out DIR --names a_wide,b_stairs [--scale 100]
"""
import argparse
import json
import os
import sys
import time

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--names", required=True)
ap.add_argument("--scale", type=int, default=100)
ap.add_argument("--samples", type=int, default=0)
a = ap.parse_args(argv)
SC = bpy.context.scene
SC.render.resolution_percentage = a.scale
if a.samples:
    SC.eevee.taa_render_samples = a.samples
os.makedirs(a.out, exist_ok=True)
D = bpy.data


def group(name, role=None):
    return [o for o in D.objects if o.get("breakable_group") == name and (role is None or o.get("role") == role)]


def set_hidden(objs, hide):
    for o in objs:
        o.hide_render = hide
        o.hide_viewport = hide


def tree(root):
    out = [root]
    for c in root.children_recursive:
        out.append(c)
    return out


def world_of(o):
    """hidden (hide_viewport) objects are not evaluated, so their matrix_world is stale: rebuild it from the chain."""
    if o.parent is None:
        return o.matrix_basis.copy()
    return world_of(o.parent) @ o.matrix_parent_inverse @ o.matrix_basis


def explode(cells, center_fn, k=0.6, lift=0.0):
    for c in cells:
        p = world_of(c).translation.copy()
        cen = center_fn(p)
        d = p - cen
        M = world_of(c)
        M.translation = p + d * k + Vector((0, 0, lift * (p.z - cen.z if lift else 0)))
        c.matrix_world = M


def temp_light(name, loc, target, energy, size):
    ld = D.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    ld.color = (0.75, 0.82, 1.0)
    o = D.objects.new(name, ld)
    SC.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return lambda: D.objects.remove(o)


def setup(name):
    """returns an undo callable"""
    undo = []
    proof = D.collections["SIZE_PROOF"]
    proof.hide_render = name != "d_overhead"
    if name == "d_overhead":
        hide = [o for o in D.objects if o.get("role") in ("intact", "piece")
                and o.get("breakable_group", "").startswith(("Vault", "RoofSlates", "RoofTimber"))]
        hide += list(D.collections["ROOF_static_ruin"].all_objects)
        hide += [o for o in D.objects if o.name.startswith("Clerestory_")]
        hide += tree(D.objects["DRAGON"])
        hide += list(D.collections["FX_haze"].all_objects)
        set_hidden(hide, True)
        undo.append(lambda: set_hidden(hide, False))
        ld = D.lights.new("PlanFill", "SUN")
        ld.energy = 0.6
        ld.color = (0.8, 0.85, 1.0)
        ld.angle = 0.2
        lo = D.objects.new("PlanFill", ld)
        SC.collection.objects.link(lo)
        lo.rotation_euler = (0.25, 0.15, 0.0)
        undo.append(lambda: D.objects.remove(lo))
    if name == "e_pillar":
        g = "Pier_E2"
        set_hidden(group(g, "intact"), True)
        cells = group(g, "cell")
        set_hidden(cells, False)
        root = D.objects[g]
        base = world_of(root).translation.copy()
        explode(cells, lambda p: Vector((base.x, base.y, p.z * 0.8 + base.z * 0.2)), k=1.6, lift=0.18)
        undo.append(temp_light("ExplodeKey", base + Vector((-6.0, -6.0, 9.0)), base + Vector((0, 0, 9.0)), 1200, 6.0))
    if name == "f_roof":
        for g in ("Vault", "VaultRib0", "VaultRib1", "VaultRib2", "RoofSlates_E", "RoofSlates_W"):
            set_hidden(group(g, "intact"), True)
            cells = group(g, "cell")
            set_hidden(cells, False)
            cen = sum((world_of(c).translation for c in cells), Vector()) / max(1, len(cells))
            explode(cells, lambda p, cen=cen: Vector((p.x * 0.4, cen.y + (p.y - cen.y) * 0.5, p.z + 3.0)), k=0.45)
        undo.append(temp_light("RoofKey", Vector((0, 20.0, 18.0)), Vector((0, 38.0, 38.0)), 7000, 10.0))
    return undo


times = {}
for name in a.names.split(","):
    cam = D.objects.get("CAM_" + name)
    if cam is None:
        print("[stills] no camera", name)
        continue
    snap = [(o, o.hide_render, o.hide_viewport, o.matrix_basis.copy()) for o in D.objects if o.get("breakable_group")]
    undo = setup(name)
    undo.append(lambda snap=snap: [(setattr(o, "hide_render", hr), setattr(o, "hide_viewport", hv),
                                    setattr(o, "matrix_basis", M)) for o, hr, hv, M in snap])
    SC.camera = cam
    ex0 = SC.view_settings.exposure
    if name == "d_overhead":
        SC.view_settings.exposure = ex0 + 0.6
    SC.render.filepath = os.path.join(os.path.abspath(a.out), f"{name}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    times[name] = round(time.time() - t0, 1)
    print(f"[stills] {name} {times[name]} s", flush=True)
    SC.view_settings.exposure = ex0
    for u in undo:
        u()
json.dump(times, open(os.path.join(a.out, "times.json"), "w"), indent=1)
