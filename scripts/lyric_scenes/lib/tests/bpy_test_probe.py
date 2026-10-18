"""Headless Blender test for lib/assets/probe.py. Through the slice:
    scripts/server/render.sh t-libprobe --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_probe.py
Fixtures: the importer test's hand-written glTFs (KHR_materials_unlit; a missing texture at 180 m scale), plus a .glb
and an .fbx exported from Blender here (rig + action + skinned mesh; plain 2 m cube)."""
import json
import os
import sys
import tempfile

import bpy

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(T)))
from lib.assets import probe  # noqa: E402
from lib.tests.bpy_test_importer import gltf, png_bytes  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_probe] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def export_rigged_glb(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    arm = bpy.data.armatures.new("RigData")
    rig = bpy.data.objects.new("Rig", arm)
    sc.collection.objects.link(rig)
    with bpy.context.temp_override(object=rig, active_object=rig, selected_objects=[rig], selected_editable_objects=[rig]):
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for nm, h, t, par in (("Root", (0, 0, 0), (0, 0, 1), None), ("Arm", (0, 0, 1), (0, 0, 2), "Root")):
            b = arm.edit_bones.new(nm)
            b.head, b.tail = h, t
            if par:
                b.parent = arm.edit_bones[par]
        bpy.ops.object.mode_set(mode="OBJECT")
    me = bpy.data.meshes.new("BodyMesh")
    me.from_pydata([(-0.2, -0.2, 0), (0.2, -0.2, 0), (0.2, 0.2, 0), (-0.2, 0.2, 0),
                    (-0.2, -0.2, 2), (0.2, -0.2, 2), (0.2, 0.2, 2), (-0.2, 0.2, 2)], [],
                   [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    body = bpy.data.objects.new("Body", me)
    sc.collection.objects.link(body)
    body.parent = rig
    vg = body.vertex_groups.new(name="Arm")
    vg.add(list(range(8)), 1.0, "REPLACE")
    body.modifiers.new("Armature", "ARMATURE").object = rig
    pb = rig.pose.bones["Arm"]
    pb.rotation_mode = "XYZ"
    for f, a in ((1, 0.0), (24, 0.8)):
        pb.rotation_euler = (a, 0, 0)
        pb.keyframe_insert("rotation_euler", frame=f)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_animations=True)


def export_cube_fbx(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("CubeMesh")
    me.from_pydata([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (0, 2)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    ob = bpy.data.objects.new("Crate", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.ops.export_scene.fbx(filepath=path)


def main():
    td = tempfile.mkdtemp(prefix="libprobe_")
    tex = os.path.join(td, "albedo.png")
    open(tex, "wb").write(png_bytes())
    unlit, huge = os.path.join(td, "unlit.gltf"), os.path.join(td, "huge_missing.gltf")
    gltf(unlit, True, "albedo.png")
    gltf(huge, False, "missing_tex.png", scale=180.0)
    rigged, crate = os.path.join(td, "rigged.glb"), os.path.join(td, "crate.fbx")
    export_rigged_glb(rigged)
    export_cube_fbx(crate)
    broken = os.path.join(td, "broken.glb")
    open(broken, "wb").write(b"not a glb")
    out = os.path.join(td, "probe.json")

    rc = probe.main(["--out", out, unlit, huge, rigged, crate, broken])
    J = json.load(open(out))
    F = J["files"]
    check("exit 0 without --strict; every file has a record", rc == 0 and len(F) == 5)
    u = F[unlit]
    check("unlit material flagged", u["materials"][0]["unlit"] and any("unlit" in x for x in u["flags"]), str(u["flags"]))
    check("quad: 1 mesh, 2 tris, 1 m wide", u["totals"]["tris"] == 2 and abs(u["size_m"][0] - 1.0) < 1e-3, str(u["size_m"]))
    h = F[huge]
    probs = h["materials"][0]["images"][0]["problems"]
    check("missing texture flagged on the material's image", any("missing" in p for p in probs), str(probs))
    check("180 m import flagged as centimetre units", any("centimetre" in x for x in h["flags"]), str(h["flags"]))
    r = F[rigged]
    check("rig: armature with 2 bones, root listed", r["armatures"] and r["armatures"][0]["bones"] == 2 and
          r["armatures"][0]["roots"] == ["Root"], str(r["armatures"]))
    check("rig: action with keys + slot", r["actions"] and r["actions"][0]["fcurves"] >= 3 and r["actions"][0]["slots"],
          str(r["actions"]))
    m = [x for x in r["meshes"] if x["name"] == "Body"]
    check("rig: skinned mesh, 2 m tall, per-mesh bbox in metres", m and m[0]["skinned"] and abs(m[0]["dims"][2] - 2.0) < 1e-3
          and r["longest_axis"] == "Z", str(r["meshes"]))
    check("rig: the importer's bone widget is listed apart, not counted", r["bone_widgets"] and len(r["meshes"]) == 1 and
          r["totals"]["polys"] == 12 and abs(r["size_m"][2] - 2.0) < 1e-3,      # glTF stores the 6 quads as 12 tris
          f'{r["bone_widgets"]} {r["size_m"]}')
    check("rig: clean (no flags)", r["flags"] == [], str(r["flags"]))
    c = F[crate]
    check("fbx: 6 polys, 2 m cube", c["totals"]["polys"] == 6 and all(abs(x - 2.0) < 1e-3 for x in c["size_m"]),
          str(c.get("size_m")))
    check("broken file: error recorded, others still probed", "error" in F[broken] and F[broken]["flags"])
    rc2 = probe.main(["--out", out, unlit, "--strict"])
    rc3 = probe.main(["--out", out, rigged, "--strict"])
    check("--strict: exit 1 on flags, 0 when clean", rc2 == 1 and rc3 == 0, f"{rc2} {rc3}")
    print(f"[bpy_test_probe] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  ({td})", flush=True)
    sys.exit(1 if FAILS else 0)


main()
