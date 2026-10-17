"""Headless Blender test for render/patch.py + qa/time_ab.py (tiny 64x36 EEVEE renders, ~1 min). Through the slice:
    scripts/server/render.sh t-libqa --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_qa.py
Prints PASS/FAIL per check and exits 1 on any failure."""
import json
import os
import sys
import tempfile

import bpy

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LIB, "qa"))
sys.path.insert(0, os.path.join(LIB, "render"))
import patch  # noqa: E402
import time_ab  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_qa] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def mean_red(path):
    im = bpy.data.images.load(path)
    px = list(im.pixels)
    bpy.data.images.remove(im)
    return sum(px[0::4]) / (len(px) / 4)


def scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 64, 36, 100
    sc.eevee.taa_render_samples = 1
    sc.render.image_settings.file_format = "PNG"
    sc.world = sc.world or bpy.data.worlds.new("W")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    me = bpy.data.meshes.new("gem")
    me.from_pydata([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], [],
                   [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
    mat = bpy.data.materials.new("red")
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1, 0, 0, 1)
    em.inputs["Strength"].default_value = 5
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    me.materials.append(mat)
    fog = bpy.data.collections.new("FOG")
    sc.collection.children.link(fog)
    gem = bpy.data.objects.new("Gem_1", me)
    sc.collection.objects.link(gem)
    gem.scale = (3, 3, 3)
    gem.hide_render = False
    gem.keyframe_insert("hide_render", frame=1)            # a KEYED visibility, like per-shot existence
    other = bpy.data.objects.new("FogBox", me)
    fog.objects.link(other)
    other.location = (0, 0, -50)
    for name, loc in (("CAM_a", (0, -12, 0)), ("CAM_b", (0, 12, 0))):
        cd = bpy.data.cameras.new(name)
        c = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(c)
        c.location = loc
        c.rotation_euler = (1.5708, 0, 0 if loc[1] < 0 else 3.14159)
    sc.camera = bpy.data.objects["CAM_a"]
    sc.frame_set(1)
    return sc


def main():
    sc = scene()
    td = tempfile.mkdtemp(prefix="libqa_")
    gem = bpy.data.objects["Gem_1"]
    fc = patch._keyed_fcurves(gem, "hide_render")
    check("keyed fcurve found on 5.x slotted action", len(fc) == 1, f"n={len(fc)}")

    # 1. keyed hide_render is overridden by mute and restored by undo
    u = patch.apply(sc, {"hide_prefix": ["Gem_"], "hide_collections": "FOG", "exposure_offset": 1.5,
                         "set": {"scene.eevee.taa_render_samples": 2, "objects['FogBox'].location": (1, 2, 3)}})
    sc.frame_set(1)
    check("hidden keyed object stays hidden after frame_set", gem.hide_render is True)
    check("fcurve muted", fc[0].mute is True)
    check("collection hidden", bpy.data.collections["FOG"].hide_render is True)
    check("exposure offset", abs(sc.view_settings.exposure - 1.5) < 1e-6)
    check("set scene path", sc.eevee.taa_render_samples == 2)
    check("set data path", tuple(bpy.data.objects["FogBox"].location) == (1, 2, 3))
    sc.render.filepath = os.path.join(td, "hidden.png")
    bpy.ops.render.render(write_still=True)
    u()
    sc.frame_set(1)
    check("undo restores", gem.hide_render is False and fc[0].mute is False and sc.view_settings.exposure == 0
          and sc.eevee.taa_render_samples == 1 and not bpy.data.collections["FOG"].hide_render
          and tuple(bpy.data.objects["FogBox"].location) == (0, 0, -50))
    sc.render.filepath = os.path.join(td, "shown.png")
    bpy.ops.render.render(write_still=True)
    r_h, r_s = mean_red(os.path.join(td, "hidden.png")), mean_red(os.path.join(td, "shown.png"))
    check("render honours the patch (red gone when hidden)", r_s > 0.3 and r_h < 0.05, f"shown {r_s:.3f} hidden {r_h:.3f}")

    try:
        patch.apply(sc, {"hide_colections": ["FOG"]})
        check("typo key rejected", False)
    except KeyError:
        check("typo key rejected", True)

    # 2. time_ab end to end
    out = os.path.join(td, "times.json")
    time_ab.main(["--cams", "a,b", "--cam-prefix", "CAM_", "--variants",
                  json.dumps({"no_gem": {"hide_prefix": ["Gem_"]}, "s4": {"eevee": {"taa_render_samples": 4}}}),
                  "--out", out, "--bookend", "--save", os.path.join(td, "ab")])
    res = json.load(open(out))
    check("time_ab run count (3 variants x 2 shots + bookend 2)", len(res["runs"]) == 8, str(len(res["runs"])))
    check("time_ab summary keys", set(res["variants"]) == {"base", "no_gem", "s4"} and res["delta"]["base"] == 0)
    rb = mean_red(os.path.join(td, "ab", "base__a.png"))
    rn = mean_red(os.path.join(td, "ab", "no_gem__a.png"))
    check("time_ab variant really applied (saved renders differ)", rb > 0.3 and rn < 0.05, f"{rb:.3f} vs {rn:.3f}")
    check("time_ab restored the scene afterwards", gem.hide_render is False and sc.eevee.taa_render_samples == 1)
    print(f"[bpy_test_qa] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  (artefacts {td})", flush=True)
    sys.exit(1 if FAILS else 0)


main()
