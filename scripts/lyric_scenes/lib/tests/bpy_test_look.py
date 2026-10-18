"""Headless Blender test for lib/look.py. Through the slice:
    scripts/server/render.sh t-liblook --wait --collect -P ~/.local/bin/blender -b --factory-startup \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/tests/bpy_test_look.py
Includes two 16×16 Cycles CPU renders of the world only (no geometry, 4 spp) — the ground truth for the HDRI mapping."""
import math
import os
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib import look as L  # noqa: E402

FAILS = []


def check(name, cond, info=""):
    print(f"[bpy_test_look] {'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAILS.append(name)


def render_px(sc, path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path)
    px = np.empty(len(im.pixels), np.float32)
    im.pixels.foreach_get(px)
    w, h = im.size
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[:, :, :3].sum(2)


def main():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    td = tempfile.mkdtemp(prefix="liblook_")

    # --- colour management ---
    check("bare 'Punchy' resolves to 'AgX - Punchy'", L.color(sc, look="Punchy") == "AgX - Punchy")
    ok = []
    for nm in L.AGX_LOOKS:
        ok.append(L.color(sc, look=nm) == nm)
    check(f"all {len(L.AGX_LOOKS)} documented AgX looks exist in {bpy.app.version_string}", all(ok))
    try:
        L.color(sc, look="Medium Contrast")
        check("an unknown look raises", False)
    except ValueError:
        check("an unknown look raises", True)
    L.color(sc, view="Standard", look=None, exposure=-1.5)
    check("view + exposure", sc.view_settings.view_transform == "Standard" and sc.view_settings.exposure == -1.5)

    # --- EEVEE defaults (strict) ---
    L.eevee_defaults(sc, samples=24, res=(1280, 720), pct=50, fps=25, look="High Contrast")
    r, ee = sc.render, sc.eevee
    check("eevee_defaults", r.engine == "BLENDER_EEVEE" and ee.taa_render_samples == 24 and r.resolution_x == 1280 and
          r.fps == 25 and r.use_motion_blur and r.motion_blur_position == "START" and ee.use_raytracing and
          ee.ray_tracing_options.resolution_scale == "2" and r.image_settings.color_depth == "8" and
          sc.view_settings.look == "AgX - High Contrast")
    L.eevee_volumes(sc, start=0.5, end=4000, tile="2", samples=96)
    check("eevee_volumes", ee.use_volume_custom_range and ee.volumetric_end == 4000 and ee.volumetric_tile_size == "2")
    try:
        L._strict_set(ee, {"no_such_setting": 1}, "eevee")
        check("strict setter raises on a renamed property", False)
    except AttributeError:
        check("strict setter raises on a renamed property", True)

    # --- compositor chain (5.x API) ---
    sc.world = bpy.data.worlds.new("W")
    P = L.post_chain(sc, mist=dict(start=30, depth=1100, color=(0.02, 0.024, 0.03), k=0.55),
                     glare=[dict(type="Fog Glow", threshold=1.2, strength=0.4),
                            dict(type="Streaks", threshold=8.0, strength=0.3, streaks=2, angle=15)],
                     lens=(-0.008, 0.012), vignette=0.72, grain=0.05)
    cg = P["tree"]
    names = {n.name for n in cg.nodes}
    check("post_chain is the scene's compositor", sc.compositing_node_group == cg and r.use_compositing)
    check("chain nodes carry stable names", {"Haze", "Glare0", "Glare1", "Lens", "Vignette", "Grain"} <= names, str(sorted(names)))
    g0, g1 = cg.nodes["Glare0"], cg.nodes["Glare1"]
    check("glare options set via input sockets", g0.inputs["Type"].default_value == "Fog Glow" and
          g1.inputs["Type"].default_value == "Streaks" and g1.inputs["Streaks"].default_value == 2 and
          abs(g1.inputs["Streaks Angle"].default_value - math.radians(15)) < 1e-6)
    check("mist pass on + world mist range", sc.view_layers[0].use_pass_mist and sc.world.mist_settings.depth == 1100 and
          P["rl"].outputs["Mist"].is_linked)
    out = P["out"].inputs["Image"]
    check("chain ends at the group output", out.is_linked and out.links[0].from_node.bl_idname == "ShaderNodeMix")
    fc = [f for ly in cg.animation_data.action.layers for st in ly.strips for cb in st.channelbags for f in cb.fcurves][0]
    check("grain W follows the frame (linear extrapolation, no driver)",
          abs(fc.evaluate(1) - 1) < 1e-6 and abs(fc.evaluate(3000) - 3000) < 1e-3, f"{fc.evaluate(3000)}")
    lnk = {(lk.from_node.name, lk.to_node.name) for lk in cg.links}
    check("order: haze → glare0 → glare1 → lens → vignette", ("Haze", "Glare0") in lnk and ("Glare0", "Glare1") in lnk
          and ("Glare1", "Lens") in lnk and ("Lens", "Vignette") in lnk)

    # --- HDRI world: a synthetic sky with ONE bright spot; render toward where hdri_rotation says it went ---
    w, h = 64, 32
    px = np.full((h, w, 4), 0.02, np.float32)
    px[..., 3] = 1
    px[20:23, 9:12, :3] = 400.0                                         # off-centre: catches a mirrored/shifted mapping
    px[21, 10, :3] = 500.0                                              # (boss v1's formula was 180° off — measured)
    im = bpy.data.images.new("spot", w, h, float_buffer=True)
    im.pixels.foreach_set(px.ravel())
    hdr = os.path.join(td, "spot.hdr")
    im.filepath_raw, im.file_format = hdr, "HDR"
    im.save()
    bpy.data.images.remove(im)
    az0, el, (ix, iy) = L.hdri_key(bpy.data.images.load(hdr))
    check("hdri_key finds the planted spot", (ix, iy) == (10, 21), f"{(ix, iy)}")
    want = math.radians(118)
    rot = L.hdri_rotation(az0, want)
    world, nodes = L.hdri_world(hdr, rot_deg=math.degrees(rot), camera=1.0)
    check("hdri_world: env + mapping + camera-ray mix", nodes["env"].image is not None and "mix" in nodes and
          abs(nodes["mapping"].inputs["Rotation"].default_value[2] - rot) < 1e-6)
    sc.render.engine = "CYCLES"
    sc.cycles.device, sc.cycles.samples = "CPU", 4
    sc.cycles.use_denoising = False
    sc.render.resolution_x = sc.render.resolution_y = 16
    sc.render.resolution_percentage = 100
    sc.render.use_compositing = False
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = "OPEN_EXR", "32"
    sc.view_settings.view_transform = "Standard"
    cam = bpy.data.objects.new("C", bpy.data.cameras.new("C"))
    sc.collection.objects.link(cam)
    cam.data.lens = 50
    sc.camera = cam
    d = np.array([math.cos(want) * math.cos(el), math.sin(want) * math.cos(el), math.sin(el)])
    from mathutils import Vector
    cam.rotation_euler = Vector(d).to_track_quat("-Z", "Y").to_euler()
    lum = render_px(sc, os.path.join(td, "toward.exr"))
    cy, cx = np.unravel_index(int(np.argmax(lum)), lum.shape)
    check("render toward the predicted direction sees the spot near frame centre",
          lum.max() > 20 and abs(cx - 7.5) <= 3 and abs(cy - 7.5) <= 3, f"max {lum.max():.1f} at ({cx},{cy})")
    cam.rotation_euler = Vector(-d).to_track_quat("-Z", "Y").to_euler()
    lum2 = render_px(sc, os.path.join(td, "away.exr"))
    check("render away from it sees only the dim sky", lum2.max() < 1.0, f"max {lum2.max():.2f}")

    print(f"[bpy_test_look] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  ({td})", flush=True)
    sys.exit(1 if FAILS else 0)


main()
