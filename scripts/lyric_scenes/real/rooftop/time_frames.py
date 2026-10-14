"""Per-frame render timing harness (v2.2). Renders each frame twice and reports the WARM (second) time, so
shader compilation doesn't pollute the number. Optional config variants override render settings.
    blender -b bemyone.blend --python time_frames.py -- OUTDIR TAG FRAMES [k=v ...]
    e.g.  ... -- /tmp/t base 1021,2341   |   ... -- /tmp/t s1 1021 samples=1   |   ... nocomp=1 nohaze=1"""
import os, sys, time
import bpy
a = sys.argv[sys.argv.index("--") + 1:]
out, tag, frames = a[0], a[1], [int(x) for x in a[2].split(",")]
kv = dict(x.split("=", 1) for x in a[3:])
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
ee = sc.eevee
if "samples" in kv:
    ee.taa_render_samples = int(kv["samples"])
    if sc.animation_data and sc.animation_data.action:       # per-shot keyed samples would override → drop for the test
        pass
if kv.get("nocomp") == "1":
    sc.render.use_compositing = False
if kv.get("nohaze") == "1":
    bpy.data.objects["HazeBox"].hide_render = True
if "voltile" in kv:
    ee.volumetric_tile_size = kv["voltile"]
if "volsamples" in kv:
    ee.volumetric_samples = int(kv["volsamples"])
if "shadowres" in kv:
    ee.shadow_resolution_scale = float(kv["shadowres"])
if kv.get("nofastgi") == "1":
    ee.use_fast_gi = False
if "trough" in kv:
    ee.ray_tracing_options.trace_max_roughness = float(kv["trough"])
if kv.get("nomb") == "1":
    sc.render.use_motion_blur = False
if "glareq" in kv:
    for n in sc.compositing_node_group.nodes:
        if n.bl_idname == "CompositorNodeGlare":
            n.inputs["Quality"].default_value = kv["glareq"]
if "hide" in kv:
    for nm in kv["hide"].split(","):
        for o in bpy.data.objects:
            if o.name.startswith(nm):
                o.hide_render = True
                if o.animation_data:
                    o.animation_data_clear()
res = []
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"{tag}_f{f:04d}.png")
    for k in range(2):
        t0 = time.time()
        bpy.ops.render.render(write_still=(k == 1))
        dt = time.time() - t0
    res.append((f, dt))
print(f"[timing] {tag} samples={ee.taa_render_samples} " + " ".join(f"{f}:{dt:.2f}s" for f, dt in res) +
      f" | mean {sum(d for _, d in res) / len(res):.2f}s")
