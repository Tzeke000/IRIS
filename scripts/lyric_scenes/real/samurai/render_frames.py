"""Render frames of samurai.blend one by one (used by render.sh).

    blender -b samurai.blend --python-exit-code 1 --python render_frames.py -- \
        --out DIR (--frames 301,988,1000-1010 | --start S --end E) [--pct 100]
        [--engine EEVEE|CYCLES] [--samples N] [--overwrite]

* Frames are Blender frames (1-based; timeline index = frame - 1).
* RESUME: an existing, non-trivial DIR/f_####.png is skipped unless --overwrite.
  Each frame is written to a temp name and renamed, so a kill never leaves a half PNG.
* Optional per-frame EEVEE volume range (scene["vol_end"], if the build stored one):
  volumetric_end is not animatable, so it is applied before each frame.
Prints 'TIME <frame> <seconds>' per rendered frame.
"""
import argparse
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=0)
ap.add_argument("--pct", type=int, default=0)
ap.add_argument("--engine", default="")
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args(argv)

sc = bpy.context.scene
if a.pct:
    sc.render.resolution_percentage = a.pct
if a.engine == "CYCLES":
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    if a.samples:
        sc.cycles.samples = a.samples
elif a.engine == "EEVEE":
    sc.render.engine = "BLENDER_EEVEE"
if a.samples and sc.render.engine == "BLENDER_EEVEE":
    sc.eevee.taa_render_samples = a.samples

frames = []
if a.frames:
    for part in a.frames.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            frames += list(range(int(lo), int(hi) + 1))
        else:
            frames.append(int(part))
else:
    frames = list(range(a.start or sc.frame_start, (a.end or sc.frame_end) + 1))

vol_end = list(sc.get("vol_end", []))
os.makedirs(a.out, exist_ok=True)
done = skipped = 0
for f in frames:
    dst = os.path.join(a.out, f"f_{f:04d}.png")
    if not a.overwrite and os.path.exists(dst) and os.path.getsize(dst) > 1024:
        skipped += 1
        continue
    if vol_end and 0 <= f - 1 < len(vol_end):
        sc.eevee.volumetric_end = vol_end[f - 1]
    sc.frame_set(f)
    tmp = os.path.join(a.out, f".tmp_f_{f:04d}.png")
    sc.render.filepath = tmp
    t = time.time()
    bpy.ops.render.render(write_still=True)
    os.replace(tmp, dst)
    done += 1
    print(f"TIME {f} {time.time() - t:.2f}", flush=True)
print(f"[render_frames] rendered {done}, skipped {skipped} existing -> {a.out}", flush=True)
