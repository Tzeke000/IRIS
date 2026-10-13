"""Render frames of bemyone.blend (resumable). Launched by render.sh (which itself runs under
scripts/server/render.sh + the GPU lock):

    blender -b bemyone.blend --python-exit-code 1 --python render_frames.py -- \
        --out DIR --frames 1-150 [--scale 100] [--samples 48] [--times times.txt]

--frames "a-b" renders an animation range (skips frames already on disk: resume);
"a,b,c" renders individual stills. Files: <out>/f_####.png (#### = scene frame).
"""
import argparse
import os
import sys
import time

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--times", default="")
    ap.add_argument("--no-motion-blur", action="store_true")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    sc.render.resolution_percentage = a.scale
    if a.samples:
        sc.eevee.taa_render_samples = a.samples
    if a.no_motion_blur:
        sc.render.use_motion_blur = False
    os.makedirs(a.out, exist_ok=True)
    sc.render.use_overwrite = False
    sc.render.use_placeholder = False
    times = []
    t_last = [time.time()]

    def _pre(scene, *_):
        t_last[0] = time.time()

    def _write(scene, *_):
        times.append((scene.frame_current, time.time() - t_last[0]))

    bpy.app.handlers.render_pre.append(_pre)
    bpy.app.handlers.render_write.append(_write)
    t0 = time.time()
    if "-" in a.frames and "," not in a.frames:
        lo, hi = (int(v) for v in a.frames.split("-"))
        sc.frame_start, sc.frame_end = lo, hi
        sc.render.filepath = os.path.join(os.path.abspath(a.out), "f_####")
        todo = [f for f in range(lo, hi + 1) if not os.path.exists(os.path.join(a.out, f"f_{f:04d}.png"))]
        if todo:
            sc.frame_start = todo[0]
            bpy.ops.render.render(animation=True)
    else:
        for f in [int(v) for v in a.frames.split(",")]:
            path = os.path.join(os.path.abspath(a.out), f"f_{f:04d}.png")
            if os.path.exists(path):
                continue
            sc.frame_set(f)
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
    wall = time.time() - t0
    msg = " ".join(f"{f}:{t:.2f}s" for f, t in times)
    print(f"[render_frames] {len(times)} frames in {wall:.1f}s | {msg}")
    if a.times:
        with open(a.times, "a") as fh:
            for f, t in times:
                fh.write(f"{f} {t:.3f}\n")


if __name__ == "__main__":
    main()
