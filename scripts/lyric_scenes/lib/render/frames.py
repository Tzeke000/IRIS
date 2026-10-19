"""Blender-side frame / still renderer for every scene (replaces the 4 render_frames.py families + boss render_stills.py).
Launched by lib/render/scene.sh (always inside a scripts/server/render.sh unit):

    blender -b SCENE.blend --python-exit-code 1 --python /ABS/lib/render/frames.py -- --out DIR
        (--frames 1-60,301,988 | --start S --end E | --cams a_wide,c_hall [--cam-prefix CAM_] [--at FRAME])
        [--pct N] [--samples N] [--engine EEVEE|CYCLES] [--overwrite] [--times FILE] [--vram] [--prefix f_]
        [--patch JSON|FILE]          whole-run patch (lib/render/patch.py spec), e.g. '{"render":{"use_compositing":false}}'
        [--cam-props JSON|FILE]      {"c_hall": {"exposure_offset": 0.3, "hide_collections": ["FX_haze"]}} — merged OVER the
                                     camera object's own custom props (exposure_offset, hide_collections, patch=JSON)
        [--hook FILE.py]             scene-only still setup: `def setup(scene, name) -> list[callable]` (undo callables)
        [--anim]                     frames only: render each CONTIGUOUS run of missing frames as ONE animation render
                                     (render.render(animation=True)) instead of one still render per frame

* Frames → DIR/f_####.png (Blender frames, 1-based). Cams → DIR/<cam>.png (at --at, else the file's current frame).
* ATOMIC: each image renders to DIR/.tmp_<name>.<pid>.png, is checked to be a complete PNG, then os.replace()d.
  RESUME skips only COMPLETE PNGs (lib/render/pngcheck.py: signature + IEND), so a torn file from any older runner
  is re-rendered instead of shipped. Stills (--cams) always overwrite; frames only with --overwrite.
* Per-frame scene props if the build stored them (one value per frame, index = frame-1): vol_start, vol_end,
  vol_dist, vol_tile, spp (spp ignored when --samples is given — look-dev override).
* Forces PNG output (the completeness check and the mux expect it).
* --anim (spooky v3, 10-19 — MEASURED): a still render rebuilds the render depsgraph, re-evaluates every modifier/GN tree
  and re-uploads every mesh to the GPU for EACH frame; an animation render keeps the depsgraph and only re-evaluates
  what is time-dependent. Spooky v2, 15 consecutive frames: 6.5 s/frame (stills) → 2.9 s/frame (anim), pixels equal
  (max 1–3/255, 0 px over 8/255). render.use_persistent_data gave nothing in either mode. Same atomic contract: Blender
  writes DIR/.tmp_f_####.<pid>.png, a render_write handler checks + os.replace()s it; per-frame props are applied by
  a frame_change_pre handler; any frame that did not land complete → exit 1.
Prints 'TIME <name> <seconds>' per image and a final '[frames] rendered N, skipped M' line; exit 1 if anything failed.
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch  # noqa: E402
import pngcheck  # noqa: E402


def parse_frames(spec):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def per_frame_props(sc, f, override_samples):
    def at(key):
        v = sc.get(key)
        if v is None:
            return None
        v = list(v)
        return v[f - 1] if 0 <= f - 1 < len(v) else None
    ee = sc.eevee
    for key, attr, conv in (("vol_end", "volumetric_end", float), ("vol_start", "volumetric_start", float),
                            ("vol_dist", "volumetric_sample_distribution", float),
                            ("vol_tile", "volumetric_tile_size", lambda x: str(int(x)))):
        v = at(key)
        if v is not None:
            setattr(ee, attr, conv(v))
    v = at("spp")
    if v is not None and not override_samples:
        ee.taa_render_samples = int(v)


def vram_line(tag):
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        mine = [ln for ln in out.splitlines() if ln.split(",")[0].strip() == str(os.getpid())]
        print(f"VRAM {tag} {mine[0].split(',')[1].strip() if mine else '?'}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("VRAM ?", e, flush=True)


def render_one(bpy, sc, out_dir, name, times_file=None, vram=False):
    """Render the current scene state to out_dir/<name>.png atomically. Returns seconds."""
    dst = os.path.join(out_dir, name + ".png")
    for old in glob.glob(os.path.join(out_dir, f".tmp_{name}.*.png")):   # leftovers of a killed run (frame is ours)
        os.remove(old)
    tmp = os.path.join(out_dir, f".tmp_{name}.{os.getpid()}.png")
    sc.render.filepath = tmp
    t = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t
    if not pngcheck.is_complete_png(tmp):
        raise RuntimeError(f"render of {name} did not produce a complete PNG ({tmp})")
    os.replace(tmp, dst)
    print(f"TIME {name} {dt:.2f}", flush=True)
    if vram:
        vram_line(name)
    if times_file:
        with open(times_file, "a") as fh:
            fh.write(f"{name}\t{dt:.2f}\n")
    return dt


def contiguous_runs(frames):
    """[1,2,3,7,8,10] → [[1,2,3],[7,8],[10]] (sorted, de-duplicated) — one animation render per run"""
    runs = []
    for f in sorted(set(frames)):
        if runs and f == runs[-1][-1] + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def render_anim(bpy, sc, out_dir, frames, prefix="f_", times_file=None, vram=False, override_samples=0):
    """Render `frames` (sorted ints) as one animation render per contiguous run. Returns the number rendered."""
    runs = contiguous_runs(frames)
    r = sc.render
    saved = dict(frame_start=sc.frame_start, frame_end=sc.frame_end, frame_step=sc.frame_step, filepath=r.filepath,
                 use_overwrite=r.use_overwrite, use_placeholder=r.use_placeholder,
                 use_file_extension=r.use_file_extension)
    state = {"t": time.time(), "errors": []}
    pid = os.getpid()

    def on_frame(scene, *_):
        per_frame_props(scene, scene.frame_current, override_samples)

    def on_write(scene, *_):
        f = scene.frame_current
        name = f"{prefix}{f:04d}"
        tmp = scene.render.frame_path(frame=f)
        try:
            if not pngcheck.is_complete_png(tmp):
                raise RuntimeError(f"render of {name} did not produce a complete PNG ({tmp})")
            os.replace(tmp, os.path.join(out_dir, name + ".png"))
        except Exception as e:  # noqa: BLE001 — a handler exception does not fail the render: record it, fail after
            state["errors"].append(str(e))
            return
        now = time.time()
        dt, state["t"] = now - state["t"], now
        print(f"TIME {name} {dt:.2f}", flush=True)
        if vram:
            vram_line(name)
        if times_file:
            with open(times_file, "a") as fh:
                fh.write(f"{name}\t{dt:.2f}\n")

    bpy.app.handlers.frame_change_pre.append(on_frame)
    bpy.app.handlers.render_write.append(on_write)
    try:
        r.filepath = os.path.join(out_dir, f".tmp_{prefix}####.{pid}")
        r.use_overwrite, r.use_placeholder, r.use_file_extension = True, False, True
        sc.frame_step = 1
        for run in runs:
            for old in glob.glob(os.path.join(out_dir, f".tmp_{prefix}*.png")):    # leftovers of a killed run
                fn = os.path.basename(old)
                if any(fn.startswith(f".tmp_{prefix}{f:04d}.") for f in run):
                    os.remove(old)
            sc.frame_start, sc.frame_end = run[0], run[-1]
            state["t"] = time.time()
            bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(on_frame)
        bpy.app.handlers.render_write.remove(on_write)
        for k, v in saved.items():
            setattr(sc if k.startswith("frame_") else r, k, v)
    bad = [f for f in frames if not pngcheck.is_complete_png(os.path.join(out_dir, f"{prefix}{f:04d}.png"))]
    if state["errors"] or bad:
        raise RuntimeError(f"[frames] anim render: {len(bad)} frame(s) not complete {bad[:10]}; errors {state['errors'][:3]}")
    return len(frames)


def load_hook(path):
    spec = importlib.util.spec_from_file_location("scene_render_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    import bpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--cams", default="")
    ap.add_argument("--cam-prefix", default="")
    ap.add_argument("--at", type=int, default=0)
    ap.add_argument("--pct", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--engine", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--times", default="")
    ap.add_argument("--vram", action="store_true")
    ap.add_argument("--prefix", default="f_")
    ap.add_argument("--patch", default="")
    ap.add_argument("--cam-props", default="")
    ap.add_argument("--hook", default="")
    ap.add_argument("--anim", action="store_true")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
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
    if sc.render.image_settings.file_format != "PNG":
        print(f"[frames] file_format {sc.render.image_settings.file_format} → PNG", flush=True)
        sc.render.image_settings.file_format = "PNG"
    run_undo = patch.apply(sc, a.patch) if a.patch else None
    done = skipped = 0
    t_all = time.time()
    if a.cams:
        cam_over = patch.load_spec(a.cam_props)
        hook = load_hook(a.hook) if a.hook else None
        if a.at:
            sc.frame_set(a.at)
        for name in [c for c in a.cams.split(",") if c]:
            cam = bpy.data.objects.get(a.cam_prefix + name) or bpy.data.objects.get(name)
            if cam is None or cam.type != "CAMERA":
                raise SystemExit(f"[frames] no camera {name!r} (prefix {a.cam_prefix!r})")
            undo = [patch.apply(sc, patch.camera_spec(cam, cam_over.get(name)))]
            if hook and hasattr(hook, "setup"):
                undo += list(hook.setup(sc, name) or [])
            sc.camera = cam
            sc.frame_set(sc.frame_current)
            render_one(bpy, sc, out, name, a.times or None, a.vram)
            done += 1
            for u in reversed(undo):
                u()
    else:
        frames = parse_frames(a.frames) or list(range(a.start or sc.frame_start, (a.end or sc.frame_end) + 1))
        if a.anim:
            todo = sorted({f for f in frames
                           if a.overwrite or not pngcheck.is_complete_png(os.path.join(out, f"{a.prefix}{f:04d}.png"))})
            skipped = len(set(frames)) - len(todo)
            if todo:
                done = render_anim(bpy, sc, out, todo, a.prefix, a.times or None, a.vram, a.samples)
            frames = []
        for f in frames:
            name = f"{a.prefix}{f:04d}"
            if not a.overwrite and pngcheck.is_complete_png(os.path.join(out, name + ".png")):
                skipped += 1
                continue
            per_frame_props(sc, f, a.samples)
            sc.frame_set(f)
            render_one(bpy, sc, out, name, a.times or None, a.vram)
            done += 1
    if run_undo:
        run_undo()
    print(f"[frames] rendered {done}, skipped {skipped} complete -> {out} ({time.time() - t_all:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
