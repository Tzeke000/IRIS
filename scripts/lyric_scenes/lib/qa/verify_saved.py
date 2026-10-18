"""Verify the SAVED .blend in a FRESH Blender process (the arcade scar: checks on the in-memory scene passed while the
file on disk was broken — relative textures went magenta). Runs lib/checks + lib/camera + lib/physics against what a
render worker will actually load, writes JSON (lib/report.Report — saved even if a check crashes), exits 1 on failure.

    lib/render/scene.sh /ABS/real/<scene>/scene.env verify [ARGS…]          → $OUT_DIR/verify_saved.json (VERIFY_ARGS too)
    scripts/server/render.sh t-verify --wait --collect -P ~/.local/bin/blender -b --factory-startup /ABS/scene.blend \
        --python-exit-code 1 --python /ABS/scripts/lyric_scenes/lib/qa/verify_saved.py -- --out /ABS/verify.json [ARGS…]

ALWAYS (cheap, no render):
  images      relative / missing images (not packed), missing linked libraries                  (arcade magenta)
  finite      every keyframe value, every mesh vertex, every object matrix at the sample frames  (gotcha 2 at source)
  camera      scene.camera set; each camera not inside / within --min-cam m of render-visible geometry at the sample
              frames (scene camera at every sample, other cameras = stills at frame_start). Volume-only objects and
              --ignore prefixes are skipped (haze boxes).                                  (police camera-in-building)
  cuts        if scene["cuts"] (lib/camera.bake_shots writes it): no camera channel moves inside the START shutter of a
              shot's last frame, and motion blur position is START                                (rooftop v3.1 5–7)
  instancers  every GN Instance-on-Points / collection / vertex instancer emits > 0 instances at some sample frame where
              it is render-visible (--allow-empty NAMES). Counted with the background depsgraph, which honours
              hide_viewport: a hide_viewport kit source counts 0 → keep kit sources hide_render-only (gotcha 1 house rule)
OPTIONAL:
  --ground NAMES --grounded NAMES [--edges] [--tol 0.05]   objects must rest on the ground (lib/physics.floating);
        --edges = building mode: every vertical bottom edge vs the terrain (rooftop verify_world, lowest_columns)
  --direction PREFIX:+Y[,…] [--min-dot 0.9]                 movers face their motion (lib/checks.direction_check)
  --view-min D                                               first hit along each camera's view axis ≥ D m (boss v1)
  --render F,… [--pct 10]   tiny EXR renders, compositing off AND on: any NaN/inf pixel fails (gotcha 2: NaN → glare
        white-out; "on" bad but "off" clean = the glare is spreading a NaN from a shader)
  --extra FILE.py           scene rules: FILE defines verify(scene, R, args) and records via R.check(cond, key, msg)
Names: comma lists; an object matches a name if its name starts with it or it is in a collection of that name.
Sample frames: frame_start, middle, frame_end, every cut and the frame before it, --frames, --every K.
"""
import argparse
import importlib.util
import math
import os
import sys
import tempfile

import numpy as np

try:
    import bpy
except ImportError:                       # the pure helpers are unit-tested outside Blender
    bpy = None

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LIB))
from lib import camera as CM  # noqa: E402
from lib import checks as K  # noqa: E402
from lib import physics as PH  # noqa: E402
from lib.report import Report  # noqa: E402

AXES = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


# ------------------------------------------------------------------------------------------------ pure helpers

def parse_list(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_frames(s):
    out = []
    for part in parse_list(s):
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def sample_frames(start, end, cuts=(), extra=(), every=0, cap=400):
    """sorted unique frames: start, middle, end, each cut c and c−1, extra, and every K-th frame (≤ cap)."""
    fs = {start, end, (start + end) // 2}
    for c in cuts:
        fs |= {c - 1, c}
    fs |= set(extra)
    if every and every > 0:
        fs |= set(range(start, end + 1, every))
    fs = sorted(f for f in fs if start <= f <= end)
    if len(fs) > cap:
        fs = [fs[int(round(i))] for i in np.linspace(0, len(fs) - 1, cap)]
    return fs


def parse_direction(s):
    """'Car_:+Y,Drone:-X' → [('Car_', (0,1,0)), ('Drone', (-1,0,0))]"""
    out = []
    for item in parse_list(s):
        name, _, ax = item.rpartition(":")
        if not name or ax.upper() not in AXES:
            raise ValueError(f"--direction wants NAME:AXIS with AXIS in {list(AXES)}, got {item!r}")
        out.append((name, AXES[ax.upper()]))
    return out


def bad_pixels(px):
    """(count of non-finite values, count of values > 1e4) in a flat float pixel array"""
    a = np.asarray(px, np.float32)
    fin = np.isfinite(a)
    return int((~fin).sum()), int((np.abs(np.where(fin, a, 0)) > 1e4).sum())


# ------------------------------------------------------------------------------------------------ bpy helpers

def match(objs, names):
    if not names:
        return []
    out = []
    for o in objs:
        if o.name.startswith(tuple(names)) or any(c.name in names for c in o.users_collection):
            out.append(o)
    return out


def hidden_collections():
    """names of collections that are render-hidden themselves or through an ancestor in the scene tree"""
    out = set()

    def walk(c, hidden):
        for ch in c.children:
            h = hidden or ch.hide_render
            if h:
                out.add(ch.name)
            walk(ch, h)
    walk(bpy.context.scene.collection, False)
    return out


def render_visible(o, hidden=None):
    """not hide_render, and in at least one collection that renders"""
    hidden = hidden_collections() if hidden is None else hidden
    return not o.hide_render and any(c.name not in hidden for c in o.users_collection)


def volume_only(o):
    mats = [s.material for s in getattr(o, "material_slots", []) if s.material]
    if not mats or o.type != "MESH":
        return False
    for m in mats:
        if not m.node_tree:
            return False
        out = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        if out is None or out.inputs["Surface"].is_linked or not out.inputs["Volume"].is_linked:
            return False
    return True


def world_aabb(o):
    from mathutils import Vector
    pts = np.array([list(o.matrix_world @ Vector(c)) for c in o.bound_box])
    return pts.min(0), pts.max(0)


def tree_has(ng, idname, seen=None):
    seen = seen if seen is not None else set()
    if ng is None or ng.name in seen:
        return False
    seen.add(ng.name)
    for n in ng.nodes:
        if n.bl_idname == idname:
            return True
        if n.bl_idname == "GeometryNodeGroup" and tree_has(n.node_tree, idname, seen):
            return True
    return False


def instancers():
    out = []
    for o in bpy.context.scene.objects:
        gn = [m for m in o.modifiers if m.type == "NODES" and m.show_render and tree_has(m.node_group, "GeometryNodeInstanceOnPoints")]
        if gn or (o.type == "EMPTY" and o.instance_type == "COLLECTION" and o.instance_collection) or \
                (o.type == "MESH" and o.instance_type in ("VERTS", "FACES")):
            out.append(o)
    return out


# ------------------------------------------------------------------------------------------------ checks

def check_images(R):
    bad = K.bpy_bad_images()
    libs = [(lb.name, bpy.path.abspath(lb.filepath)) for lb in bpy.data.libraries
            if not os.path.isfile(bpy.path.abspath(lb.filepath))]
    R.add("details.images", {"bad_images": bad, "missing_libraries": libs})
    R.check(not bad and not libs, "images", f"{len(bad)} bad image(s) {bad[:5]}, {len(libs)} missing librar(ies) {libs[:3]}")


def check_finite(R, sc, frames):
    bad = []
    for a in bpy.data.actions:
        for ly in a.layers:
            for st in ly.strips:
                for cb in st.channelbags:
                    for fc in cb.fcurves:
                        co = np.empty(len(fc.keyframe_points) * 2, np.float32)
                        fc.keyframe_points.foreach_get("co", co)
                        if not np.isfinite(co).all():
                            bad.append(f"action {a.name}: {fc.data_path}[{fc.array_index}]")
    for me in bpy.data.meshes:
        if me.users and len(me.vertices):
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            if not np.isfinite(co).all():
                bad.append(f"mesh {me.name}: {int((~np.isfinite(co)).sum())} non-finite coords")
    for f in frames:
        sc.frame_set(f)
        for o in sc.objects:
            if not np.isfinite(np.array(o.matrix_world)).all():
                bad.append(f"object {o.name} matrix at frame {f}")
    R.add("details.finite", bad[:200])
    R.check(not bad, "finite", f"{len(bad)} non-finite value group(s) {bad[:5]}")


def camera_positions(sc, frames):
    """[(camera object, frame, position)]: scene camera at every frame, other cameras at frame_start"""
    out = []
    for f in frames:
        sc.frame_set(f)
        if sc.camera is not None:
            out.append((sc.camera, f, np.array(sc.camera.matrix_world.translation)))
    sc.frame_set(sc.frame_start)
    for o in sc.objects:
        if o.type == "CAMERA" and o != sc.camera:
            out.append((o, sc.frame_start, np.array(o.matrix_world.translation)))
    return out


def check_camera(R, sc, frames, min_d, ignore, view_min):
    if sc.camera is None:
        R.check(False, "camera", "scene has no camera")
        return
    hits, views = [], []
    for cam, f, p in camera_positions(sc, frames):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        hid = hidden_collections()
        near = []
        for o in sc.objects:
            if o.type not in ("MESH", "CURVE", "FONT", "SURFACE", "META") or not render_visible(o, hid) or volume_only(o) \
                    or (ignore and o.name.startswith(tuple(ignore))):
                continue
            lo, hi = world_aabb(o)
            if np.all(p >= lo - min_d) and np.all(p <= hi + min_d):
                near.append(o)
        if near:
            bad = K.bpy_camera_clear([p], K.bpy_bvh(near, dg), min_d)
            if bad:
                hits.append(dict(camera=cam.name, frame=f, distance=round(bad[0][1], 3), near=[o.name for o in near][:6]))
        if view_min and cam.data.type != "ORTHO":
            nm, dist = CM.view_hit(cam, sc, dg, skip=lambda ob: volume_only(ob) or not render_visible(ob, hid)
                                   or bool(ignore and ob.name.startswith(tuple(ignore))))
            if nm is not None and dist < view_min:
                views.append(dict(camera=cam.name, frame=f, hit=nm, distance=round(dist, 3)))
    R.add("details.camera_inside", hits[:100])
    R.check(not hits, "camera", f"{len(hits)} camera sample(s) inside or within {min_d} m of geometry {hits[:3]}")
    if view_min:
        R.add("details.view_hit", views[:100])
        R.check(not views, "view_hit", f"{len(views)} camera(s) see geometry closer than {view_min} m {views[:3]}")


def check_cuts(R, sc, cuts):
    if not cuts:
        R.add("details.cuts", "no scene['cuts'] — skipped")
        return
    cam = sc.camera
    v = (CM.cut_violations(cam, cuts) + CM.cut_violations(cam.data, cuts)) if cam is not None else []
    mb_ok = not sc.render.use_motion_blur or sc.render.motion_blur_position == "START"
    R.add("details.cuts", dict(cuts=list(cuts), violations=v[:50], motion_blur_position=sc.render.motion_blur_position))
    R.check(not v and mb_ok, "cuts", f"{len(v)} camera channel(s) move across a cut {v[:3]}"
            + ("" if mb_ok else f"; motion_blur_position {sc.render.motion_blur_position} (cut holds need START)"))


def check_instancers(R, sc, frames, allow):
    rows, bad = [], []
    for o in instancers():
        counts, vis = {}, 0
        for f in frames:
            sc.frame_set(f)
            if not render_visible(o):
                continue
            vis += 1
            counts[f] = K.bpy_instances(o, bpy.context.evaluated_depsgraph_get())
        rows.append(dict(name=o.name, counts=counts, visible_samples=vis))
        if vis and max(counts.values()) == 0 and not o.name.startswith(tuple(allow)):
            bad.append(o.name)
    R.add("details.instancers", rows[:200])
    R.check(not bad, "instancers", f"{len(bad)} instancer(s) emit 0 instances at every visible sample {bad[:5]}"
            " (source hide_viewport / hidden? keep kit sources hide_render only)")


def check_grounding(R, sc, ground, grounded, tol, edges):
    sc.frame_set(sc.frame_start)
    dg = bpy.context.evaluated_depsgraph_get()
    gobs = match(sc.objects, ground)
    tobs = [o for o in match(sc.objects, grounded) if o not in gobs and o.type == "MESH"]
    if not gobs or not tobs:
        R.check(False, "grounding", f"--ground matched {len(gobs)} object(s), --grounded {len(tobs)}")
        return
    gfn = PH.ground_from(gobs, dg)
    if edges:
        rows = []
        for o in tobs:
            V, _ = K.bpy_world_tris([o], dg)
            cols = PH.lowest_columns(V)
            fl = K.floating(cols, gfn, gap=(tol, 15.0))
            if len(fl):
                rows.append((o.name, int(len(fl)), int(len(cols))))
        R.add("details.grounding", dict(mode="edges", floating=rows[:100]))
        R.check(not rows, "grounding", f"{len(rows)} object(s) with floating bottom edges {rows[:3]}")
    else:
        bad = PH.floating(tobs, gfn, tol=tol, dg=dg)
        R.add("details.grounding", dict(mode="objects", checked=len(tobs),
                                        floating=[(n, None if not math.isfinite(g) else round(g, 3)) for n, g in bad][:100]))
        R.check(not bad, "grounding", f"{len(bad)} of {len(tobs)} object(s) not resting on the ground (> {tol} m) {bad[:3]}")


def check_direction(R, sc, spec, min_dot, step):
    rows, bad = [], []
    frames = list(range(sc.frame_start, sc.frame_end + 1, max(1, step)))
    for prefix, ax in spec:
        for o in match(sc.objects, [prefix]):
            pos, nose = [], []
            for f in frames:
                sc.frame_set(f)
                M = np.array(o.matrix_world)
                pos.append(M[:3, 3])
                nose.append(M[:3, :3] @ np.asarray(ax, float))
            r = K.direction_check(np.array(pos), np.array(nose), min_dot=min_dot)
            r["worst_frame"] = None if r["worst_frame_index"] is None else frames[r["worst_frame_index"]]
            rows.append(dict(name=o.name, **r))
            if not r["ok"]:
                bad.append(o.name)
    R.add("details.direction", rows[:200])
    R.check(not bad and rows, "direction", f"{len(bad)} of {len(rows)} mover(s) not facing their motion {bad[:5]}")


def check_render(R, sc, frames, pct):
    r = sc.render
    keep = (r.filepath, r.resolution_percentage, r.image_settings.file_format, r.image_settings.color_depth,
            r.use_compositing)
    td = tempfile.mkdtemp(prefix="verify_render_")
    rows, bad = [], []
    try:
        r.resolution_percentage = pct
        r.image_settings.file_format, r.image_settings.color_depth = "OPEN_EXR", "32"
        for f in frames:
            sc.frame_set(f)
            for comp in (False, True):
                r.use_compositing = comp
                r.filepath = os.path.join(td, f"f{f}_{'comp' if comp else 'raw'}.exr")
                bpy.ops.render.render(write_still=True)
                im = bpy.data.images.load(r.filepath)
                px = np.empty(len(im.pixels), np.float32)
                im.pixels.foreach_get(px)
                bpy.data.images.remove(im)
                nf, big = bad_pixels(px)
                rows.append(dict(frame=f, compositing=comp, nonfinite=nf, over_1e4=big))
                if nf:
                    bad.append((f, "comp" if comp else "raw", nf))
    finally:
        (r.filepath, r.resolution_percentage, r.image_settings.file_format, r.image_settings.color_depth,
         r.use_compositing) = keep
    R.add("details.render", dict(dir=td, rows=rows))
    R.check(not bad, "render_nan", f"non-finite pixels {bad[:5]} (raw clean + comp bad = glare spreading a shader NaN)")


def run_extra(R, sc, path, args):
    spec = importlib.util.spec_from_file_location("verify_extra", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify(sc, R, args)


def main(argv=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if argv is None and "--" in sys.argv else (argv or [])
    ap = argparse.ArgumentParser(prog="verify_saved.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--every", type=int, default=0)
    ap.add_argument("--cuts", default="", help="override scene['cuts']")
    ap.add_argument("--min-cam", type=float, default=0.3)
    ap.add_argument("--ignore", default="", help="object name prefixes the camera may be inside (haze boxes)")
    ap.add_argument("--view-min", type=float, default=0.0)
    ap.add_argument("--allow-empty", default="")
    ap.add_argument("--ground", default="")
    ap.add_argument("--grounded", default="")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--direction", default="")
    ap.add_argument("--min-dot", type=float, default=0.9)
    ap.add_argument("--dir-step", type=int, default=1)
    ap.add_argument("--render", default="")
    ap.add_argument("--pct", type=int, default=10)
    ap.add_argument("--extra", default="")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    if not bpy.data.filepath:
        raise SystemExit("verify_saved.py must run on a SAVED .blend (blender -b FILE.blend --python …)")
    cuts = parse_frames(a.cuts) if a.cuts else [int(c) for c in sc.get("cuts", [])]
    frames = sample_frames(sc.frame_start, sc.frame_end, cuts, parse_frames(a.frames), a.every)
    with Report(a.out, "verify_saved") as R:
        R.add("file", dict(path=bpy.data.filepath, mtime=os.path.getmtime(bpy.data.filepath), scene=sc.name,
                           blender=bpy.app.version_string, saved_with=".".join(map(str, bpy.data.version)),
                           frames=[sc.frame_start, sc.frame_end], samples=frames, cuts=cuts))
        steps = [("images", lambda: check_images(R)),
                 ("finite", lambda: check_finite(R, sc, frames)),
                 ("camera", lambda: check_camera(R, sc, frames, a.min_cam, parse_list(a.ignore), a.view_min)),
                 ("cuts", lambda: check_cuts(R, sc, cuts)),
                 ("instancers", lambda: check_instancers(R, sc, frames, parse_list(a.allow_empty)))]
        if a.ground or a.grounded:
            steps.append(("grounding", lambda: check_grounding(R, sc, parse_list(a.ground), parse_list(a.grounded),
                                                               a.tol, a.edges)))
        if a.direction:
            steps.append(("direction", lambda: check_direction(R, sc, parse_direction(a.direction), a.min_dot, a.dir_step)))
        if a.render:
            steps.append(("render_nan", lambda: check_render(R, sc, parse_frames(a.render), a.pct)))
        if a.extra:
            steps.append(("extra", lambda: run_extra(R, sc, a.extra, a)))
        for key, fn in steps:
            try:
                fn()
            except Exception as e:                      # a crashing check is a failed check, the others still run
                R.check(False, key, f"check crashed: {type(e).__name__}: {e}")
        R.tick("checks")
    failed = [x["key"] for x in R.failed]
    print(f"[verify_saved] {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} → {a.out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
