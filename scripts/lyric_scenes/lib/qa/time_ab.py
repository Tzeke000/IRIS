"""Warm-frame render timing + A/B variants — the one harness (replaces rooftop/time_frames.py and boss
v3_tools/time_ab{,2,3,4,5}.py, which each hard-coded their variant ifs and output path).

ALWAYS through the capped slice (absolute paths; units start in $HOME):
    scripts/server/render.sh t-ab --wait --collect -P ~/.local/bin/blender -b SCENE.blend --python-exit-code 1 \
        --python /ABS/scripts/lyric_scenes/lib/qa/time_ab.py -- \
        (--cams a,b [--cam-prefix CAM_] | --frames 1021,2341 | both = every cam at every frame) \
        [--variants variants.json | --variants '{"s1": {"eevee": {"taa_render_samples": 1}}}'] \
        --out /ABS/times.json [--pct 100] [--samples N] [--warm each|once|none] [--repeat 1] [--bookend] [--save DIR]

variants.json = {name: patch spec} (see lib/render/patch.py), e.g.
    {"no_fog": {"hide_collections": ["V3_EXT_FOG"]}, "s1": {"eevee": {"taa_render_samples": 1}},
     "no_gems": {"hide_prefix": ["V3Gem_"]}, "nocomp": {"render": {"use_compositing": false}},
     "nomb": {"render": {"use_motion_blur": false}}, "glare_med": {"glare_quality": "Medium"},
     "volend400": {"set": {"scene.eevee.volumetric_end": 400}}}
A "base" variant (no patch) always runs first. --bookend runs base again at the end (catches GPU drift — the boss
time_ab3 trick). --warm each (default) renders every variant×shot twice and reports the second (warm) time, because
a variant that changes visibility can compile new shaders; --warm once = one low-res warm-up per shot (faster, boss style).
--save DIR writes each timed render as DIR/<variant>__<shot>.png so qa/crops.py diff can prove a cut is invisible.
Output JSON: runs, per-variant medians, mean, delta vs base. Treat ±0.3 s as noise on the shared V100.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "render"))
import patch  # noqa: E402


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


def plan(variants, shots, repeat=1, bookend=False):
    """[(rep, variant_name, shot)] in run order; 'base' first in every repeat, again at the end with bookend."""
    names = ["base"] + [v for v in variants if v != "base"]
    order = [(r, v, s) for r in range(repeat) for v in names for s in shots]
    if bookend:
        order += [(repeat, "base", s) for s in shots]
    return order


def summarize(runs):
    """runs: [{"variant","shot","s"}] → {"variants": {v: {shot: median, "_mean": mean_of_medians}}, "delta": {v: Δ vs base}}"""
    by = {}
    for r in runs:
        by.setdefault(r["variant"], {}).setdefault(r["shot"], []).append(r["s"])
    summ = {}
    for v, shots in by.items():
        med = {s: round(statistics.median(x), 3) for s, x in shots.items()}
        med["_mean"] = round(sum(med.values()) / len(med), 3)
        summ[v] = med
    base = summ.get("base", {}).get("_mean")
    delta = {v: (round(m["_mean"] - base, 3) if base is not None else None) for v, m in summ.items()}
    return {"variants": summ, "delta": delta}


def table(summary, shots):
    cols = [str(s) for s in shots]
    w = max([len(v) for v in summary["variants"]] + [8])
    lines = [f"{'variant':{w}s} " + " ".join(f"{c[-12:]:>12s}" for c in cols) + f" {'mean':>8s} {'Δbase':>8s}"]
    for v, m in summary["variants"].items():
        d = summary["delta"][v]
        lines.append(f"{v:{w}s} " + " ".join(f"{m.get(c, float('nan')):12.2f}" for c in cols) +
                     f" {m['_mean']:8.2f} {'' if d is None else f'{d:+8.2f}'}")
    return "\n".join(lines)


def main(argv):
    import bpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="")
    ap.add_argument("--cam-prefix", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--warm", default="each", choices=["each", "once", "none"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--bookend", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    D = bpy.data
    variants = patch.load_spec(a.variants)
    for v, spec in variants.items():
        patch.normalize_spec(spec)                       # fail BEFORE spending GPU time on a typo
    if a.pct:
        sc.render.resolution_percentage = a.pct
    if a.samples:
        sc.eevee.taa_render_samples = a.samples
    cams = []
    for c in [x for x in a.cams.split(",") if x]:
        ob = D.objects.get(c) or D.objects.get(a.cam_prefix + c)
        if ob is None or ob.type != "CAMERA":
            raise SystemExit(f"[time_ab] no camera {c!r} (prefix {a.cam_prefix!r})")
        cams.append((c, ob))
    frames = parse_frames(a.frames)
    if cams and frames:
        shots = [(f"{c}@{f}", ob, f) for c, ob in cams for f in frames]
    elif cams:
        shots = [(c, ob, None) for c, ob in cams]
    elif frames:
        shots = [(f"f{f}", None, f) for f in frames]
    else:
        shots = [(f"f{sc.frame_current}", None, sc.frame_current)]
    shot_by = {s[0]: s for s in shots}
    tmp = os.path.join(a.save or "/tmp", f".time_ab_{os.getpid()}.png")
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    def set_shot(name):
        _, ob, f = shot_by[name]
        if ob is not None:
            sc.camera = ob
        if f is not None:
            sc.frame_set(f)

    def render(path=None):
        sc.render.filepath = path or tmp
        t0 = time.time()
        bpy.ops.render.render(write_still=bool(path))
        return time.time() - t0

    if a.warm == "once":
        pct0 = sc.render.resolution_percentage
        sc.render.resolution_percentage = 25
        for name, _, _ in shots:
            set_shot(name)
            render()
        sc.render.resolution_percentage = pct0
    runs = []
    cur, undo = None, None
    for rep, v, name in plan(variants, [s[0] for s in shots], a.repeat, a.bookend):
        if v != cur:
            if undo:
                undo()
            undo = patch.apply(sc, variants.get(v, {})) if v != "base" else None
            cur = v
        set_shot(name)
        if a.warm == "each":
            render()
        out = os.path.join(a.save, f"{v}__{name}.png") if a.save and rep == 0 else None
        dt = render(out)
        runs.append({"rep": rep, "variant": v, "shot": name, "s": round(dt, 3)})
        print(f"[time_ab] {v:16s} {name:20s} {dt:7.2f} s", flush=True)
    if undo:
        undo()
    res = summarize(runs)
    res.update({"blend": bpy.data.filepath, "runs": runs, "shots": [s[0] for s in shots],
                "args": vars(a), "resolution": [sc.render.resolution_x, sc.render.resolution_y,
                                                 sc.render.resolution_percentage],
                "samples": sc.eevee.taa_render_samples, "variants_spec": variants})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(table(res, res["shots"]))
    print(f"[time_ab] wrote {a.out}")
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
