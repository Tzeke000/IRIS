"""BOSS FIGHT — animatic v1 build: arena_v3.blend → fight_v1.blend (knight vs dragon blocked through the whole song).

    scripts/lyric_scenes/lib/render/scene.sh /home/iris/IRIS/scripts/lyric_scenes/real/boss_fight/anim/animatic.env build
      (= blender -b --factory-startup arena_v3.blend --python build_anim.py -- --out fight_v1.blend)
    extra args: --stop-after STAGE   --report PATH   --debug-dir DIR   --audit-step N   --no-save

Stages (stages/*.py, run in order, each `run(S)` on one shared context S):
  s10_setup        collections, frame range, characters (knight re-imported + rest-normalised; dragon re-rooted), sword
  s20_knight       UAL clips retargeted → NLA BASE_A/BASE_B (+ ARMS overlays), root path + ground
  s30_dragon       native clips + T-rex donors (COMBINE) + hand-keyed moves (COMBINE), root path (ground + flight)
  s40_contacts     solve contact pins (root nudges so the planned contact really happens on its frame)
  s50_flinch       closed-form damped springs per contact → FLINCH (COMBINE) + body shove on the root
  s60_track        evaluate both rigs every frame → key points (head, jaw, hands, blade, tail…) for FX + cameras
  s70_fx           projectiles (stop on contact), shield, spikes, magic placeholders, impact flashes
  s75_debris       pillars / ceiling / roof / colossus arm: kinematic→dynamic release, Bullet bake → keys
  s80_cameras      shot list → one baked camera (cuts on event frames) + impact kicks
  s90_audit        no-pass-through audit (physics OFF), grounding, camera clearance, timing → build report
NEVER overwrites arena_v3.blend (asserted).
"""
import argparse
import importlib
import os
import sys
import time
import types

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # scripts/lyric_scenes
BOSS = os.path.dirname(HERE)
for p in (SCENES, BOSS):
    if p not in sys.path:
        sys.path.insert(0, p)

import anim.common as C  # noqa: E402
from lib.report import Report  # noqa: E402

STAGES = ["s10_setup", "s20_knight", "s30_dragon", "s40_contacts", "s50_flinch", "s55_feet", "s70_fx",
          "s75_debris", "s80_cameras", "s90_audit"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.OUT_BLEND)
    ap.add_argument("--report", default=os.path.join(C.OUT_DIR, "fight_v2_build_report.json"))
    ap.add_argument("--stop-after", default="")
    ap.add_argument("--debug-dir", default=os.path.join(C.REPO, "scratch", "boss_anim_v2", "debug"))
    ap.add_argument("--audit-step", type=int, default=1)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--post", default="", help="debug: exec this python file with S after the stages")
    ap.add_argument("--no-debris", action="store_true", help="skip the rigid-body stage (iteration only)")
    ap.add_argument("--audit-soft", action="store_true", help="record audit failures without failing the build")
    ap.add_argument("--slice-end", type=int, default=0, help="audit/track only up to this frame (0 = whole song)")
    a = ap.parse_args(argv)
    assert os.path.abspath(a.out) != os.path.abspath(C.ARENA), "refusing to overwrite arena_v3.blend"
    assert os.path.abspath(bpy.data.filepath) == os.path.abspath(C.ARENA), f"open arena_v3.blend first, got {bpy.data.filepath}"
    os.makedirs(a.debug_dir, exist_ok=True)
    S = types.SimpleNamespace(args=a, sc=bpy.context.scene, R=Report(a.report, name="fight_v2"), E=C.load_events(),
                              N=C.NF, contacts=[], debug=a.debug_dir, t0=time.time())
    S.R.add("events", len(S.E))
    with S.R:
        for st in STAGES:
            mod = importlib.import_module(f"anim.stages.{st}")
            t = time.time()
            mod.run(S)
            S.R.add(f"stage_s.{st}", round(time.time() - t, 1))
            S.R.log(f"stage {st} done in {time.time() - t:.1f} s")
            if a.stop_after and st.startswith(a.stop_after):
                break
        write_meta(S, a.report.replace("_build_report.json", "_meta.json").replace("report.json", "meta.json"))
        if a.post:
            exec(compile(open(a.post).read(), a.post, "exec"), {"S": S, "bpy": bpy})
        if not a.no_save:
            S.sc.frame_set(1)
            bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=False)
            S.R.add("saved", a.out)
            S.R.log("saved", a.out)


def write_meta(S, path):
    """everything the STRICT audit (audit_strict.py, run on the saved blend) needs to know about intent: registered
    contacts + windows, airborne windows, held props, side/back-step clips"""
    import json
    E = S.E
    f = {k: e.f for k, e in E.items()}
    K, D = S.K, S.D
    cons = [dict(a=c.a, b=c.b, lo=c.window[0], hi=c.window[1], key=c.key, kind=c.kind) for c in getattr(S, "contacts", [])
            if c.kind != "rubble"]
    sk = K.plan["sword_keys"]
    held = []
    for (fa, ia), (fb, _ib) in zip(sk, sk[1:] + [(S.N + 1, False)]):
        if ia:
            held.append(dict(prop="K_Sword", hands=["R"], lo=fa + 3, hi=fb - 2))
    held.append(dict(prop="FX_gem", hands=["R"], lo=f["gem_out"] + 32, hi=f["gem_crush"] - 2))
    held.append(dict(prop="PRIZE_GoldRPG7", hands=["L", "R"], lo=K.plan["rpg"]["lifted"] + 2, hi=S.N))
    sit = [w for w in K.plan["air"] if w[0] >= f["slam"]][:2]
    meta = dict(version="v2", contacts=cons, air_k=[list(w) for w in K.plan["air"]], sit_k=[],
                air_d=[list(w) for w in D.plan["air"]], held=held,
                thrown_k=[[f["slam"], f["thrown"] + 34]], side_clips=["Strafe", "RunBack", "W_Strafe", "Back", "SHUFFLE"],
                events=f)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1)
    S.R.add("meta", path)


main()
