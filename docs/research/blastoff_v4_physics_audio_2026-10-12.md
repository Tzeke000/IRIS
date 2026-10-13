# Blast Off v4: Blender's own physics for debris, and a stem-based audio visualizer (2026-10-12)

This is research plus small tests only. Nothing in `scripts/lyric_scenes/real/rocket3/` was touched, and
nothing was built or rendered for the scene. It follows up on `docs/research/blastoff_v4_2026-10-12.md` §6 (debris)
and §2/§8 (LEDs and speaker). Tests live in `scratch/scenes/blastoff4_research/physics/` and `.../audio/`.
Big outputs are on `/data/iris_assets/blastoff/stems/`.

Two asks from Zeke:
- *"You might not have to calculate all the debris yourself, have Blender do it."*
- *"Research how an audio visualizer should work: a kick-specific one, bass, hi-hats, snare and whatever else. Research music like this, how to make it and how to understand it."*

---

## TL;DR

**Physics.** Blender's Bullet can do it, and the no-tunnel test passes:
- **Result:** 24 chunks at 120–600 m/s hit a moving 24 × 17 × 38 m box. 0 tunnelled, and **0 vertex-inside events over all frames**. The minimum clearance to the visual box was 0.45 m. The bake took 0.16 s.
- **Settings:** `substeps_per_frame=30`, `solver_iterations=20`, `use_split_impulse=True`, and a hidden proxy BOX collider padded 0.5 m. The chunks get LINEAR, extrapolated kinematic keys released at frame 4.
- **The earlier "Bullet tunnels" finding was wrong.** That chunk missed the box by about 70 m (§1.1). It does not hold up.
- **Crumble works too.** It uses the **Cell Fracture** extension (GPL, run offline at build time) plus breakable FIXED constraints. It also passes the no-penetration proof.
- **Production scale:** 200 chunks × 1000 frames bakes in 18 s, plus 6 s to write keyframes. Render workers get plain keys and never touch Bullet.
- **Recommendation:** **use Blender physics for the debris and crumble, baked to keys, with the vertex-inside assert kept as a build gate.** Keep the analytic numpy path only as the fallback (§1.7).

**Audio.** Demucs `htdemucs_ft` split the song on the V100:
- **Cost:** 48 s wall time for the 199 s song including model load, about 0.5 GiB peak GPU.
- **Kicks, using the stems and a per-onset classifier:**
  - 59 % of detected kicks land on the beat grid. The old fixed-band detector managed 37 %.
  - 83 % of beats in drop sections get a kick. The old detector: 25 %.
  - **63 % of the old "snares" were actually kicks.**
- **Surprises on this song:**
  1. Demucs puts the kick's low end in the **drums** stem in some sections and in the **bass** stem in others, so neither stem alone is a kick detector.
  2. In the drops the kick's punch is at **150–400 Hz**, and the sub under it is a sustained bass.
  3. The band-split kick-vs-snare test inside the drums stem works only partly: 60 % of "snares" share a kick onset.
  4. The real first-drop impact is at **56.07 s**, about 1.5 s **before** the old timeline's first entry (57.57 s).
- **Visualizer rules (§2d):** each element gets its own channel and its own attack/release.
  - kick: punches and flashes (attack 0, release ~120 ms), with on-beat and offbeat kicks kept separate
  - bass: sustained glow and size (30 ms / 300 ms)
  - snare: side flashes, only for snares *not* on a kick
  - hats: sparkle (release ~50 ms)
  - vocals: colour and text (gated)
  - risers: build intensity
  - spectrum: per-band normalised and log-spaced

---

## Part 1: Blender's own physics for asteroid debris vs the giant speaker

### 1.1 Correction first: the earlier test did not show tunnelling

`t_rigidbody.py` (in the first research doc) reported the 400 m/s chunk "passed straight through". I re-ran it
unchanged and printed its full 3-D path (`physics/t_old_diag.py`, `t_old_diag2.py`):

```
Chunk2 400  (x, y, z) per frame: f12 (63.0, 43.4, -14.0)  f15 (81.7, 0.8, -18.8)  f16 (88.0, -13.3, -20.4)
```

It crossed the speaker plane at **x = 82 m**. The box half-width is 12 m, so it **missed**. The old pass/fail test only looked at y.
Its 430 m/s did not come from the velocity injection either. The four chunks were created **overlapping** (radii 4–12 m,
centres 8–14 m apart), so Bullet's overlap recovery blew them apart. Run alone, the same chunk ends with **zero** velocity,
because its two Bezier keys *end* at the release frame and it is easing to a stop when let go.

So the conclusions "Bullet has no CCD, so it tunnels" and "velocities are inconsistent" were both artefacts of the test. The real
limits are measured in §1.3.

### 1.2 What 5.2.1 exposes (RNA dump, `-b --factory-startup`)

| Setting | Range / default | Effect here |
|---|---|---|
| `rigidbody_world.substeps_per_frame` | 1–32767, default 10 | **The tunnelling control.** Distance moved per substep = v / (fps · substeps) |
| `solver_iterations` | 1–1000, default 10 | Contact/constraint accuracy. 20 was enough |
| `use_split_impulse` | default False | Separates the penetration fix from the velocity, so no extra bounce energy. Turned on |
| `time_scale` | 0–100, default 1 | **No effect on tunnelling** for kinematically injected chunks (§1.3) |
| `rigid_body.collision_shape` | BOX / SPHERE / CAPSULE / CYLINDER / CONE / CONVEX_HULL / MESH / COMPOUND | Chunks: CONVEX_HULL. Collider: BOX |
| `use_margin`, `collision_margin` | default 0.04 m | Margin 0 on both sides; the proxy pad does the job instead |
| `kinematic` (keyable) | | Animated passive collider = keyed speaker. Kinematic→dynamic release injects the velocity |
| initial velocity / CCD | **not exposed** | Velocity comes from kinematic keys. CCD is unnecessary at 30 substeps (§1.3) |

### 1.3 The no-tunnel test (`physics/t_bullet_notunnel.py`)

**Setup**
- **Speaker:** a 24 × 17 × 38 m visual box with keyed kick recoil (3 m back every 0.68 s), a ±2 m drift and an ±8° yaw.
- **Collider:** a hidden child **proxy BOX** = visual + `pad` on every side, PASSIVE + kinematic.
- **Chunks:** 24 lumpy convex rocks, 2–12 m, at 120–600 m/s, aimed at the front and sides of the moving cabinet. Zero gravity, damping 0.
- **Velocity injection:** chunk location keys are **LINEAR with LINEAR extrapolation**, and `rigid_body.kinematic` is keyed True→False at frames 4→5. This measured **0.00–0.04 % speed error**, against the earlier test's random speeds.

**Proof**
- After the bake, every frame, every chunk's world vertices are moved into the speaker's moving local frame and tested against the **visual** box. Any vertex inside fails the script with an assert (`--python-exit-code 1`).
- A chunk counts as "tunnelled" if it ends up behind the back face with its velocity unchanged.

`matrix.sh` results:

| substeps | iters | scale | time_scale | pad m | contacts | tunnelled | worst depth into visual box | inside events | bake s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 1 | 1 | 0 | 21 | **3** | 8.50 m | 54 | 0.04 |
| 3 | 10 | 1 | 1 | 0 | 24 | 0 | 4.10 m | 22 | 0.07 |
| 10 (default) | 10 | 1 | 1 | 0 | 24 | 0 | 0.23 m | 8 | 0.08 |
| 10 | 10 | 1 | 1 | 0.5 | 24 | 0 | 0.23 m | 2 | 0.08 |
| 10 | 10 | **0.1** | 1 | 0 | 24 | 0 | 0.23 m | 6 | 0.08 |
| 10 | 10 | 1 | **0.1** | 0 | 24 | 0 | 0.23 m | 8 | 0.09 |
| 30 | 20 | 1 | 1 | 0 | 24 | 0 | 0.18 m | 1 | 0.15 |
| **30** | **20** | 1 | 1 | **0.5** | 24 | 0 | **−0.47 m (clear)** | **0** | **0.15** |
| 30 | 20 | 0.1 | 1 | 0.5 | 24 | 0 | −0.46 m | 0 | 0.15 |
| 30 | 20 | 1 | 0.25 | 0.5 | 24 | 0 | −0.47 m | 0 | 0.15 |
| 60 | 20 | 1 | 1 | 0 | 24 | 0 | 0.00 m | 1 | 0.23 |
| 60 | 20 | 1 | 1 | 0.5 | 24 | 0 | −0.45 m | 0 | 0.26 |
| 120 | 20 | 1 | 1 | 0 | 24 | 0 | 0.003 m | 2 | 0.92 |

What the table shows:
- **Real tunnelling needs very few substeps.** At 1 substep a 600 m/s chunk moves 20 m per step, more than the 17 m box, so it passes. From 3 substeps up nothing tunnels, but chunks sink in deep. Rule of thumb:
  **v / (fps · substeps) must stay well under the smaller of (chunk radius, collider half-thickness).** At 600 m/s, 30 fps and 30 substeps that is 0.67 m per step.
- **Bullet always allows some penetration** while it resolves a contact. Even 120 substeps left 3 mm inside the box on sampled frames.
  The fix is the **thick hidden proxy**: collide against a box 0.5 m bigger than the visible cabinet, so any residual
  sink stays inside the pad. At 30 substeps the worst sink into the proxy was 3.2 cm, so a 0.5 m pad leaves 0.47 m of clearance from the visible box.
- **`time_scale` and 1/10 scale change nothing.** That is expected. With kinematic injection, the motion per frame is fixed by
  the keys, and the number of substeps per frame is unchanged. So the step length in "frame space" is identical, and Bullet's
  contact handling is scale-free to first order. `time_scale` only matters for slow motion of physical
  (gravity-driven) motion. To get "slow the sim, then remap", key the injection 10× slower over 10× more frames and then
  scale the baked keys in time. That is the same thing as raising substeps 10×, so just raise substeps.
- **Sampling caveat:** the proof samples whole frames. Between frames, contacts at 600 m/s resolve inside one frame, so a
  1/30 s motion-blur window could smear a chunk across the pad. That is fine visually; it is not penetration.

**Working settings (copy these):**
```python
rbw.substeps_per_frame, rbw.solver_iterations, rbw.use_split_impulse = 30, 20, True
rbw.effector_weights.gravity = 0.0
proxy = box(visual_half_extents + 0.5)        # hidden (hide_render), parented to the keyed speaker
# proxy: PASSIVE, kinematic=True, collision_shape='BOX', use_margin=True, collision_margin=0.0, restitution 0.6
# chunk: ACTIVE, 'CONVEX_HULL', margin 0, linear/angular damping 0, restitution 0.5, friction 0.5
# velocity: key location at F0 and REL+2 along v/fps per frame; ALL location keys LINEAR + fc.extrapolation='LINEAR';
#           rigid_body.kinematic keyed True @REL, False @REL+1 (booleans step automatically)
with bpy.context.temp_override(scene=sc):
    bpy.ops.ptcache.bake_all(bake=True)     # works in -b; 0.15 s for 24 chunks x 120 frames
```

### 1.4 Bake to keys so render workers never re-simulate (`--keys`, `t_reload_check.py`)

v3 renders in resumable chunks that start mid-timeline, so the sim must become ordinary keyframes.
- **The stock `bpy.ops.rigidbody.bake_to_keyframes` fails in `-b`** with
  `anim.keyframe_insert_by_name.poll() failed, context is incorrect`. A `temp_override(selected_objects=…)` alone is not enough.
  It works with **real selection (`select_set`) + active object + a borrowed `VIEW_3D` area override** from
  `window_manager.windows[0].screen`. That took 0.65 s for 24 × 120. Read the operator source
  (`5.2/scripts/startup/bl_operators/rigidbody.py`) before relying on it: it calls `keyframe_insert_by_name` once per frame,
  which is the "ops in a loop" anti-pattern, then removes the objects from the world and thins out flat keys.
- **Faster and context-free: a numpy bake.** Sample `matrix_world` once per frame (euler kept compatible with the previous
  frame), call `rigidbody.object_remove`, then write the F-curves with `keyframe_points.add` + `foreach_set` (5.x channelbag API).
  That took 0.085 s for 24 × 120, 7.7× faster.
- Both paths reproduced the sim **exactly** (max drift 0.0 m). The vertex-inside proof was re-run on the keys and still showed 0 events.
- **Fresh-process check (what a render worker does):** `t_reload_check.py` opens the saved .blend, confirms
  0 objects still have a rigid body, and jumps to frames **97, 3, 60, 120, 15, 42 out of order**. Max drift was 0.00000 m.
- **Scale:** 200 chunks × 1000 frames gave a **bake of 17.9 s + numpy keys of 5.7 s**, all 200 contacting, 0 tunnelled, 0 inside events,
  worst depth −0.34 m. v4's debris (about 60 chunks + about 8 crumbling rocks × 23 fragments, about 1000 frames) is in this range.

### 1.5 Crumble (`physics/t_crumble.py`, `crumble_sheet.jpg`)

**Options checked in 5.2.1:**
- **Cell Fracture** is no longer bundled. It is the extension **"Cell Fracture" 0.2.1** on extensions.blender.org,
  GPL-3.0-or-later, Blender ≥ 4.2. I downloaded it (sha256 `878bbd46…fb2`) to `/data/tmp_dl/cellfrac/` and **loaded it straight
  from the unpacked folder** with `importlib` + `register()`. Nothing was installed into the user's Blender. Operator:
  `bpy.ops.object.add_fracture_cell_objects(source={'VERT_OWN'}, source_limit=24, source_noise=0.6, margin=0.001,
  use_recenter=True, use_remove_original=True, collection_name="Frags")`. It runs headless and made 23 convex cells from a
  r ≈ 6 m lumpy rock in **0.12 s**. ⚠ **It is not deterministic unless you seed both `random.seed()` and
  `mathutils.noise.seed_set()`.** Without that, cell count and results changed run to run (22 vs 26 cells).
  GPL code is fine for a build-time tool. Only *assets* fall under the CC0/PD/CC-BY rule.
- **Geometry Nodes has no fracture node in 5.2.1.** Of 274 `GeometryNode*` types, the closest are ConvexHull, MeshBoolean,
  SDFGridBoolean and SplitToInstances. A GN Voronoi shatter is DIY, and GN geometry **cannot be a Bullet rigid body**. So
  a GN shatter would need its own collision code, which puts it back on the analytic path. Not recommended for this job.
- **Pre-fractured fragments "spawned on impact":** this is just the no-glue case below. All fragments fly in formation (same
  linear keys) until they hit, then scatter. Measured cohesion before impact: 1.000 (fully intact).

**Test:** 23 fragments, released as one rock at 300 m/s into the keyed cabinet proxy (pad 0.5), substeps 30.
- With glue: FIXED constraints (`use_breaking=True`) between neighbouring cells, 115 constraints in an `rbw.constraints` collection.
  Note: linking an empty into that collection **auto-creates** its `rigid_body_constraint`, so check before calling `constraint_add`.
- "Spread" = rms distance of the fragment centres from their mean, relative to frame 1.

| Glue | breaking_threshold | spread just before impact | spread at end (frame 90) | fragment speeds p10/p50/p90 | inside events | bake s |
|---|---|---|---|---|---|---|
| none | | 1.000 | 47.9 (full scatter) | 26 / 67 / 143 m/s | **0** | 0.62 |
| 115 FIXED | 1e4 | 1.000 | 19.0 (chunky crumble) | 18 / 26 / 67 | **0** | 0.62 |
| 115 FIXED | 1e6 | 1.000 | 17.2 (chunky crumble) | 21 / 30 / 59 | **0** | 0.64 |
| 115 FIXED | 1e8 | 1.000 | 1.0 (bounces off **whole**) | 42 / 96 / 132 | **0** | 2.16 |

Raising solver iterations to 60 or the pad to 1.0 m changed nothing important. The contact sheet (workbench, 480×300,
rendered via `render.sh`) shows frames 16/20/22/26/35. The no-glue rock bursts into a spray. The 1e6-glued rock
breaks into clusters that slide down the cabinet face. For artistic control: **no glue = "shattered into gravel", ~1e6 =
"cracked into boulders", ≥1e8 = "it survives and ricochets"**. Mix them per rock.

### 1.6 What the speaker gets from this for free
- **Hit reactions** can be read off the bake. The contact frame and point are where a chunk's velocity changes (`deflected`
  in the test). Feed them to the keyed `HitAmt/HitPos` LED flash and the speaker flinch impulse from the first doc §6.
- **The flinch must be keyed BEFORE the bake.** The proxy follows the speaker's keys, and Bullet never pushes a kinematic body.
  So: (1) key the speaker's music recoil, (2) bake, (3) read the contact frames, (4) add the hit flinch as an
  *additive* NLA layer on the visual mesh only. If the flinch moves the proxy, re-bake.
- **For the speaker mesh asset** (Sousinho CC-BY cabinet, 152 islands, front +Y), the collider stays a BOX proxy fitted to
  the asset's bounds + 0.5 m. Never use MESH shape on a 7k-tri asset: it is slow, and thin parts are where real tunnelling lives.

### 1.7 Recommendation: Blender physics, with the analytic path as fallback

| | **Blender Bullet, baked to keys** (recommended) | Analytic numpy (first doc §6) |
|---|---|---|
| Chunk-vs-chunk collisions | **free** (debris clacks off other debris) | must be written (O(n²) spheres) |
| Rotation, rolling, sliding off faces | real, from the convex hull | approximate (spin from slip) |
| Crumble | **Cell Fracture + breakable glue, tested** | pool swap at contact (no real breakage) |
| Determinism | deterministic for a fixed .blend (identical across repeat runs once Cell Fracture is seeded) | fully seeded |
| Tunnelling | none at 30 substeps + 0.5 m proxy (**proved per frame, per vertex**) | none by construction (substeps + push-out) |
| Cost | 18 s bake for 200 × 1000, + keys; render cost 0 | < 1 s; render cost 0 |
| Art direction | indirect (threshold, restitution, aim). Contact frames must be read from the bake, then re-aimed and re-baked | direct (exact contact frame on a beat) |
| Risk | Bullet changes between Blender versions → **keep the vertex-inside assert as a build gate**; bake only in the build step | more code to own |

**Plan:**
1. Bake the debris and crumble with Bullet in `build_scene.py` (Bullet must not run in render workers).
2. Convert to keys with the **numpy** baker.
3. Run the per-vertex inside assert over all debris frames and fail the build on any event.
4. **Beat-sync the hits by measuring, not predicting.** Release velocities are exact, but in the 24-chunk test the
   measured contact frame differed from my straight-line estimate by a median of +6 frames, ranging from −52 to +48.
   Chunk-vs-chunk collisions on the way in, the moving cabinet, and my estimate ignoring both all contribute.
   So: bake, read each chunk's contact frame from the bake (the frame its velocity changes), shift that chunk's release
   to put the contact on the nearest kick, and re-bake. Each loop costs well under a second.

Keep the analytic solver only if a hit must land on an exact frame that Bullet's bounce won't give.
Honest trade-off: Bullet is less "aimable". A ricochet's exit direction comes out of the solver, so camera framing of the bounce
is found by looking at the bake, not by design.

---

## Part 2: understanding this music, and a visualizer that hears each element

### 2a. How hardstyle/EDM like "Blast Off" is built (short, practical)

- **Grid:** 4/4, built in **8/16/32-bar phrases**. 8 bars says "something changes", 32 says "new section"
  ([Cymatics](https://cymatics.fm/blogs/production/edm-song-structure), [SampleFocus](https://samplefocus.com/blog/edm-song-structure-an-essential-guide/)).
  At 175.8 bpm, 1 bar = 1.365 s, 8 bars = 10.9 s, 16 bars = 21.8 s, 32 bars = 43.7 s. Visual changes belong on 8-bar boundaries, cuts on bars, hits on beats.
- **Arc:** Intro → Break/Verse → **Build** (risers, snare rolls that speed up 1/4→1/8→1/16, filter sweeps) → **Drop** → Break →
  Build → Drop → Outro. Energy ebbs and flows with drops sandwiched between builds and breakdowns
  ([MusicRadar](https://www.musicradar.com/how-to/3-successful-pop-and-edm-track-arrangement-structures-explained),
  [Inspired by Beatz](https://www.inspiredbybeatz.com/en/edm-arrangement-from-loop-to-exciting-track/)).
- **The hardstyle kick is a tuned bassline, not just a drum.** It has:
  - a short attack click ("tok")
  - a punch with a fast **pitch drop** (the "snap")
  - a **distorted, pitched tail** that carries the bass note, often resampled and layered with a reversed tail
  ([musicproductionwiki](https://musicproductionwiki.com/articles/how-to-recreate-the-hardstyle-kick),
  [EDMProd](https://www.edmprod.com/how-to-make-hardstyle/), [ADSR/Serum](https://www.adsrsounds.com/serum-tutorials/hardstyle-kick-sound-design-with-serum/),
  [Native Instruments](https://blog.native-instruments.com/hardstyle-music)).
  **Measured on Blast Off:** drop kicks sweep from a median **75 Hz** about 5 ms after the onset to **43 Hz** about 120 ms in (375 kicks).
  Zeke's style tag is "deathstep with hardcore phonk, huge kicks, laser synth stabs, female vocal, 170–200 bpm"
  (`state/tzeke_songs/blast_off/style.txt`).
- **Sidechain pumping:** a compressor on the bass/pads is keyed by the kick, so they duck on every hit and swell back. That makes
  room for the kick and makes the groove "breathe" ([MusicRadar](https://musicradar.com/how-to/sidechain-bass-and-drums),
  [Sonarworks](https://www.sonarworks.com/soundid-reference/blog/learn/maximize-your-mixs-clarity-and-groove-with-sidechain-compression/)).
  **Measured here:** inside drops the bass stem dips **−3.2 dB one frame after each on-beat kick** and is back (+1–3 dB) by
  frames 3–4. That is a short pump (about 70–100 ms), not a long tempo-synced one.
- **Leads/screeches/stabs:** heavily distorted, detuned or FM leads ("razor-edged laser stabs"). They sit in the
  **other** stem and drive its spectral centroid up.
- **FX:** **uplifters/risers** (noise and pitch rising over 4–16 bars into a drop), **downlifters** (falling sweeps after a
  drop), **impacts** (a boom plus a crash on the drop's first downbeat), and **snare rolls**. These are the "build"
  signals. They mostly live in the other stem plus the high band, so they show as a rising slope, not as onsets.
- **Vocals:** a commanding female vocal (vocals stem). In drops she is often chopped. In breaks she is often the only lead.

### 2b. Frequency territory, and why fixed bands bleed

| Element | Where it lives (mixing-guide starting points) | Notes |
|---|---|---|
| Kick | sub 40–60 Hz, thump 60–100, body 100–200, click/beater 2–5 kHz | [ProduceLikeAPro](https://producelikeapro.com/blog/how-to-eq-14-common-instruments/), [Loopmasters](https://loopmasters.com/articles/2041-EQ-Ranges-For-A-Better-Mix-Part-1) |
| **Hardstyle kick (measured)** | adds energy in **every** band from 66 Hz to 16 kHz: +6 dB at 66–80 Hz, +2–3 dB through 150–900 Hz, +3.5 dB at 9–13 kHz | per-band mix lift at drop kicks vs between kicks (`metrics.json: kick_lift_db_by_band`) |
| Sub / bass | 30–120 Hz (sub), harmonics to ~1 kHz; distorted reese/hoover bass much higher | in this song the bass stem is a sustained sub through the drops |
| Snare | body 120–400 Hz, ring 400–800, attack ~1–4 kHz, crack/snap 5–10 kHz | |
| Clap | 1–3 kHz body, noise to ~8 kHz | |
| Hats / cymbals | clank 200–300 Hz, sizzle 6–10 kHz, air to 16 k | |
| Vocals | fullness ~120 Hz, boom ~240, presence ~5 kHz (4–7), sibilance 4–7 k, air 10–15 k | [Sonicbids cheat sheet](https://blog.sonicbids.com/the-ultimate-eq-cheat-sheet-for-every-common-instrument) |
| Leads / screeches | 300 Hz–8 kHz, distortion harmonics everywhere | |

**Why fixed bands make mush:** every element above spans several bands, and a distorted kick spans *all* of them.
So:
- a "bass band" fires on kicks *and* on sustained sub notes
- a "mid band" (the old snare) fires on kick distortion, leads and vocals
- a "high band" fires on kick clicks, hats and vocal sibilance

That is measured, not assumed: **63 % of the old timeline's "snares" coincide with a kick** (±1 frame). Old `kick`
(<150 Hz mix flux) gets only 25 % of drop beats, because the drop's sub is *sustained*: no flux, so no onset.

### 2c. Better than fixed bands: stems, HPSS, per-element onsets (tested on Blast Off)

**Tools and licences**
- **Demucs v4 `htdemucs` / `htdemucs_ft`** (Hybrid Transformer Demucs; drums / bass / other / vocals). The code is MIT. The
  original repo (facebookresearch/demucs) is **archived** (2025-01-01). The maintained fork is `adefossez/demucs`. `pip install demucs`
  gave **4.1.0**, which pulls weights from Hugging Face `adefossez/HTDemucs(-ft)`. ⚠ **The HF model cards declare no
  licence.** The repo's MIT is the only licence on record. The weights are used locally and are not redistributed, and the stems are of Zeke's own song.
  If the strict "models CC0/PD/CC-BY only" rule is meant to cover analysis models too, this is a **MIT-but-unstated** case.
  Flagging it rather than deciding it. `htdemucs_ft` is a bag of 4 fine-tuned models (better, 4× the compute).
- **Drum-part splitters (kick/snare/hat stems):** they exist, but **none passes the licence gate**.
  - LarsNet weights are **CC BY-NC 4.0** ([repo](https://github.com/polimi-ispl/larsnet)).
  - DrumSep (inagoy, HDemucs) has no stated licence.
  - The MDX23C DrumSep (aufr33/jarredou) has "license terms not formally documented" and its original release is gone
    ([mdxnet-infer](https://pypi.org/project/mdxnet-infer/), [HF mirror](https://huggingface.co/vincewin/drumsep)).
  - Not used. The in-stem onset classifier below replaces it.
- **Beat/downbeat trackers:** madmom's pretrained models are **CC BY-NC-SA 4.0**
  ([paper](https://arxiv.org/pdf/1605.07008), [PyPI](https://pypi.org/project/madmom)). BeatNet and all-in-one build on madmom.
  Excluded. **librosa** (ISC) `beat_track` on the drums+bass stems gave 568 beats at 178.2 bpm, and **99.3 % of them sit
  within ±1 frame of the old 175.8-bpm grid**. The existing grid is right; keep it.
- **HPSS** (librosa `decompose.hpss`, margin 2) on the drums stem gives a percussive part for hats, without the tonal ring of the kick tail.

**What ran** (`audio/analyze_stems.py`, separate venv `~/venvs/audio`: torch 2.5.1+cu124, demucs 4.1.0, librosa 0.11;
`pip cache purge` done, about 2 GB freed)

| Model | Wall time (whole 199 s song, incl. load + write) | GPU inference | Peak GPU (60 s excerpt) |
|---|---|---|---|
| htdemucs | 18.2 s | about 8 s (≈30× realtime) | 0.54 GiB |
| htdemucs_ft | 48.2 s | about 4 × 8 s | 0.54 GiB |

The stems are in `/data/iris_assets/blastoff/stems/{htdemucs,htdemucs_ft}/blast_off/{drums,bass,vocals,other}.wav` (float32,
44.1 kHz, 70 MB each). Stem quality was judged **from measurements and plots, not by listening** (I can't). The
stems are internally consistent: the bass stem is a clean sustained sub in the drops, the vocal stem is −31 dB
average and gated in by the vocal lines, and the drums stem holds the transients. ⚠ The original wav is **48 kHz** and the stems are
44.1 kHz. All analysis maps through seconds, never through sample indices.

**Findings on this song**
1. **Demucs routes the kick by section.** Around 56–60 s the drums stem has almost nothing below 150 Hz (about −10 dB), and the kick's
   low end is in the bass stem. Around 89 s the drums stem carries +43 dB of sub. **A drums-stem-only low-band kick detector
   is therefore wrong in whole sections, and so is a bass-only one.**
2. **The kick's punch in the drops is 150–400 Hz.** The first detector (30–150 Hz flux on drums+bass) hit only 64 % of drop
   beats. The final one is a **per-onset classifier**:
   - Every drums-stem onset (librosa SuperFlux-style, about 5.8 ms hop) gets its **added dB per band**: max over the 46 ms after,
     minus the mean of the 23 ms before, for low / low-mid / mid / presence / high, plus the bass stem's low band.
   - Kick = low, low-mid or bass-low rises ≥ 8 dB.
   - Snare/clap = mid or presence rises ≥ 6 dB, not dwarfed by the low-mid.
   - Onsets are snapped to video frames only at the end.
3. **The kick pattern is 8ths.** Kicks fall on the beat (386) and on the "&" (200), and rarely on the 16ths (58).
   So the output gives **`kick_onbeat`** (382, within ±1 frame of the grid) and **`kick_offbeat`** (262).
4. **Kick vs snare inside the drums stem only half-separates.** 60 % of "snares" share a kick onset, and in drops snares are
   **evenly split between odd and even beats** (93 vs 95). So there is no 2-and-4 backbeat to find; most drop "snares" are the kick's own
   crack or a clap layered on it. Usable channel: **`snare_solo`** (173 snares *not* on a kick: fills, rolls, break snares).
   A true split needs a drum-part separator with an acceptable licence, or NMF with templates. That is not done here.
5. **The first drop hits at 56.07 s** (bass stem +18 dB in 3 frames; `impact`). The old timeline's first entry is **57.57 s**,
   about 1.5 s (≈ 1 bar) late. The old timeline snaps entries to "the kick that starts it", and that kick detector missed the sustained-sub
   drop. **v4's 57.6 s explosion and every effect keyed to `entries[0]` should be re-checked against 56.07 s** (by eye/ear; Zeke's call).
6. **Sections:** a per-bar label (drop / build / break / vocal) from stem levels + kicks + riser is in the npz, but it
   **flickers bar to bar**. It needs smoothing to 8-bar phrases before anything hangs on it. The riser signal (other-stem
   brightness + mix high band, slope over 2 bars) did *not* fire on the build right before the first drop. Treat it as unvalidated.

**Old vs new** (`audio/metrics.json`, htdemucs_ft; htdemucs gives nearly identical kick numbers, 0.61 / 0.84, but fewer snares)

| Measure | Old (mix bands) | v1 low-band (drums+bass) | **New (stem onset classifier)** |
|---|---|---|---|
| kicks detected | 434 | 743 | **644** (382 on-beat + 262 offbeat) |
| kicks on the beat grid (±1 frame) | 37 % | 49 % | **59 %** (the rest are 8th offbeats, by design) |
| drop beats that get a kick | 25 % | 64 % | **83 %** |
| snares | 586 (63 % are kicks) | | 432, of which **173 not on a kick** |
| hats | (no hat channel) | | 1024 onsets (HPSS percussive 7–16 kHz, minus kick-click residual) |

**Output npz** (`/data/iris_assets/blastoff/stems/blastoff_elements_30fps.npz`, 892 KB, 5976 frames, same indexing as
`timeline_30fps.npz`: value i = frame_start + i):
- kick: `kick`, `kick_onbeat`, `kick_offbeat` (bool), `kick_amp`, `kick_env`
- snare and hats: `snare`, `snare_solo`, `snare_env`, `hat`, `hat_env`
- stems: `bass_env`, `vocal_env`, `vocal_on` (gate), `other_env`, `lead_bright` (other-stem centroid), `riser`
- sections and impacts: `impact` (drop entries), `section` + `section_names`, `beat_librosa`
- spectrum: `spec32` (5976×32, full mix, log-spaced 30 Hz–16 kHz, **per-band normalised**, attack 0 / release 0.18 s) + `spec32_edges_hz`
- raw levels: `stem_db` (+ `stem_db_order`)
- envelope settings: `attack_release_s`

**Plot:** `scratch/scenes/blastoff4_research/audio/first_drop_old_vs_new.png` (900 px). It covers 53.6–65.6 s and shows the
spectrogram, old vs new kick (red = on-beat, salmon = offbeat), old vs new snare (light = on a kick, dark = alone), hats,
stem envelopes and stem dB. Dotted blue = the 56.07 s impact; dashed black = the old entry.

### 2d. Visualizer design: what each element drives (rocket and speaker LEDs)

Principle: **one element, one channel, one visual job, its own envelope.** Mush comes from sending everything to everything,
and from symmetric smoothing. Each channel is baked as keyed scalars (the v3/v4 "no handlers" rule); drivers only scale them.

| Element (npz key) | Visual job | Attack / release | Gate / rule |
|---|---|---|---|
| **Kick on-beat** (`kick_onbeat`, `kick_amp`) | the PUNCH: LED-ring flash, cabinet "breath" (scale +1 %), cone excursion, camera shake on big ones, exposure bump | **0 / 120 ms** (exp decay) | full strength in drops; 30 % in builds; skip the camera shake if the previous kick was < 300 ms ago |
| **Kick offbeat** (`kick_offbeat`) | a smaller second pulse: alternate LED segments (A/B chase), not a full flash | 0 / 80 ms | never the camera |
| **Bass** (`bass_env`) | sustained GLOW and SIZE: base emission of rings/strips, rocket-plume glow, speaker cone rest position, the haze tint | **30 ms / 300 ms** | can **duck 3 dB** with the measured pump (multiply by `1 - 0.3*kick_env`) so the kick flash reads against it |
| **Snare alone** (`snare_solo`) | SIDE flashes: the speaker's side LED matrices, tower-corner strobes left/right alternating | 0 / 100 ms | only `snare_solo`; snares on a kick add nothing (they ARE the kick) |
| **Hats** (`hat_env`) | SPARKLE: random-pixel twinkle on the LED panel (seeded noise threshold driven by the env), lens-flare glints | 0 / 50 ms | cap density; at 16ths in builds it becomes a shimmer, fine |
| **Vocals** (`vocal_env`, `vocal_on`) | COLOUR + TEXT: hue shift of the whole LED palette while she sings, lyric/HUD text opacity | 20 ms / 250 ms | gate with `vocal_on`; never flash on syllables |
| **Lead/screech** (`lead_bright`, `other_env`) | laser sweep speed and colour temperature, the spectrum panel's top rows | 30 / 300 ms | |
| **Riser / build** (`riser`, + the bar grid) | INTENSITY RAMP: strobe rate 1/4→1/8→1/16 on the grid, LED chase speed, haze density, pad-light brightness. Black out the last beat before the drop | slow (0.3 / 0.3 s) | validate by eye first (§2c-6) |
| **Impact** (`impact`) | the drop's first downbeat: white-out flash + shockwave + laser burst | one-shot, 0 / 400 ms | only at `impact` frames that are real drops (check the list) |
| **Spectrum** (`spec32`) | the rocket's spectrum panel and the speaker's side matrices (32 columns, log-spaced) | 0 / 180 ms (baked) | per-band normalised already, so the highs don't vanish |

**Anti-mush rules:**
1. **Asymmetric envelopes.** The attack is instant (0–30 ms), the release depends on the element. Never use a linear moving average on hits.
2. **Percentile normalisation per channel** (5–99 %), never by the max, so one spike does not crush the song.
3. **Don't double-count:** a snare on a kick or a hat on a kick adds nothing. The kick owns that frame.
4. **Priorities per frame:** impact > kick > snare > hats. Lower channels are attenuated while a higher one is above 0.5.
5. **Section gain:** multiply every hit channel by a section gain (drop 1.0, build 0.5–0.9 ramping, break 0.2) from the
   smoothed 8-bar section, so the breakdown actually breathes.
6. **Structural changes on the bar grid,** never on onsets: palette switches, camera cuts, which LED group is "hot".
7. **Photosensitivity:** kick and offbeat flashes in drops land at about 5.9 Hz; 16th hats/strobes in builds run at about 11.7 Hz.
   Full-field flashes in the 3–30 Hz range are a photosensitive-epilepsy risk, so flag them on the publish and keep the *full-field*
   white flash to the on-beat kick at most.

**Hook into the v4 plan:** load `blastoff_elements_30fps.npz` next to `timeline_30fps.npz` and bake its channels onto the
same control empty the LED shader already reads (first doc §2). The `spec32` image replaces the old 40-bar `bars`. The beat grid,
`bars`, `segs` and `T_*` can stay from the old timeline; only the element channels and the first drop time change.

---

## Files

- `scratch/scenes/blastoff4_research/physics/`
  - `t_bullet_notunnel.py` (no-tunnel proof, `--keys --keymode ops|numpy`)
  - `matrix.sh` + `m_*.json`
  - `t_reload_check.py`
  - `t_crumble.py` + `cr_*.json` + `crumble_sheet.jpg`
  - `t_old_diag*.py` (the correction)
  - `scale_n200_f1000.json`
- `scratch/scenes/blastoff4_research/audio/`
  - `analyze_stems.py` (`STEMS_MODEL=htdemucs` for the comparison)
  - `metrics.json`, `metrics_htdemucs.json`
  - `first_drop_old_vs_new.png`
  - `gpu_mem.py`, `pump.py`, `explore*.py`
- `/data/iris_assets/blastoff/`
  - `run_demucs.sh`, `demucs.log`
  - `stems/htdemucs*/blast_off/*.wav`
  - `stems/blastoff_elements_30fps.npz`
- `/data/tmp_dl/cellfrac/` (Cell Fracture 0.2.1 zip + unpacked, used by `t_crumble.py`)
- Venv `~/venvs/audio` (5.8 GB, on the root disk). The Demucs weights are cached under `~/.cache/huggingface/hub/`.

## Sources
- Blender 5.2.1 RNA and `bl_operators/rigidbody.py`, read locally. Cell Fracture: https://extensions.blender.org/add-ons/cell-fracture/
- Demucs: https://github.com/facebookresearch/demucs (archived; fork github.com/adefossez/demucs). Rouard, Massa, Défossez, *Hybrid Transformers for Music Source Separation*, ICASSP 2023 (arXiv:2211.08553)
- LarsNet: https://github.com/polimi-ispl/larsnet · DrumSep: https://huggingface.co/vincewin/drumsep · https://pypi.org/project/mdxnet-infer/
- madmom licence: https://arxiv.org/pdf/1605.07008 · https://pypi.org/project/madmom
- Structure: https://cymatics.fm/blogs/production/edm-song-structure · https://samplefocus.com/blog/edm-song-structure-an-essential-guide/ · https://www.musicradar.com/how-to/3-successful-pop-and-edm-track-arrangement-structures-explained · https://www.inspiredbybeatz.com/en/edm-arrangement-from-loop-to-exciting-track/
- Hardstyle kick: https://musicproductionwiki.com/articles/how-to-recreate-the-hardstyle-kick · https://www.edmprod.com/how-to-make-hardstyle/ · https://www.adsrsounds.com/serum-tutorials/hardstyle-kick-sound-design-with-serum/ · https://blog.native-instruments.com/hardstyle-music
- Sidechain: https://musicradar.com/how-to/sidechain-bass-and-drums · https://www.sonarworks.com/soundid-reference/blog/learn/maximize-your-mixs-clarity-and-groove-with-sidechain-compression/
- EQ ranges: https://producelikeapro.com/blog/how-to-eq-14-common-instruments/ · https://loopmasters.com/articles/2041-EQ-Ranges-For-A-Better-Mix-Part-1 · https://blog.sonicbids.com/the-ultimate-eq-cheat-sheet-for-every-common-instrument
