# Neon samurai duel: "Pull Me Close" in true 3-D (Blender 5.2, EEVEE)

Zeke 2026-10-10: *"super realistic looking"*. It is night in heavy rain, on a small stone plaza
between old wooden townhouses. Neon signs are mounted on the walls, paper lanterns hang under the
eaves, and a torii gate stands in the fog at the end of the north lane. Two duelists fight there, and
their faces are never shown:

- **The ronin** wears a straw kasa, a kimono with wide sleeves, an obi and hakama. His blade edge glows **cyan**.
- **The hood** wears a deep hood and a long split coat. His blade edge glows **magenta**.

The song is intimate and tense, so the piece is staged as a dangerous slow dance as much as a fight.
The Tzeke000 emblem appears on two large shop lanterns on the east wall. There are no real brands or places.
The "PULL ME CLOSE" neon on the west wall uses outline tubes made from Bebas Neue.

| File | What |
|---|---|
| `choreo.py` | **Pure numpy, no bpy.** Plans the whole song. It covers the duel geometry (distance, axis, midpoint), the 167 clashes and their attack types, sword key poses, analytic contact templates, the reach clamp, footwork, pelvis height, torso lean, hands, every camera shot, and the light, neon and lightning curves. Run it with `~/venvs/iris-v100/bin/python choreo.py` to print a report. |
| `build_scene.py` | Builds `samurai.blend` deterministically in about 10 s on the CPU. It contains the rig and garments, the street, the Geometry Nodes effects and the lights, and it **bakes** every reaction as keyframes. There are no handlers, no drivers and no simulations. A self-test at the end compares the evaluated bone positions against the plan (0.00 cm). |
| `render_frames.py` | Renders frames one at a time. It skips finished PNGs (resume), and writes to a temp name before renaming. |
| `render.sh` | `build` · `frames [S] [E] [PCT]` (chunked, resumable) · `stills F1,F2 [PCT]` · `mux [S] [E]` · `all` |

```bash
R=scripts/lyric_scenes/real/samurai/render.sh
$R build                      # writes scratch/scenes/samurai/samurai.blend
$R frames 1 5675              # the whole song -> scratch/scenes/samurai/frames/ (to resume, run it again)
$R mux                        # -> scratch/scenes/samurai/pull_me_close_samurai.mp4 (H.264 crf17 + AAC 320k)
OUT_DIR=/tmp/x $R frames 1261 1380 50     # a 50 % test slice that stays apart from the full render
```
Environment variables: `OUT_DIR`, `BLEND` (always `scratch/scenes/samurai/samurai.blend` unless you set it), `CHUNK` (60
frames per Blender launch), `SAMPLES` (EEVEE TAA; 32 is baked in), `PCT`, `MP4`, and `VRAM_LOG` (samples the
Blender process's GPU memory once a second). Every Blender launch runs as `flock /tmp/iris_gpu_render.lock nice -n 15 …`,
one chunk at a time, so other GPU jobs can interleave. Frame f is timeline index f-1, which is audio time (f-1)/30.

## The characters (how "realistic" is done)
The figures are procedural and built in this script. **No external asset and no licence is involved.**

- **Rig.** It is built from `choreo.BONES`: a spine chain, arms and legs, plus 7 *spring bones* (four skirt
  panels, two sleeve bags and the hood).
- **Bone rotations are computed, not constrained.** The script does analytic two-bone IK for the arms and
  legs in numpy, with swing-only orientations so limbs don't twist. It uses the conjugation
  `basis = Cᵀ·W` against Blender's own rest matrices, then bulk-bakes quaternions. This avoids the
  roll-dependent IK `pole_angle` (measured: `cos(elbow, pole)` was only 0.69). Arms may stretch up to 1.3×
  for the few frames of the iai draw and the noto, so the hand stays on the hilt.
- **Garments are lofted meshes with computed skin weights.** These are the kimono torso with sloped
  shoulders, sleeves with hanging bags, a pleated hakama, the obi, the collar, a woven straw kasa with chin
  cords, the hooded coat with its front slit, boots, and gloved fists. The fists are modelled in *sword
  space* around the grip and mapped into the hand's rest frame, so they close around the hilt. The skirt hem
  follows the legs left and right, and the panel bones add the lag.
- **Follow-through.** Damped springs are integrated over the whole song in world space, so the skirt
  panels, sleeve bags and hood swing and settle after every lunge.
- **Footwork.** A plant/step solver keeps feet planted. Steps land on **kicks** (drops), on **beats** or on the
  sung **"close"** (intro). The attacker's front foot **stamps on the clash** (fumikomi) with a splash. The
  pelvis sinks wherever a planted foot would be out of leg reach.
- **Sword contact.** Six attack templates are used: shomen, kesa (right and left), do, tsuki and kiriage,
  plus a symmetric drop-entry clash, a bind, and the final pass-through cut. Each template is **solved** by a
  grid search for the defender's tsuba and the attacker's distance, so both blades cross at one point and all
  four wrists stay within reach. Across 167 clashes the measured blade gap at contact is a median 0.0 cm (90th percentile
  0.7 cm). The reach clamp at clashes averages 0.2 cm.
- **Motion principles.** Anticipation: wind-up keys before each strike, and big wind-ups in every build. Strikes
  ease in so they accelerate into contact, then a hit-stop and an eased-out rebound. Sword paths **orbit the
  chest pivot**, so the arcs come from the blade rotation. Overlap and follow-through come from the spring
  bones.

## Story → music
| Section | Picture | Music hook |
|---|---|---|
| intro 0–15.6 s | crane down the gate lane, the crest lantern, rain on the kasa; the duelists stand 9 m apart and stalk around each other | lanterns warm up slowly |
| "pull me close, close, close" (17–19 s, 24–26 s, 37–40 s) | **each "close" is one step closer**: the step lands on the word, with a splash | lantern and neon swell on every "close" |
| "I like that **dangerous pose**" | the hand finds the hilt on "dangerous"; the **iai draw** (with saya-biki) on "pose" | |
| "skin on skin" | blade tips reach out and **touch** (tiny spark) | |
| build `T_pre` | A rises to jodan and B to hasso, holding the wind-up; a slow push-in | neon ramps up and **blazes on "glows"** |
| `T_black` (half beat) | neon and lamps die; only a faint lightning glint | |
| drop entries | symmetric clash: sparks, a **rain shockwave** (radial spray plus a ground ripple ring), **lightning**, and rain **hit-stop** (it slows to 8 % and eases back) | |
| drops | attacker and defender exchanges: **snare = blade clash** (spark burst, spark light, camera jolt), **kick = stomp/step** (splash, ripple ring), attacker swaps every 2 bars, the pair circles | neon breathes with the sub-bass and kick; one tube stutters |
| short calm gaps | **binds**: blades locked near the guards, pushing, then shoved apart on the blackout | |
| 128–134 s | the 1-second hit at 130.5 s is one huge clash, then a bind | |
| BIG drop 161.4–175.6 | a flurry, a clash on every snare (9-frame minimum gap), alternating kesa cuts; lightning on the strobe hits | **"the whole room glows"**: neon at full |
| 175.6–181 | they back off to opposite sides of the plaza | |
| 181.2 mini-drop | **pass-through cut**: both charge, cut as they pass (sparks, shockwave), and end **back to back** | |
| outro | zanshin, chiburi flick, noto (the sheathing slides the saya), standing back to back | neon fades and the lanterns stay |

The cameras follow 64 shots and cut on bar downbeats: every 2 bars in the drops, every bar in the big
drop. The camera holds a CONSTANT key before each cut, so motion blur never smears across a cut. The shot types are
establishing crane, crest insert, kasa close-up, hand on the hilt, feet in the puddle, tips touching, profile
(the gate lane behind), low puddle angle, over-the-shoulder, orbit, high, an 85 mm telephoto down the lane,
the charge, back-to-back and the final crane-up. Words come from `words.json`. The trailing "we'll be right
back / see you next time" is a whisper hallucination and is ignored.

## Look and tech
* **EEVEE** (5.2): 32 TAA samples, screen-space raytracing, fast GI, AgX Medium High Contrast,
  motion blur (shutter 0.5, centred) and per-shot DOF.
* **Volumetric haze**: a world volume (density 0.0065, anisotropy 0.6) with volumetric shadows, so the
  street lamps' cones and the gate light show shafts.
* **Rain**: Geometry Nodes streaks, about 63k in the plaza and lanes plus about 8.5k in bead curtains dripping off the
  six plaza eaves. They are stateless (`Time` is a baked input; drops wrap at zero scale), thin triangular prisms with
  a semi-transparent glossy water material. Motion blur adds the streak, so they pick up neon colour instead of
  reading as white sticks. The ground splashes (3.5k) re-roll their positions every cycle by ID.
* **Wet ground shader**: flagstones in the plaza and cobbles in the lanes, puddle mask, two layers of
  voronoi rain ripples, plus a keyed shockwave ring and 4 keyed footstep ripple slots.
* **Effect bursts** use one stateless GN "Burst" tree (ballistic points drawn as velocity-aligned streaks):
  4 spark slots with point lights, 4 footstep-splash slots and 1 shockwave spray. A slot switches to its next
  event **one frame early**: the origin and seed change while age = -1 frame, which is dead. Otherwise the
  motion-blur subframes would see live sparks at the old origin and draw a long dotted line across the frame.
* **Lightning**: three jagged bolt meshes (visibility keyed), a sun flash and the sky clouds lit by the flash.
* **Compositor**: bloom, light streak glare (sparks only), slight lens dispersion and a vignette.

## Measured (2026-10-10, V100 shared with the live voice model and the rocket render; 1920×1080, 32 samples)
| run | s/frame |
|---|---|
| calm intro, frames 576–600 (25 consecutive) | **3.08** mean (2.90–3.44) |
| big drop, frames 4931–4955 (flurry, sparks, lightning) | **3.14** mean (2.83–3.37) |
| first frame of every Blender launch | +2 s (scene sync / shaders) |
| the same clip at **50 %** resolution | 3.1. The cost is **CPU-side scene evaluation**, not pixels: about 72k realised rain instances and two rigs, evaluated three times for motion blur. Halving the resolution does not speed up tests. |

Peak GPU memory of the Blender process: **2.9 GB**. The full song is 5675 frames × about 3.15 s plus about 95 launches,
which comes to **about 5.3 h** on the shared GPU (the budget was ≤ 8 h).

## Known weak spots
* **Close-ups of hands and feet.** The bodies are procedural: lofted cloth, gloved fists and simple boots. They read well as
  silhouettes in medium and wide shots (the camera mostly stays there), but hands are dark fists without fingers.
  The one "hand on the hilt" insert works only because it is dark.
* **Cloth is skinning plus spring bones, not a simulation.** There is no cloth collision, so in the deepest lunges
  the hakama hem can pass through a shin for a frame or two.
* **IK limits.** On a handful of frames of the fastest lunges and the final charge a foot falls up to about 18 cm
  short of its plant (the pelvis is clamped at 0.80 m). Arms stretch up to 1.3× for the 2–6 frames of the
  iai draw and the noto so the hand stays on the hilt.
* **Drop-entry lightning** whitens 1–2 frames. That frame, plus the budgeted strobe hits in the big drop, makes this a
  photosensitivity flag for the publish, as with the other scenes.
* **Off-screen reflections.** The wet ground uses screen-space raytracing (no planar probe), so a neon sign
  outside the frame does not appear in a puddle.
