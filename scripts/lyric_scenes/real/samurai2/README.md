# Neon samurai duel v2: real characters ("Pull Me Close", Blender 5.2, EEVEE)

Zeke 2026-10-10 asked for **real human models instead of the hand-built mannequins**. v2 is the v1 scene
(`../samurai/README.md`: the street, rain, effects, the 64 shots, lights and music sync are unchanged), with the
two fighters replaced by downloaded, licensed characters. They are driven on their own skeletons by a retarget solver.

| Fighter | Asset (all CC BY 4.0; credits block in `ASSETS.md`) | Plus |
|---|---|---|
| **Ronin** (cyan blade) | Renderpeople **"Eric Rigged 001"**: a photoreal scan with an 89-bone rig and real fingers. He wears **"Japanese Clothes"** by Takeru: haori, hakama and jinbei. | v1's procedural woven kasa. The face stays under the brim. |
| **Opponent** (magenta blade) | **"Death Samurai Character"** by Alexandru Jardan: full lacquered yoroi with 78 rigid parts, a skull mask and a kabuto. | Our fitted armature, with finger chains fitted to his gloves. |

Assets stay on `/data/iris_assets/characters/` and are never committed. Fallback: the MPFB2 CC0 bodies in `mpfb_base/`.

| File | What |
|---|---|
| `choreo.py` | v1's planner (pure numpy). The **planning skeleton is re-measured from Eric**, and each fighter has his own shoulders and reach. The contact templates are solved per attacker. The pass-through cut and kiriage distance are retuned. |
| `figures.py` | **New (bpy).** Imports the assets into one canonical rest frame, fits the garments and the armour rig, adds the spring bones, calibrates the grips, and runs the per-frame solver (`FigureSolver`). |
| `build_scene.py` | v1's builder. The mannequin section is replaced by `figures.py`. It also adds asset look-dev (wet cloth, urushi lacquer) and a camera-tracking rim pair. It self-verifies (below) and writes `samurai2.blend` and `samurai2_solve.npz` (per-frame solver report). |
| `render_frames.py`, `render.sh` | Same as v1. Chunked, resumable, GPU lock, mux. The defaults point to `scratch/scenes/samurai2/`. |

```bash
R=scripts/lyric_scenes/real/samurai2/render.sh
$R build                     # ~40 s CPU -> scratch/scenes/samurai2/samurai2.blend (+ _solve.npz)
$R frames 1 5675             # whole song, resumable -> scratch/scenes/samurai2/frames/
$R mux                       # -> scratch/scenes/samurai2/pull_me_close_samurai2.mp4
OUT_DIR=/tmp/x $R frames 1290 1394 50     # a half-res test slice
```

## How the characters are done (`figures.py`)
* **Canonical rest.** Each import is normalised: Eric is scaled to 1.80 m; the clothes ×0.0105 (the research fit); the armour is used at its natural size, about 1.82 m of body under a 1.95 m crest. World transforms are baked into the data (`Armature.transform` / `Mesh.transform`), so the armature object is identity, faces +Y, and has its hip joint over the origin. That is choreo's standing frame, so the armature object takes v1's location and yaw keys unchanged. The glTF importer leaves objects in QUATERNION rotation mode, and yaw keys would be silently ignored, so the build resets it.
* **Bone maps** go from our ~17 joints to each skeleton. Eric: `hip_02`, `spine_01/02/03`, `shoulder_*` (clavicle), `upperarm/lowerarm/hand_*`, `lowerarm_twist_*`, `upperleg/lowerleg/foot/ball_*`, plus 15 finger bones per hand. Joint positions are read from **bone heads**, never from the glTF tails, which point anywhere.
* **Death Samurai rig.** Body joints come from mesh slices (arm-tube centres, pants leg centres, feet, skull). Finger chains are fitted **geodesically** on each glove: the fingertips are the five strongest maxima of geodesic distance from the cuff; the joints are centroids of geodesic bands from each tip. All 59 plates are **bone-parented rigidly**, so a plate never bends and there is no deform cost. The 4 kusazuri and 2 sode hang on spring bones. Only the sleeves, pants, knee ropes and gloves are skinned.
* **Garments.** Weights come from the nearest body vertex near the skin and blend to bone-segment weights farther away. Then:
  * The yoke loses upper-arm influence inside the shoulder line, and a half-rotation **shoulder helper bone** carries the sleeve seams. This addresses the bunching in jodan.
  * The sleeve bags are **deflated toward the arm** and partly hung on a gravity spring. Kimono sleeves skinned to a raised arm otherwise become wings.
  * **Each hakama leg hangs on its own spring bone** that follows its leg. With no cross-leg mixing, the stripes no longer swirl in lunges.
  * The haori skirt hangs on four panel springs.
  * Faces fully covered by an outer layer are deleted (ray test), which stops poke-through and saves deform time. The scan's shirt under the kimono's under-arm slits is tinted to a dark inner layer.
* **The solver** (v1's `Rig`, generalised):
  * The spine and head use world-delta rotations from the plan; the head looks at the opponent.
  * The arms and legs use analytic two-bone IK on **each body's own lengths**, oriented by **hinge frames**: the upper arm and forearm share the elbow axis, so the elbow never candy-wraps.
  * **Clavicle protraction** (≤ 24°) gives extra reach before anything else.
  * A **pelvis drop** handles out-of-reach planted feet, and a **chest yaw assist** handles out-of-reach grips (the iai draw and the noto). All three act only where needed and are smoothed.
  * **Grip:** at build time the grip curl is posed and the deformed hand skin is read back. The script fits the largest empty tunnel through the fist, on an axis diagonal across the palm like a real power grip. The curl is scaled until that hole is about 1.5 cm in radius, the size of the hilt. Per frame, the hole is put on the hilt axis at v1's grip stations. The roll about the hilt is chosen by alternating with the IK elbow: the hand aligns with the forearm, and forearm twist beyond 75° is priced. The forearm twist is shared with the twist bone (Eric) or with the forearm itself (armour).
  * The fingers blend from the grip curl to a relaxed curl using choreo's grip weight.
  * Springs are v1's damped world-space integrators. The skirt panels follow the thighs, the hakama legs follow the legs, the sode follow the upper arms, and the sleeve bags follow gravity.
  * The poses are baked with `foreach_set`. There are no constraints, drivers or handlers.

## Measured (2026-10-10; V100 shared with the arcade render and the live voice model)
* **Self-verification in the build:**
  * The evaluated pose matches the solver to 0.20 cm (12 frames).
  * The **fist hole sits on the hilt axis to 0.02 cm on average, 0.07 cm at worst** (40 evaluated gripping hand-frames).
  * **Blade gap at the 170 clashes (from the baked sword F-curves): median 0.0 cm, 90th percentile 0.3 cm, max 3.5 cm.** The max is the bind at 3814. v1 had 0.0 / 0.7 / 6.7 cm.
* **Hands on the hilt** (all gripping frames):
  * The miss is 0 on all but a handful of frames: the ronin's right hand at the iai draw (frames 667-670, ≤ 13.7 cm), the opponent's at 666 and in the noto (≤ 5 cm). The planned draw needs more reach than a real arm has. v1 stretched its arms 1.3× there.
  * Wrist bend: median 17-36°, 95th percentile 67-83°.
  * Forearm twist: 95th percentile 112-146°, past anatomical comfort in the hardest cuts.
* **Feet:** the plant miss is 0 except during the fast retreat at frames 5297-5304 (≤ 16 cm for 2-4 frames). The pelvis drop is capped at 18 cm.
* **Speed:** 1080p, 32 TAA samples, motion blur: **2.92 s/frame** mean for frames 576-600 (calm) and the same for 4931-4955 (big drop), 2.6-3.25 s. The first frame of each launch costs about +2.2 s. v1 was 3.08 / 3.14. The per-frame cost is still dominated by the rain instances, not by the characters: the armour is rigid, and only about 70k triangles are skinned.
* **VRAM:** peak 3.0 GB for the Blender process (v1 2.9 GB).
* The full song is 5675 × about 2.95 s plus launches, about **4.8 h**.

## Known weak spots
* **Wide kimono sleeves without a cloth sim.** The sleeves are deflated and sprung, but with both arms overhead (jodan) the haori still bunches into sleeve "wings" around the raised arms. This shows mostly from behind.
* **Forearm twist** gets high on some strikes (the grip roll is constrained by the hilt), so a wrist can look wrung for a few frames. The twist bone hides most of it on Eric. On the armour the kote rotate with it.
* **The iai draw and noto reach** (a few frames, see above): the hand leaves the hilt by up to 14 cm during the fast draw. It is motion-blurred, but visible if frozen.
* **The opponent's arms are short** (0.46 m shoulder→wrist, Eric 0.52-0.54). The plan uses his real reach, so he fights a little closer.
* **No cloth collision.** The hakama hem can graze a shoe in the deepest steps, and armour plates can interpenetrate for a frame in extreme poses.
* **Eric's face** is a real scan. It is hidden by the kasa brim and the lighting, but a few frames of the kasa close-up show a cheek.

> **STATUS 2026-10-11: WORK IN PROGRESS** — the builder was cut off mid-build by an API session limit. Not render-ready; resume from here (see the music-scenes memory note).
