# Rocket launch — "Hit a Bump" in true 3-D (Blender 5.2, EEVEE)

Zeke 2026-10-10: *"super realistic looking"*. A generic heavy-lift rocket (core + two side
boosters, no real agency marks) with the Tzeke000 emblem on the fuselage and the tower banner.
It launches at dawn, music-synced to the song's timeline (119 s, 148 bpm, 3570 frames at 30 fps).

| File | What |
|---|---|
| `build_scene.py` | Builds `rocket.blend` **deterministically** from `timeline_30fps.npz` (see `../../export_timeline.py`). Every music reaction is baked as keyframes, so there are no handlers and no Python drivers. |
| `render_frames.py` | Renders frames one at a time. It skips finished PNGs (resume), writes to a temp name and renames, and applies the per-frame EEVEE volume range. |
| `render.sh` | `build` · `frames [S] [E] [PCT]` (chunked, resumable) · `stills F1,F2 [PCT]` · `mux [S] [E]` · `all` |

```bash
R=scripts/lyric_scenes/real/rocket/render.sh
$R build                    # ~5 s, writes scratch/scenes/rocket/rocket.blend
$R frames 1 3570            # full song → scratch/scenes/rocket/frames/ (resume: just re-run)
$R mux                      # → scratch/scenes/rocket/hit_a_bump_rocket.mp4 (H.264 crf17 + AAC 320k)
$R frames 958 1065 50       # a 50%-res slice; OUT_DIR=… keeps it apart from the full render
```
Env: `OUT_DIR`, `CHUNK` (60 frames per Blender launch), `SAMPLES` (EEVEE TAA, 48 baked in), `PCT`, `MP4`.
Every Blender launch runs as `flock /tmp/iris_gpu_render.lock nice -n 15 …`. The lock is released between
chunks, so other GPU jobs can interleave. Frame f is timeline index f-1, and the audio is offset to match in `mux`.

## Story → music (drop entries come from the npz, so the scene re-times itself)
| Section | Picture | Music hook |
|---|---|---|
| intro 0–32.9 s | Dawn pad, floodlit rocket, LOX vapour, searchlights, 7-segment LED clock "HIT A BUMP / T-00:00:33", **T-0 on entry[0]** | beat = clock colon blink; build `T_pre` = vapour thickens, beacons ×2/×4, camera creep; the swing arms retract one bar early (anticipation); `T_black` = pad lights cut |
| drop 1 | IGNITION (green TEA-TEB flash, base fire, trench fireball, billowing steam), liftoff, tower clear, chase | kick = plume pulse + camera jolt; one-frame exposure pop on budgeted `T_strobe` hits |
| calm | Climbs out of a cloud deck at dawn | sub-bass breathes in the plume |
| drop 2 (59.7 s) | MAX-Q: vapour cone bursts on the entry and then flickers | cone density pulses on each kick |
| drop 3 (72.8 s) | BOOSTER SEP: sep-motor puffs, boosters peel off and tumble | |
| drop 4 (92.5 s) | FAIRING SEP: pyro puffs, halves hinge open, payload revealed | |
| calm 99.2 s | MECO, stage separation, coast; `T_black` blackout | exposure dip ("silence") |
| BIG drop 102.4 s | Upper-stage ignition (green flash) and the **sun breaking Earth's limb** with a flare, over the curved Earth | camera push + plume/sun flare on every bar downbeat (`T_one_age`) |
| outro | Engine cutoff, drift over the limb | |

Throughout: snare = an ice-chip burst shed from the tank (GN, stateless: seed = snare index) plus pad-flood
flicker before launch. Hi-hat = ice-crystal glints around the vehicle (GN, constant topology, so the motion blur stays clean).

## Look / tech
* **EEVEE** (5.2), 48 TAA samples, raytracing, volumetric shadows, motion blur (shutter 0.5, START, with
  CONSTANT keys before every cut so nothing smears across cuts), per-shot DOF, AgX Medium-High Contrast.
* Compositor: streak glare (threshold 80, the sun only), bloom, slight lens dispersion, vignette.
* World: 5.x multiple-scattering sky (altitude and sun keyed from the pad to 100 km). Earth surface,
  clouds and night-side city lights come from a **ray-sphere hit in the world shader**, with a custom sun
  disc that the planet occludes, so the sunrise breaks the real limb.
* Plumes: an emissive shell with shock diamonds plus an emissive **volume** sheath that balloons with altitude.
  Steam, fire and separation puffs are **Geometry-Nodes "billow" meshes** (noise-displaced sphere clusters).
  Froxel volumes looked scaly at 1080p.
* `volumetric_end` can't be animated. The build stores one value per frame (about 1.8× the camera→rocket
  distance) and `render_frames.py` applies it, so close shots get fine froxels.

## Measured (V100 shared, 1920×1080, 48 samples, steady state)
calm pad ≈ 2.6–2.7 s/frame · ignition/steam drop ≈ 4.3–4.7 s · max-Q 2.6–2.8 s · big drop 2.7–2.8 s ·
the first frame per launch costs +1.5 s (shader compile) · peak VRAM ≈ 4.3 GB (Blender process).
Full song ≈ 3 h on an idle GPU.

Strobe note: the full-frame pops (ignition, `T_strobe` hits) are single frames on the budgeted hits.
Flag the publish for photosensitivity like the other scenes.
