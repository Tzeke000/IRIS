# Rocket launch v2: "Hit a Bump" in photoreal 3-D (Blender 5.2, EEVEE)

v1 (`../rocket/`) got Zeke's verdict "it might need some help with photo realism". v2 rebuilds it
along the ranked fix list in `docs/research/rocket_photoreal_2026-10-10.md`, and follows Zeke's
"look for what is already online first". The vehicle is a real-looking heavy lifter assembled
from **NASA's public-domain SLS Block 1 hardware model**, the **Shuttle SRB** (stretched to 5
segments) and the **Orion capsule**. The tower is **NASA's Mobile Launcher**. The steam, vapour
and clouds are **JangaFX CC0 VDB simulations**. The Earth uses **NASA Blue Marble / Black Marble**.
Scrub comes from **Poly Haven (CC0)**. No agency marks: the model's NASA/ESA/flag decals are not
used, and the Tzeke000 emblem is on the core stage. URLs and licences: `ASSETS.md`.

| File | What |
|---|---|
| `prep_assets.py` | Run once. Turns the downloads (`/data/iris_assets/…`) into one library `.blend`: joins the 4,600 NASA parts into 12 assemblies, drops the sub-0.35 m bolt meshes (1.6M tris), and maps materials to semantic slots. It also stretches the SRB, scales the Orion capsule and the ML, and packs the scrub. About 5 min on CPU. |
| `build_scene.py` | Builds `rocket2.blend` **deterministically** from `timeline_30fps.npz` (about 7 s). Every music reaction is baked as keyframes: no handlers, no Python drivers. |
| `render_frames.py` | Renders frame by frame with resume (temp name, then rename). Applies the per-frame EEVEE volume **start and end** (a slab around the subject). `--vram` logs this process's GPU MB. |
| `render.sh` | `prep` · `build` · `frames [S] [E] [PCT]` (chunked, resumable) · `stills F1,F2 [PCT]` · `mux [S] [E]` · `all` |
| `ASSETS.md` | Every third-party asset: URL, licence, what it's used for, plus what was looked at and rejected. |

```bash
R=scripts/lyric_scenes/real/rocket2/render.sh
$R prep                     # once (needs the downloads listed in ASSETS.md)
$R build                    # → scratch/scenes/rocket2/rocket2.blend
$R frames 1 3570            # full song → scratch/scenes/rocket2/frames/ (resume: just re-run)
$R mux                      # → scratch/scenes/rocket2/hit_a_bump_rocket2.mp4
STILLS=/tmp/look $R stills 301,988,1003 50   # look-dev stills at 50 %
```
Env: `OUT_DIR`, `CHUNK` (60), `SAMPLES` (32 baked in), `PCT`, `MP4`, `ASSETS`, `LIB`, `VRAM=1`.
Every Blender launch runs as `flock /tmp/iris_gpu_render.lock nice -n 15 …`, one chunk at a time.

## Story → music (same song, same timeline, T-0 = drop entry 1 = frame 988)

| Section | Picture (SLS sequence) | Music hook |
|---|---|---|
| intro 0–32.9 s | Night pad under xenon floods: a 400 mm view across the lagoon, the LED countdown clock, a 300 mm tilt up to the emblem, a low pad remote. Deluge water starts sheeting off the ML deck at T-10 s. Hydrogen burn-off sparks fire one bar before the engines. The **four RS-25s start on the bar before the build** (translucent plumes, Mach disks, the first steam) | build `T_pre` = vapour thickens, beacons double/quadruple (hi-hats strobe them), camera creep; `T_black` = floods cut, the core engines keep burning |
| drop 1 (32.9 s) | **SRB ignition on the entry**: a 2-frame overexposed pop, then the camera stops down. The trench blast rolls east, steam rises around the ML, ice sheets off the tank. Shots: 1.3 km 135 mm, tower-top remote, 5 km 800 mm tracking with lag and heat shimmer, onboard booster camera | kick = plume pulse + jolt (scaled down on long lenses), one-frame exposure pop on `T_strobe` hits |
| calm | Climbing out of a CC0-VDB cumulus deck. The sun grazes the tops by the end | sub-bass breathes in the plume |
| drop 2 (59.7 s) | MAX-Q: condensation **cone shroud** (volume) on the upper stack and the booster noses. Onboard camera looking up into it, plus a long-lens air-to-air | kicks flicker the cone |
| drop 3 (72.8 s) | BOOSTER SEP: boosters peel and tumble. Onboard core camera, then a far long lens | |
| calm | The plume balloons into the **sunlit "jellyfish"** over a dark Earth | |
| drop 4 (92.5 s) | **Launch-abort-tower jettison**: three SAJ panels peel off and the tower and ogive yank away, revealing Orion | |
| calm 99.2 s | MECO, core separation (onboard aft view), coast, blackout | exposure dip |
| BIG drop 102.4 s | Upper-stage ignition and the sun breaking the limb, over NASA Blue/Black Marble | push on every bar downbeat |
| outro | Engine cutoff, drift | |

Snare = ice flakes shed from the tank (white tumbling flakes, no glints) plus flood flicker before launch.

## Look / tech
* **EEVEE** 5.2, 32 TAA samples, ray tracing, volumetric shadows, tile 2 / 96 slices with a
  **per-frame custom range** (long lenses get a ±500 m slab at kilometre range), motion blur
  (0.5, START, CONSTANT keys before cuts), per-shot DOF, AgX Medium High Contrast.
* Materials: dielectric paint and foam with CC0 scan roughness and normal (ambientCG,
  Poly Haven), no albedo above 0.75, frost bands that shed after liftoff, plume soot building in
  flight, the deluge wetting the pad and the ML.
* Light: the plume IS the key. Chains of 2600 K point lights down each SRB plume plus a 40 m
  area bounce, xenon spots on the stack, sky searchlights. Steam is lit from inside, through the
  VDB flames grid plus a falloff from the plume axis. The exposure ride clips at ignition.
* Compositor (5.x node group): mist-pass atmospheric haze, distance-scaled heat shimmer
  (Displace), streak glare on true highlights only, fog glow, dispersion, keyed barrel
  distortion for the onboard shots, vignette, per-frame film grain.

## Measured
(see the end of this file, filled in from `scratch/scenes/rocket2/` test logs)

> **STATUS 2026-10-11: WORK IN PROGRESS** — the builder was cut off mid-build by an API session limit. Not render-ready; resume from here (see the music-scenes memory note).
