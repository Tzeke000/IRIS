# Rocket launch v4 — "Blast Off", the booster-3 cut (Blender 5.2, EEVEE) — built 2026-10-13

Zeke's brief + song map (10-12/10-13) → `build_scene.py --rave --v4` (args: `scratch/scenes/blastoff4_env.sh`). v3
(`../rocket3/`) stays reproducible. Every music reaction is still baked to keys (no handlers, no Python drivers).

| Time (Zeke's map, snapped to bars) | Picture |
|---|---|
| 0:00–0:05 intro | tower + deck + lightning-tower LEDs breathe/strobe with kicks + hats; ignition white-strobe; liftoff on "blast off" (5.47 s); each tower LEVEL lights as the stack passes it |
| 0:05–0:33 | strips + nose chase rainbow to the beat; climb set re-cut |
| 0:33 build | the LED rings BOOT bar by bar, bottom → top; the 32-band spectrum panel on the core |
| 0:43.8 (bar 8 of the build) | ALL LEDs RED, pulsing on the kicks (alternate-halves warning blink); camera-feed HUD "⚠ BOOSTER 3 FAULT — EXPLOSION IMMINENT"; booster 3's plume sputters on the snares |
| 0:53.3 | booster 3 separates early: pivots on its forward attach, rips free, cartwheels STILL BURNING |
| 0:56.07 false drop | it explodes ~55 m off the core (JangaFX Mid-Air Explosion, flash, shock shell, casing + embers); the feed shows SIGNAL LOST; the stack TUMBLES (flight model: 27.5° overshoot, saturated TVC, gimbals swing, plumes bend, RCS puffs, sag + drift) and recovers (<2° by 1:05.6) |
| 1:07 halftime drop | rainbow REBOOT sweep bottom → top; vapour collar |
| 1:28 hardstyle drop | boosters 1+2 separate (kick + screech drive the LEDs) |
| 1:50 "blast off" | fairing opens, the SPEAKER comes out: Sousinho cabinet (CC-BY) + v3's moving cones, LED rings/edges, side 32-band LED matrices, blinders, 4 moving heads (CC-BY) |
| 1:50–2:33 breakdowns | the synth drives the speaker; Earth's city lights + an aurora breathe with the bass; asteroid GLIMPSE at 2:02, then from 2:17 it closes in with a TRAIL |
| 2:33–3:05 drop 2 | LASERS (GN fans of emissive beams + a 12-beam tunnel at the rock; drop 2 ONLY), sound-wave rings pin it |
| 3:04.0 (one bar before the drop ends) | it SHATTERS: Cell Fracture + Blender rigid bodies (60 substeps, 20 iterations, split impulse, 1 m proxy), baked to keys; debris bounces / crumbles off the speaker in the outro |
| 3:05 outro | the speaker drifts, the last hit, fade |

**Build gates (fail the build):** countdown board segments |x| < 2.7 (measured 2.575); booster 3 bursts 50–80 m off the core;
tumble overshoot 25–35°; recovered < 5° by 1:06; gimbal ≤ 8°; NO debris vertex inside the speaker on any frame (0 events;
closest 0.23 m) and ≥ 4 measured speaker contacts.

**Measured (10-13, V100, EEVEE, production per-shot spp/tiles, one worker):** pad/steam ~8–16 s/frame, climb + tumble
~4–4.7 s, boosters/high ~2.5–3 s, space + speaker + lasers ~1.1–2.8 s → ~21.6k worker-seconds for 5,976 frames
(~6 h solo). Three capped workers ran the 0:40–1:15 timing clip at ~1.75x one worker's throughput → full render
≈ 3.5 h (est. 3–4.5 h) + the HUD pass (328 frames, ~20 s at 25 %, ~1–2 min at 100 %) + mux. Bullet bake 23 s, build ~45 s.

**v4.1 (Iris's review, 10-13):** (1) booster-3 fireball rebuilt: EmberGen grids are normalised (measured with Blender's
bundled `openvdb`: temperature 0–2.2, flames 0–1.1), so temperature maps 0.25–2.2 → 1900–3300 K through a Blackbody node
(1500–2500 K read salmon-pink under AgX), emission = flames² × a keyed glow, dark grey-brown denser smoke, playback starts
5 sim frames in (the first frames are near-empty), an orange 1900 K point flash that lights the stack's flank (irradiance
~100 W/m² peak, ~0.3 s); (2) the 1:07 reboot: a keyed `SweepZ` wavefront in the LED shader (rainbow below, red above, a
±7 m white-hot band + each ring flares as it's reached, one bar), no white strobe and a capped level during that bar,
and a new shot `R0_reboot` cut on the 66.97 s entry: 11 mm, 30 m off the stack near its base looking up the full
height; (3) speaker-bound debris ~2× (radius 6–11.5 m), re-baked at 60 substeps / 1.0 m pad: 0 inside, closest 0.14 m.

**Known weak spots:** the first ~0.1 s of the blast is a near-white fireball + flash (then orange-brown); the TVC moment arm
is 1.75x SLS's so the recovery fits the music (flight_model.py docstring); the post-blast HUD wording "⚠ BOOSTER 3 LOST —
ATTITUDE ERROR" is a placeholder (edit `post_hud.py` WARN3, re-run `render.sh hud` + `mux` only); drop-2 laser fans are
dense in the over-the-shoulder shot (M8_tunnel).

**Files (v4):** `flight_model.py` (numpy flight model, self-test: `python3 flight_model.py`), `post_hud.py` (HUD
compositor pass), `par_frames.sh` (N capped workers), `render.sh hud` / `mux` (HUD frames replace their beauty frames).

```bash
source scratch/scenes/blastoff4_env.sh
R=scripts/lyric_scenes/real/rocket4/render.sh
$R build                                                                  # ~45 s (Bullet bake ~23 s), no render: may run bare
scripts/lyric_scenes/real/rocket4/par_frames.sh 1 5976 3 100 r4full      # 3 capped workers; resumable (re-run = resume)
scripts/lyric_scenes/real/rocket4/par_frames.sh wait r4full              # blocks until all three units exit
scripts/server/render.sh r4post bash -c "$PWD/$R hud && $PWD/$R mux"    # HUD pass + MP4 (render.sh defaults = v4 paths)
```

---

# Rocket launch v3 — RAVE cut of "Blast Off" (built on v2, Blender 5.2, EEVEE)

**v3 (10-11, Zeke's brief after watching v2):** `build_scene.py --rave` adds, on top of everything below:
liftoff on the first sung "blast off" (`--ign`, snapped to a bar, not a drop entry); neon LED rings and strips on
the core, boosters and upper stage, plus a white-hot LED nose, all music-driven after liftoff; flames that surge on EVERY
kick; the payload is a **giant LED speaker** (procedural: two proud woofers in LED rings, an LED edge frame, VU
meters) that climbs out of the fairing and grows 0.3 → 2.4 (38 m tall); an **asteroid** (NASA PD model of Bennu
+ an ambientCG CC0 rock scan, glowing Voronoi cracks) is pinned by neon sound-wave rings (one per kick, 720 m/s)
on the last drop and shatters on a bar (`--boom`): chunks, a JangaFX CC0 fireball, a flash and a blast ring,
while the Earth's surface and horizon jitter on every kick. Shots R*/S*/L*/M*/N* are the new ones.
Render ONLY through `scripts/server/render.sh` (capped slice). Env + args: `scratch/scenes/blastoff3_env.sh`.

---

# (v2) Rocket launch: "Hit a Bump" in photoreal 3-D (Blender 5.2, EEVEE)

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
