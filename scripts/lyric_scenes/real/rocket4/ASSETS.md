# Rocket v2: third-party assets (URL + licence)

Licence gate (Zeke 2026-10-10): **CC0, public domain or CC-BY (credited) only.** No NC, no ND,
nothing unlicensed. Everything below was downloaded on 2026-10-10 into its own directory under
`/data/iris_assets/` and is treated as untrusted data. Nothing from these downloads is executed:

- the NASA `.blend` is opened with `--disable-autoexec` and `use_scripts=False`;
- `prep_assets.py` asserts that the `.blend` has no text blocks, then takes only meshes;
- zips and 7z archives were listed first and checked for absolute or `..` paths, and only data
  files were extracted (`.vdb .png .txt .mp4 .jpg`). The `.ember` project files were skipped.

`prep_assets.py` builds `/data/iris_assets/rocket2/rocket2_lib.blend` (and `SRB_ALL.png`) from the
downloads. `build_scene.py` reads that library plus the VDB, texture and Earth folders.

## Used in the render

| Asset | Local path (under `/data/iris_assets/`) | Source URL | Licence / credit |
|---|---|---|---|
| **SLS Block 1, GNC-markings model** (RS-25 engines, core-stage plumbing, booster attach hardware, LVSA, ICPS, stage adapter, ESM/SAJ panels, ogive, launch abort tower) | `nasa3d/sls_block1_x/Block 1 (GNC markings).blend` from `nasa3d/Space Launch System Block 1/Block 1 (GNC markings).7z` | https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Space%20Launch%20System%20Block%201 (also https://science.nasa.gov/3d-resources/space-launch-system-block-1/) | NASA 3D Resources: *"free and without copyright"* (repo README). NASA media guidelines apply: no NASA insignia or logos, no implied endorsement. **The model's own decal textures (NASA, ESA and flag logos ship in the scanline variant) are NOT used.** Every surface gets our own material. |
| **Space Shuttle Solid Rocket Booster** (booster body, stretched from 4 to 5 segments) + its texture `SRB_ALL` (luminance only) | `nasa3d/Space Shuttle Parts/Solid Rocket Booster.glb` | https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/Space%20Shuttle%20Parts | NASA 3D Resources, as above. The texture has no logos (checked). |
| **Mobile Launcher (assembled)** (launch tower and deck) | `nasa3d/Mobile Launcher/Mobile Launcher (assembled).glb` | https://science.nasa.gov/3d-resources/mobile-launcher/ | NASA 3D Resources, as above |
| **Orion Capsule (no FBC)**: crew module, revealed at abort-tower jettison | `nasa3d/orion_capsule/orion_no_fbc.stl` | https://science.nasa.gov/3d-resources/orion-capsule/ | NASA 3D Resources, as above |
| **JangaFX Ground Explosion VDB**: launch steam (trench blast, pad clouds, first steam at engine start), density + flames grids | `jangafx/x_GroundExplosionVDB/` | https://jangafx.com/software/embergen/download/free-vdb-animations (Mediafire link on that page) | **CC0** (`LICENSE.txt` in the zip: "Creative Commons - Public Domain - CC0") |
| **JangaFX Industrial Chimney Smoke VDB**: LOX/LH2 boil-off vapour pouring down the vehicle (flipped upside down) | `jangafx/x_IndustrialChimneySmokeVDB/` | same page | **CC0** |
| **JangaFX Cloud Pack VDB** (10 cumulus): the cloud deck on the climb | `jangafx/x_CloudPackVDB/` | same page | **CC0** |
| **NASA Blue Marble Next Generation**, July 2004, topography + bathymetry, 5400×2700 | `nasa_earth/world.topo.bathy.200407.3x5400x2700.jpg` | https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg | NASA Earth Observatory / NASA GSFC: NASA imagery, generally not copyrighted. Credit NASA. |
| **NASA Blue Marble cloud layer** (`cloud_combined_8192.tif`) | `nasa_earth/cloud_combined_8192.tif` | https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_8192.tif | NASA Earth Observatory, as above |
| **NASA Black Marble 2016** (night lights, 3 km) | `nasa_earth/BlackMarble_2016_3km.jpg` (an 8192 px copy in `nasa_earth/derived/`) | https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_3km.jpg | NASA Earth Observatory, as above |
| Poly Haven **concrete_floor_02** (pad + trench concrete, 2k) | `polyhaven/tex_concrete_floor_02/` | https://polyhaven.com/a/concrete_floor_02 | **CC0** (https://polyhaven.com/license) |
| Poly Haven **rusty_painted_metal** (Mobile Launcher steel roughness/normal/grime, 2k) | `polyhaven/tex_rusty_painted_metal/` | https://polyhaven.com/a/rusty_painted_metal | **CC0** |
| Poly Haven **coast_sand_01** (beaches, 2k) | `polyhaven/tex_coast_sand_01/` | https://polyhaven.com/a/coast_sand_01 | **CC0** |
| Poly Haven **shrub_01 / shrub_02 / shrub_04** (scrub scatter around the pad) | `polyhaven/mdl_shrub_0*/` | https://polyhaven.com/a/shrub_01, …/shrub_02, …/shrub_04 | **CC0** |
| ambientCG **PaintedMetal017** (vehicle paint/metal roughness + normal + wear luminance, 2K) | `ambientcg/PaintedMetal017/` | https://ambientcg.com/view?id=PaintedMetal017 | **CC0** (https://docs.ambientcg.com/license/) |
| Tzeke000 emblem (on the core stage and the countdown clock) | `state/tzeke_songs/_assets/tzeke000_symbol.png` | Zeke's own | Zeke's |
| Bebas Neue font (clock labels, from v1) | `assets/fonts/BebasNeue-Regular.ttf` | repo asset | SIL OFL (unchanged from v1) |

The only procedural geometry left: the plain core-stage tank (a lathe; NASA ships the tank only
inside a 3ds Max `.max`, which Blender can't read), the ESM body under the panels, the flame
deflector and trench walls, lightning towers + catenary wires, sphere tanks, water tower, flood
masts, the terrain heightfield, the plumes and the countdown clock.

## Downloaded but NOT used in the render (kept for later; same licences)

- NASA: `Saturn V.glb` (rejected: it carries a US flag and lettering), `External tank.glb`,
  `Crawler.glb`, `Block 1 (GNC markings scanline).7z.001/.002` (contents are `.max` and `.dwf`
  only, which Blender can't import; its textures include NASA, ESA and flag logos).
- Poly Haven CC0: `asphalt_02`, `metal_grate_rusty` (textures); `grass_bermuda_01`,
  `modular_chainlink_fence`, `modular_electricity_poles`, `modular_industrial_pipes_01` (models).
- ambientCG CC0: `PaintedMetal009`, `Concrete034`, `Metal029`.
- NASA `world.200407.3x5400x2700.jpg` and `world.topo.bathy.200412…` (alternate Blue Marble months),
  `cloud_combined_2048.jpg`.

## Looked at online, not downloadable here

- **Sketchfab** (CC-BY, downloadable): "SpaceX Falcon Heavy V4" by clarence365
  (uid a91bccdf8e624997892bab2f0fbec28f, 274k faces), "Falcon Heavy" by adam.hnevkovsky
  (1738b8ce…), "LT SpaceX Falcon Heavy" by dunealex, "Falcon Launch-pad" by ajd6735
  (5cb616f0…), "Merlin 1D" by KUBAHA (e5caa613…). Sketchfab's download API needs an account
  token, and none is configured on this machine. With a token, any of these could be pulled
  (credit the author in this file; CC-BY).
- The **Sketchfab CC0 filter** returned nothing relevant for rocket / launch pad / launch tower.
- **BlenderKit** needs its add-on plus an account (not set up).
- **NASA 3D Resources** has no separate Orion service module, ICPS tank or core-stage tank as an
  importable format, and no flame trench or lightning tower.

## v3 additions (10-11)
- **Bennu (1999 RQ36) asteroid** — NASA 3D Resources, public domain: https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Models/1999%20RQ36%20asteroid → `/data/iris_assets/nasa3d/bennu/bennu.glb`
- **Rock030** — ambientCG, CC0: https://ambientcg.com/view?id=Rock030 → `/data/iris_assets/ambientcg/Rock030/`
- The speaker is procedural (no third-party model).

## v4 additions (10-13) — credits for the video description (CC-BY needs them)
| Asset | Local path | Source | Licence / credit line |
|---|---|---|---|
| **"Stage Speaker - black"** by **Sousinho** (the speaker cabinet, 4K PBR; its static woofer cones are cut out and replaced by our animated ones) | `/data/iris_assets/blastoff/sf_stage_speaker_gltf/` | https://sketchfab.com/3d-models/stage-speaker-black-f3209a6a45b844df92560099f982a508 | **CC BY 4.0** — credit: *"Stage Speaker - black" by Sousinho (Sketchfab), CC BY 4.0* |
| **"Moving Head Beam (high Poly)"** by **Lord Henry** (the four moving heads on the speaker's lid) | `/data/iris_assets/blastoff/sf_moving_head/model.glb` | https://sketchfab.com/3d-models/moving-head-beam-high-poly-ce283a1277bb4e98bd29b157075e003f | **CC BY 4.0** — credit: *"Moving Head Beam (high Poly)" by Lord Henry (Sketchfab), CC BY 4.0* |
| **JangaFX Mid-Air Explosion** VDB (booster 3's explosion; density / flames / temperature, 126 frames; a padded symlink sequence is made at build time in `.../midair_explosion/seq/`) | `/data/iris_assets/jangafx/midair_explosion/ex/` | https://jangafx.com/software/embergen/download/free-vdb-animations | **CC0** (`LICENSE.txt` in the archive) |
| **B612 / B612 Mono** (the HUD font; Airbus cockpit-display typeface) | `/data/iris_assets/fonts/b612/` | fetched 10-13 with the tumble research (see `docs/research/blastoff_v4_tumble_2026-10-13.md`) | **SIL OFL 1.1** (`OFL.txt` alongside) |

Direction checks (blender.md rule): the speaker's front is **+Y** in the asset (a +Y marker render from +Y shows the
woofers + horn, from -Y the plain back; numerically the woofer-cone islands sit at max Y) → turned 180° to face local -Y.
The moving head's lens points **+Z** at rest (top-view marker render). Both renders: `/tmp` scratch, re-runnable from
`build_scene.py`'s import code.

Tools used at BUILD time only (not assets, nothing of theirs is in the picture): **Cell Fracture 0.2.1** (Blender
extension, GPL-3.0-or-later, loaded per run from `/data/tmp_dl/cellfrac/x`) and Blender's built-in Bullet rigid bodies.
The per-element music channels (`/data/iris_assets/blastoff/stems/blastoff_elements_30fps.npz`) were analysed from
Demucs `htdemucs_ft` stems of Zeke's own song (analysis only; the weights' licence is unstated, flagged to Zeke 10-12).
