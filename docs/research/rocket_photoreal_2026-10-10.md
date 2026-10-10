# Rocket launch scene: photorealism research brief (2026-10-10)

Scope: research, references and skills only. No scene code was changed. Subject: `scripts/lyric_scenes/real/rocket/`
(EEVEE 5.2, 3570 frames at 1080p, budget about 5–8 s/frame on a shared V100). There is a short arcade section at the end.
Review note on the stills: the rocket "might need some help with photo realism", specifically the rocket, the smoke and
flames, and the world build.

Companion material:
- **Skills** (local, `.claude/skills/`): `blender-fire-smoke-realism`, `blender-photoreal-lookdev`,
  `blender-terrain-and-cityscapes`. Every API detail in them was run on Blender 5.2.1 headless (tests in
  `scratch/skill_tests/<skill>/`).
- **Reference photos** (local only, not in the repo): `scratch/refs/rocket/` (90) and `scratch/refs/arcade/` (25),
  contact sheets in `scratch/refs/sheets/`, and source URL + licence per file in `scratch/refs/SOURCES.md`.
  They're mostly NASA (generally public domain) and Wikimedia Commons PD/CC0/CC-BY(-SA); the arcade set adds Openverse CC-BY/BY-SA Flickr photos.
- Technique notes with a URL per claim: `scratch/refs/research_notes_techniques.md`.

---

## 1. Critique: what reads as CG (against real launch footage)

Compared with the NASA/Commons references (night and dawn launches from LC-39A/B and SLC-40, Stennis RS-25 tests,
Shuttle/Saturn/Falcon/Delta IV liftoffs, ISS views of the limb):

**Smoke and steam (the biggest tell)**
1. The ignition and liftoff steam is made of **opaque, evenly lit spheres** (`f1060`, `f1100`), with hard silhouettes,
   no wisps and no internal shadowing. It reads as cotton wool or popcorn. Real deluge steam is a **volumetric, self-shadowing,
   cauliflower cloud hundreds of metres across**. It blasts **sideways out of the flame trench**, its core is lit orange from
   inside by the plume, the far side is dim grey-blue, and the edges are wispy. Real clouds also dwarf the vehicle within
   seconds (refs `b02`, `b03`, `c07`).
2. Near the flame, the steam **glows uniformly** like a lamp (`f1003` crop). Real steam is lit *by* the flame: brightest
   where the plume enters it, with a falloff, and the near side is often in silhouette against the glow.
3. Under the pad there's a **striped/zig-zag column** (`f1060`). This is froxel/slice aliasing of the plume volume.

**Plume**
4. Three identical, uniform white "tubes" with hard edges and no expansion (`f1181`, `f1371`). An RP-1/LOX plume at sea level
   is a blinding yellow-white core with a turbulent orange edge, flicker and a sooty trailing edge, and it lights everything
   around it. At night the camera **blows out** (refs `b01`, `b11`).
5. In vacuum (`f3200`, `f3300`) the plume is a **solid white cone**. A real upper-stage plume at altitude is enormous,
   translucent and soft. At twilight it is the sunlit "jellyfish" bell (refs `c04` 1–2, 12).
6. The Max-Q vapour effect is two **thin white rings** (`f1795`). A real vapour cone is a conical shroud of
   condensation hugging the vehicle and streaming aft (ref `c01`).
7. The ice "chips" and hi-hat glints read as **fairy-dust sparkles** (`f1795`, `f1951`, `f2900`). Real ice shed at liftoff
   is white flakes and sheets that tumble and fall, not points of light.

**Vehicle and pad**
8. Vehicle surfaces are uniform satin with no breakup: no panel seams, stringers, soot streaks or
   **frost bands on the LOX sections**. Under warm light the body turns uniform tan-orange and looks like a toy (`f2200`, `f2790`).
   Separated boosters are plain white sticks (`f2330`).
9. The pad is **blank blocks on a flat plane**: no flame trench, deluge pipes, water tower detail or cabling.
   The lightning-mast poles cut straight through the foreground (`f0988`, `f1003`) as flat black bars with no
   depth cue, which looks like a compositing error.
10. **Lighting is flat.** Before launch the rocket is evenly lit (`f0901`) instead of hot xenon searchlight pools against a dark
   sky (ref `b07`). At ignition the tower and rocket barely pick up the flame's light.

**World**
11. An **endless flat ground and a featureless ruler-straight horizon**, with no haze between camera and subject. Real pads sit on
   scrubby flat coast with water, and launch photos carry heavy atmospheric depth.
12. The cloud deck (`f1501`, `f2051`) is uniform soft noise with no cumulus structure, scale or shadowing.
13. The limb shots (`f3080`+) are the strongest part. The surface below still lacks real texture (land, clouds and night
   lights at the right scale).

**Camera**
14. Mostly **18–55 mm lenses a few tens of metres from the vehicle**. Real launch imagery is
   (a) **long telephoto from kilometres away** (tracking mounts use 20–150 inch, about 500–3800 mm, lenses), giving
   compression, heat shimmer and haze, with slight tracking lag, and (b) **wide remote cameras at ground level** with steam
   rushing at the lens. Our ascent "chase cam" angles have no real-world equivalent. That's fine for a music video, but
   they need onboard-camera cues (fisheye, frost on the lens, exposure pumping) to read as footage.
15. Everything is pin-sharp and clean: no grain, no heat haze, and highlights never clip the way real sensors do.

## 2. Fix list ranked by visual impact per render cost

| # | Fix | Impact | Cost / how | Skill |
|---|---|---|---|---|
| 1 | **Steam → real volumes.** Keep the GN billow meshes as *shapes*; render them as **pyroclastic volumes** (Mesh to SDF Grid → Volume Cube density = noise − sdf). Or use CC0 **JangaFX VDBs** (Dust Shockwave / Ground Explosion for the trench cloud, High-Res Smoke Plume for the column), placed and time-scaled. | ★★★★★ | Volume Cube at 96–160³ per cluster. For crisp close steam, render a **separate Cycles volume layer at 50 % resolution** (it's soft, so upscaling is invisible) and composite it over the EEVEE beauty pass. Distant clouds stay EEVEE with a tight `volumetric_start/end` and `tile_size` 2–4. Cycles needs `volume_bounces` 4–8 or steam turns grey. | fire-smoke §4, §7 |
| 2 | **Light the scene with the plume.** A large blackbody area/point light (1800–2200 K) at the engines, intensity ramped with thrust. Illuminate the steam from the plume side via volume emission falloff from the plume axis. Use an exposure ride so ignition **clips**. Add xenon searchlight hot pools before launch. | ★★★★★ | Nearly free (lights + keyed `view_settings.exposure`). | lookdev §1, §6 |
| 3 | **Plume shading.** Sea level: overexposed yellow-white core, turbulent orange edge (4D noise flicker), soot fringe, subtle Mach diamonds (first disk ≈ 5–7 m below a 1 m nozzle). Altitude: progressively wider, translucent, low emission; a twilight sunlit "jellyfish" bell near staging. Vacuum: huge, faint, soft. | ★★★★ | Cheap: shader work on the existing shell plus volume. | fire-smoke §8 |
| 4 | **Camera language.** Re-block key shots as (a) 400–1200 mm tracking shots from 1–5 km with slight lag, haze and **heat shimmer** above the pad (compositor Noise → Displace, masked), and (b) low wide remotes with steam rushing in. Add grain and slight dispersion to everything. Fake onboard cams for the ascent (fisheye, frost, vibration). | ★★★★ | Free to cheap. | lookdev §7, §8 |
| 5 | **Pad and world build.** Flame trench + deflector, deluge water sheeting **before** ignition, a real water tower, cabling, a fence line; GN terrain with low relief, scrub scatter and a coastline with water that **reflects the plume** (refs `b01`, `c07` show this); a distant town glow. Fix the foreground masts: give them lattice detail or move them out of the frame centre. | ★★★ | Medium (geometry); instances are cheap. | terrain §3–§5 |
| 6 | **Vehicle look-dev.** Dielectric white paint (albedo ~0.75) with roughness variation from a CC0 scan; panel seams; soot streaks on the booster bases; **LOX frost bands** that shed after liftoff; carbon interstage; bare-metal nozzles; LOX vent vapour **pouring down** the vehicle before launch. | ★★★ | Cheap (shaders + textures at 1–2k). | lookdev §2–§4 |
| 7 | **Replace sparkles with ice.** Snare/hi-hat → falling white flakes and sheets with motion blur, plus beacon/strobe flicker on the tower. No point glints. | ★★ | Cheap (same GN, different instance and material). | — |
| 8 | **Max-Q cone** as a conical condensation shroud (volume or a soft alpha-faded shell around the vehicle) that streams aft and flickers on kicks. | ★★ | Cheap. | fire-smoke §8 |
| 9 | **Orbit world**: NASA Blue Marble/Black Marble on the ray-sphere Earth, a cloud layer with parallax and shadow, a thin atmosphere rim. The 5.x Multiple Scattering sky at 100 km already draws the lit limb. | ★★ | Cheap (textures). | terrain §2, §6 |
| 10 | **Cloud deck**: replace the soft slab with clustered cumulus volumes (Disney cloud CC-BY-SA or JangaFX Cloud Pack CC0 instanced), lit by the low sun. | ★★ | Medium (VRAM). | fire-smoke §6 |

Budget note: EEVEE shots measure about 2.6–4.7 s/frame now. Items 2, 3, 4, 6, 7 and 8 add almost nothing. Item 1 is the only one
that can blow the budget. Restrict Cycles to the steam-heavy drop-1 shots (about 600 frames), render that layer at half resolution,
and keep everything else in EEVEE. Measured on the same pyroclastic test volume at 960×540 on the shared V100
(`scratch/skill_tests/blender-fire-smoke-realism/t3_eevee_vs_cycles_gpu.py`):

| Renderer and settings | Time per frame | Look |
|---|---|---|
| EEVEE, tile 8 | 0.65 s | soft and blurry |
| EEVEE, tile 2 + custom range | 0.83 s | most of the detail back |
| Cycles OptiX, 64 spp, 8 bounces | 3.8 s | crispest, best self-shadowing |

All three already read as smoke rather than cotton balls, so **the volume conversion matters more than the engine**.
Try EEVEE with tile 2 and a tight range first. Add a half-resolution Cycles steam layer (~4 s/frame) only for the
close ignition shots.

## 3. Recommended assets (licences checked 2026-10-10)

| Asset | Use | Licence |
|---|---|---|
| JangaFX free VDB packs (Dust Shockwave, Ground Explosion, High-Res Smoke Plume, Fire Plume, Cloud Pack) — jangafx.com/software/embergen/download/free-vdb-animations | trench steam, plume smoke, clouds | **CC0** |
| Poly Haven HDRIs `syferfontein_0d_clear_puresky`, `aarfontein_dawn_2`, `umhlanga_sunrise`, `industrial_sunset_puresky` | reflections and dawn lighting reference | **CC0** |
| Poly Haven textures `concrete_floor_02`, `hangar_concrete_floor`, `metal_grate_rusty`, `rusty_painted_metal`, `coast_sand_01`, `asphalt_02` | pad, trench, apron, ground | **CC0** |
| ambientCG `PaintedMetal001–017`, `Concrete030–048`, `Metal027–029` | vehicle paint roughness/normal, steel | **CC0** |
| NASA Blue Marble Next Generation / Black Marble 2016 | Earth day/night from altitude | NASA, generally not copyrighted (credit NASA GSFC; no NASA insignia, no implied endorsement) |
| Disney cloud data set | hero cumulus | CC-BY-SA 3.0 (credit + share-alike) |
| ~~OpenVDB.org sample VDBs~~ | — | no licence stated for the files, so avoid |

Downloader for the CC0 libraries (md5-checked, safe zip extraction):
`scratch/skill_tests/blender-photoreal-lookdev/fetch_cc0.py`. Large VDBs belong on `/data`.

## 4. Real-world facts to match (short)

- RP-1/LOX: yellow-orange opaque sooty flame, **green TEA-TEB flash** at Merlin ignition. LH2/LOX: almost invisible, faint blue,
  diamonds visible. CH4/LOX: translucent blue-violet. Solid boosters: blinding white flame + dense white smoke.
- Deluge steam is white **wet steam** (Shuttle: water from T−6.6 s, 300k gallons used in 41 s), billowing sideways from the trench.
- Vapour cone ≈ transonic, about 25–35 s after liftoff, before Max-Q.
- Frost only on uninsulated LOX sections. LOX vent vapour flows downward.
- Tracking lenses 500–3800 mm. Liftoff stills at 1/500–1/8000 s (little blur); video at 180° shutter.

Sources are listed per claim in `scratch/refs/research_notes_techniques.md` (Wikipedia Shock diamond / Sound suppression
system / Space jellyfish, L3Harris RS-25 brochure, NASA NTRS 19960016118, NASA KSC imagery PDFs, Blender 5.0–5.2 release notes
and manual).

## 5. Arcade: what would push it from "close" to "real"

The arcade is already convincing in the hero shots: DOF, worn control-panel scratches, CRT curvature and reflections.
Against the arcade references (`scratch/refs/arcade/`: Japanese game centres, US retro arcades, CRT shadow-mask and
aperture-grille macros, a bowling-alley carpet, neon), the remaining tells are:

1. **Every cabinet is the same model.** Real rows mix silhouettes, heights, widths, side-art styles, ages and brands, with a
   sit-down driver, a claw machine or a change machine breaking the rhythm. Make 3–4 cabinet variants and shuffle them.
2. **Too clean and too uniform.** Add sticker residue, a faded marquee, a dusty bezel, scuffed kick plates, coin-door price
   stickers, burn marks and finger grease on the glass (roughness map), and uneven T-molding glow (one dead section).
3. **CRT image too crisp and saturated at distance.** Add phosphor bloom/halation around bright pixels, slight RGB
   convergence error at the edges, a visible shadow-mask or aperture-grille pattern in the close-ups (macro refs `c06`,
   `o08`), raised blacks (an unlit CRT is dark grey, not black), and a faint reflection of the room on the curved glass.
4. **The room is under-dressed.** Add a ceiling (dark drop tiles or exposed ducts), cables, a trash can, stools, posters,
   floor cable covers, and mixed colour temperatures (cool fluorescent spill from a corner, warm exit sign).
5. **Camera.** A low-light handheld feel: a touch more grain in the shadows, slight exposure breathing, occasional focus
   hunting, and halation on neon. The blacklight carpet glow should be subtle (UV-reactive specks, not a lit floor).

## 6. Process notes

- All GPU tests ran under `flock /tmp/iris_gpu_render.lock nice -n 15`. Everything else was CPU at 320–480 px.
- Reference photos are for looking at, not for the repo. `scratch/` and `.claude/` are git-ignored.
