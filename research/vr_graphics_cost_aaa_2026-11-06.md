# VR graphics cost: what AAA PC VR does, what Godot 4.4–4.7 supports, and why head-shake costs more (2026-11-06)

Context: Godot 4.7.2, Vulkan, OpenXR via SteamVR, Jolt, RTX 3060 desktop. Measured: ~12 ms still, 23–24 ms
fast head-shake (17–18 ms with VRS_XR foveation MEDIUM). Already done: 2 cascades / 60 m, small objects cast no shadow,
city HLOD, visibility-range fades, occlusion culling. Numbers in brackets like [3] point to the Sources list at the end.
"(unsourced)" marks my own inference.

---

## 0. Most important findings

> **Correction (vrperf-1106, read from the project + the tower's log):** on the tower this game runs Godot's **Mobile** renderer (on D3D12)
> (`project.godot`: `renderer/rendering_method="mobile"`, MSAA 3D = 2×; Settings ▸ Renderer only picks the Windows driver,
> Vulkan or D3D12). So finding 1 below is already done; TAA/FSR2/SSAO/SSR don't exist on it. The `features` tag in
> project.godot still says "Forward Plus" (cosmetic). Also: `xr_render.gd` (xrfix-1104) reads the 4.7 source and finds
> `foveation_level` is NOT Compatibility-only on Vulkan when the runtime offers `XR_FB_foveation` (Meta's runtime); on
> SteamVR, the knob that acts is `Viewport.vrs_mode = VRS_XR` + `vrs_min_radius`/`vrs_strength`, which is what the
> game's Foveation setting drives.

1. **Godot's own docs say Forward+ is the wrong renderer for VR** (already followed here — see the correction above). "The current recommendation is to use the Mobile renderer
   for any desktop VR project … [Forward+] isn't well optimized for XR right now" [9]. The renderer table rates Forward+
   for XR as "Supported, but poorly optimized. Use Mobile or Compatibility instead", with the "highest base cost" [10].
   Mobile loses SSAO/SSR/SSIL/SDFGI/volumetric fog/**TAA/FSR2**, caps lights at 8 omni/spot per mesh, and uses RGB10A2 colour
   and 24-bit depth [10]. It keeps MSAA, glow, lightmaps and VRS. **An A/B test on the Mobile renderer is the biggest
   single lever we haven't pulled.**
2. **Valve's budget rule:** aim for 90 % of the frame, not 100 %. "My mental model of GPU budget changed from last year's
   11.11ms to now 10.0ms per frame" [2]. Valve scales **resolution + MSAA first**, then turns on radial density masking or
   foveation. It never toggles shadows or specular at runtime [2].
3. **Reprojection is a safety net, not a budget.** Vlachos: "Please DO NOT rely on reprojection to maintain framerate" [2].
   Meta: "Don't rely on Asynchronous SpaceWarp (ASW) to hit rendering frame rate" [4]. When an app misses, SteamVR Motion
   Smoothing **halves the app's rate (90 → 45)** and synthesises the in-between frames [5].
4. **The 23–24 ms reading may be partly a quantisation artifact.** OpenXR's `xrWaitFrame` blocks the app loop to the
   display's frame interval [7], so a wall-clock frame time snaps to multiples of the refresh period once the app
   misses its slot. At 90 Hz, 2 × 11.1 = 22.2 ms. **Check SteamVR's Advanced Frame Timing graph and Godot's measured GPU
   time before optimising (see §4.6).**
5. **Godot 4.7 does not stream textures** (that only arrives, opt-in, in 4.8 dev 5 [19]). So "texture streaming on head
   turn" is *not* the cause here. The remaining rotation-specific suspects are **first-sight pipeline compiles**
   (mitigated but not eliminated by 4.4 ubershaders [11][12]), **content variance across headings** (the city HLOD, shadow
   casters entering the cascades), and **VRAM oversubscription** (only on the 8 GB RTX 3060 variant).
6. FFR MEDIUM cut the shake case by ~6 ms. That means the shake cost is largely **pixel/fill-bound**, not CPU or draw-call
   bound (inference, consistent with Meta's "ALU-bound apps benefit most" [3]).

---

## 1. Frame budget per refresh

GPU target = ~90 % of the interval (Valve's 10 % idle rule [2]); 80 % is a safer working target (unsourced; it adds
headroom for the compositor and the Windows desktop, which "needs a slice of the GPU every few VR frames" [2]).

| Refresh | Interval | 90 % (Valve high threshold) | 80 % | 70 % (Valve "low": raise quality) | 2× interval (half-rate) |
|---|---|---|---|---|---|
| 72 Hz | 13.9 ms | 12.5 ms | 11.1 ms | 9.7 ms | 27.8 ms |
| 80 Hz | 12.5 ms | 11.3 ms | 10.0 ms | 8.8 ms | 25.0 ms |
| 90 Hz | 11.1 ms | 10.0 ms | 8.9 ms | 7.8 ms | 22.2 ms |
| 120 Hz | 8.3 ms | 7.5 ms | 6.7 ms | 5.8 ms | 16.7 ms |
| 144 Hz | 6.9 ms | 6.3 ms | 5.6 ms | 4.9 ms | 13.9 ms |

Valve's adaptive-quality thresholds at 90 Hz: high = 90 % (10.0 ms), low = 70 % (7.8 ms), prediction = 85 % (9.4 ms) [2].
**8 ms** is a 90 Hz / 70–80 % target, or 120 Hz at ~96 % (too tight at 120).

---

## 2. Techniques: savings and Godot 4.4–4.7 support

| Technique | Typical saving (sourced) | Godot 4.4–4.7 (OpenXR, Vulkan) |
|---|---|---|
| **Hidden-area / stencil mesh** | **17 % fewer pixels** (Vive) [1] | OpenXR `XR_KHR_visibility_mask` is implemented in Godot's OpenXR module [21] |
| **Fixed foveated rendering / VRS** | Meta: up to **25 %** on pixel-heavy apps; example scene 6.5 / 11.5 / 21 % at Low / Med / High [3]. Godot VRS on a desktop RTX 4090: +10.7 % (1080p) to +23.1 % (8K) [13]. NVIDIA MRS gave Valve **~5–10 %** [2] | **Yes** on Forward+ and Mobile: `Viewport.vrs_mode = VRS_XR` [13]. On desktop SteamVR the density map is generated by Godot from `vrs_min_radius` (default 20 %) and `vrs_strength` [14]. `foveation_level` / `foveation_dynamic` are "**Compatibility renderer only**" [8]. Vulkan can take the density map from `XR_FB_foveation_vulkan` (PR #99768, merged 2025-04, 4.5) [15], but that's a Meta/Quest runtime extension; whether SteamVR exposes it is (unsourced). Needs NVIDIA Turing+ (RTX 3060 OK) [13] |
| **Eye-tracked foveation** | n/a for most PC headsets | Meta eye-tracked foveation PR #112994 is post-4.5 and Quest-runtime only [15] |
| **Radial density masking** (Valve: skip 2×2 quads in a checker in the periphery, reconstruct) | **5–15 %** in Aperture Robot Repair; "almost always a big savings on low-end GPUs" [2] | Not built-in (unsourced). VRS_XR is the Godot equivalent |
| **Render scale / dynamic resolution** | Valve's ladder runs from 0.65× to 1.4× res, with MSAA 4×→8× [2]; Alyx auto-fidelity reportedly drops to ~65 % [22] | `OpenXRInterface.render_target_size_multiplier` (set **before** init) [14], and `Viewport.scaling_3d_scale` [16]. No built-in dynamic res for XR (unsourced) |
| **MSAA** | Valve: "4xMSAA minimum quality", 8× if perf allows; "forward renderers win … because MSAA just works" [1] | **Yes**, all renderers [10]. Godot: 2× or 4×, "8× MSAA is usually too demanding" [17]. Only one MSAA level is precompiled at a time [11] |
| **TAA** | Smears in VR: blur in still scenes that "increases when the camera is moving" + ghosting [17]. In an HMD the frame is *always* moving because of head-pose jitter [20] | Forward+ only [10]. With multiview, colour/depth/velocity are texture arrays [23]. Godot warns that many post effects "have not yet been updated to support stereoscopic rendering" [9] |
| **FSR1 / FSR2** | FSR2 is "more demanding than using TAA at native resolution" [17]; "slowest" scaling mode [16] | Forward+ only [10][16]. A user report shows FSR2 badly distorted in a headset (4.3) [24]. **Avoid in VR** |
| **Single-pass / multiview stereo** | Valve: instancing to double geometry "BETTER. Half the API calls" [1] | **Yes**, Godot 4 renders both eyes with multiview [25]. Multiview is a precompiled feature for XR projects [11] |
| **Shadow budgets** | Valve: shadows are *not* runtime-adjustable in adaptive quality; budget them statically [2]. Godot: an object spanning all 4 splits is "rendered five times in total" [18] | Cascade count/distance, shadow-map size (2048 instead of 4096 on low-end), avoid PCSS, dual-paraboloid omni shadows [18]. **Positional (omni/spot) shadows are cached** and re-rendered only "if any object affecting the shadow map has changed" [18]; directional cascades follow the camera (re-render each frame, unsourced) |
| **Baked lighting / lightmaps** | Source 2 (Alyx) bakes indirect light into lightmaps for static geometry and probe volumes for dynamic objects by default [26] | `LightmapGI` with Static bake mode, recommended when you need more lights than the renderer allows [18]. Supported in Forward+ and Mobile [10] |
| **LOD / HLOD / impostors** | Alyx's lower presets show "more aggressive LOD" [27] | Auto mesh LOD + visibility ranges/HLOD [28]. Fade modes force **alpha-blended transparent rendering** during transitions ("has a performance impact"); "Disabled" fade "provides the best performance" [28] |
| **Occlusion culling** | Biggest gains on Mobile renderer; Forward+ already has a depth prepass so gains are smaller [29] | Yes: Embree CPU raster of occluders into a low-res buffer [29]. Cost scales with occluder detail; tune *Occlusion Rays Per Thread* / BVH quality [29] |
| **Batching / instancing** | Meta PC guideline: **500–1,000 draw calls**, **1–2 M triangles** per frame [4] | MultiMesh, auto instancing of identical mesh+material in Forward+ (unsourced detail) |
| **Shader/pipeline warm-up** | — | 4.4 ubershaders + pipeline precompile (PR #90400) [12]; 4.5+ shader baker on export [11]. See §4.2 |
| **Reprojection / ASW / Motion Smoothing** | Safety net only [2][4]. Interleaved reprojection gives the app ~18 ms per frame [2] | Runtime-side. `OpenXRInterface.display_refresh_rate` + `get_available_display_refresh_rates()` let the app pick the refresh [14]. SteamVR per-app "Throttling Behavior" / `preferHalfFramerate` [6] |
| **Submit depth buffer** | Better reprojection quality | Project setting; "will disable stencil support"; leave off unless it helps [8] |

### Valve's adaptive-quality ladder (Aperture Robot Repair, Vive 1512×1680 per eye at 1.0) [2]

| Level | MSAA | Res scale | GPU that holds it |
|---|---|---|---|
| +6 | 8× | 1.4 | GTX 980 Ti (95 % of the time) |
| +2 | 8× | 1.0 | |
| 0 (default) | 4× | 1.0 | GTX 970 stays ≥ here |
| −2 | 4× | 0.81 | |
| −4 | 4× | 0.65 + radial density masking | GTX 680 stays ≥ here |

Vlachos later raised the floor to ~0.8 because "in-game text will be very difficult to read" at 0.65, and below that
he turns on the interleaved-reprojection hint [2]. Rules: **drop aggressively** (frame > 90 % → drop; extrapolated
> 90 % → drop 2 levels) and **raise conservatively** (3 frames < 70 %) [2].

---

## 3. What AAA PC VR titles do (and run at)

**Half-Life: Alyx (Source 2)**
- Min spec: **GTX 1060 / RX 580, 6 GB VRAM**, i5-7500 / Ryzen 5 1600, 12 GB RAM [30]. An RTX 3060 is well above that.
- Lighting: Source 2's default is baked (indirect light in lightmaps, light probe volumes for dynamic objects) [26].
  Source 2 games generally use forward rendering, "like Half-Life: Alyx" [26].
- Resolution: has automatic fidelity scaling. `vr_fidelity_level_auto 0` disables it; levels 1–8 with 3 reportedly
  native. These come from community testing, not Valve docs [22].
- Oct 26 2026 patch: optional desktop **foveated rendering** (eye tracking recommended, not required; needs GTX 16 /
  RTX 20+, AMD RX 6000+, Intel Arc) [31].
- ComputerBase (Rift S, 350 % SS = 3080×3320): RTX 2070 Super got 132 / 108 / 96 / 92 fps at Low / Med / High / Ultra.
  Medium → Ultra changes little visually; lower presets have "more aggressive LOD" and more flicker [27]. An RTX 3060 is
  roughly 2070 Super class (unsourced).
- Index refresh: 120/144 Hz modes make Alyx run at a lower resolution; PC Gamer's tester found 90 Hz "more than
  suitable" [32].

**Valve's own published practice (The Lab / Robot Repair, the Alyx engine team's precursor)** [1][2]: single-pass forward
renderer "because we want 4xMSAA and 8xMSAA", up to 18 dynamic shadowing lights, adaptive quality, hidden-area mesh,
8× anisotropic on colour+normal maps, "Running Start" (submit 2 ms before vsync: 1.5–2 ms saved).

**Asgard's Wrath 2 (UE4, Quest 2)**: 72 Hz with *dynamic* fixed foveated rendering in GPU-heavy areas; a later Quest 3 toggle
raises resolution and LOD distance [33]. No PC-VR postmortem with numbers was found for Asgard's Wrath 1.

**Boneworks / Bonelab, Lone Echo**: no primary rendering postmortem with numbers was found. Omitted rather than guessed.

**Typical 3060 settings**: no credible measured source was found for "3060 + Index/Quest in Alyx". Build guides suggest a
3060-class card for 90 fps at medium [34], which is secondary. A sane starting point for our game, extrapolated from
Valve's ladder (unsourced): **90 Hz, 1.0× render scale, MSAA 2–4×, VRS_XR on, adaptive scale 0.8–1.1**.

---

## 4. Why fast head rotation costs more than holding still

Sort the suspects into **rate-dependent** costs (they only happen while the view sweeps) and **content-dependent**
costs (the shake just visits more expensive headings).

### 4.1 Content variance across headings (most likely; test first)
The per-frame cost depends on what's in the frustum *that frame*. A left-right shake averages over every heading it
sweeps, so if one direction faces the city HLOD, the playground, or many shadow casters, the shake's average and peak
are higher than the still pose (inference). **Test:** hold still facing 8 headings and record the GPU time per heading.
If the worst heading ≈ the shake number, it's content, not motion.

### 4.2 First-sight pipeline compiles
- Before 4.4, Godot had no option "other than generating them when an object shows up inside the camera's view" [11];
  PR #90400 calls this "pretty much the entire root" of first-play stutter [12].
- Since 4.4: specialisation constants are read from push constants in an **ubershader**, so "a version of the shader
  [is] already … used for drawing immediately while the specialized version is generated in the background" [12].
  The docs say the *Specialization* category can't cause stutters "but may reduce framerate" when many compile per
  frame [11]. So a first sweep may run slower on the generic ubershader for a few frames (inference from [11][12]).
- The **Draw** category (an on-demand compile at draw time) still causes stutter and "should not occur" [11]. Features
  detected only at runtime (MSAA level, multiview, lightmaps, reflection probes, shadow modes, …) cause compiles if first
  toggled mid-game. Material overrides set at runtime aren't caught at mesh load [11].
- **Fix:** a warm-up scene that uses every feature for a frame; keep hidden instances of spawned effects; load meshes on
  threads; watch the pipeline-compilation monitors for **Surface/Draw spikes** during a shake [11].
- Same class of problem in UE (PSO precaching): it still causes "rare one-time hitches on first use" [35].
- Signature: a *one-time* spike that's gone on the second shake (inference). A *persistent* 2× cost is something else.

### 4.3 Texture streaming / VRAM
- Godot ≤ 4.7 has **no texture streaming**. All mips are resident at load, so turning the head triggers no texture
  uploads [19][36]. Mip streaming is new and opt-in in 4.8 dev 5 [19].
- VRAM oversubscription: the 3060 ships as 12 GB (192-bit) or a later **8 GB (128-bit)** variant [37]. When over
  budget, the WDDM video memory manager must page allocations in, and allocations "may succeed but take a long time"
  [38][39]. A shake that brings paged-out resources back into view would then cost (inference). **Check:** dedicated VRAM
  use vs capacity in Task Manager / `nvidia-smi` while shaking.

### 4.4 LOD swaps and visibility-range fades
Auto mesh LOD and visibility ranges are distance-based, so pure rotation shouldn't change LOD (inference). But
**fade transitions render alpha-blended transparent geometry** [28]: anything mid-fade inside the swept view costs extra.
With "Disabled" fade mode this goes away [28].

### 4.5 Shadows
- Directional cascades are fit to the view frustum, so rotating changes the caster set per split (unsourced, standard
  PSSM behaviour). A heading that faces many casters costs more (falls under 4.1).
- Omni/spot shadows are **cached** and only re-render when an affecting object changes [18], so they're unlikely to be
  rotation-sensitive. Moving or animated objects near those lights defeat the cache.

### 4.6 SteamVR / OpenXR behaviour when the frame is missed (and the 2× signature)
- `xrWaitFrame` "throttles the application frame loop" to the display. If a frame interval has passed it "may return
  immediately or block until the beginning of the next frame interval", and the runtime may change throttling based
  on submission/completion timing [7].
- SteamVR Motion Smoothing kicks in automatically on dropped frames and runs the app at **half rate** (90 → 45) with
  synthetic frames in between [5]. Interleaved reprojection likewise gives the app ~18 ms per frame [2]. Per-app
  **Throttling Behavior** and `preferHalfFramerate` / `preferMotionSmoothingMode` exist as per-app settings [6].
- **So yes: a wall-clock frame time that lands on ~2× the refresh interval (22.2 ms at 90 Hz, 25 ms at 80 Hz, 16.7 ms
  at 120 Hz) is the signature of a throttled / reprojecting app.** It doesn't mean the GPU took that long (inference from
  [5][7]). Note: 23–24 ms fits 90 Hz × 2 roughly, while the FFR figure of 17–18 ms fits 120 Hz × 2. **Find out the
  headset refresh first.**
- **How to tell:** SteamVR → Developer → *Advanced Frame Timing*. The GPU cyan line = vsyncs before the frame was ready
  (should be 1); **red = reprojection active**; brown "Other" = GPU bubbles or other apps; the CPU cyan line = throttling
  level [40] (Valve wiki, quoted via search; the page is bot-walled). In Godot, measure real GPU time with
  `RenderingServer.viewport_set_measure_render_time` + `viewport_get_measured_render_time_gpu` [41], not the frame
  delta.
- Rotation-only reprojection is eye-centred and narrows inter-camera distance during rotation; "a leisurely head turn is
  in the ballpark of 100 degrees/second" [2]. Shake artifacts are most visible exactly when the app is reprojecting.

### 4.7 CPU-side work
Godot culls on the CPU per frame. Occlusion culling cost grows with occluder detail [29]. Script `_process` work
doesn't depend on head speed (inference). A CPU-bound frame shows as a high CPU time with low GPU time in Advanced Frame
Timing [40]. Meta's PC guidance: keep script/logic to 1–3 ms [4].

### 4.8 Occlusion-culling lag
No source found that Godot's raster occlusion uses previous-frame data. (unsourced: it rasterises occluders for the
current camera each frame; no temporal lag expected.)

---

## 5. Suggested order of attack (generic, from the sources; inference — the game-specific ranked plan is §7)

1. **Measure correctly**: headset refresh, Advanced Frame Timing (red/reprojection?), Godot GPU time per heading.
2. **A/B the Mobile renderer** (Godot's own XR recommendation) [9][10].
3. Keep **VRS_XR on** and tune `vrs_strength` / `vrs_min_radius` [14]. Turn off **TAA / FSR2 / screen-space effects**,
   use MSAA 2× or 4× [17].
4. **Adaptive render scale** à la Valve (0.8–1.1, drop fast, raise slow, 90 % threshold) via `scaling_3d_scale` [2][16].
5. Visibility-range fade → **Disabled** for anything that's often mid-fade in view [28].
6. Pipeline monitors during a shake; add a warm-up scene [11].
7. Lightmaps for static lighting where shadows aren't gameplay [18][26].

---

## 6. Measured here (vrperf-1106, simulated headset on the server) — why the head-shake costs more

`vr.sh xr renderbudget` with the new **RB6 head shake**: yaw ±60° at 2.5 Hz, advanced 1/90 s per rendered frame (each
frame sees what a 90 Hz headset frame sees), 360 frames, at 4 vantages; plus a **still sweep** of the same headings
(held at −60…+60° in 15° steps, 40 frames each). Baseline = main + the probe only. Draws/prims carry over to the tower;
the V100's GPU ms don't (it barely feels draw count).

**1. Motion itself costs nothing extra.** Binned by heading, the shake's frames draw the same as standing still at that
heading (e.g. temple spawn +45°: still 1,154 vs shake 1,181 draws; +60°: 1,233 vs 1,239; table +30°: 952 vs 951), and the
V100 GPU ms match within noise. **Pipeline compiles during the shake: 0** at every vantage after the first cycle (one at
the spawn in the first cycle in one run, none in the other). VRAM 2.5 GB (textures 2.1 GB) — far under either 3060 variant.

**2. It's WHERE the shake points.** The draw count swings 2–2.6× across a ±60° shake:

| vantage (facing) | draws at −60…+60° (still, colour+shadow) | cheapest | worst | shake mean / worst frame |
|---|---|---|---|---|
| temple spawn (N) | 650 · 577 · 474 · 494 · 552 · 663 · 964 · 1,154 · 1,233 | 474 | **1,233** (+60°, toward the avatar room) | 797 / 1,251 |
| table (N) | 698 · 659 · 525 · 523 · 618 · 699 · 952 · 1,088 · 1,068 | 523 | **1,088** (+45°) | 784 / 1,116 |
| temple centre (S) | 862 · 499 · 370 · 665 · 636 · 606 · 627 · 808 · 834 | 370 | 862 | 685 / 900 |
| Lumen avenue | 170 · 232 · 253 · 246 · 220 · 224 · 230 · 166 · 124 | 124 | 253 | 203 / 262 |

perf-1105's ceilings and Zeke's "steady ~12 ms" were both taken at single headings; the shake visits the worst ones every
cycle and the frame-time graph shows those frames. Draws correlate with the render thread's CPU time (r = 0.58–0.87 per
vantage), not with the V100's GPU time (r ≈ 0.1).

**3. Who owns the worst heading** (RB_ATTR_SHAKE: hide each subtree, Δ draws; temple spawn turned +60°, total 654 colour +
570 shadow): the **AvatarRoom** (40 m west) 270 colour + 166 shadow — its display avatars (VRMs, 30–80 draws each) drawn
*through solid stone walls* (occlusion culling was off at the temple) and casting sun shadows under a roof; the world-level
loose meshes (columns etc.) 46 + 135; the shooting range 57 + 33; the water bar 42 + 23; destinations' teleport hub 47 + 14;
relics 37 + 19; the playground 0 + 44 (shadow only). At the table (+45°) the same room is 208 + 163.

**4b. Renderer mismatch (found 11-06):** the server sim renders **Forward+ on Vulkan**, while Zeke's tower runs the
**Mobile renderer on D3D12** (tower game log: `D3D12 12_0 - Forward Mobile - RTX 3060`; his renderer override picked D3D12).
Only draws/prims carry over; a D3D12-vs-Vulkan A/B on the tower is an open, possibly large lever.

**4. What this does NOT explain:** foveation MEDIUM cutting the shake 23 → 17 ms on the tower says those frames are also
pixel-heavy there; the V100 can't show that. And whether 23–24 ms is GPU time or a frame-interval reading (§4.6) needs the
tower's SteamVR Advanced Frame Timing.

## 7. Ranked plan for VR Iris (est. = estimate; only the tower can measure ms)

| # | change | where it helps | est. tower saving | look cost | effort | status |
|---|---|---|---|---|---|---|
| 1 | Avatar-room interior: walls/roof → occluders, occlusion on at the temple; displays cast no sun shadow | the worst shake headings (−400 draws there) | 1–3 ms on the bad headings (CPU render thread + vertex) | none (inset occluders; the doorway stays see-through) | small | **done** (vrperf-1106) |
| 2 | Chess: 64 squares → 1 mesh, each piece's ~5 parts → 1 mesh | table/south headings (−~180 colour + shadow draws) | 0.3–1 ms there | none (wood grain phase on the pieces may shift) | small | **done** |
| 3 | Labels end at their legible distance (250 × text height) | everywhere in the temple | small, many headings | none past legibility | small | **done** |
| 4 | Planar mapping on the yard, platform and step tops (4 texture fetches instead of 12) | every temple frame, most pixels | 0.5–2 ms (fill-bound frames) | none (same texels, same scale) | small | **done** |
| 5 | Same treatment for the bunk room / gallery displays and the shooting range / water bar / relics clusters (merge parts, MultiMesh) | remaining heading spikes | 0.5–1 ms | none | medium | next |
| 6 | Fixed foveation LOW/MEDIUM as the default with a smaller full-rate radius tuned on the headset | every frame | 3–6 ms measured (medium) | periphery shading rate — Zeke's eyes (his "circle" complaint) | small | Zeke's call |
| 7 | Valve-style adaptive render scale *at startup / between scenes* (Godot can't change the XR swapchain live, xr_render.gd) | every frame | ∝ pixels | resolution | medium | needs design |
| 8 | Bake the temple's static light (LightmapGI) and drop the sun's shadow to 1 split near the player | every frame | 1–2 ms | baked look must match day/night cycle — hard here (moving sun) | large | later |
| 9 | Grass: shorter range / fewer far tufts | yard views | 0.3–0.5 ms | far grass thins | small | Zeke's eyes |
| 10 | 90 Hz vs 72 Hz decision + Motion Smoothing policy | — | budget 13.9 → 11.1 ms | — | none | Zeke |

## Sources

1. Alex Vlachos (Valve), "Advanced VR Rendering", GDC 2015 — https://media.steampowered.com/apps/valve/2015/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
2. Alex Vlachos (Valve), "Advanced VR Rendering Performance", GDC 2016 — https://media.steampowered.com/apps/valve/2016/Alex_Vlachos_Advanced_VR_Rendering_Performance_GDC2016.pdf (GDC Vault listing: https://gdcvault.com/play/1023522/Advanced-VR-)
3. Meta, Fixed Foveated Rendering (Unity) — https://developers.meta.com/horizon/documentation/unity/os-fixed-foveated-rendering/
4. Meta, PC VR Performance Guidelines — https://developers.meta.com/horizon/documentation/native/pc/dg-performance-guidelines/
5. Road to VR, "SteamVR Gets Motion Smoothing" (Vlachos quotes) — https://roadtovr.com/steamvr-motion-smoothing-asw-alex-vlachos/
6. Steamworks, VR preferences (`preferHalfFramerate`, `preferMotionSmoothingMode`, `preferMinRefreshRate`) — https://partner.steamgames.com/doc/steamhardware/steamframe/vrpreferences
7. Khronos, OpenXR `xrWaitFrame` — https://registry.khronos.org/OpenXR/specs/1.0/man/html/xrWaitFrame.html
8. Godot docs, OpenXR settings — https://docs.godotengine.org/en/stable/tutorials/xr/openxr_settings.html
9. Godot docs (4.7), Setting up XR — https://docs.godotengine.org/en/4.7/tutorials/xr/setting_up_xr.html
10. Godot docs (4.7), Overview of renderers — https://docs.godotengine.org/en/4.7/tutorials/rendering/renderers.html
11. Godot docs (4.7), Reducing stutter from shader (pipeline) compilations — https://docs.godotengine.org/en/4.7/tutorials/performance/pipeline_compilations.html
12. Godot PR #90400, "Ubershaders and pipeline pre-compilation (and dedicated transfer queues)", merged 2024-10-03 — https://github.com/godotengine/godot/pull/90400
13. Godot docs, Variable rate shading — https://docs.godotengine.org/en/stable/tutorials/3d/variable_rate_shading.html
14. Godot docs, OpenXRInterface class — https://docs.godotengine.org/en/stable/classes/class_openxrinterface.html
15. Godot PR #99768, "OpenXR: Use the XR_FB_foveation_vulkan extension to get the density map for VRS" (merged 2025-04-19, mirror) — https://git.objectionable.solutions/Sara/behaviour-tree-test/commit/98aced4e65418d3e5b1161cf8bd7c6f405f60907 (GitHub: https://github.com/godotengine/godot/pull/99768); W4 Games on Vulkan foveation in 4.5 — https://www.w4games.com/blog/w4-games-news-1/where-every-millisecond-counts-optimizing-godot-for-standalone-xr-159
16. Godot docs, Resolution scaling — https://docs.godotengine.org/en/stable/tutorials/3d/resolution_scaling.html
17. Godot docs, 3D antialiasing — https://docs.godotengine.org/en/stable/tutorials/3d/3d_antialiasing.html
18. Godot docs, 3D lights and shadows — https://docs.godotengine.org/en/stable/tutorials/3d/lights_and_shadows.html
19. Godot blog, Dev snapshot 4.8 dev 5 (mip texture streaming, opt-in) — https://godotengine.org/article/dev-snapshot-godot-4-8-dev-5/
20. Kim et al., "Head movement based temporal antialiasing for VR HMDs", WSCG 2017 — https://dspace5.zcu.cz/handle/11025/29549?locale=en
21. Godot OpenXR visibility mask extension (source mirror) — https://remotebranch.eu/Stowage/godot/src/commit/edde380e9cb1545d77e99bba99aaf55f55e7dc86/modules/openxr/extensions/openxr_visibility_mask_extension.h
22. Steam Community, Alyx `vr_fidelity_level_auto` discussions (community-sourced) — https://steamcommunity.com/app/546560/discussions/0/3408678529102884400 ; https://steamcommunity.com/app/546560/discussions/0/592884013152077960
23. Godot `RenderSceneBuffersRD` class docs (multiview texture arrays, `get_use_taa`; mirror) — https://remotebranch.eu/Stowage/godot/src/commit/f392a9c4f81e6ea5f55c3fa7af0ca94de44ce463/doc/classes/RenderSceneBuffersRD.xml
24. Godot forum, "Unable to change Viewport Properties in XR" (FSR2 distortion in headset) — https://forum.godotengine.org/t/unable-to-change-viewport-properties-in-xr/90470
25. Godot blog, XR progress update June 2021 (multiview) — https://godotengine.org/article/godot-xr-progress-update-june-2021
26. Valve Developer Community, Source 2 Level Design: Lighting — https://developer.valvesoftware.com/wiki/Source_2/Docs/Level_Design/Lighting
27. ComputerBase, Half-Life: Alyx GPU benchmarks (2020-03) — https://www.computerbase.de/2020-03/half-life-alyx-benchmark-test/
28. Godot docs, Visibility ranges (HLOD) — https://docs.godotengine.org/en/stable/tutorials/3d/visibility_ranges.html
29. Godot docs, Occlusion culling — https://docs.godotengine.org/en/stable/tutorials/3d/occlusion_culling.html
30. Steam store, Half-Life: Alyx system requirements — https://store.steampowered.com/app/546560/HalfLife_Alyx/
31. Mixed News, "Half-Life: Alyx adds foveated rendering on PC" (2026-10-28) — https://mixed-news.com/en/?p=38532
32. PC Gamer, best GPU/headset for Half-Life: Alyx — https://pcgamer.com/half-life-alyx-best-gpu-best-headset
33. UploadVR, Asgard's Wrath 2 review (72 Hz, dynamic FFR) — https://uploadvr.com/asgards-wrath-2-review/ ; Mixed News, Quest 3 enhanced graphics — https://mixed-news.com/en/asgards-wrath-2-quest-3-enhanced-graphics-update-screenshot-comparison/
34. Build-gaming-computers, Half-Life: Alyx PC build (secondary) — https://www.build-gaming-computers.com/half-life-alyx-pc-build.html
35. Epic, "Game engines and shader stuttering: Unreal Engine's solution" — https://www.unrealengine.com/en-US/tech-blog/game-engines-and-shader-stuttering-unreal-engines-solution-to-the-problem ; PSO precaching docs — https://dev.epicgames.com/documentation/unreal-engine/pso-precaching-for-unreal-engine
36. godot-proposals #3177, texture streaming (deferred) — https://github.com/godotengine/godot-proposals/issues/3177 ; Godot, "What's missing in Godot for AAA" — https://godotengine.org/article/whats-missing-in-godot-for-aaa/
37. Guru3D, RTX 3060 8 GB / 128-bit variant — https://www.guru3d.com/news-story/geforce-rtx-3060-with-8gb-128-bit-memory-bus-memory-released.html
38. Microsoft, Process residency budgets (WDDM) — https://learn.microsoft.com/en-us/windows-hardware/drivers/display/process-residency-budgets
39. NVIDIA, Advanced API Performance: Memory and Resources — https://developer.nvidia.com/blog/advanced-api-performance-memory-and-resources
40. Valve Developer Community, SteamVR/Frame Timing — https://developer.valvesoftware.com/wiki/SteamVR/Frame_Timing
41. Godot docs, RenderingServer (`viewport_set_measure_render_time`, `viewport_get_measured_render_time_gpu`) — https://docs.godotengine.org/en/stable/classes/class_renderingserver.html
