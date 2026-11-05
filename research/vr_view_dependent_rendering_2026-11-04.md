# Zeke's FOV-ring render design: what it's called, where it's explained, what Godot 4.7 already has (2026-11-04)

Helper research for VR Iris / LUMEN (Godot 4.7.2 Mobile renderer, OpenXR, PC VR at 72 Hz). This **adds to**
`research/vr_render_lod_streaming_2026-11-04.md` (the 8 ms plan, mirror, Lumen HLOD, NYC walls). It doesn't repeat that doc.
Nothing in the game was changed. Claims marked *(unverified)* weren't checked against a primary source.

## Summary (5 lines)

1. **Zeke's design already exists, as several established pieces:** *frustum culling with a guard band* (the margin),
   *screen-space-error / view-dependent LOD* (lower detail at the edges and far away), *HLOD + impostors* (the placeholders),
   *visibility ranges with hysteresis + dither cross-fade* (the "seamless" swap), and on the network side *interest management /
   relevancy* (each client only gets what's near it; the server simulates only where players are).
2. **The one part that's new is cutting quality by *angle from the view centre inside* the FOV.** Engines don't do that with LOD,
   because a head turns at 300–600 °/s, faster than any load. The angular piece that does exist is **foveated rendering**, and it only
   lowers the *shading rate* (pixels), never lights, textures or geometry. **Zeke's rule ("never lower quality inside the real FOV")
   is exactly the rule fixed foveation breaks.**
3. **Probable cause of "the circle" Zeke saw:** Godot's XR foveation. We turn on `VRS_XR` (`xr_session.gd:14`) and never set
   `vrs_min_radius` (default **20 %**) or `vrs_strength` (default **1.0**). Both properties exist on `OpenXRInterface` in 4.7.2
   (checked with `--doctool`), so only a small central disc gets full shading. `openxr/foveation_level=3` (HIGH) may add runtime
   foveation on top. **First fix: foveation off or a wide radius; measure; then decide.**
4. **Godot 4.7 already has almost all the pieces:** frustum culling from both eyes' real projections, `extra_cull_margin`,
   automatic mesh LOD (screen-space error), visibility ranges (HLOD) with margins + fade, occlusion culling, VRS foveation, the OpenXR
   visibility mask, and `MultiplayerSynchronizer` visibility filters for interest management. It **lacks**: built-in streaming /
   world partition, impostor baking, Nanite-style virtualized geometry, and any "eccentricity LOD".
5. **Recommended mapping:** the FOV + margin → engine frustum + a small guard band (`extra_cull_margin`, ~5° Valve margin for our own
   script culls); "lower quality at the edge" → **don't**; use distance/screen-size LOD + HLOD instead. "Placeholders outside / behind" →
   visibility-range HLOD that is distance-based but view-independent, so turning never pops. Multiplayer → per-peer visibility
   filters on the server + server-side simulation sleep/disable for empty areas (the destinations pattern we already use).

---

## 1. Zeke's sentence, piece by piece

| Zeke's words | established name(s) | does it exist in engines? | Godot 4.7 |
|---|---|---|---|
| "inside the headset's real FOV render at full quality" | **view-frustum culling** with the runtime's per-eye projection; **"never degrade the fovea"** | yes, everywhere | yes: culling uses a frustum combined from both eyes' runtime projections (see the 8 ms doc §2.2) |
| "a margin 10–15 % wider at lower quality, +5 % lower still" | **guard band / over-scan / cull margin** (wider than the view, so motion between cull and display doesn't show holes); the *lower-quality* part = **view-dependent LOD** / **eccentricity-based LOD** / **foveated LOD** (research, not shipping engines) | guard band: yes. Lower quality by angle: only as **foveated rendering** (shading rate), and in research papers for geometry | `GeometryInstance3D.extra_cull_margin`; VRS for shading; no angle-LOD |
| "outside = placeholders swapped in seamlessly" | **HLOD** (hierarchical LOD: merged proxy meshes), **impostors** (billboards / octahedral impostors), **cross-fade / dithered LOD transitions**, **hysteresis** | yes (Unreal HLOD, Unity LOD Group cross-fade, SpeedTree, Horizon/Spider-Man proxies) | visibility ranges (`visibility_range_begin/end` + margins + fade mode) = Godot's HLOD; no impostor baker built in |
| "behind you ≈ nothing" | frustum culling already draws **nothing** behind you; what stays *loaded* is decided by **streaming**, which is distance-based, not direction-based | yes | culling yes; streaming = our own code (`destinations.gd`) |
| "each player renders only what they see" | that's just per-client rendering (every client already only renders its own view) | — | already true |
| "the server keeps detail only where players are" | **interest management / relevancy / area of interest (AOI)**; **simulation LOD / dormancy**; Valve **PVS-based networking**; Unreal **Net Relevancy / Replication Graph / Net Dormancy** | yes | `MultiplayerSynchronizer` visibility (`public_visibility`, `add_visibility_filter`, `set_visibility_for`); `PROCESS_MODE_DISABLED` for empty areas (we do this for real cities) |

## 2. The pieces, with the best explanations

### 2.1 Frustum culling + guard band (Zeke's FOV + margin)
- **What it is:** draw only what's inside the view frustum. A *guard band* or *cull margin* widens the culling frustum a little past
  the visible one, so objects already exist when the view moves onto them.
- **Why VR needs a margin:** the pose is predicted again after culling runs. Valve (Vlachos, GDC 2015 "Advanced VR Rendering",
  slides [S1]): "We re-predict transforms and update our global cbuffer right before submit … You must conservatively cull on the
  CPU by about 5 degrees." Meta's late-latching post says the same about the cull pose versus the render pose [S2]. That **is**
  Zeke's "margin", at ~5° rather than 10–15 %, and it's drawn at **full** quality: it's a correctness margin.
- **Godot:** culling uses both eyes' real projections (engine source, 8 ms doc §2.2). `GeometryInstance3D.extra_cull_margin` grows an
  object's AABB for culling ("The extra distance added to the GeometryInstance3D's bounding box (AABB) to increase its cull box" [S3]);
  it pads objects, not the camera frustum. Godot already culls with the *predicted* render pose, so the engine needs no margin; **our own script
  culls** (the mirror's `glass_in_view`) do.
- **Map to Zeke:** FOV = the runtime's per-eye projection (never `XRCamera3D.fov`). Margin = ~5° on script culls. The *quality* in
  the margin should be **the same** as in the FOV, because the margin's whole job is to be invisible when you turn.

### 2.2 Lower detail by angle: view-dependent / eccentricity LOD (the part that's research, not practice)
- **Names:** *perceptually driven simplification*, *gaze-contingent LOD*, *foveated LOD*, *eccentricity-dependent LOD*.
  Classic papers: Luebke & Hallen, EGWR 2001 "Perceptually Driven Simplification for Interactive Rendering" [S4] (eye tracker ⇒
  "simplified more aggressively in the periphery than at the center of vision"); Murphy & Duchowski, Eurographics 2001 "Gaze-Contingent
  Level Of Detail Rendering" [S5]; Swafford et al. 2016 (foveated resolution, AO, tessellation, ray-casting) [S6]. Survey:
  Jabbireddy et al. "Foveated Rendering: Motivation, Taxonomy, and Research Directions" [S7]: early works choose an object's LOD "based
  on its distance from the gaze point".
- **What they found:** peripheral vision has low acuity but is **very sensitive to motion and flicker**. The survey [S7]: "the human
  visual system is sensitive to temporal aliasing artifacts even at higher eccentricities … The peripheral motion sensitivity of the
  human visual system poses a critical challenge to all foveated rendering techniques." A LOD swap at the edge is a sudden change in
  the periphery, which is exactly what people notice. So any eccentricity LOD needs gaze tracking (the high-detail
  zone follows the eye) and hidden transitions.
- **Why shipping engines don't do it with a fixed ring:** the eyes move inside the headset. With **fixed** foveation the "centre"
  is the lens centre, not where you look, so whatever you glance at in the "margin" is low quality. And a head turn brings the margin
  into the centre within a frame or two, faster than LOD fades or streaming can catch up. Distance and screen size don't change when
  you turn your head, so LOD based on them never pops on a turn. **That's why engines pick LOD by screen-space error, not angle**:
  Unreal "it is not calculated by distance, but by screen size" [S8]; Godot "mesh LOD selection uses a screen-space metric … The
  default mesh LOD threshold of 1 pixel is tuned to look perceptually lossless" [S9]. No shipping engine found (Godot, Unreal, Unity)
  lowers mesh LOD by angle from the view centre; it exists only in research.
- **Godot:** automatic mesh LOD is screen-space-error based (`mesh_lod_threshold`, in pixels; the quality governor raises it under
  load, `quality_governor.gd:101`). No angle-based LOD exists.

### 2.3 Foveated rendering (what actually does the "ring")
- **Fixed foveated rendering (FFR):** lower *shading rate* toward the lens edges, where the lens blurs anyway (Meta Quest, NVIDIA VRS,
  Godot `VRS_XR`). **Eye-tracked foveated rendering (ETFR / dynamic):** the full-rate zone follows the gaze (Quest Pro, PSVR2, Vive Pro
  Eye; OpenXR `XR_META_foveation_eye_tracked`, also present in Godot 4.7.2's binary).
- **What it changes:** the number of pixel-shader invocations (1×1, 2×2, 4×4 blocks), i.e. resolution of *shading*. It does **not**
  remove lights or swap textures or meshes. But at 4×4 a small light's highlight or a lit window can collapse into one coarse
  block and vanish or smear, and fine texture detail turns into blur, which **could** be what "no lights + simplified textures" looks
  like. *(Unverified: every doc says VRS lowers only shading rate; none says it removes lights. Other candidates for "no lights":
  OmniLight/SpotLight distance fade, or the Mobile renderer's per-object light limit. Neither depends on angle, though, and he saw
  a **ring**, so the A/B in §3 decides.)*
- **Docs:** Meta FFR "enables the edges of an application-generated frame to be rendered at a lower resolution" and warns the "jump
  between FFR levels can be fairly noticeable" [S10]. Godot VRS: "decreasing this shading cost by reducing the resolution of per-pixel
  shading … geometry edges remain as sharp"; "XR automatically generates a texture suited for foveated rendering"; and **"the texture
  LOD bias is set globally, so this will also affect areas of the viewport with full shading rate"** [S11].
- **Research:** Guenter et al. 2012 (Microsoft Research, "Foveated 3D Graphics": gaze-centred layers, ~5–6× speed-up, extra
  antialiasing against "twinkling" outside) [S12]; Patney et al. 2016 (NVIDIA): "filtering peripheral regions reduces contrast,
  inducing a sense of **tunnel vision**", so they filter before and after shading against undersampling aliasing [S13]. NVIDIA's
  Multi-Res / Lens-Matched Shading are the fixed, lens-tuned version *(named only; not opened)*.
- **Godot 4.7.2 (verified in the binary/doctool):** `Viewport.vrs_mode = VRS_XR` uses a density map from the XR interface.
  `OpenXRInterface.vrs_min_radius` (default 20.0) and `vrs_strength` (default 1.0) shape that map (`XRVRS` has the same plus
  `vrs_render_region`). `foveation_level` / `foveation_dynamic` drive the runtime's `XR_FB_foveation` (now also on Vulkan). Glow
  disables VRS (we already turn glow off).
- **Map to Zeke:** his rule "never lower quality inside the FOV" means **fixed foveation must be off or very gentle** (a large
  `vrs_min_radius` so only the lens-blurred rim is coarse). That part of the "margin at lower quality" idea *is* FFR, and it's
  the part he disliked when he saw it. Real gains with no visible cost: the **visibility mask** (hidden-area mesh, ~15 % per Valve),
  which only removes pixels you physically can't see.

### 2.4 Placeholders swapped in seamlessly: HLOD, impostors, cross-fade, hysteresis
- **HLOD:** distant clusters are replaced by one merged proxy mesh + one material (Unreal World Partition HLOD; Spider-Man's Manhattan
  tiles). Godot's equivalent is **visibility ranges**: a far node shows past a distance, its detailed children inside it.
- **Impostors:** a pre-rendered picture of an object on a quad. *Octahedral impostors* (Ryan Brucks, Epic, 2018) store many view
  angles in one atlas so it holds up as you walk round it [S17]. Godot: community plugins only, e.g. SIsilicon's Godot 4 port [S18]
  (licence check needed before shipping, 8 ms doc §2.1).
- **Seamless swap = cross-fade + hysteresis:** Unity LOD Group "Cross Fade" / SpeedTree, Unreal *dithered LOD transition*, Godot
  `visibility_range_fade_mode` (Self/Dependencies, alpha = costs transparency) or per-material **distance-fade Dither** (cheap):
  "Godot currently only supports alpha-based fading for visibility ranges. You can however use dithering instead by using several
  different materials for different LOD levels" [S14]; with fading off, begin_margin "acts as a hysteresis distance". Unreal HLOD:
  "combine them into a single proxy mesh and Material at long view distances" [S15]. Unity `LODFadeMode` CrossFade/SpeedTree [S16].
- **Map to Zeke:** "placeholders outside the view" are **distance**-based in every engine, not view-based, so turning around never
  shows a placeholder. "Behind you ≈ nothing" is already true for *drawing* (frustum culling); keeping things *loaded* behind you is
  cheap and avoids pops on a quick turn.

### 2.5 Occlusion culling
- Godot: CPU-rasterised occluders (`OccluderInstance3D`, baked). Covered in the 8 ms doc §2.3 (on in real cities, off in Lumen).
  GPU-driven alternative: two-phase hierarchical-Z culling with reprojected depth (Ubisoft, Haar & Aaltonen, SIGGRAPH 2015
  "GPU-Driven Rendering Pipelines", Assassin's Creed Unity) [S19], which Godot doesn't ship.

### 2.6 Nanite / virtualized geometry
- Epic's Nanite (Karis, Stubbe & Wihlidal, SIGGRAPH 2021 "A Deep Dive into Nanite" [S20]): "It intelligently does work on only the
  detail that is visible on-screen and no more"; "clusters are swapped on the fly at varying levels of detail based on the camera
  view" [S21]. It's the most complete version of "full detail only where it matters", but it's error-based (screen size), not
  angle-based. **VR: the sources conflict.** Epic's page lists "Stereo rendering for Virtual Reality" and forward rendering as
  unsupported, while forum quotes of the UE 5.3 notes describe instanced-stereo support. Either way it needs the deferred renderer, so
  it's a poor fit for us. Godot has no equivalent (proposal #2793 / #6109, status unconfirmed); mesh LOD + visibility ranges are the
  practical stand-in.

### 2.7 Streaming / world partition
- Unreal World Partition: "Streaming sources are components that define a position in the world and trigger the loading of cells
  around them" [S22], i.e. by **distance**, with HLOD covering unloaded cells. Godot has no built-in system (building block:
  `ResourceLoader.load_threaded_request`; community addons Chunx / Open World Database / Chunk Loader, unchecked). Ours is
  `destinations.gd` (build over frames, hide + `PROCESS_MODE_DISABLED` past 350 m).
- **Why streaming isn't view-direction based** *(reasoning; no source says it outright)*: loading takes hundreds of ms to seconds;
  a 180° head turn takes a fraction of a second.

### 2.8 Multiplayer: interest management / relevancy
- **Names:** *interest management*, *area of interest (AOI)*, *relevancy*, *network culling*, *PVS-based networking*.
- **Engines:**
  - Unreal relevancy: `NetCullDistanceSquared` = "Square of the maximum distance from the client's viewport that this actor is relevant
    and replicates" [S23]; the **Replication Graph** (grid spatialization, "designed to scale well with large numbers of players")
    [S24]; **Net Dormancy** (idle actors leave the gathered list) [S25]; the newer **Iris** replication system's grid filter +
    prioritization [S26].
  - Unity NGO network visibility (`CheckObjectVisibility`, `NetworkShow/Hide`; a hidden object is despawned on that client) [S27];
    Netcode for Entities ghost relevancy + importance scaling [S28].
  - Photon Fusion AOI / interest management: interest built from "sphere, box, cone" shapes over a grid [S29]. **The cone is the
    only view-direction-shaped option found anywhere.**
  - Mirror spatial hashing (grid); its docs advise a visibility range ~10 % larger than the camera far clip [S30].
  - Source: entities transmit only while in the client's PVS [S31]. Quake's network PVS assumes the viewpoint is "anywhere in that
    leaf, looking in any direction" (Abrash, ch. 70) [S32]: **position-based on purpose.**
- **Server-side "detail only where players are":** *simulation LOD / dormancy*: the server stops simulating areas nobody is in
  (sleeping bodies, disabled processing) and wakes them when a player arrives. We already do this client-side for real cities.
- **Why servers don't cull by *view direction*:** latency (the client turns faster than a round trip) and pop-in. The closest
  real system is **VALORANT's fog of war** [S33]: "the game server withholds information about the positions of enemies until a client
  needs to display it", using "a PVS system [that] pre-computes which parts of a scene (our map) can see which other parts", and
  it sends an enemy "just before the enemy came around the corner". So it's **position + line of sight, sent early**, not the
  player's gaze. Gaze-based server culling would pop whenever latency exceeds head-turn time.
- **Simulation LOD in Godot:** RigidBody3D sleeping ("will not move and will not calculate forces until woken up") and
  `PROCESS_MODE_DISABLED` ("Never processes") [S34].
- **Godot 4.7:** `MultiplayerSynchronizer.public_visibility`, `add_visibility_filter(callable)` ("should take a peer ID int and
  return a bool"), `set_visibility_for(peer, bool)`, `update_visibility()` [S35]. **Ours:** no interest management at all (every update to every peer, vrgameskills finding #5), which is
  the real multiplayer gap.

## 3. What to do in our game (ordered)

1. **Foveation first (fixes the circle, maybe):** default `VRS_XR` **off** in Settings, or keep it with `vrs_min_radius` ~50–60 and
   `vrs_strength` ~0.5. Set `openxr/foveation_level` to 0 unless measured. Log at startup which foveation path is live (VRS map,
   `XR_FB_foveation`, runtime). A/B at the New York spot Zeke saw, with his eyes. (xrfix_1104 is on this.)
2. **Visibility mask on:** confirm `XR_KHR_visibility_mask` is active (free ~15 %, invisible).
3. **The margin, done the established way:** engine culling is already right. For our script culls (mirror) use per-eye
   `get_projection_for_view` + ~5°. Don't add a lower-quality ring.
4. **Placeholders, done the established way:** distance-based HLOD for Lumen (8 ms doc step 2), dither fade + margins (step 6),
   screen-space mesh LOD with `mesh_lod_threshold` ≤ 1 in VR (cars, step 8).
5. **Multiplayer:** per-peer visibility on the server by area (temple / Lumen / each real city; Godot visibility filters or our own
   per-area send lists), plus server-side sleep of props in areas with no players. That's Zeke's "server keeps detail only where players
   are", and it also fixes the n² traffic.
6. **Later, only with eye tracking:** dynamic foveation (`XR_META_foveation_eye_tracked`) on headsets that have it. Quest 3 has no
   eye tracking (Quest Pro does), so this isn't for Zeke's headset now.

## 4. What Godot 4.7.2 already has (checked) vs not

| piece | Godot 4.7.2 | where / note |
|---|---|---|
| frustum culling from both eyes' runtime projections | ✅ | engine source (8 ms doc §2.2) |
| cull margin | ✅ `GeometryInstance3D.extra_cull_margin` | per instance |
| screen-space-error mesh LOD | ✅ auto LOD on import + `mesh_lod_threshold` | per viewport |
| HLOD | ✅ visibility ranges (begin/end, margins, fade Self/Dependencies) + material distance-fade Dither | we use it in `city_kit.gd` |
| impostors | ❌ (plugins only) | trees use our own 4-view impostor |
| occlusion culling | ✅ `OccluderInstance3D` (CPU, Embree) | on in real cities only |
| fixed foveation | ✅ `VRS_XR` + `vrs_min_radius` / `vrs_strength`; `foveation_level` / `foveation_dynamic` (XR_FB) | **defaults 20 % / 1.0 — never tuned by us** |
| eye-tracked foveation | ✅ strings for `XR_META_foveation_eye_tracked` in 4.7.2 | headset must support it |
| visibility mask | ✅ OpenXR `XR_KHR_visibility_mask` extension | verify it's live on the tower |
| XR render scale | ✅ `OpenXRInterface.render_target_size_multiplier` | the XR-native knob (vs `scaling_3d_scale`, which breaks XR for us) |
| streaming / world partition | ❌ | our `destinations.gd` |
| virtualized geometry (Nanite) | ❌ | — |
| network interest management | ✅ `MultiplayerSynchronizer` visibility filters | we don't use MultiplayerSynchronizer (custom `net.gd`), so we'd build per-area send lists |

## 5. Sources
Quotes were read from the page or PDF unless marked *(snippet)* = search-result text only, page not loaded.
- [S1] Vlachos, GDC 2015 "Advanced VR Rendering" slides: https://media.steampowered.com/apps/valve/2015/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
- [S2] Meta, "Optimizing VR graphics with late latching" *(snippet)*: https://developer.oculus.com/blog/optimizing-vr-graphics-with-late-latching
- [S3] Godot `GeometryInstance3D`: https://docs.godotengine.org/en/stable/classes/class_geometryinstance3d.html
- [S4] Luebke & Hallen, EGWR 2001: https://diglib.eg.org/items/2bee1720-f9cc-4b83-9bb0-40d3a4555305 (quote from the 2000 UVA tech report *(snippet)*: https://libraopen.lib.virginia.edu/public_view/x633f100h)
- [S5] Murphy & Duchowski, Eurographics 2001 short paper, DOI 10.2312/egs.20011004 *(abstract snippet)*: https://diglib.eg.org/items/ce489bf6-4ad5-4beb-a373-0724eb6e25ce
- [S6] Swafford et al., SAP 2016, DOI 10.1145/2931002.2931011: https://purehost.bath.ac.uk/ws/files/146010777/user_metric_computational.pdf
- [S7] Jabbireddy, Sun, Meng & Varshney, "Foveated Rendering: Motivation, Taxonomy, and Research Directions": https://arxiv.org/pdf/2205.04529
- [S8] Unreal, Creating and using LODs: https://dev.epicgames.com/documentation/en-us/unreal-engine/creating-and-using-lods-in-unreal-engine
- [S9] Godot, Mesh level of detail: https://docs.godotengine.org/en/stable/tutorials/3d/mesh_lod.html
- [S10] Meta, Fixed Foveated Rendering: https://developers.meta.com/horizon/documentation/unity/os-fixed-foveated-rendering/
- [S11] Godot, Variable rate shading: https://docs.godotengine.org/en/stable/tutorials/3d/variable_rate_shading.html · OpenXR settings (foveation level / dynamic; "disabled if post effects are used such as glow, bloom, or DOF"): https://docs.godotengine.org/en/stable/tutorials/xr/openxr_settings.html
- [S12] Guenter et al., "Foveated 3D Graphics", SIGGRAPH Asia 2012 *(abstract snippet)*: https://www.microsoft.com/en-us/research/publication/foveated-3d-graphics
- [S13] Patney et al., NVIDIA 2016: https://research.nvidia.com/publication/2016-12_towards-foveated-rendering-gaze-tracked-virtual-reality
- [S14] Godot, Visibility ranges (HLOD): https://docs.godotengine.org/en/stable/tutorials/3d/visibility_ranges.html
- [S15] Unreal HLOD: https://dev.epicgames.com/documentation/en-us/unreal-engine/hierarchical-level-of-detail-in-unreal-engine
- [S16] Unity `LODFadeMode` *(snippet)*: https://docs.unity3d.com/520/Documentation/ScriptReference/LODFadeMode.html
- [S17] Brucks, "Octahedral Impostors" (2018; page body didn't load): https://shaderbits.com/blog/octahedral-impostors
- [S18] SIsilicon Godot-Octahedral-Impostors: https://github.com/SIsilicon/Godot-Octahedral-Impostors
- [S19] Haar & Aaltonen, SIGGRAPH 2015 *(not quoted)*: https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
- [S20] Karis, Stubbe & Wihlidal, SIGGRAPH 2021 *(PDF > 10 MB, not read)*: https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf
- [S21] Unreal, Nanite: https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine
- [S22] Unreal, World Partition: https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition-in-unreal-engine
- [S23] Unreal, Actor relevancy: https://dev.epicgames.com/documentation/en-us/unreal-engine/actor-relevancy-in-unreal-engine
- [S24] Unreal, Replication Graph: https://dev.epicgames.com/documentation/en-us/unreal-engine/replication-graph-in-unreal-engine
- [S25] Unreal, Net Dormancy: https://dev.epicgames.com/documentation/en-us/unreal-engine/actor-network-dormancy-in-unreal-engine
- [S26] Unreal Iris filtering / prioritization *(snippet)*: https://dev.epicgames.com/documentation/en-us/unreal-engine/iris-filtering-in-unreal-engine
- [S27] Unity NGO object visibility *(snippet)*: https://docs.unity3d.com/Packages/com.unity.netcode.gameobjects%402.5/manual/basics/object-visibility.html
- [S28] Unity Netcode for Entities: https://docs.unity3d.com/Packages/com.unity.netcode@1.4/api/Unity.NetCode.GhostRelevancy.html · https://docs.unity3d.com/Packages/com.unity.netcode@1.1/manual/optimizations.html
- [S29] Photon Fusion AOI sample + interest-management add-on *(snippet)*: https://doc.photonengine.com/fusion/v2/technical-samples/fusion-multi-peer-area-of-interest · https://doc.photonengine.com/ja-jp/fusion/v2/addons/interest-management/architecture
- [S30] Mirror, Spatial Hashing interest management: https://mirror-networking.gitbook.io/docs/manual/interest-management/spatial-hashing
- [S31] Valve Developer Wiki, Networking Entities: https://developer.valvesoftware.com/wiki/Networking_Entities
- [S32] Abrash, *Graphics Programming Black Book* ch. 70: https://github.com/othieno/GPBB/blob/master/book/gpbb.2.70.pdf
- [S33] Riot, "Demolishing wallhacks with VALORANT's fog of war": https://www.riotgames.com/en/news/demolishing-wallhacks-valorants-fog-war
- [S34] Godot `RigidBody3D` / `Node`: https://docs.godotengine.org/en/stable/classes/class_rigidbody3d.html · https://docs.godotengine.org/en/stable/classes/class_node.html
- [S35] Godot `MultiplayerSynchronizer`: https://docs.godotengine.org/en/stable/classes/class_multiplayersynchronizer.html
- OpenXR `XR_META_foveation_eye_tracked`: https://registry.khronos.org/OpenXR/specs/1.1/man/html/XR_META_foveation_eye_tracked.html
- Godot 4.7.2 class reference, dumped locally with `--doctool` (members + defaults of `OpenXRInterface`, `XRVRS`): checked 11-04.
- Our code: `scripts/xr_session.gd:14-18`, `project.godot:70`, `scripts/quality_governor.gd:101`, `scripts/destinations.gd`, `scripts/net.gd`.
- Companion doc: `research/vr_render_lod_streaming_2026-11-04.md` (Valve 5° margin, hidden-area mask ~15 %, HLOD, visibility ranges, occlusion, VRChat mirrors).

## 6. Looked for, not found or not verified
- **Meta "frustum padding"**: no source with that name or a number.
- **Any shipping engine with angle-from-centre (eccentricity) mesh LOD**: none found; research only (§2.2).
- **A Godot proposal for view-dependent LOD**: none found. A user-facing Godot eye-tracked-foveation setting: not confirmed (the
  extension strings are in the 4.7.2 binary).
- **Not read in full:** Brucks impostors, Karis Nanite slides, Haar/Aaltonen slides, Guenter full paper, NVIDIA MRS/LMS pages.
- **Why "no lights" in the periphery:** no doc says VRS removes lights. Distance fade or Mobile per-object light limits are
  hypotheses, but they aren't angle-based. The foveation A/B in §3 is the test.

