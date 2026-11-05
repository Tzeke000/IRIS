# VR render cost: far vs near, LOD, streaming, culling — and the road to 8 ms (2026-11-04)

Helper research for VR Iris / LUMEN (Godot 4.7 Mobile, OpenXR, RTX 3060, 72 Hz). The code was read only; nothing was
changed or committed. **Estimates are marked (est.)**; everything else is a measurement, a quote, or a file:line.

## Summary (5 lines)

1. **The mirror is the biggest single cost we know of: +8 ms (low) to +12 ms (high) on Zeke's headset.** The size of the
   viewport barely matters (low→high is ~10× the pixels for +4 ms), so most of the cost is the fixed cost of **two extra
   full scene passes** (cull, shadows, draws). Shrink what the mirror draws; don't just shrink its resolution.
2. **The temple is the most expensive place in the whole game, more than any city street.** temple_spawn draws 1,957
   calls / 3.83 M prims, against 24–383 calls on every city and real-city vantage (`research/cities4_1101/perf/cities4_vr.txt`).
   Lumen is never streamed out. Real cities *are* hidden from the temple (`destinations.gd:192-203`), but Lumen is not.
3. **No hand-written "circle cull" exists in our code.** The circle Zeke saw while swinging is almost certainly the
   **XR Tools comfort vignette** (on by default; it closes to 35 % at ≥ 10 m/s). Distance pop-in comes from
   visibility ranges, which are **spheres measured from the centre of each 110 m cell**, not from the FOV.
4. **NYC "no bottoms / see-through walls" is a data bug in the party-wall pass** (`tools/city2/osm_district.py:195-221`).
   It drops a wall wherever three sample points find a neighbour 0.8 m away, so ends of walls and 0.15–0.8 m gaps go
   missing: Midtown has **194 party-wall edges with open stretches (~970 m of missing wall, 153 buildings)**; Shinjuku has 292.
5. **Ranked plan to 8 ms:** instrument → mirror diet → Lumen HLOD/streaming from the temple → shadows (2 splits, 60 m,
   off when invisible) → verify foveation + visibility mask → smaller detail cells with dither fade → swing-hitch trace
   → cars + NYC walls (looks, not ms).

---

## 1. Where the frame goes today (what we already know)

| source | number |
|---|---|
| Zeke, SteamVR, 11-04 | 13–14 ms GPU normal · ~22 ms mirror Low · ~26 ms mirror High · budget 13.9 ms (72 Hz) · **target ≈ 8 ms** |
| VR cityperf (Monado, V100 stand-in, 11-01) | temple_spawn **1,957 draws / 3.83 M prims** · north_high_back 1,763 / 2.03 M · every street/rooftop/real-city vantage 24–383 draws, 32 k–230 k prims |
| Meta PC-VR guideline | 500–1,000 draws and 1–2 M tris per frame for both eyes ([Meta](https://developers.meta.com/horizon/documentation/native/pc/dg-performance-guidelines/)) |

So the temple is **2–4× over** Meta's draw and triangle budget, and every city street is well inside it. The two
outlier vantages are the only ones that look *at* the temple/yard + Lumen core from outside. The Lumen A/B
(`research/lumen_1101/perf/ab_table.md`: 1,936 → 1,957) shows the 11-01 Lumen dressing isn't the cause. The bulk is
something older.

**What we can't yet say: which subtree owns those 1,957 draws.** `xr_probe.gd:250` records
`RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME`, which (I believe; verify) adds shadow passes to the colour pass. A plausible
reading, unverified: the temple's Meshy props and display pieces have many surfaces each (`iris_eye_body.glb` 171
surfaces / 71 k tris, `gyroscope.glb` 47 / 45 k, `robot_arm.glb` 28 / 32 k, VRM avatars 13–67 surfaces; counted
11-04 from the glTF files). The sun is in the default **PSSM 4 splits / 100 m** (`world.gd:456-462`; nothing sets mode or
distance outside a real city). Godot docs: a caster in all four splits is drawn **5×**
([godot-docs PR 8723](https://github.com/godotengine/godot-docs/pull/8723.diff)). ~400 surfaces × ~5 passes ≈ 2,000 draws,
which is the order of what we see. Step 0 below exists to prove or kill that reading before anyone optimises.

## 2. What the research says, mapped to our code

### 2.1 LOD chains, HLOD, impostors
- **Godot visibility ranges = HLOD.** The distance is "from the camera to the **centre of the instance's AABB**".
  Fade mode *Self* "forces transparent rendering … during its fade transition, so it has a performance impact".
  *Disabled* + margins = hysteresis, the cheapest. Dithering (per-material Object Dither distance fade) is the fast
  alternative to alpha fade ([Godot docs](https://docs.godotengine.org/en/stable/tutorials/3d/visibility_ranges.html)).
  A MultiMeshInstance3D can stand in as the far level for a group of houses.
- **Ours:** `city_kit.gd:265-293`. Three range classes (220 / 400 / district `vis_end` 480 m from `city_big.gd:92`),
  `visibility_range_end_margin = 8 %`, fade mode default (Disabled = a hard pop). With 110 m cells and AABB-centre
  distance, a piece at a cell corner pops up to ~78 m (55·√2) early or late compared with its own distance. That's the
  "pop-in" you see at the sides while turning near a cell boundary. City v2 already learned this (`city2.gd:15`
  "a MultiMesh is culled + ranged as ONE box … hence the small 3D cells", 60 m cells).
- **Unreal World Partition HLOD:** a grid of cells, and distant clusters of static meshes are replaced by **one proxy
  mesh + one material** "to reduce the number of draw calls per frame"
  ([Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition---hierarchical-level-of-detail-in-unreal-engine)).
  City v2 already does this for real cities: the shell *is* the far HLOD (`city2.gd:24-28`). **Lumen has no HLOD.**
  Its full kit (towers + window shader + every module batch) is the only level.
- **Octahedral impostors:** capture the object from many angles into an atlas and draw one quad that picks the view in
  the shader. Godot 3 reference implementation: [wojtekpil](https://github.com/wojtekpil/Godot-Octahedral-Impostors)
  (MIT, Godot 3.2). For Godot 4 there's an "Octahedral Impostor LOD" plugin that uses visibility ranges, under a custom
  licence ([asset library](https://godotengine.org/asset-library/asset/edit/24616)); **check that licence before use**,
  since the game may be sold. Our trees already use a 4-view impostor level (`kit/tree_kit.gd:3-20`). Impostors pay off
  for **unique, detailed** silhouettes (Meshy props, landmark towers). Lumen's boxy towers are cheaper as a merged shell
  mesh than as impostors.

### 2.2 Frustum culling with the real XR frustum
- Godot builds the culling frustum from **both eyes' runtime projections**: `CameraData::set_multiview_camera` →
  `Projection::create_combined_projection(main transform, each eye's projection + offset)`
  ([renderer_scene_render.cpp](https://raw.githubusercontent.com/godotengine/godot/master/servers/rendering/renderer_scene_render.cpp)).
  We don't override it, so engine culling should already follow each headset's real per-eye FOV.
- Other engines have had exactly the bug Zeke describes. Unreal (PICO OpenXR) merged per-eye FOV angles in eye space,
  and on toe-out/canted displays the combined frustum came out narrower than the real view, culling objects at the edge
  ([UE forum](https://forums.unrealengine.com/t/pico-openxr-fix-sps-mono-culling-frustum-incorrectly-culling-edge-objects/2763119)).
  Unity had a Vive left-edge cull bug ([Unity](https://issuetracker.unity3d.com/issues/openvr-culling-in-left-eye-slightly-off)).
  **I found no Godot report of this**, but it's cheap to test on his headset (§5 test F).
- Vlachos (Valve, GDC "Advanced VR Rendering Performance", transcript in `/home/iris/untrusted/yt_lod/`): "we
  conservatively cull on the CPU by about **five degrees** in each direction out of our frustum" because the head moves
  between simulate and render. Godot culls with the predicted render pose, so this matters most for **our own** gates
  (mirror `glass_in_view`, any future script cull). Widen those by 5°.
- **Precedent inside our code:** the mirror's old visibility test used the XRCamera3D's *nominal* 75° frustum and froze
  the mirror at the edges of the real FOV (`mirror.gd:172-186`, fixed 10-31 to use
  `get_projection_for_view` per eye). A grep for `is_position_in_frustum` / `unproject_position` / `.fov` on the XR path
  found no other user (11-04).

### 2.3 Occlusion culling
- Godot rasterises occluders on the CPU (Embree) into a low-res buffer. It has a CPU cost, and "large open scenes with
  few or no objects blocking the view will generally not benefit much". **The Mobile renderer gains most because it has
  no depth prepass**, so occlusion directly cuts shading overdraw. Moving or hiding occluders triggers a
  several-frame rebuild ([Godot docs](https://docs.godotengine.org/en/stable/tutorials/3d/occlusion_culling.html)).
- **Ours:** on only inside a real city (`destinations.gd:231`), with an occluder prism per building (`city2.gd:421-450`).
  Measured: draws 359 → 236 on a side street. **Lumen has none** (`city2.gd:423` "Lumen: off, as before";
  `city_kit.gd` makes no occluders). Lumen is a street-canyon city, the case occlusion is built for. Note: the
  `use_occlusion_culling` project setting isn't set (`project.godot`), so the toggle relies on the runtime viewport
  property. It measured as working, so fine.
- Mirrors: "occlusion culling doesn't help … a mirror facing a wall will still try to render everything behind the
  wall" ([VRChat wiki](https://wiki.vrchat.com/wiki/Mirror)).

### 2.4 Streaming / destination-based unloading
- **Ours, real cities:** done right. Built over frames ≤ 4 ms each (`destinations.gd:157-176`), hidden +
  `PROCESS_MODE_DISABLED` past 350 m outside the slot (`:192-203`), far plane 1,200 m inside a city (`:213-222`, which
  cut "1737 calls / 2.0 M tris" looking back at Lumen + the temple).
- **Ours, Lumen + the old core:** never hidden. `city.gd` / `city_big.gd` build once and stay visible from everywhere.
  This answers Zeke's "why are we drawing the big cities from the temple?": the **real** cities aren't drawn there, but
  **Lumen is**, out to 480 m per cell, fogged to ~97 % at 450 m (`city_big.gd:92` comment). Fog doesn't cull
  (`destinations.gd:46`).
- **Industry pattern:** Spider-Man (Insomniac) builds Manhattan as a streamed tile grid with low-LOD proxies for
  everything not loaded. Ruskin's GDC 2019 "Technical Postmortem" covers frame breakdown + streaming
  ([GDC Vault](https://gdcvault.com/play/1026495/-Marvel-s-Spider-Man)). I couldn't get the slide numbers (Vault
  paywall), so no figures are quoted. Unreal World Partition: load/unload grid cells, and **HLOD stands in for unloaded
  cells** (Epic docs above). Same shape for us: Lumen as cells + an always-on skyline proxy.

### 2.5 Mesh merging / MultiMesh trade-offs / draw budgets
- A MultiMesh is culled **only as a whole** (its combined AABB). The fix is "several MultiMeshes for different areas"
  ([Godot docs](https://docs.godotengine.org/it/stable/tutorials/performance/using_multimesh.html)). Separate
  MeshInstance3Ds with the same mesh+material are auto-instanced *and* culled per instance, at a CPU cost.
- **Mesh LOD on a MultiMesh** is chosen for the whole batch from its AABB (Godot `_fill_render_list`; from reading
  the engine source in the past, **verify**). One car LOD gets picked per 110 m cell, not per car.
- Budget: every draw is issued per eye in stereo unless multiview ([Meta](https://developers.meta.com/horizon/documentation/native/pc/dg-performance-guidelines/));
  Godot Mobile renders both eyes with multiview, so the CPU cost stays near 1×. The GPU still rasterises both.

### 2.6 Shadows
- Directional shadows redraw **every frame**. 4 splits is the slowest mode, 2 splits a compromise, orthogonal the
  fastest. `directional_shadow_max_distance` (default 100 m) bounds which casters join
  ([DirectionalLight3D docs](https://docs.godotengine.org/en/4.3/classes/class_directionallight3d.html), forum report
  "chopped off about 100 draw calls" by mode + distance).
- **Ours:** sun 4 splits / 100 m everywhere except real cities (2 splits, `destinations.gd:232-237`). The moon also has
  `shadow_enabled = true` (`world.gd:68`). Day/night fades **`shadow_opacity`** (`daynight.gd:98, :106`). **I believe
  opacity 0 still renders the shadow map** (opacity is a shading term; verify with SHADOW draw counts). The moon is
  hidden (`visible = false`) when its energy is ~0 (`:105`), which does skip it. Through twilight, both directional
  shadows are live.
- The home-room omni light (its room script, lines 68-69, 6 m, shadowed) is fine: omni shadows only update when something in range moves.

### 2.7 Foveated rendering + visibility mask
- Godot: Forward+ and Mobile both support VRS. **XR mode "automatically generates a texture suited for foveated
  rendering"**, and on unsupported hardware there's no visual difference. Desktop support starts at NVIDIA Turing, so the
  RTX 3060 qualifies. `--verbose` prints the support ([Godot VRS docs](https://docs.godotengine.org/en/stable/tutorials/3d/variable_rate_shading.html)).
  The docs don't name drivers. **Whether it's active on D3D12 vs Vulkan on the tower is unverified.**
- **Ours:** `xr_session.gd:14` `VRS_XR`, glow forced off (`:17-18`), `foveation_level=3` (`project.godot:70`, which only
  drives Compatibility). Never A/B-measured.
- **Hidden-area / visibility mask.** Vlachos: the stencil "hidden area mesh" is "another fifteen percent perf boost",
  since 15–20 % of the projected pixels are never seen through the lenses. Godot's OpenXR module has an
  `openxr_visibility_mask_extension` ([source mirror](https://remotebranch.eu/Stowage/godot/src/commit/edde380e9cb1545d77e99bba99aaf55f55e7dc86/modules/openxr/extensions/openxr_visibility_mask_extension.h)).
  **Check that it's enabled and that SteamVR offers `XR_KHR_visibility_mask`** in the game log at startup.

### 2.8 Mirrors in VR
- VRChat: "every object reflected in a mirror has to be rendered again … a mirror reflecting the whole world will
  **double** the performance cost". Off by default, on when near, **limit reflect layers** ("people only"), lower res,
  disable pixel lights in mirrors, one mirror at a time ([VRChat wiki](https://wiki.vrchat.com/wiki/Mirror),
  [creator docs](https://creators.vrchat.com/worlds/components/vrc_mirrorreflection)).
- **Ours (`mirror.gd`):** proximity (14 m) + per-eye clip-space view gate (`:161-186`), per-eye SubViewports at 512 /
  1024 / 1600 px tall (`:64-67`), MSAA 2× on each, far plane 60 m (`:230`), EVERYTHING mode culls only the
  first-person layer (`:61`). Each eye is a **separate** viewport render (not multiview), so each pays its own cull +
  **its own 4-split sun shadow pass** + draws. That fits the +8 ms fixed cost. The temple mirror sits at (-3.4, 0, 11.6)
  (`world.gd:154`); what it faces decides how much of the yard + Lumen falls inside its 60 m.

### 2.9 Profiling Godot XR
- In-engine GPU time: `RenderingServer.viewport_set_measure_render_time(rid, true)`, then
  `viewport_get_measured_render_time_gpu(rid)` ([godot #75346](https://github.com/godotengine/godot/issues/75346); one
  report of zeros on 4.0.1, so cross-check). Do it **per viewport**: the mirror SubViewports have their own RIDs, so the
  mirror's own GPU ms can be read directly.
- Split counts: `Viewport.get_render_info(RENDER_INFO_TYPE_VISIBLE / _SHADOW, …)` gives colour vs shadow draws and prims.
  Today we only log the total.
- Headset level: the SteamVR frame-timing graph (app GPU ms, reprojection %). OpenXR has no standard app-GPU-time query
  (Unity thread, dated). The Monado runs (V100) are good for *counts*. Only the tower is good for ms.
- Valve adaptive quality (Vlachos): scale resolution first, then MSAA, keep hysteresis. Ours already does
  (`quality_governor.gd`), but note it **measures main-loop wall time**, not GPU ms (`:47-50`), and only reacts at
  1.3× budget (18 ms), so it will never chase an 8 ms target.

---

## 3. Ranked plan to ≈ 8 ms (biggest win first)

ms figures are **(est.)** from the numbers above. Real gains come only from the §4 protocol on the tower.

| # | change | where | expected | confidence |
|---|---|---|---|---|
| 0 | **Instrument first:** split VISIBLE vs SHADOW draws/prims; per-viewport GPU ms (main + each mirror eye); a `perf toggle` probe command that hides one subtree (Lumen root, BigCity, temple props, grass, trees, avatars) or flips a light's shadow; log SteamVR render resolution % | `xr_probe.gd:231-256`, `harness/xr/xr_cityperf.py` | 0 ms, but it tells us where the 1,957 draws live | — |
| 1 | **Mirror diet.** (a) mirror cameras: far 60 → ~12–15 m (`mirror.gd:230`), cull mask drops the city/far-scenery layers (put Lumen + yard scenery on their own visual layer); (b) `vp.msaa_3d` off on Low; (c) `positional_shadow_atlas_size = 0` on the mirror viewports; sun shadows in the mirror: no per-viewport switch, so give the mirror a reduced scene layer or accept it; (d) AVATARS mode as the default "On" (VRChat practice); (e) optional: update each eye every other frame on Low | `mirror.gd:61-67, :160-171, :230` | **−4 to −9 ms while the mirror is on** (est.; the measured fixed cost is ~8 ms, resolution ~4 ms) | high that it's big; mix unknown |
| 2 | **Stop drawing Lumen at full detail from the temple.** Build a **Lumen skyline HLOD**: one merged shell mesh per ~200 m block (tower boxes + roofs, vertex colour, no window shader, no shadows) at build time. Show it, and hide the full kit (visible off like `destinations.gd:200-203`), whenever the player is > ~60 m outside `BOUNDS`. Then add **occluder prisms for Lumen towers** (reuse `city2.gd:_occluder`) and turn occlusion on inside Lumen too | `city_big.gd:75-110, :92`, `city.gd`, `destinations.gd:213-237` | temple_spawn draws 1,957 → < 700 (est.); **−1.5 to −4 ms at the temple** (est.), −0.5 to −1.5 ms in Lumen streets from occlusion (est.; the real-city side street went 359 → 236 draws) | medium until step 0 shows the split |
| 3 | **Shadows.** (a) sun 2 splits + `directional_shadow_max_distance` 100 → 50–60 m everywhere (blend splits off); (b) `shadow_enabled = false` whenever `shadow_opacity < 0.01` (sun and moon), instead of fading opacity alone; (c) `cast_shadow = OFF` on small temple props and display pieces, or a single low-poly shadow proxy each (the tree_kit level-3 trick, `kit/tree_kit.gd:20`) | `world.gd:68, :456-462`, `daynight.gd:98, :106`, prop builders | **−1 to −3 ms** at the temple (est.; casters drawn ×5 → ×3) | medium-high |
| 4 | **Verify foveation + visibility mask on the tower.** Log VRS support (`--verbose`), A/B `vrs_mode` XR vs disabled at the temple and a street; confirm `XR_KHR_visibility_mask` is active in the log. If VRS turns out off on the D3D12 path, A/B Vulkan | `xr_session.gd:14`, `gpu_profile.gd` | 0 to −2 ms if fill-bound (est.; Vlachos: hidden-area mask ≈ 15 %) | unknown: unverified on this runtime |
| 5 | **Check SteamVR's per-app render resolution.** GPU ms scales with pixels when fill-bound; an auto-supersample > 100 % is a silent tax. Record it in every measurement; set the game's `render_target_size_multiplier` deliberately | SteamVR settings; `quality_governor.gd` | could be the largest lever of all, or zero | unknown, so measure |
| 6 | **Smaller detail cells + dither instead of pop.** Detail (220 m) and mid (400 m) batches on 55 m cells (draws rise slightly; ranges stop drifting ±78 m); `distance_fade_mode = DITHER` on the kit materials over the last 8 % of range (no transparency). Also stops edge pop-in | `city_kit.gd:21, :265-293` | ≈ 0 ms (it's a look fix; slightly more draws) | high for the pop, none for the ms |
| 7 | **Swing frame drops: trace before fixing.** Record per-frame ms + draws along a scripted 25 m/s swing path. Suspects, unverified: first-sight pipeline compiles of kit materials; the governor itself changing `scaling_3d_scale` mid-swing (buffer re-allocation; `quality_governor.gd:100`); visibility-range batches popping in bulk at cell crossings; city movers / web strands CPU | §4 test D | removes hitches, not mean ms | — |
| 8 | **Cars.** They're voxel-remeshed + decimated to 2.4–4.2 k tris with **UVs/textures dropped, colour baked per face** (`props/city_cars/SOURCE.md`), so they look unrendered close up by design. On top of that, `generate_lods=true` on import (`sedan.glb.import:27`) and the MultiMesh picks one LOD per 110 m cell; under load the governor's ladder sets `mesh_lod_threshold = 4` (`quality_governor.gd:101`). Fix: turn import LODs off for cars (they are already low), and add a near tier (< 25 m) with a textured ~10–15 k model | `city_big.gd:907-970`, `city_kit.gd:97` | ≈ 0 ms (est.; few cars near) | high it's the asset + LOD, verify with `mesh_lod_threshold = 0` |
| 9 | **NYC party walls (§6).** Sample the whole edge, keep a wall unless a neighbour *touches* it (≤ 0.2 m), and cut the wall only over the covered interval | `tools/city2/osm_district.py:195-221`, `city2_gen.gd:150-160` | 0 ms (looks) | high (measured) |

**Arithmetic toward 8 ms (est.):** 13.5 ms today at the temple without the mirror → −2.5 (Lumen HLOD + occlusion)
→ −1.5 (shadows) → −1 (foveation/mask, if it's actually off today) ≈ 8.5 ms. The rest has to come from step 0's
attribution (temple props/avatars) or from render resolution (step 5). **With the mirror on, 8 ms is only reachable in
AVATARS mode or at a very short mirror far plane.** A full-world mirror is a second render of the world by definition
(VRChat's own words).

## 4. Measurement protocol (tower, real headset; Monado only for counts)

**Fixed conditions, recorded with every row:** build id; SteamVR render resolution %; refresh 72 Hz; governor **OFF**
(`graphics/auto_quality=false`), render scale fixed at 1.0; time of day fixed (`daynight` at one phase; record whether
moon/sun shadows are live); no other players; tower idle apart from SteamVR. Settle 10 s after arriving, then 600
frames, ×3 repeats; report mean / p95 / p99 GPU ms, SteamVR reprojection %, frames > 20 ms, and VISIBLE vs SHADOW
draws/prims from the probe (step 0).

**Vantages (each held still, head level, facing the named way):**

| tag | where | facing |
|---|---|---|
| A temple_centre | (0, 0, 0) | north (toward Lumen), then south |
| B mirror | 1.5 m in front of the mirror at (-3.4, 0, 11.6), looking at it | the glass |
| C table | `TABLE_POS` (-6.5, 0, 7.5) | the table (hands near props) |
| D swing_path | scripted 30 s path through Lumen's avenue at ~25 m/s, 20–40 m up (a recorded web swing, replayed) | along the path; log **per-frame** ms + draws, mark hitches |
| E nyc_street | `mid_street_side` vantage from `xr_cityperf.py` (Midtown) | along the street |

**Toggles, one at a time against the baseline at every vantage (on → off → on):**
1. mirror: off / AVATARS / EVERYTHING × low / medium / high (B only, plus A facing it)
2. sun shadows: 4 splits / 2 splits / off; moon shadow on/off (twilight phase)
3. Lumen full kit hidden (A, B, C), and later Lumen HLOD on
4. occlusion culling on/off (D, E)
5. `vrs_mode` XR / disabled
6. comfort vignette on/off (D: does the "circle" go away)
7. `mesh_lod_threshold` 0 / 1 / 4 (cars at D and the precinct)
8. temple subtrees hidden one at a time: props, avatars/pedestals, grass, trees, gallery, avatar room

**Answer rule:** a change counts only if its delta is larger than the spread of the three repeats.

## 5. The "custom circle" cull: where it is (and isn't)

**There is no hand-written angular/circle cull in the game code.** Searched 11-04: every `dot(` / `cos(deg_to_rad` /
`is_position_in_frustum` / `unproject_position` / `.fov` / `visible =` in `scripts/` (the hits are aim assists,
gestures, Iris's own sight model `rig_iris.gd:82-83, :1868`, and the mirror's per-eye gate). The candidates, most likely
first:

1. **The XR Tools comfort vignette: a literal circle that narrows while moving.** `rig_xr.gd:94-99` adds
   `addons/godot-xr-tools/effects/vignette.tscn` to the camera on layer 8 (the player's eyes only). It's **on by
   default** (`settings_manager.gd:27` `"vignette": true`; toggle in `settings_menu.gd:109`, "Comfort vignette while
   moving"). In auto mode (`vignette.gd:9-14, :152-185`) it closes toward **inner radius 0.35** as camera speed nears
   **10 m/s** or rig rotation nears **20°/s**, then reopens after 1 s. Web-swinging is well past 10 m/s. The ring is
   drawn in clip space with `depth_test_disabled` (`vignette.gdshader`), so it blacks out the edges of each eye's real
   FOV whatever the headset. It *feels* like a narrower view with things popping at the rim as it breathes open.
   **Falsifier:** turn the setting off and swing (protocol toggle 6). If the circle stays, it's not this.
2. **Visibility ranges are spheres, not cones.** `city_kit.gd:267, :292`: 220 / 400 / 480 m from the centre of each
   110 m cell's AABB, hard pop (fade Disabled, 8 % hysteresis). It reads as a "circle" of detail round you, and at the
   sides of the view the nearer edge of a cell can be in view while its centre is still out of range. Fix = §3 row 6.
3. **The engine frustum.** Godot uses both eyes' runtime projections (§2.2), so it should match each headset. Unreal and
   Unity both shipped a too-narrow combined frustum on canted/toe-out displays at one point. **Test F:** put a bright
   pole at the extreme left/right edge of each eye (turn the head slowly until it's at the lens edge) with the vignette
   off and ranges out of play (< 50 m). If it vanishes while still visible in that eye, that's an engine combined-frustum
   bug for this headset, and the work-around is to widen culling (e.g. `extra_cull_margin` on the city batches).
4. **Already fixed:** the mirror's old 5-point test against the nominal 75° Camera3D frustum (`mirror.gd:172-186`, 10-31).
   Rule for any future script cull: use `XRInterface.get_projection_for_view(i, …)` per eye, never `XRCamera3D.fov`,
   and add Valve's ~5° margin.

## 6. NYC: buildings with no bottoms / see-through walls

`city2_gen.gd:150-160` starts each wall at `y0 = max(min_h, nb)`, where `nb` is the edge's neighbour height from
`osm_district.py:195-221`. **The wall below `nb` is never built** (a party wall: the neighbour is assumed to cover it),
and if `nb ≥ h` the wall is skipped entirely. Shells are hollow and back-face culled, so any stretch where that
assumption is wrong shows the inside of the block: a missing bottom, or a see-through wall.

The tool decides `nb` from **three sample points (25 / 50 / 75 %) 0.8 m outside the edge**. Two failure shapes:
- the neighbour covers the middle but not the **ends** of the edge (corners, setbacks) → open wall at the ends;
- the neighbour stands **0.15–0.8 m away** (a light-well, a mapping gap) → *both* walls are dropped → a see-through crack.

Measured 11-04 against the shipped data (`/tmp/lodscan/party2.py`: 20 samples per party-wall edge; covered = inside
another footprint, at least as tall, 0.15 m outside or inside the edge):

| district | party-wall edges | edges with an open stretch | ~open wall | buildings |
|---|---|---|---|---|
| midtown_nyc | 1,997 | **194 (10 %)**, 11 wholly open | ~970 m | 153 of 557 |
| shinjuku_tokyo | 518 | 292 (56 %), 270 wholly open | ~3,500 m | 148 of 317 |
| dubai_downtown | 1,011 | 119 (12 %) | ~350 m | 43 |
| loop_chicago | 428 | 37 (9 %) | ~270 m | 31 |

At the tool's own 0.8 m distance, Shinjuku's wholly-open count drops 270 → 1. Those are real 0.15–0.8 m gaps with both
walls removed. Midtown's open stretches mostly stay (176 at 0.8 m), so they're the uncovered-ends case.

**Fix (data, no frame cost):** sample every ~1 m including the ends, at ≤ 0.2 m out. Store `e` as covered *intervals*
per edge (or split the edge at coverage changes), and build the wall over the uncovered intervals. Keep the wall
whenever in doubt: a hidden internal wall costs 2 triangles, a missing one costs the illusion. A regression test can run
the same check in `city2_test`. My coverage check is an approximation (point-in-polygon with heights, ignoring
`building:part` subtleties), so treat the counts as order-of-magnitude.

## 7. Sources

- Godot: [visibility ranges](https://docs.godotengine.org/en/stable/tutorials/3d/visibility_ranges.html) ·
  [occlusion culling](https://docs.godotengine.org/en/stable/tutorials/3d/occlusion_culling.html) ·
  [VRS](https://docs.godotengine.org/en/stable/tutorials/3d/variable_rate_shading.html) ·
  [MultiMesh](https://docs.godotengine.org/it/stable/tutorials/performance/using_multimesh.html) ·
  [DirectionalLight3D](https://docs.godotengine.org/en/4.3/classes/class_directionallight3d.html) ·
  [docs PR 8723 (5× shadow draws)](https://github.com/godotengine/godot-docs/pull/8723.diff) ·
  [renderer_scene_render.cpp](https://raw.githubusercontent.com/godotengine/godot/master/servers/rendering/renderer_scene_render.cpp) ·
  [GPU time issue #75346](https://github.com/godotengine/godot/issues/75346) ·
  [OpenXR visibility mask ext (mirror)](https://remotebranch.eu/Stowage/godot/src/commit/edde380e9cb1545d77e99bba99aaf55f55e7dc86/modules/openxr/extensions/openxr_visibility_mask_extension.h)
- Meta [PC VR performance guidelines](https://developers.meta.com/horizon/documentation/native/pc/dg-performance-guidelines/)
- Valve, Alex Vlachos, GDC "Advanced VR Rendering" / "Advanced VR Rendering Performance" (YouTube auto-transcripts
  `ld5LbDEyqYw`, `DdL3WC_oBO4`, kept untrusted in `/home/iris/untrusted/yt_lod/`): 5° conservative cull, hidden-area
  mesh ≈ 15 %, adaptive quality
- Epic [World Partition HLOD](https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition---hierarchical-level-of-detail-in-unreal-engine)
- Insomniac, Ruskin, GDC 2019 [Marvel's Spider-Man: A Technical Postmortem](https://gdcvault.com/play/1026495/-Marvel-s-Spider-Man) (topics only; no figures quoted)
- VRChat [mirror wiki](https://wiki.vrchat.com/wiki/Mirror) · [VRC_MirrorReflection](https://creators.vrchat.com/worlds/components/vrc_mirrorreflection)
- Combined-frustum bugs elsewhere: [UE PICO OpenXR](https://forums.unrealengine.com/t/pico-openxr-fix-sps-mono-culling-frustum-incorrectly-culling-edge-objects/2763119) ·
  [Unity OpenVR](https://issuetracker.unity3d.com/issues/openvr-culling-in-left-eye-slightly-off)
- Impostors: [wojtekpil/Godot-Octahedral-Impostors](https://github.com/wojtekpil/Godot-Octahedral-Impostors) (MIT, Godot 3) ·
  [Octahedral Impostor LOD (Godot 4, custom licence)](https://godotengine.org/asset-library/asset/edit/24616)
- Our data: `research/cities4_1101/perf/cities4_vr.txt`, `research/lumen_1101/perf/ab_table.md`, `research/city2_report_2026-10-31.md`

## ★ Zeke's design decision (2026-11-04, after reading the summary) — this OVERRIDES the plan above where they differ
- The "circle" he saw in New York was NOT the comfort vignette: the periphery had no lights and simplified textures while the centre was normal ⇒ a rendering-quality falloff (foveation-like). His eyewitness outranks the code-reading guess.
- **Inside the headset's REAL field of view: full quality at the player's own graphics settings — never turned down there.** Use each headset's actual per-eye FOV.
- A margin **10–15 % wider than the FOV: lower quality**; a further **~5 %: lower still**; **outside that: placeholders**, switched **seamlessly** to full detail as they come into view; **behind the player: essentially nothing but a super-simple stand-in.**
- **Multiplayer:** every client renders only what ITS player sees; the server keeps full detail/simulation only where players are (everyone in New York ⇒ the temple is placeholders; players split across the temple / New York / Dubai ⇒ each of those areas is live).
