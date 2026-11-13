# How shaders (and the "make it look nicer" stuff) should load into LUMEN — 2026-11-13

Zeke's ask (11-13 01:0x): *"some games let you start playing with the main shaders done, and the other stuff that makes
textures and everything look better loads in as you play."* Research only; nothing was run on the box/lanes.
Helper: shaderload_1113. Engine context: Godot 4.7.2 custom build (ed1daf0bf, carries the #123312 Mobile mutex fix).

## TL;DR

1. **The pattern has two halves, and Godot already does the shader half for us.** The game draws immediately with a
   *generic* shader (an **ubershader** / fallback) while the *specialized* fast versions compile on background threads
   and swap in when ready ("async pipeline compilation with ubershader fallback": Dolphin 2017, Godot 4.4 #90400,
   Unreal "PSO precaching", Unity 6 `GraphicsStateCollection.WarmUpProgressively`). The other half, *"textures look
   better as you play"*, is **streaming**: low mips/low LODs first, then higher detail as it loads (texture/mip
   streaming, mesh LOD/HLOD).
2. **In Godot the specialized shader does not look better.** It looks identical and runs faster. So for us, "quality
   streams in" is really about **textures, LODs and districts**, not shaders. The only shader cost we must pay up front
   is the **ubershader pipeline per mesh+material** (otherwise it's a hitch in the headset). That's what our cover's
   compile counters already wait for (`client_load.gd:53-54`; they deliberately leave out SPECIALIZATION, which is the
   background part).
3. **Three things we are not using yet, in order of value:**
   (a) the **shader baker** (4.5+): bakes SPIR-V/DXIL into the PCK, so a first launch skips shader compilation. PR
   #102552: D3D12 went from over a minute to ~2 s. ⚠ It **doesn't work in `--headless` exports**, and ours are headless
   (`harness/tools/engine_patch/export.sh:37`). It is also off in `export_presets.cfg` (no `shader_baker` key).
   (b) the **feature-prewarm list**: MSAA level, shadow modes, reflection probes, multiview etc. are only compiled when
   first seen (Godot docs), so warm them under the cover with the player's real settings.
   (c) **mip-level texture streaming** exists, but only in **Godot 4.8** (#113429, merged 2026-08-27, Forward+/Mobile
   only, not Compatibility). Not in 4.7.2. Until we move up, "textures sharpen in" has to be our own LOD/visibility
   code, which we partly have (`lod_kit.gd`, chunked `city2.gd` build).
4. **The Compatibility renderer (the client-smoke VM) has none of this.** No ubershaders, no background pipeline
   compiles. Every first-seen shader compiles **on the main thread**. The only help is the GL program-binary disk
   cache (`drivers/gles3/shader_gles3.cpp:547 _load_from_cache`). That's why tonight's stall was so severe there
   (`state/helpers/smokehang_1113.report.md`: the loading camera drew the whole city, so a software-GL VM compiled and
   drew everything at once). **The smoke VM measures the worst case, not the 3060's behaviour.** Keep that in mind when
   using it as a gate.
5. **Plan:** keep "compile what's visible at spawn behind the cover" (we do this). Add feature-prewarm, the shader
   baker (needs a non-headless export step), and a **post-cover background warm queue** that spreads the rest (other
   districts, effects, LOD-high meshes) over play at ≤ 1 job per quiet window (LoadGate already does this). Add a
   **first-60-s hitch gate** to the smoke. Mip streaming waits for 4.8.

---

## 1. What the pattern is called and how it works

| Approach | What the player sees | Who uses it | Source |
|---|---|---|---|
| **Compile everything at the loading screen** ("shader precompilation step") | long first launch, then smooth (if the list was complete) | many PC ports; Unreal bundled PSO cache; Unity `ShaderVariantCollection.WarmUp` | Unreal "PSO Precaching" doc; Unity manual `shader-prewarm` |
| **Ubershader fallback + async specialization** | plays at once; the first draw of a material uses a generic, slower but already-compiled shader, and the fast version swaps in silently | Dolphin "Hybrid Ubershaders" (2017); Godot 4.4+ (#90400) | dolphin-emu.org/blog/2017/07/30/ubershaders (from memory, the page returned 403 to the fetcher); godotengine/godot#90400 |
| **Async compile + skip/substitute the draw** | the object is missing or drawn with a default material for a few frames | Unreal `r.PSOPrecache.ProxyCreationDelayStrategy` 0 = skip (default), 1 = default material; Dolphin "async skip drawing" | Unreal PSO Precaching doc |
| **Precache at asset load, not draw** | the cost moves into loading (threadable) | Unreal (PostLoad of a primitive component), Godot 4.4+ (ArrayMesh load + surface-cache creation) | Unreal doc; #90400 "Pipeline compilation at loading time" |
| **Texture/mip streaming** | blurry textures sharpen within ~seconds | every open-world engine; Godot 4.8 #113429 | godotengine/godot#113429 |
| **Mesh LOD / HLOD streaming** | far things are low-poly proxies; detail loads as you approach | Godot importer LODs + visibility ranges; Spider-Man tiles (see `research/vr_render_lod_streaming_2026-11-04.md` §2.4) | |

Two key facts, both cited:
- **Unreal's own guidance is still "hold the loading screen until outstanding precompiles reach zero"**
  (`FShaderPipelineCache::NumPrecompilesRemaining()`), because skipping or swapping materials is unacceptable for some
  things (landscape). Async is the safety net, not the target. That holds even more in VR: a skipped draw is
  missing geometry in your face, and a stalled frame is a reprojected judder.
- **Unity 6 says the old warmup (`ShaderVariantCollection.WarmUp`) can create the wrong PSOs on Vulkan/D3D12/Metal**
  because it lacks render state. Its replacement, `GraphicsStateCollection`, records real PSOs during a play-through
  and warms them, in full or progressively (`WarmUpProgressively` returns a job handle; it needs
  `SystemInfo.supportsParallelPSOCreation`). The lesson for us: **warm by actually drawing the real thing with the
  real settings**, which is what our cover does, rather than by listing shaders.

## 2. What Godot 4.7 offers (per renderer)

### 2.1 Ubershaders + pipeline precompilation (Forward+ and Mobile only)
- Mechanism (#90400, DarioSamo, merged 2024-10-03 for 4.4): an ubershader is the normal shader with **specialization
  constants read from push constants**, so one pipeline per mesh+material can draw any combination. At draw time the
  renderer asks for the specialized pipeline. If it isn't compiled yet, it **starts it on a background thread and draws
  with the ubershader**.
  Verified in 4.7.2 source: `render_forward_mobile.cpp:2567-2606` tries the specialized pipeline without waiting
  (`PIPELINE_SOURCE_SPECIALIZATION`), then the ubershader *with* waiting (`PIPELINE_SOURCE_DRAW`). If the ubershader
  isn't compiled either, the frame **stalls**. That's the "Draw" counter, which the docs say "should not happen".
  `pipeline_hash_map_rd.h` queues compiles as `WorkerThreadPool` template tasks named "PipelineCompilation".
- Ubershader pipelines are precompiled at two points (#90400; docs "Reducing stutter…"):
  **Mesh** = when an ArrayMesh loads (threadable: a background `ResourceLoader` load pays it off the main thread) and
  **Surface** = the first frame an instance is in the tree, **even invisible**. Mobile's mesh path waits for the
  ubershaders it started unless `p_background_compilation` (`render_forward_mobile.cpp:410-415`).
- **Not precompiled** (docs list): **2D canvas items**; **material overrides** (not visible at mesh load); and
  **features first enabled mid-game**: MSAA level (and differing MSAA between viewports), reflection probes, separate
  specular, motion vectors, normal/roughness buffers, lightmaps, VoxelGI, SDFGI, **multiview** (XR!), 16/32-bit shadows,
  omni shadow modes. Also **effects spawned only during play** (keep a hidden instance).
- Ubershaders can be turned off by a **driver workaround** on some GPUs (`disable_ubershaders` from
  `RD::get_driver_workarounds()`, `render_forward_mobile.cpp:3560`). On such a device every new pipeline is a stall.
  Not the case on the 3060 as far as I know; it's an argument for keeping the cover anyway.
- Specialized compiles cost **framerate, not stutter**. The docs: "Specialization: background work that doesn't stutter
  but can lower framerate if many occur per frame". So they still eat CPU workers in the first seconds of play.
- **The 4.7 Mobile deadlock** (pitfalls skill §11; #123312 merged 2026-10-08, milestone 4.8, label `cherrypick:4.7`;
  Forward+ equivalent #105138 in 4.5): `ShaderData::is_valid()` holds the global shader mutex while waiting on a compile,
  and HIGH-priority ubershader pipeline tasks fill the pool waiting for the same mutex. We ship the patched engine. Keep
  the bootloop proof per engine bump. **Any background-warm design must keep LoadGate's rule "one threaded mesh job,
  never beside a prewarm draw"** (`scripts/kit/load_gate.gd:10-17`) at least until we're on an official build with the
  fix. It also protects the frame time.

### 2.2 The two disk caches (both on by default, neither set in `project.godot`)
- **Shader cache** (`rendering/shader_compiler/shader_cache/enabled`): compiled shader *variants* (SPIR-V/DXIL) in
  `user://`. Saves the *shader* compile on the 2nd launch.
- **Pipeline cache** (`rendering/rendering_device/pipeline_cache/enable`, `save_chunk_size_mb`): Vulkan
  `VkPipelineCache` on disk. Saves *pipeline* creation on the 2nd launch. ProjectSettings.xml (4.7.2): to simulate a
  first run you must turn it off **and** delete the driver's own cache. The docs: driver caches are "usually deleted
  when the driver is updated", and pipelines are hardware+driver specific, so they **can't be shipped** to players.
- That's why our cover already has "first launch ~30 s" vs "last time N s" (`client_load.gd:43-46,326`,
  `FIRST_WARM_S` = 3060 measured 31 s on 11-11).

### 2.3 The shader baker (4.5+, #102552, DarioSamo/RandomShaper)
- An export plugin that scans scenes/resources for shaders and bakes them into the **intermediate format**
  (SPIR-V for Vulkan, DXIL for D3D12) in the PCK, used as a read-only fallback shader cache. **It removes the
  first-launch shader compile, not pipeline compiles.** The docs: "not a replacement for pipeline precompilation".
  Gains: small on Vulkan on many-core CPUs, huge on D3D12 ("over a minute … to ~2 s", #102552). Costs: slower export,
  PCK + a few MB.
- Limits (docs): off by default (per preset: *Shader Baker > Enabled*); bakes only the driver(s) the preset targets
  (Windows: Vulkan + D3D12); **unsupported with Compatibility, web, and `--headless` exports**. One forum report
  (4.6) of "No loader found for resource" export errors with it on.
- **For us:** our export is `--headless` (`harness/tools/engine_patch/export.sh:37`) and the Windows preset has no
  `shader_baker` option set (`export_presets.cfg`, grep finds none). So **we're not baking today, and turning the box
  on would silently bake nothing.** We need a windowed export (e.g. editor on a virtual display with a real Vulkan
  device: the V100 on iris-home, or the tower). **Unverified:** whether a Linux editor can bake **DXIL** for the
  Windows preset (D3D12 baking may need the Windows-side DXIL signer). Test that before relying on D3D12 bakes.
  `gpu_profile.gd:24` offers `d3d12` as a player renderer, so that's where the baker would matter most.

### 2.4 Compatibility (GLES3), which the client-smoke VM uses (per Iris's brief)
- **No ubershaders, no pipeline precompile, no background compile** (docs + #90400 "No changes are expected for
  Compatibility"). The GL driver compiles/links on the **main thread** on first use. Godot's only help is the
  program-binary disk cache (`shader_gles3.cpp:547-610`, `glProgramBinary`) on the 2nd launch.
- Docs' advice for it: the legacy way, i.e. **render every material/particle at least one frame in the frustum during
  loading**. Our cover does exactly this, which is also why a wrongly-masked loading camera (tonight) is so expensive
  there.
- Our `COUNTERS` (`client_load.gd:53`) read `RENDERING_INFO_PIPELINE_COMPILATIONS_*`, which **are RD-renderer counters.
  On Compatibility I'd expect them to stay 0**, so "quiet frames" is trivially satisfied and only the frame-time
  "calm" check (`client_load.gd:277`) really gates the cover there. That's my inference from the RD-only docs, **not
  verified**: check a Compatibility boot log's `compiles under it` (`client_load.gd:256`). If it's always 0, the smoke VM
  isn't testing the compile-quiet logic at all.

## 3. Texture / asset streaming in Godot 4.7

| Need | 4.7.2 native? | What we have / need |
|---|---|---|
| Mip-level texture streaming ("textures sharpen in") | **No.** Added in **4.8** (#113429, merged 2026-08-27, in 4.8 dev 5, 2026-09-10): "Texture2D Streamed" import type, min/max res, a VRAM budget, `TextureStreaming` singleton, project setting needs a restart. Forward+/Mobile only; Compatibility has no feedback path. A follow-up fix exists (#123662 triplanar/custom feedback). | Wait for 4.8 (or a later backport). Today textures load at full res. Our cities' cars are baked vertex colour (no textures), so the gain is mostly in the temple/props. |
| Mesh LOD | **Yes.** Importer `meshes/generate_lods` + `mesh_lod_threshold` | `lod_kit.gd` gives code-built meshes LOD twins + small-object far ranges (`lod_kit.gd:1-15`); the quality governor coarsens `mesh_lod_threshold` (`quality_governor.gd:8`). |
| Visibility ranges / HLOD | **Yes** (`visibility_range_*`, fade modes) | Used in ~174 places in `scripts/`; FarLayer impostors (`far_layer.gd`). |
| Background resource loading | **Yes**, `ResourceLoader.load_threaded_request` (3 call sites) | Gated by `LoadGate` (one job client-wide, compile-quiet before each, `load_gate.gd`). Measured 10-31: boot is CPU + pipelines, not disk (`client_load.gd:15-17`). |
| World/chunk streaming | **No** built-in | Ours: `city2.build_async` spreads a district build over frames with a per-frame budget and keeps it hidden until done (`city2.gd` "THE SAME BUILD SPREAD OVER FRAMES"); `city2.warm` builds a tiny district under the ground for 2 s to compile its pipelines (`city2.gd:68-84`); deferred loads (cars, flycars, tree species) go through LoadGate (`client_load.gd:37-41`). |

## 4. Recommended plan for LUMEN

### 4.1 What happens behind the cover (keep, plus three additions)
Already right: cover first (`world.gd:47`), build with yields, render the real world behind the cover until compile
counters are quiet for 20 frames and frames are calm (`client_load.gd:265-300`), effect prewarms (`shards.gd:940`,
fire_fx), the city2 warm district, LoadGate. Add:
1. **Feature prewarm with the player's real settings.** Before the cover lifts, make sure every *feature* the session
   will use has been seen once **in the XR viewport** (multiview, its MSAA, shadow atlas size/modes, any reflection
   probe, the mirror's SubViewport with *its* MSAA). Rule: **settings changes only under a cover** (the docs; our
   governor already avoids recompiling knobs, `quality_governor.gd:6`). The mirror is a separate viewport: if its MSAA
   differs from the main view's, the docs say that alone stutters.
2. **"Things you'll see in the first 60 s" list.** Everything visible from spawn plus the hands, held items, UI boards,
   the avatar (VRM/MToon), and **2D canvas** (text boards, debug panel: canvas is never precompiled) gets one draw
   behind the cover. Effects that can trigger early (fire, sparks, shards, splash) stay in their prewarm queues.
3. **Shader baker on the Windows export** (2.3). Needs a non-headless export. Payoff = shorter *first* launch only.
   Measure it: first-launch cover time with a wiped `user://` shader cache **and** driver cache (the docs' recipe).

### 4.2 What to defer and stream in during play
- **Other districts/cities, their effects, LOD0 of far things, flycars/cars/trees:** already deferred via LoadGate.
  Extend it into an explicit **background warm queue** after the cover lifts. Each item = (load on a worker → add
  hidden → one draw at -400 m / in a 2×2 SubViewport like `shards.gd` `_touch_root` → free), **one at a time, only
  after N quiet frames, only when the last 90 frames are under budget, paused while the player is moving fast**. This
  is Unity's `WarmUpProgressively` idea done our way.
- **Prefer warming by destination**: before a teleport/travel to a city, run its `city2.warm`-style mini-district for
  a few seconds (a travel fade is a natural mini-cover).
- **Textures:** until 4.8, ship sensible max sizes per import + mipmaps on. On 4.8, try "Texture2D Streamed" for big
  temple/prop textures with a VRAM budget from `gpu_profile.gd`'s estimate.

### 4.3 Keeping VR smooth while the background works
- Specialization compiles and warm draws compete for the same `WorkerThreadPool` (`max_threads.windows=8`,
  `project.godot:100-107`). Budget: **≤ 1 warm job per quiet window**. Rule: if a frame misses budget × 1.3
  (`quality_governor.gd` MISS_X), pause the queue for 2 s.
- Never let a warm draw be *seen*: render it 1 cm big / under the floor / in a disabled-update SubViewport, as shards
  and city2 already do.
- On a GPU where ubershaders are disabled by a driver workaround, or on Compatibility: **everything new stalls**. There
  the cover must hold longer (the existing "calm" rule does this) and the background queue should stay **off**
  (warm only under covers/travel fades).
- Keep the hang watchdog (`hang_watchdog.gd`). A background warm is exactly where the 4.7 deadlock bit us (11-06,
  `load_gate.gd:10-17`).

### 4.4 How to test it (no headset)
1. **Cold-cache first boot time**: wipe `user://` shader cache + set `pipeline_cache/enable=false` via override.cfg
   in a scratch worktree (+ clear the NVIDIA cache on the tower when testing there). Record `[load] cover off after
   … s` (`client_load.gd:256`). Do 3 runs. Report the baseline vs baker-on vs feature-prewarm-on.
2. **Hitch count in the first 60 s after the cover lifts** (new smoke check): per-frame ms + per-frame delta of all five
   `PIPELINE_COMPILATIONS_*` counters (boot_probe.gd already reads all five, `boot_probe.gd:16-18`). **Gate:**
   0 frames > 2× budget with a DRAW/SURFACE/MESH/CANVAS compile in them, and ≤ 2 frames > 50 ms total. SPECIALIZATION
   counts are logged, not gated. A DRAW count > 0 is always a bug (docs).
3. **Warm run** (2nd launch): the cover must be ≥ 3× shorter than cold. If not, the disk caches aren't working
   (path/permissions on the player's PC).
4. **Smoke VM caveat:** Compatibility + software GL measures *main-thread* compile cost and draw cost, not the
   ubershader path. Keep it as a "boots and gets online" gate. Put the hitch gate on the **real GPU** (tower, Vulkan
   Mobile; also D3D12 once offered). Verify first whether the compile counters are 0 on Compatibility (§2.4).
5. Regression guard for tonight's class (per the "every root cause gets a guard" rule): smokehang's
   `farlayer_cam_unit` + the runtime guard in `client_load.gd` cover the loading camera. Add a cover-time assert:
   `compiles_under_cover` > 0 while `state == "building"` (before the build finishes) = something drew the world
   behind the cover too early.

### 4.5 Build order
1. **Measure first** (cheap): the 60-s hitch log + cold/warm cover times on the 3060 (Mobile/Vulkan) and the smoke VM.
   Check the Compatibility counter question. This tells us whether anything below is worth it.
2. **Feature prewarm + first-60-s draw list** (biggest likely hitch win in the headset; small code in `client_load.gd`).
3. **Background warm queue** after the cover (extends LoadGate; reuse shards/city2 draw tricks) + **travel-fade warm**
   per destination.
4. **Windowed export step + shader baker** for Windows (first-launch time; prove DXIL from Linux or bake on the tower).
5. **Hitch gate in the smoke** (after 1–3 so the threshold has a baseline).
6. **Engine 4.8** (when stable + the deadlock fix is native): drop the patched engine, try mip streaming.

## Sources
- Godot docs, "Reducing stutter from shader (pipeline) compilations" (latest/4.7):
  https://docs.godotengine.org/en/latest/tutorials/performance/pipeline_compilations.html
- godotengine/godot#90400 "Ubershaders and pipeline pre-compilation" (4.4): https://github.com/godotengine/godot/pull/90400
- godotengine/godot#102552 "Add shader baker to project exporter" (4.5): https://github.com/godotengine/godot/pull/102552
- Godot 4.5 release notes: https://godotengine.org/releases/4.5/
- godotengine/godot#113429 "Implement texture mip-level streaming" (4.8): https://github.com/godotengine/godot/pull/113429 ;
  Godot 4.8 dev 5 article: https://godotengine.org/article/dev-snapshot-godot-4-8-dev-5/ ; proposal godot-proposals#3177
- godotengine/godot#105138 (Forward+ mutex scope, 4.5), #123312 (Mobile, merged 2026-10-08, cherrypick:4.7)
- Godot 4.7.2 source: `servers/rendering/renderer_rd/forward_mobile/render_forward_mobile.cpp` (2567-2606, 410-415, 3560),
  `servers/rendering/renderer_rd/pipeline_hash_map_rd.h`, `drivers/gles3/shader_gles3.cpp` (547-610),
  `doc/classes/ProjectSettings.xml` (pipeline_cache, shader_cache)
- Unreal Engine docs, "PSO Precaching": https://dev.epicgames.com/documentation/en-us/unreal-engine/pso-precaching-for-unreal-engine
- Unity 6 manual: https://docs.unity3d.com/Manual/shader-prewarm.html , https://docs.unity3d.com/Manual/shader-pso-trace.html ,
  `GraphicsStateCollection` https://docs.unity3d.com/ScriptReference/Rendering.GraphicsStateCollection.html
- Dolphin, "Ubershaders: A Ridiculous Solution to an Impossible Problem" (2017): https://dolphin-emu.org/blog/2017/07/30/ubershaders/
  (403 to the fetcher tonight; summarised from prior knowledge)
- Project: `scripts/client_load.gd`, `scripts/kit/load_gate.gd`, `scripts/city2.gd`, `scripts/shards.gd`,
  `scripts/lod_kit.gd`, `scripts/quality_governor.gd`, `scripts/gpu_profile.gd`, `scripts/boot_probe.gd`, `project.godot`,
  `export_presets.cfg`, `harness/tools/engine_patch/export.sh`; tonight's `state/helpers/smokehang_1113.report.md`.
