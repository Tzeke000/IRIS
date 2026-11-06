# Sources for the 11-06 skill batch (godot-engine-pitfalls extension, vr-multiplayer-merge-test-deploy, vr-anticheat-plausibility)

Skills live under `.claude/skills/` (git-ignored, local to the server); this file is the tracked record of what backs them.
Tags as in `gameskills2_sources_2026-11-05.md`: **[measured]** we ran it · **[doc]** official docs/spec/source ·
**[secondary]** third-party write-up · **[anecdote]** forum/user report. "Our game" = the separate private VR project and
its harness (read only); the measured lessons come from the 11-05/11-06 helper reports in `state/helpers/` (local).

## godot-engine-pitfalls (§5 rewritten, §11–§18 new)
- godotengine/godot PR #123312 "Reduce Forward Mobile singleton mutex scope to prevent shader compilation deadlock" — merged 2026-10-08, milestone 4.8, labels bug / cherrypick:4.7 / topic:rendering; links #102877, #123054, #123060; the author reports stock 4.7.2 hanging on Quest 3 cold launches **[doc]** (fetched 11-06).
- godotengine/godot PR #105138 "Renderer: Reduce scope of mutex locks to prevent common deadlocks" — merged 2025-04-15, milestone 4.5 (the Forward+ fix #123312 ports) **[doc]**.
- Godot 4.7.2-stable source: `scene_shader_forward_mobile.cpp` (is_valid under singleton_mutex), `shader_rd.cpp`, `worker_thread_pool.cpp`, `pipeline_hash_map_rd.h` (HIGH-priority pipeline tasks) **[doc]** — read by bootlock_1106.
- Our bootloop measurements: official pool 2 = 16/16 hung, pool 4 ≈ 1/12, pool 8 0/20 on the server (but a hang on the Windows desktop at 8); patched pool 2 = 0/18; the game-side LoadGate 10/10 hung (main-thread holder) **[measured]** (bootlock_1106, bootlock2_1106, towerperf_1106).
- Windows template cross-built with llvm-mingw 20261104 (github.com/mstorsjo/llvm-mingw, Apache-2.0 WITH LLVM-exception) + Godot's `misc/scripts/install_*.py` deps; an ANGLE-less first build was silent **[measured]** (wintemplate_1106).
- Patched editor packs the official templates when `custom_template` is empty; `.text`-section verify **[measured]** (engineswap_1106).
- Godot class reference `WorkerThreadPool` (stable): "Every task must be waited for completion … so that any allocated resources inside the task can be cleaned up"; ERR_BUSY for a task waiting on an older one; ERR_INVALID_PARAMETER if already awaited **[doc]**.
- Exit heap corruption bisect (autoload set, 3/3 → 0/6 after claiming the task; headless clean vs windowed abort) **[measured]** (exitcrash_1106).
- Godot docs "Background loading": `load_threaded_get` before completion blocks like `load()` **[doc]**. The "threaded load stalls without the main loop" figure (IN_PROGRESS after 150 ms, 213 ms block) is **[measured]** (bootchunks_1106); the engine-side reason was not traced.
- godotengine/godot PR #99895 (Jolt module, merged for 4.4): StaticCompoundShape; shape changes committed only when the body is in a tree; in-tree changes rebuild the compound; deferred compound planned **[doc]**. godot-jolt/godot-jolt issue #660 (in-tree shape edits slow at 16–64 shapes; build out-of-tree) **[anecdote]**.
- Godot 4.7.2 `modules/jolt_physics/shapes/jolt_shape_3d.cpp` `JoltShape3D::try_build()` builds once and caches `jolt_ref` **[doc]** (read 11-06).
- Compound/private-space/off-tree numbers (cars 492 → ≤ 27 ms chunks, Kowloon 293 → 28 ms; 2,560 tile bodies ≈ 1.2 ms/tick) **[measured]** (bootchunks_1106, r4land_1106).
- Khronos OpenXR spec, `xrRequestExitSession` / `xrEndSession` man pages (1.0 + 1.1): xrEndSession only in STOPPING else XR_ERROR_SESSION_NOT_STOPPING; xrRequestExitSession walks the session through STOPPING to EXITING **[doc]**.
- Godot 4.7.2 `modules/openxr/openxr_api.cpp`: EXITING → `on_state_exiting` (emits instance_exiting), INSTANCE_LOSS_PENDING "TODO … We should queue exiting Godot" **[doc]** (read 11-06). Godot class reference `OpenXRInterface`: signals instance_exiting / session_stopping / session_loss_pending; `render_target_size_multiplier` "Must be set before the interface has been initialized" **[doc]**.
- SteamVR logs + Windows event 1000 signatures at exit (vrserver in the NVIDIA D3D user-mode driver; "didn't quit in time"; safe-mode add-on block) **[measured]** (exitcrash_1106).
- Godot 4.7.2 `modules/openxr/extensions/openxr_fb_foveation_extension.cpp`: swapchain foveation info only if `foveation_level > 0` at creation (l.156); `is_foveation_with_subsampled_images_enabled` flips live (l.250); `xrUpdateSwapchainFB` (l.304) **[doc]** (read 11-06); the head-locked symptom + `xr fovlive` 35/35 **[measured]** (integ_1105, foveation_1105).
- Meta OS Fixed Foveated Rendering docs (level changes noticeable) **[doc]**, via foveation_1105's reading.
- Render region vs scaled 3D buffer ("custom region must be contained within the framebuffer rectangle") **[measured]** (xrfix-1104 commit in the project).
- Godot command-line tutorial: `--rendering-method` valid values forward_plus/mobile/gl_compatibility; `--rendering-driver` is the driver only **[doc]**; Forward+ observed in the log when only the driver was passed **[measured]** (towerperf_1106).
- Pose-per-frame swim haptics fix (4–9× spikes) and the wall-clock grab gate (P1/P2 at 2–4 fps; `slow` probe) **[measured]** (triage_1106, deployfix_1106).
- SteamVR null-headset standby + `power.turnOffScreensTimeout` / `pauseCompositorOnStandby` **[measured]** (towerperf_1106).
- Engine A/B triage (8 suites, 0 engine-caused; draws identical on official vs patched) **[measured]** (triage2_1106).
