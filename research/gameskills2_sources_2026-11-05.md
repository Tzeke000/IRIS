# Sources for the second batch of game-dev skills (2026-11-05)

Research for six skills under `.claude/skills/` (git-ignored, local): `godot-engine-pitfalls`, `vrgame-graphics-lookdev`,
`vrgame-avatars-ik`, `vrgame-audio-voice`, `game-minigames-open-world`, `game-design-core`. They fill gaps around the
first six `vrgame-*` skills (sources: `research/vrgame_skills_sources_2026-11-04.md`).

Tags, same as the first batch: **[measured]** someone ran it and reported numbers · **[doc]** official documentation,
standard or primary text · **[default]** a shipped engine's/game's chosen value · **[opinion]** a respected developer's
recommendation · **[anecdote]** forum/user reports · **[secondary]** a wiki/press/third-party summary. YouTube
transcripts were fetched as untrusted data (auto-subs, read only). Each skill's sections keep their own "Dead ends".
"Our game" references point at the separate (private) VR project and its harness, read only.

## godot-engine-pitfalls
- Godot docs, "Reducing stutter from shader (pipeline) compilations" (stable) **[doc]**: 4.4 ubershaders and pipeline precompilation (Forward+/Mobile only); the remaining stutter sources (material overrides, first Surface frame, mid-game feature changes such as MSAA, 2D); the pipeline-compilation monitors by step; the 4.5 shader baker.
- Godot blog, "UID changes coming to Godot 4.4" (Jan 2025) **[doc]**: `.uid` sidecars for scripts and shaders, commit them, move them with the file, path fallback "should not be relied upon".
- Godot docs, "Background loading" + `ResourceLoader` class ref **[doc]**: `load_threaded_request`/`get_status`/`get`; get-before-done blocks like `load()`.
- Godot docs, "Using Jolt Physics" (4.4+) **[doc]**: setting renames (`physics/jolt_physics_3d`), areas vs static bodies, kinematic contacts, estimated contact impulses, ray face_index (+25 % memory), single-body joint inversion, margins, Baumgarte position-only.
- Godot docs, "Exporting for dedicated servers" **[doc]**: dedicated_server tag forces `--headless`, Strip Visuals/Keep/Remove, `OS.has_feature`/`DisplayServer.get_name()`.
- Godot docs, command-line tutorial + godotengine/godot PRs #68461 + #90431 (`--import` implies --editor --quit; export flags imply it; 4.2 series, exact release unverified) **[doc]**.
- Godot docs, "OpenXR settings" / `OpenXRInterface` **[doc]** (via the existing vrgame-performance skill): VRS_XR on Mobile/Forward+; glow/bloom/DOF disable foveation.
- Godot class ref `MultiMesh` / `MultiMeshInstance3D` **[doc]**: format flags before `instance_count`, `custom_aabb`, single-AABB culling.
- Godot GDScript warning system (`debug/gdscript/warnings/*`, INFERENCE_ON_VARIANT, UNTYPED_DECLARATION) **[doc]**.
- Our incidents, **[measured]** (project at `/data/projects/vr_iris`): `research/client_hang_1102.md` (4.7.2 Forward Mobile worker-pool deadlock, minidump stacks, 3/6 vs 0/6 hang table, the LoadGate fix); `scripts/client_load.gd` header (650 ms + 170 ms first frames, 144 pipeline compiles, threaded prefetch net-negative); the 11-05 merge log in Iris's live list (stale class cache after the carry merge produced 4 cc fails + 6 parity fails; stale import-script imports after the vehicles merge, 74 files purged; shared git stash across worktrees); `harness/cc.sh` (the compile sweep pattern; a parse error hid for hours 10-23); `scripts/arrow.gd:395` (untyped literal vs `Array[RID]`).

## vrgame-graphics-lookdev

- **[doc]** Godot docs, OpenXR settings (`tutorials/xr/openxr_settings.html`). Foveation Level is Compatibility-only, VRS_XR is for Mobile/Forward+, Foveation Dynamic is Compatibility-only, and "disabled if post effects are used such as glow, bloom, or DOF". Supported §4, §8.
- **[doc]** Godot docs, Introduction to global illumination. LightmapGI/ReflectionProbe work on all renderers; VoxelGI/SDFGI/SSIL are Forward+ only; cost ranking; mobile = LightmapGI + probes. Supported §2.
- **[doc]** Godot docs, Using LightmapGI. Bake modes, directional lightmaps, shadowmask, texel size 0.2 default, probes subdiv 8, no reflections, max 8 LightmapGI nodes. Supported §2.
- **[doc]** Godot docs, Lights and shadows. Mobile limit of 8 omni + 8 spot per mesh and 256 + 256 per view (fixed); emission doesn't light other objects unless baked or SS-GI. Supported §2, §4, the checklist.
- **[doc]** Godot docs, Visibility ranges (HLOD). Fade modes Disabled/Self/Dependencies, margins, visibility parent, combines with mesh LOD. Supported §3.
- **[doc]** Godot docs, 3D antialiasing. Roughness limiter (Forward+ and Mobile, on by default); MSAA doesn't fix specular aliasing; TAA/FSR2 are Forward+ only and blur/ghost; alpha-to-coverage on materials. Supported §1, §5.
- **[doc]** Godot docs, StandardMaterial3D. Alpha Edge Blend = A2C, Edge Clip = A2C + alpha-to-one, MSAA ≥2×, the edge threshold below the scissor threshold. Supported §5.
- **[doc]** Godot docs, Volumetric fog. "Only supported in the Forward+ renderer"; quad fog works on Mobile. Supported §5.
- **[doc]** Meta developer docs, OS Fixed Foveated Rendering (Unity/Unreal/native pages, Dec 2024 and Apr 2026). Levels, "very noticeable" when High is used app-wide, ~6.5/11.5/21 % gains in an ALU-bound example, keep high-contrast items and text central, compositor layers unaffected, change levels only at transitions. Supported §4. *(Read through search summaries, not full pages.)*
- **[doc]** W3C, Understanding WCAG 2.1 SC 2.3.1. 3 flashes per second, general flash 10 % / 0.80, red flash definition, 0.006 sr area, fine-pattern exception. Supported §6.
- **[doc]** ITU-R BT.1702-1 (2018), via search snippets. 20 cd/m² opposing change with the darker below 160 cd/m²; risk for sequences over 5 s. Supported §6. *(Full text not read.)*
- **[secondary]** ASA rulings / Ofcom Guidance Note Annex 1 references. Harding FPA as the broadcast test; automated testing doesn't guarantee compliance. Supported §6. The Ofcom ¼-screen area figure is **unverified** (from memory).
- **[measured]** Vlachos, *Advanced VR Rendering*, GDC 2015 (Valve slides, via search summary). Normal-map variance into roughness per mip, a geometric roughness from normal derivatives, "sparkling", normal maps vs geometry up close. Supported §1.
- **[secondary]** Kaplanyan et al., specular AA, GDC 2016 (NVIDIA PDF, via summary). Toksvig → Hill → Vlachos lineage; specular sparkle especially visible in VR. Supported §1.
- **[secondary]** Interior mapping write-ups (80.lv, Alan Zucconi, UDK Gems) on van Dongen, *Interior Mapping*, CGI 2008. Ray-cast rooms in the shader, used in Spider-Man, Overwatch and UE. Supported §3. The paper itself wasn't read.
- **[secondary]** Octahedral impostor write-ups (arycama 2024, smailikhappy, Unity Industry Toolkit BakeImpostor). Hemi-octahedral layout, 3-frame blend, depth parallax; Brucks credited on UE forums. Supported §3.
- **[doc]** Meta blog, *Art Direction for All-in-One VR Performance*. Stylisation reduces technical pressure and the uncanny-valley risk; silhouettes; hair shells over alpha cards. Supported §7.
- **[anecdote]** Owlchemy at GDC 2017 (Shacknews write-up), "believability over realism"; Schwartz in a 2015 Engadget interview. Supported §7.
- **[secondary]** Valve Developer Community wiki (Source 2 lighting, HLA sky list). Alyx uses lightmaps + light-probe volumes + cubemaps. Supported §2.
- **[anecdote]** Unreal forum thread, "VR Anti-Aliasing Nightmare (TAA vs MSAA)" (Quest). A2C banding, distant foliage flicker, vendors recommend MSAA. Supported §5.
- **[measured]** Our project, read-only: `xr_render.gd:30-45`, `xr_session.gd:14-22`, `settings_manager.gd:24`, `project.godot:63-71`, `world.gd:445-460`, `city_kit.gd:155-166`, `city.gd:166-236`, `city_skyline.gd:214-218`, the HDR header (2048×1024), and the perf-1105 render-budget report. Supported "In our game".
- **[opinion]** My own arithmetic and judgement: px/deg for sky sizes (width/360, face/90), the fwidth-filtered window mask, the geometric-roughness shader sketch, ~100 ms flash ramps, "periphery is more flicker-sensitive", the mip-bias caution, and the BC7/BC5/BC4 split (standard practice, not re-verified this session).

### Dead ends

- No primary Half-Life: Alyx art/lighting GDC talk was found. The Alyx claims rest on the Valve wiki and interviews only.
- No Owlchemy talk transcript was found. Only press write-ups.
- No primary Ryan Brucks impostor write-up came up in search (his blog wasn't reached).
- Ofcom guidance text wasn't retrieved, so its area threshold is unverified.
- No source covers what the foveation boundary looks like (the "ring"). The explanation in §4 is reasoning from how VRS works, consistent with the 11-04 headset report.
- No VR-specific photosensitivity test standard was found.
- YouTube transcripts weren't used (time budget).

## vrgame-avatars-ik


### Solvers and body solving
- RootMotion Final IK VRIK docs — http://www.root-motion.com/finalikdox/html/page16.html and IKSolverVR class reference
  (`_1_1_spine`, `_1_1_arm`, `_1_1_leg`). **[doc][default]** solve order, bodyPosStiffness 0.55, bodyRotStiffness 0.1,
  neckStiffness 0.2, maxRootAngle 25°, shoulder yaw 45°/pitch −30°, bendToTargetWeight 0.5, moveBodyBackWhenCrouching 0.5,
  bent-rest-pose and don't-use-camera warnings. Procedural-locomotion defaults are NOT documented.
- Pärtel Lang, "Developer Perspectives: Character Animation in Dead and Buried" —
  https://developers.meta.com/horizon/blog/developer-perspectives-character-animation-in-dead-and-buried/ **[opinion/doc]**
  VRIK design: CoM/CoP balance stepping, toes anchored, "empirically-found mixture" for elbows.
- Resonite VRIK component — https://wiki.resonite.com/Component:VRIK **[doc]** Resonite uses a VRIK-based solver.
- VRChat developer on auto-footstep — https://feedback.vrchat.com/vrchat-ik-20/p/ik2-auto-footstep-not-using-idle-foot-position-as-base **[anecdote]** IK 2.0 uses VRIK auto-footstep.
- V-Sekai RenIK (MIT) — https://github.com/V-Sekai/renik **[default]** gait thresholds, crouch_ratio 0.4, shoulder 0.25, arm pole defaults.
- Godot "Design of SkeletonModifier3D" — https://godotengine.org/article/design-of-the-skeleton-modifier-3d/ **[doc]** modifier order, pose rollback.
- Godot "Inverse kinematics returns to Godot 4.6" — https://godotengine.org/article/inverse-kinematics-returns-to-godot-4-6/ **[doc]** IK class list, determinism.
- Godot 4.7 class docs (TwoBoneIK3D, BoneTwistDisperser3D, LimitAngularVelocityModifier3D, XRBodyModifier3D, XRHandModifier3D)
  — https://docs.godotengine.org/en/4.7/classes/ **[doc]**; `scene/3d/two_bone_ik_3d.cpp` @4.7-stable **[doc]** no stretch, pole required, end bone untouched.
- Unity Animation Rigging — only general knowledge, not fetched this session **[secondary]**.

### Legs, ML estimation
- Meta, "Inside-out body tracking and Generative Legs" — https://developers.meta.com/horizon/blog/inside-out-body-tracking-and-generative-legs/ **[doc]**.
- AGRoL, Du et al. 2023 — https://arxiv.org/abs/2304.08577, https://dulucas.github.io/agrol **[measured]** real-time diffusion full body from 3 points.
- QuestSim, Winkler et al. 2022 — https://arxiv.org/abs/2209.09391 **[measured]** RL policy + physics sim from HMD/controllers; code not public (per a reimplementation thesis, **[secondary]**).
- VRChat Upright ≈ 0.65 switch — https://feedback.vrchat.com/vrchat-ik-20/p/upright-065 **[anecdote]**.
- Rec Room "Avatar Embodiment" — blog login-walled; only search-result summaries **[secondary]**.

### Calibration, FBT, scale
- VRChat IK 2.0 features and options — https://docs.vrchat.com/docs/ik-20-features-and-options **[doc][default]** Lock Hip/Head/Both, `--custom-arm-ratio 0.4537`, height vs arm-span measuring, calibration range 0.6 m.
- VRChat FBT — https://docs.vrchat.com/docs/full-body-tracking **[doc]** "not asked to T-pose", calibration flow.
- VRChat creator docs, Avatar Scaling — https://creators.vrchat.com/avatars/avatar-scaling/ **[doc]** scale = target ÷ prefab eye height on root localScale, dial 0.2–5.0 m, Udon 0.1–100 m, sync to 3 decimals, smoothed remote transitions, remeasure viewpoint/voice. Does NOT document IPD/hand/collider scaling.
- VRChat "Adjusting your avatar's height" — https://docs.vrchat.com/docs/adjusting-your-avatars-height **[doc]** Action Menu dial, Match Eye Height.
- VRChat wiki IK — https://wiki.vrchat.com/wiki/IK **[secondary]** User Real Height, auto-footsteps/force-locomotion toggles.
- Prismic247 ScaledColliderSystem — https://github.com/Prismic247/ScaledColliderSystem **[doc]** VRChat player colliders don't scale with avatars; the prefab scales the world instead.
- Godot OpenXR body tracking / Vive trackers — https://docs.godotengine.org/en/latest/tutorials/xr/openxr_body_tracking.html **[doc]** tracker role paths.
- Drillis & Contini eye height ≈ 0.935 × stature, leg ≈ 0.53 × stature — via project research **[secondary]**.

### Speed
- Alexander 1984 "Walking and running" (Am Sci 72) and 1989 "Optimization and gaits…" (Physiol Rev 69:1199) **[measured/opinion]** dynamic similarity, Fr 0.2–0.3 walking.
- Kram, Domingo & Ferris 1997, J Exp Biol 200:821 **[measured]** walk→run at Fr ≈ 0.5.
- Bohannon 1997, Age and Ageing 26:15 **[measured]** comfortable walk 1.27–1.46 m/s.
- Weyand et al. 2000, J Appl Physiol 89:1991 **[measured]** top speed 6.2–11.1 m/s.
- (These four were gathered in project research `vr_stride_speed_2026-11-05.md`; not re-fetched today **[secondary]**.)

### Hands and fingers
- OpenXR 1.1 spec, standard pose identifiers (grip, aim, grip_surface) — https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html **[doc]**.
- XR_EXT_hand_tracking_data_source — https://registry.khronos.org/OpenXR/specs/1.1/man/html/XR_EXT_hand_tracking_data_source.html **[doc]** UNOBSTRUCTED vs CONTROLLER sources.
- araghon007/openxrhands — https://github.com/araghon007/openxrhands **[anecdote]** Knuckles curl changes with controller pitch.
- Unreal forum thread on Index curl vs hand pitch — https://forums.unrealengine.com/t/openxr-hand-tracking-input-not-consistent-for-different-hand-rotations-get-motion-controller-data/1339719 **[anecdote]**.
- Godot XR Tools `xr_tools.gd` palm/aim/grip offsets (−45° default grip rotation) **[default]**.

### Networking and physics bodies
- Meta Movement SDK networking — https://developers.meta.com/horizon/documentation/unity/move-networking/ **[doc]** skeleton compression, ~6 KB/s at 30 fps (search snippet was garbled; verify on the page). Unity-Movement v78 release notes: packets −25 % **[doc]**.
- Marrow SDK 1.2.0 (BONELAB) — https://github.com/StressLevelZero/MarrowSDK **[doc]** avatar ellipses × eye height; no licence, read-only.
- AvatarStatExtender (decompiled formulas) — https://github.com/EtiTheSpirit/AvatarStatExtender **[secondary]** mass toward 82 kg, stats vs 75 kg/1.638 m.
- LabFusion — https://github.com/Lakatrazz/BONELAB-Fusion **[doc]** 73 floats + 8 ellipses synced per avatar.
- Road to VR interview with SLZ — https://roadtovr.com/bonelab-custom-avatar-mod-levels-items-vehicles/ **[secondary]** stats from proportions, 1–4 m.

### Our own measurements
- `harness/xr/xr_ik.py` 120-pose sweep (old chain hinge error p95 55°, wrist bend to ~175°) — cited in `arm_ik.gd` header **[measured]**.
- S6 double-scale bug: giant walked 4.67 m/s instead of 4.00 — `rig_xr.gd` comment, avatarscale-1105 **[measured]**.
- Bundled avatar leg lengths (jennifer 0.552 m … bruno 1.115 m) — `vr_stride_speed_2026-11-05.md` **[measured]**.

### Opinion / judgment in the skill (not sourced)
Spine split weights, pole heuristic weights, step thresholds (0.18 × leg, 0.25–0.35 s), crouch/prone thresholds (0.65/0.4),
≤ 5 % arm stretch, 60–80 bytes per 3-point packet, "godot-xr-tools has no full-body IK" (my reading of the addon) **[opinion]**.

### Dead ends
- VRIK procedural-locomotion default values: not in public docs.
- VRChat docs: no statement of how IPD, hands and colliders behave under scale; behaviour inferred.
- Rec Room avatar blog: login-walled.
- QuestSim code/data: not released.
- A GDC talk specifically on BONELAB's body or a public Owlchemy body-ownership write-up: not searched for in the time box (hands-only claim is common knowledge, tagged secondary).
- No YouTube transcripts were used.
- Valve Index per-finger curl via the OpenXR interaction profile: none; only via hand tracking with the controller data source.

## vrgame-audio-voice

**[opinion]** practitioner or my own judgement · **[anecdote]** a single report · **[secondary]** a summary of a primary
source I could not read directly.

### Godot
- **[doc][default]** Godot class ref, AudioStreamPlayer3D: https://docs.godotengine.org/en/stable/classes/class_audiostreamplayer3d.html
  Supports every default (unit_size 10, max_db 3, max_distance 0, filter 5000 Hz/−24 dB, panning 1, doppler off,
  emission 45°/−12 dB, max_polyphony 1), the model descriptions, and that panning is a product with the project setting.
- **[doc]** *Sync the gameplay with audio and music*: https://docs.godotengine.org/en/stable/tutorials/audio/sync_with_audio.html
  Supports the playback_position + time_since_last_mix − output_latency formula, the "never go backwards" rule, the
  system-clock method, and that hardware and system clocks drift. It does not mention pitch_scale.
- **[doc]** *Audio streams*: https://docs.godotengine.org/en/stable/tutorials/audio/audio_streams.html
  Supports Area3D reverb bus sends, uniformity, doppler tracking (Idle/Physics), and that doppler and reverb are not
  supported on web in Sample mode. HRTF is not mentioned anywhere (no built-in HRTF is inferred from the feature list).
- **[doc][default]** ProjectSettings 4.3 (via search): `audio/driver/mix_rate` 44100 and `output_latency` 15 ms. The web
  overrides are 0 and 50. I checked these against 4.3 only, not 4.7.
- **[secondary]** godot issue #38215 (via search): ~23 ms real latency at the defaults. get_output_latency is constant
  unless the driver config changes.

### Spatializers / HRTF
- **[doc]** stechyo/godot-steam-audio README: https://github.com/stechyo/godot-steam-audio  Supports the feature list
  (occlusion, transmission, reflections, ambisonics, dynamic geometry), alpha status, Linux/Windows, the archive on
  2026-10-07, MIT licence, and that Steam Audio has proprietary parts. HRTF is not listed.
- **[secondary]** Nexus Resonance listing: https://store.godotengine.org/asset/movec/nexus-resonance/  Supports HRTF,
  Godot 4.6+ and MIT. There is one review, so I treat it as unverified.
- **[doc]** Meta XR Audio SDK features: https://developers.meta.com/horizon/documentation/unity/meta-xr-audio-sdk-features/
  Supports mono-only input, ambisonics for music and ambience, near-field <1 m, keeping the direct/reverb ratio under
  custom curves, and that binaural needs headphones.
- **[doc]** Steam Audio Unreal source docs and C API direct effect:
  https://valvesoftware.github.io/steam-audio/doc/unreal/source.html ·
  https://valvesoftware.github.io/steam-audio/doc/capi/direct-effect.html  Supports occlusion as the fraction of blocked
  rays, and transmission as a 3-band EQ applied only to the occluded part.

### Voice codec / transport
- **[doc]** RFC 7587 (Opus RTP): https://www.rfc-editor.org/rfc/rfc7587.html  Supports NB 8–12 / WB 16–20 / FB 28–40
  kbps at 20 ms, frame sizes 2.5–60 ms, FEC semantics, and DTX (continuous recommended).
- **[doc][secondary]** Xiph *Opus Recommended Settings* (TLS cert expired; content via search snippet):
  https://wiki.xiph.org/Opus_Recommended_Settings  Supports 20 ms as the default and a VoIP starting range of 10–24 kbps.
- **[doc][default]** libopus encoder CTLs: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoderctls.html
  Supports INBAND_FEC off by default (LPC layer only), PACKET_LOSS_PERC default 0, DTX off by default, complexity 0–10,
  and SIGNAL_VOICE/MUSIC. The claim that FEC needs loss % > 0 is my reading and is not stated there (✗ in the skill).
- **[doc]** WebRTC NetEQ design doc: https://webrtc.googlesource.com/src/+show/HEAD/modules/audio_coding/neteq/g3doc/index.md
  Supports deriving the target delay from inter-arrival stats, accelerate/preemptive expand, expand/merge, and FEC/NACK.
  The p95 + 20 ms recipe in the skill is my simplification **[opinion]**.
- **[doc][secondary]** ITU-T G.114 (2003): https://www.cs.columbia.edu/~andreaf/new/documents/other/T-REC-G.114-200305.pdf
  plus the 3GPP TS 22.105 annex (via search). Supports the 150 ms "most applications unaffected" figure and the 400 ms
  planning limit. The three-band labels are the 3GPP reading of it.
- **[doc][measured]** TwoVoIP: https://github.com/goatchurchprime/two-voip-godot-4  Supports MIT; libopus, RNNoise and
  SpeexDSP; RNNoise mono 48 kHz in 480-sample chunks at ~10 % of a core (author-measured, i5-1240P); Opus rates 8–48 kHz;
  the lag target; partial `push_input_chunk`; late-join headers. The 0.6 s lag default comes from a comment in our code,
  not the README.
- **[secondary]** IETF codec test results (cited in our ableton research): 96 kbps stereo is "barely perceptible",
  128 near-transparent.

### Social VR practice
- **[doc][default]** VRChat Udon player audio: https://creators.vrchat.com/worlds/udon/players/player-audio/  Supports
  voice gain 15 dB, far 25 m, volumetric 0, and the voice low-pass switched off for DJ audio.
- **[anecdote]** VRChat feedback "lower Voice Distance Far to ~10": https://feedback.vrchat.com/udon/p/lower-the-player-voice-distance-far-default-from-25-to-10
  plus a creator guide (4–10 m for small rooms).
- **[secondary]** VRChat *The Sound of Progress* (2019, Medium; 403 on fetch, snippet only): https://medium.com/vrchat/vrchat-the-sound-of-progress-3c4ab4fb5472
  Supports ~70 % volume outside the facing zone and the distance low-pass.
- **[secondary]** Rec Room *Directional Voice Chat* (2020; 401 on fetch, snippet only): https://blog.recroom.com/posts/2020/10/26/directional-voice-chat
  Supports facing-weighted voice, "no basis on any real physical effect", and that it can be turned off per room.
- **[doc]** VRChat Udon video sync example (via our stage research): https://creators.vrchat.com/worlds/examples/udon-example-scene/udon-video-sync-player
  Supports (trackTime, serverTime) sync. **[anecdote]** ProTV issue #13 says seeks blip, so users turn sync off.

### Music sync / drift
- **[doc]** RFC 5905 (NTPv4) offset and delay formulas (standard, quoted from memory, not fetched this session).
- **[secondary][anecdote]** pitch JND: 0.1 % ≈ 1.7 cents; 3–6 cents cited as thresholds (arXiv 2307.00084 overview,
  forums). RetroArch rate-control thread: 0.5 % was not detected in ABX, but flipping corrections warble:
  https://forums.libretro.com/t/default-audio-rate-control-delta-value-0-005-is-bad-and-quite-audible/15129/9
- **[opinion]** the seek threshold of 0.25–0.5 s (also flagged as a suggestion in our stage research).

### Loudness
- **[secondary]** ITU-R BS.2434 (2018): https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BS.2434-2018-PDF-E.pdf
  Supports typical game loudness of −24 ±2 LKFS console and −18 ±2 portable. **The Sony ASWG-R001 text itself was not
  found.** The −1 dBTP figure is general practice (EBU R128 family), not confirmed from ASWG.
- **[doc]** EBU R128: −23 LUFS (standard; not re-fetched).

### PA physics (from our own research, already sourced there)
- `research/locomotion_daynight_pa_mass_2026-10-26.md` §3: ESTA 58 dB voice, Bose L1 Pro8 112/118 dB, ISO 9613-1 air
  absorption, and ANSI S12.60 +15 dB SNR **[doc]** as cited there. The telephone band 300–3400 Hz is standard
  **[doc]** (ITU-T G.712 lineage, not re-fetched). The megaphone EQ, slapback, ducking times and crowd voice caps are
  **[opinion]**.

### In-game evidence (read-only)
- **[measured]** `radio_fx.gd` (makeup +2.9 dB measured; distortion −8 dB in band), `acoustics.gd` header (Godot
  max_distance tail: halved at 30 m, dead at 60), `voice_mic.gd` header (gate failure measured at −33…−45 dBFS).

### Dead ends
- Xiph Opus wiki: TLS certificate expired, so I used the search snippet.
- VRChat Medium blog: 403. Rec Room blog: 401. Both are snippet-only.
- Sony ASWG-R001 PDF: not publicly found. Only secondary references exist.
- No GDC talk specifically on Rec Room/VRChat voice proximity. Found *Environmental Audio and Processing for VR* (Sony,
  GDC 2015) https://gdcvault.com/play/1021958/ and *Spatial Audio: From Screen-Based to World-Based* https://gdcvault.com/play/1028965/
  but did not read them (video/paywall).
- godot-proposals #8609 is about exposing AudioStream to GDExtension (merged for 4.4). It enabled Steam Audio extensions
  and says nothing about HRTF.
- No YouTube transcripts were used.

## game-minigames-open-world


### Official docs
- **[doc][default]** Fortnite/UEFN Island Settings — https://dev.epicgames.com/documentation/en-us/fortnite/island-settings-in-unreal-editor-for-fortnite — Game Start Countdown default 3 s; Auto Start Off ⇒ public games start on a 90 s countdown; Force Start at Max Players (default Off, delay 0); Minimum Players default 8; Max Players default 16; Total Rounds default 1; Time Limit default 5 min; Join in Progress = Spawn on Next Round / Spawn / Spectate / team; Allow Spectating Other Teams; Last Standing Ends Game.
- **[doc]** Fortnite Box Fight tutorial — https://dev.epicgames.com/documentation/fortnite/box-fight-1-set-up-the-game-in-unreal-editor-for-fortnite — late joiner is a spectator until the next round, then spawns.
- **[doc]** UEFN round_settings_device Verse API — https://dev.epicgames.com/documentation/uefn/verse-api/fortnitedotcom/devices/round_settings_device?lang=en-US — RoundBeginEvent, EndRound, DisableEndRoundConditions (round end can be fully script-driven).
- **[doc]** VRChat Late Joiners — https://creators.vrchat.com/worlds/udon/networking/late-joiners — late joiners get latest synced variables via OnDeserialization and correct owners; "Events are not replayed"; keep important state in synced variables.
- **[doc]** VRChat Udon networking (master) — https://docs.vrchat.com/docs/udon-networking — master owns never-transferred objects; new master chosen on leave; prefer ownership checks over master checks. Creator docs: don't rely on any particular player becoming master.
- **[doc]** VRChat Networking Specs & Tricks — https://creators.vrchat.com/worlds/udon/networking/network-details — ~11 KB/s Udon out, manual sync ~280 KB/serialization, continuous ~200 B; network events not repeated for late joiners. **[anecdote]** VRChat staff in a bug report called the 11 KB/s figure out of date (https://feedback.vrchat.com/bug-reports/p/vrchat-incorrectly-reporting-amount-of-data-being-sent-across-network-causing-im).
- **[doc]** Roblox TeleportService / teleport guide — https://create.roblox.com/docs/projects/teleport.md , https://create.roblox.com/docs/reference/engine/classes/TeleportService — lobby place → reserved server per match (ReserveServerAsync / TeleportAsync with ShouldReserveServer or ReservedServerAccessCode); teleports from server scripts; secure data via data stores, not teleport data.
- **[doc]** Rockstar support: Next Job Vote Screen matchmaking — https://support.rockstargames.com/articles/6UAQQL0nk9Vh5tvR5x8VTB/new-matchmaking-added-to-the-next-job-vote-screen-in-gta-online — lobbies with Matchmaking Open merge with others voting the same job; "Remain Host" option.
- **[doc][default]** ScreamingBedWars plugin (via search summary of its docs/changelog) — https://hangar.papermc.io/ScreamingSandals/ScreamingBedWars/versions/0.2.43 — dynamic lobby countdown example 60 s → 30 s at 4 players → 15 s at 8. Plugin, not Hypixel.

### Wikis / secondary
- **[secondary]** GTA Wiki, Jobs in GTA Online — https://gta.wiki/w/Jobs_in_GTA_Online — blue coronas as world entry points, host starts the job, On Call players added; vote screen = 5 same-category jobs + Random, Refresh, Replay, Freemode; can leave. No timer or tie rule stated.
- **[secondary]** Gorilla Tag wiki / Wikipedia / Steam dev comment — https://gorillatag.fandom.com/wiki/Infection , https://en.wikipedia.org/wiki/Gorilla_Tag — Infection needs ≥4 players, else plain Tag; random first infected; round ends when all infected, then restarts. Late-join behavior unclear (wiki says joiners become infected).
- **[secondary]** VRChat wiki, Murder 4 — https://wiki.vrchat.com/wiki/Community:Murder_4 — min 3 players, up to 24; instance master sets Game Modifiers; team circles in the lobby. No late-join/spectator detail.
- **[secondary]** Among Us wiki (Voting / Ejection) — https://among-us.fandom.com/wiki/Ejection , https://breezewiki.discard.no/among-us/wiki/Voting — discussion 0–120 s, voting 0–300 s (0 = unlimited); tie or skip-majority ⇒ nobody ejected; anonymous votes option.
- **[secondary]** Minecraft wiki mirror, Spleef — https://wiki.ronlab.site/content/minecraftwiki_en_all_maxi_2025-11/Spleef — last player standing, break the floor under opponents, no block placing; rules vary by server; origin 2009 (attribution "citation needed").
- **[secondary]** Spleef plugins (planetminecraft, bukkit configs, BungeeCord spleef) — min-players-to-start setting, 90 s lobby countdown in one, force-start commands.
- **[secondary]** Fall Guys lobby size — https://www.pcgamesn.com/fall-guys/player-count , https://mp1st.com/title-updates-and-patches/fall-guys-update-1-20-jumps-out-for-lobby-size-changes-this-february-21 — 60 players chosen by playtesting for readability/perf; cut to 40 in 2023 for faster matchmaking and more round variety. (Context only; not used for numbers in the skill.)
- **[anecdote]** Roblox devforum on Player.Idled / AFK — https://devforum.roblox.com/t/do-you-know-how-to-kick-out-players-if-they-are-afk-for-5-minutes/1311610 — platform AFK kick ~20 min; Idled fires after ~2 min. Not confirmed in official docs.
- **[anecdote]** UEFN forum threads on Join in Progress bugs — https://forums.unrealengine.com/t/join-in-progress-spawn-on-next-round-not-working/2277711 — late joiners stuck spectating / spawning immediately; test your settings.

### Ours
- **[measured]** /data/projects/vr_iris/project/scripts/spleef_rules.gd, spleef.gd, spleef_arena.gd, net.gd, powers.gd, player_flat.gd (read-only) — constants and file:line refs in the skill's "In our game".
- **[secondary]** research/spleef_minigame_2026-11-04.md — prior Spleef design research and the owner's decisions.

### Opinion (no source; marked as such in the skill)
- Countdown 3–5 s, friends-hub lobby 10–30 s, results 5–10 s, AFK unready after 30–60 s, rejoin grace 30–60 s, announcer priority order, VR no-forced-spectator-camera, participation rewards per completed round.

### Dead ends
- Hypixel pre-game countdown / minimum players / queue design: no citable Hypixel source found (only third-party plugins). Not quoted.
- GTA Online vote-screen timer and tie-break: not stated on the wiki or Rockstar support.
- Rec Room Paintball / Rec Room Studio ready-up, team pads, join-in-progress: nothing citable found.
- Roblox official "implement rounds" gameplay-scripting tutorial: URL tried returned 404; community tutorials only.
- No GDC talk/postmortem found on Fall Guys round flow or "designing social hubs" with minigame lobbies.
- VRChat docs: no rate-limit guidance for master migration; Murder 4 late-join handling undocumented.
- YouTube transcripts: not used (time).

## game-design-core


### Loops and frameworks
- Hunicke, LeBlanc & Zubek, *MDA: A Formal Approach to Game Design and Game Research* (AAAI workshop, 2004). PDF fetched and text extracted: users.cs.northwestern.edu/~hunicke/MDA.pdf. **[doc]** Supports the M→D→A definitions, the designer vs player perspective, and the 8 aesthetics (Sensation … Submission), with Charades and Quake as examples.
- Jaime Griesemer interview, "Half-Minute Halo" (Joystiq, reposted Engadget, 2011-07-14). **[doc]**/**[anecdote]** Supports the full 3 s / 30 s / 3 min loop explanation. The original "30 seconds of fun" line comes from the *Making of Halo 2* documentary and is quoted via Halopedia **[secondary]**.
- Jesse Schell, *The Art of Game Design: A Book of Lenses* (2008; 3rd ed. 2019). **[doc]** Supports the lens-based review method. Cited from general knowledge and not re-read this session.
- Csikszentmihalyi, *Flow* (1990), and Jenova Chen, "Flow in Games" (CACM 2007). **[secondary]** Supports challenge tracking skill and player-chosen difficulty. Not re-read.

### Game feel and juice
- Martin Jonasson & Petri Purho, "Juice it or lose it" (talk, 2012). YouTube `Fy0aCDmgnxg`; auto-subs read as untrusted data. **[doc]** Supports "maximum output for minimum input", the order of the layers (tween, squash, sound, music, particles, shake, eyes), "eyes on anything" as cheap personality, and the source of the term "juicy" (Gamasutra, "How to prototype a game in under 7 days").
- Jan Willem Nijman (Vlambeer), "The Art of Screenshake" (INDIGO 2013), YouTube `AJdEqssNZ-U`. The transcript fetch hit **HTTP 429**. Content comes from the Game Developer write-up ("Vlambeer co-founder shares advice on building better action games") and DK Liao's itch.io recreation breakdown. **[secondary]** Supports ~30 tweaks, permanence, camera kick, shake and hit pause. "Sleep" as Nijman's name for hitstop is from memory. **Unverified.**
- Steve Swink, *Game Feel: A Game Designer's Guide to Virtual Sensation* (2008), ch. 1, definition via Wikipedia "Game feel" and Liz England's review. **[doc via secondary]** Supports the definition: real-time control, simulated space, polish.
- Meta VR comfort guidance (predictable camera motion), already cited in `vrgame-interaction-physics`. **[doc]** Supports the rule against shaking the camera in VR.
- VR hitstop (freeze only the object, never the world) is **[opinion/inference]**. I found no source that tests it.

### Onboarding
- Miyamoto & Tezuka on World 1-1 (Nintendo/Eurogamer video, 2015). Read via Game Developer, Nintendo Life and AV Club coverage. **[secondary]** Supports Goomba over Koopa, the mushroom placement, and "gradually and naturally understand".
- Koichi Hayashida on kishotenketsu (Gamasutra/Game Developer, "The secret to Mario level design", 2012), plus Wikipedia and GMTK. **[secondary]** Supports one core idea per level and introduce → develop → twist → conclude.
- *Portal* developer commentary (Valve 2007), via the Portal Wiki and nicegamehints transcripts as summarised in search. **[doc via secondary]** Supports concepts that are "very deliberately" introduced, testers seeing themselves through a portal, the orange-portal misconception room, and the funnel ride for fatigue.
- Jay Stelly, "Physical gameplay in Half-Life 2" (GDC 2006), per the Game Developer post-GDC feature. **[secondary]** Supports the early gravity-gun areas with weak enemies, many props, and safe failure.

### Difficulty
- Celeste Assist Mode: Vice ("celeste-assist-mode-change-and-accessibility", "celeste-difficulty-assist-mode"), Game Developer, and New Normative. **[secondary]**/**[doc quotes]** Supports the options (10% speed steps to 50%, stamina, invincibility, skip), the rename from "Cheat Mode", and criticism of the 2019 "intended" wording.
- The "fail fast, restart fast" threshold is **[opinion]**. I found no measured number.

### World design
- Kevin Lynch, *The Image of the City* (MIT Press, 1960). **[doc]** Supports the five elements. Cited from general knowledge and not re-read.
- BotW field level design: CEDEC 2017 (Fujibayashi, Yonezu), Matt Walker's English translation in gist.github.com/idbrii/e39fe96279aa1670319bfa521d907399, plus Game Developer and GMTK summaries. **[secondary of a doc]** Supports linear tower routing failing in playtests, heat maps, "gravity" with visibility ranking, the triangle rule at three scales, and rectangles that hide. Also supports the correction that the GDC 2017 talk was Dohta's and is a separate session.
- Disney "weenie" and Sleeping Beauty Castle: disneyblog glossary, Mouse Planet, Walt Disney Co. blog, theoryofthemeparks wayfinding post. **[secondary/anecdote]** Supports a landmark at the end of a path and helping guests stay oriented. The castle's forced perspective is well documented but **not re-verified** this session.
- "BotW uses gravity to move you forward" (Game Developer 2019) is **[opinion]** and wasn't used as a fact.

### Playtesting
- Mike Ambinder, "Valve's Approach to Playtesting: The Application of Empiricism" (GDC 2009 slides, Steam CDN PDF, text extracted). **[doc]** Supports "what people do—not what they say", "presence of observers can bias results", think-aloud "unprompted and uncorrected" with its pros and cons, and the biases in Q&A.
- GMTK, "Valve's 'Secret Weapon'" (substack). **[opinion/secondary]** Supports the "designs as hypotheses, playtests as experiments" quote and Gabe Newell's "fresh eyes".
- Chet Faliszek, "Playtesters aren't idiots, it's you" (Game Developer). **[opinion]** Supports fixing the design, not the player (headline only).

### Scope
- Derek Yu, "Finishing a Game" (makegames.tumblr.com, 2010). **[secondary]** Read via the Game Developer summary, the GameMaker interview, and a Substack quote. Supports finishing as a skill, crude tools being fine, and non-developer feedback.
- Greybox, vertical slice, cut list and Pareto are **[default]** industry practice and **[opinion]**, with no single source.

### In our game (read-only)
- `/data/projects/vr_iris/project/scripts/` headers: `destinations.gd`, `city_skylanes.gd`, `city_skyline.gd`, `spleef.gd`, `spleef_rules.gd`, `impacts.gd`, `flight.gd`, `powers.gd`, `setup_wizard.gd`, `world.gd`. **[doc]** (our code). Nothing was modified.
- `/data/projects/vr_iris/research/vr_game_best_practices_2026-10-23.md`: headings skimmed only, to avoid overlapping its comfort, interaction and perf sections.

### Dead ends
- derekyu.com/makegames/finishing.html → 404. The original Tumblr post wasn't fetched.
- Art of Screenshake auto-subs → YouTube HTTP 429.
- WebFetch on PDFs returns binary. Workaround: extract the saved PDF with pypdf (`python -I`).
- "Kleenex testers" as a Valve term: no source found. It's treated as general industry slang.
- An HL2 Ravenholm sawblade "teaching" claim: no developer source found, so it wasn't used.
- No primary transcript of the Halo 2 documentary line was found.

