# Sources for the `vrgame-*` skills (2026-11-04)

Research for six skills under `.claude/skills/vrgame-*` (multiplayer-architecture, performance, interaction-physics,
crossplay-desktop, dedicated-server, testing). Each source is tagged with what it supported and its kind:
**[measured]** someone ran it and reported numbers · **[doc]** official engine/runtime documentation ·
**[default]** a shipped engine's chosen value · **[opinion]** a respected developer's recommendation ·
**[anecdote]** forum or player reports. YouTube transcripts were fetched as untrusted data (auto-subs, read only).

## Networking / architecture
- Glenn Fiedler, "Networked Physics in Virtual Reality" (Oculus-sponsored; gafferongames.com and the Meta developer blog), plus the sample `github.com/unitycoder/oculus-networked-physics-sample` (BSD + patent grant) **[measured]**: authority + ownership sequences, host arbitration, 60 Hz state, <256 kbps/player, 100 ms jitter buffer, held objects as id + relative transform, 16-frame force-rest, 1 m/s depenetration, quantize both sides.
- Glenn Fiedler, "State Synchronization", gafferongames.com **[measured]**: priority accumulator values, 4–5 frame jitter buffer, visual error smoothing factors, at-rest handling.
- Glenn Fiedler, "Snapshot Interpolation", gafferongames.com **[measured]**: buffer ≈ 3× send interval, 350/150/85 ms at 10/30/60 pps, Hermite vs linear, extrapolation failure.
- Valve Developer Community, "Source Multiplayer Networking" + "Lag Compensation" **[default]**: 66/30 tick, `cl_interp 0.1`, interp_ratio, 1 s rewind history.
- Godot docs, "High-level multiplayer" **[doc]**: transfer modes, channels, authority, MTU warning, "treat all client input as untrusted".
- Godot class ref, `MultiplayerSynchronizer` (4.x) **[doc]**: visibility filters, replication/delta intervals.
- Photon Quantum 3 "Quantum XR" technical sample, doc.photonengine.com **[doc/opinion]**: hands as final positions, local hand override, PID physics hands, PhysicsHands collision layer.
- Photon Fusion 2 "VR Shared" + Meta hands-sync addon docs **[doc]**: rig as nested network transforms, finger-bone compression.
- "Multiplayer Networking for Games with Glenn Fiedler", Game Dev Unchained livestream (YouTube `Do2Hk3jKokM`, transcript) **[opinion]**: dedicated servers vs P2P, bandwidth scaling with players².
- edgegap.com, "Game Backend Deep Dive – Overwatch" **[secondary]**: Overwatch predicts abilities, but "can't trust the client with simulation authority". Used only for background.

## Performance / rendering
- Alex Vlachos (Valve), "Advanced VR Rendering", GDC 2015 (YouTube `JO7G38_pxU4`, transcript read) **[measured + opinion]**: 11.1 ms, 1.4× scale, 457 Mpix/s, 15.3 px/deg, MSAA 4–8×, normal-map limits in the tracked volume, running start (+15 %), 5° wider culling.
- Alex Vlachos, "Advanced VR Rendering Performance", GDC 2016 (gdcvault 1023522; slides on slideshare; Road to VR live blog) **[opinion]**: adaptive quality knobs, fixed foveated rendering / radial density masking.
- Meta, "PC performance guidelines" (developers.meta.com …/native/pc/dg-performance-guidelines) **[doc]**: 500–1,000 draws, 1–2 M tris, stereo doubles draws, ASW not a substitute.
- Godot docs, "Setting up XR" **[doc]**: Mobile renderer for desktop VR, v-sync off, raise physics ticks.
- Godot docs, "A better XR start script" **[doc]**: refresh-rate selection, physics = headset rate, session states, pose_recentered. (Note: its pause-the-tree-on-visible advice does NOT apply to multiplayer.)
- Godot docs, "OpenXR settings" + `OpenXRInterface` class ref (4.4) **[doc]**: foveation vs VRS_XR, glow/bloom/DOF disable foveation, frame synthesis not on Forward+, render_target_size_multiplier before init.
- Godot docs, "XR full screen effects" **[doc]**: screen-texture reads disable XR optimisations.
- Godot docs, "Optimizing 3D performance" **[doc]**: visibility ranges, LOD, occluders, MultiMesh, no auto-instancing of transparents.
- Godot docs, "Using physics interpolation" + quick start (4.4) **[doc]**.
- Godot blog, "Dev snapshot: Godot 4.6 dev 5" and UploadVR coverage **[doc/news]**: D3D12 default for new projects, OpenXR 1.1, D3D12 + OpenXR since 4.5.
- Reddit r/godot thread "My VR game is suffering from vertical jitter" **[anecdote]**: fixed by physics ticks = display refresh.

## Interaction / physics / comfort
- Meta, "Locomotion best practices", "Locomotion comfort & usability", "Locomotion design techniques" (developers.meta.com/horizon/design/…) **[doc/opinion]**: snap turns, vignettes, real-world speeds 1.4/2.8 m/s, quantised velocity, predictability, comfort presets, height rules.
- Meta IWSDK locomotion "slide" docs **[doc, SDK-specific]**: vignette fade ~10 Hz, only while moving.
- Godot docs, "Room scale in XR" **[doc]**: head–body gap → fade to black, never move the origin to block a step, two rig architectures.
- godot-xr-tools docs, "Pickup function" (godotvr.github.io) + Godot "Introducing XR tools" **[doc]**: velocity samples, impulse factor, ranged grab, physics layers.
- HurricaneVR docs, "Physics hands" (cloudwalker2020.github.io/HurricaneVR-Docs) **[doc]**: joint-based PD hands, MaxDistanceBehaviour (prevent/drop/sweep), throwing centre of mass.
- Another-Axiom/GorillaLocomotion, `Player.cs` (GitHub, MIT; read as untrusted text, not run) **[code]**: kinematic sphere-cast hands, `maxArmLength 1.5`, `unStickDistance 1`, velocity history.
- PhysX 5 docs, "Implicit drives" (nvidia-omniverse.github.io) **[doc]**: joint drives as implicit PD controllers that are stable at high gains.
- Malcolm Nixon, "Godot VR Physics Hands" (YouTube `THaRSaUu-xA`, transcript) **[demo]**: per-finger bone colliders, world-scale issues.
- Bastiaan Olij, "Godot XR First Person Controller part 1 – physical movement" (YouTube `v-jLvxlDSOY`, transcript fetched, skimmed only) **[demo]**.
- Godot docs, "Using Jolt Physics" (4.4) + godot-jolt README/settings **[doc]**: Baumgarte position-only correction, settings names, non-determinism warning.
- Road to VR "VR Design Unpacked: Half-Life: Alyx"; Engadget interview **[secondary]**: gravity gloves always on, avoiding linked two-hand interactions. Background only.

## Crossplay
- Schell Games blog, "The future of VR and PC crossplay" **[opinion]**: free-aim advantage, asymmetric mechanics, platform-agnostic design.
- PocketGamer.biz interview with Rec Room **[opinion]**: "feel … originally designed for whatever platform they're on", input as the biggest challenge.
- Steam community threads on Rec Room VR vs screen mode **[anecdote]**: VR-only room requests, fatigue asymmetry.
- Reddit dev thread on VR/flatscreen crossplay **[opinion]**: "roughly equivalent to designing two great games", VR-first with flat support from day one.
- Prior in-project research: `/data/projects/vr_iris/research/crossplay_desktop_vr_2026-10-23.md` (the skill condenses it; not re-researched).

## Dedicated server / security
- Godot docs, "Exporting for dedicated servers" (stable + 4.5) **[doc]**.
- Godot blog, "ENet DTLS encryption" (3.x era) + `ENetConnection` class ref (4.2) **[doc]**; Godot forum threads on DTLS cert errors **[anecdote]**: `verify=false` accepting any certificate.
- Gorilla Tag mod-menu / EAC pages (mostly low-quality SEO sites) **[anecdote]**: used only for "client-side checks are weak, server validation needed". No specific claim was relied on.

## Testing
- Khronos news, "Monado 21.0.0 is officially an OpenXR conformant implementation" **[doc]**: conformance on a simulated device.
- Meta, "XR Simulator" intro + keyboard input docs (developers.meta.com/horizon/documentation/…/xrsim-*) **[doc]**: WASD/mouse/gamepad, recorded tracking playback, the standalone simulator supports any OpenXR app.
- Godot Meta Toolkit manual, "XR Simulator" **[doc]**: launching the Meta XR Simulator from Godot.
- Godot Asset Library "Godot XR Simulator" (fork of Cafezinhu/godot-vr-simulator, MIT) **[tool]**.
- helpmetest.com "OpenXR testing strategies" **[secondary, low confidence]**: its CLI snippets looked unreliable, so not used.

## Codebase (read-only)
- `/data/projects/vr_iris/project` (Godot 4.7.2): `project.godot`, `export_presets.cfg`, `scripts/{net,props,prop,posepack,remote_avatar,rig_xr,player_flat,xr_session,quality_governor,gpu_profile,layers,strength,web_sling,powers,flight,movement_armswing,settings_manager,world}.gd`; `/data/projects/vr_iris/harness` (`vr.sh`, `mp/`, `xr/`, `probes/`, `known_flakes.txt`, `run/history.tsv`); prior research `vr_game_best_practices_2026-10-23.md`, `fullstack_audit_2026-11-03.md`, `mp_stress_coop_plan_2026-11-03.md`, `coop_grab_design_2026-11-03.md`; `profiles/vr_iris.md`.

## Dead ends (searched, nothing usable)
- **Rec Room netcode GDC talk**: the only Rec Room GDC talk found (2018, "Room for Everyone") is about UGC, not netcode.
- **Normcore ownership docs**: the search returned only unrelated packages and a Normcore 2 serialization page; the ownership guide wasn't reached.
- **Boneworks hand-controller devlog (Stress Level Zero / Brandon Laatsch)**: no primary source found; PD-hand claims rest on HurricaneVR, Quantum XR and the PhysX docs instead.
- **Half-Life: Alyx GDC talk on interactions**: none found; only reviews and interviews.
- **Overwatch GDC 2017 talk transcript** (`8QHHVpBiG-I`) and **Rocket League GDC talk** (`ueEmiDM94IE`): yt-dlp subtitle fetch failed (exit 1); not used.
- **Gorilla Tag official locomotion write-up**: none; read the MIT source instead.
- **Monado "qwerty" driver docs**: not confirmed by search. The harness already uses Monado with scripted controllers, so it wasn't needed.
- **Numeric snap-turn / vignette values from Meta**: Meta's pages give no numbers beyond the IWSDK SDK-specific ones.
