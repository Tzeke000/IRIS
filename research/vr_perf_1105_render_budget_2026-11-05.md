# VR render budget, perf-1105: what was cut, what's left to 8 ms (2026-11-05)

Helper report for Iris. Branch **`perf-1105`** (main merged in after xrfix-1104 landed) in both the game (`/data/projects/vr_iris/project`) and the harness. It's
committed, **not merged, not deployed**. Plan: `research/vr_render_lod_streaming_2026-11-04.md` (+ Zeke's design section)
and `research/vr_view_dependent_rendering_2026-11-04.md`. Mirror, render scale and foveation belong to `xrfix-1104` and
weren't touched.

**How it was measured.** The game in VR against the simulated headset (Monado, the V100 renders both eyes), using
`vr.sh xr renderbudget`. Auto quality is off, the time of day is fixed (phase 0.12, mid-morning, sun only), the mirror is
off, and each fixed vantage is measured ×3 for 150 frames. The headset view's draws are split into **colour** (VISIBLE)
and **shadow** passes. The swing is a 25 m/s web-swing path through Lumen's avenue, recorded per frame. Draw and primitive
counts carry over to the tower; **the V100's GPU ms do not** (relative only). Attribution: `RB_ATTR=1` hides each
subtree in turn (`xr_probe perf_attr`).

## Step 0: who owned the temple's draws (main, before)

At the spawn and the table, the **sun's shadow passes were 72 % of the draws and 80 % of the prims**:

| vantage (main) | colour draws | shadow draws | colour prims | shadow prims |
|---|---|---|---|---|
| temple_spawn | 613 | 1,367 | 0.96 M | 2.95 M |
| table | 632 | 1,477 | 0.92 M | 3.34 M |

Owners, largest first (Δ when hidden, temple_spawn):
- **320 firefly motes**: each a separate default SphereMesh (~4 k tris) and every one a shadow caster in all 4 splits.
  About 700 shadow draws and 2.4 M shadow prims (the world-level `@MeshInstance3D#` group).
- **Shadows of small or indoor things**: chess men (440 shadow draws looking south), the playground (277), and the
  bunk room room's furniture (179), which sits under a roof 40 m away.
- **Lumen (`City`)**: ~215–256 **colour** draws from the temple and ~0 shadow (it's past 100 m). About 10 % of the
  total, not the "bulk" the plan doc guessed.
- The moon's shadow was still rendered at opacity 0 through twilight, and so was the sun's from 0° to −0.8°.

## What changed (game, `perf-1105`)

1. **Shadows (step 2):** sun + moon now use **2 splits / 60 m** (was 4 / 100 m; split 1 = 0–15 m, no blend).
   `shadow_enabled` is now false whenever the shadow has faded out. New `scripts/render_diet.gd`: anything under 0.6 m,
   every Label3D, and anything under 2.2 m inside the roofed rooms (avatar room, bunk room, gallery) casts no sun shadow.
   Players, skinned meshes and SHADOWS_ONLY proxies are never touched. Props get the same rule at `Props.register`.
   The motes are now one low-poly MultiMesh with no shadow (same seed, same places).
2. **Only the destination you're in, at full detail (step 1):** new `scripts/city_skyline.gd`, an HLOD **stand-in for all
   of Lumen**. Every piece ≥ 8 m is merged into one mesh per 110 m kit cell (1,766 pieces → 42 meshes, 45 k verts,
   ~0.1 s to build). It casts no shadows and is drawn by **the city's own facade shading, baked per vertex** (same
   floors, bays, lit-window hash, day/night), so a tower looks the same in both. `city.gd`:
   - **Not in Lumen** (the temple, the yard, a real city): the stand-in is shown and the full city is **hidden**.
     Visuals only: colliders, props, movers, web anchors and both gates keep working.
   - **In Lumen**: full city near you; the stand-in takes over past **300 m**. The full city's big pieces end at 330 m,
     and the two overlap there, drawn identically. **Occlusion culling is on**, using inset occluder boxes for
     ground-standing towers (22 cells). `destinations.gd` now keeps occlusion on for Lumen as well as the real cities.
   - Hysteresis: in Lumen = on its ground **and** > 95 m from the temple centre; out = < 85 m, or 40 m past its ground.
     A/B switches: `IRIS_NO_LUMEN_LOD=1`, `IRIS_NO_LUMEN_OCC=1`.
   - **The temple, same rule:** stations' pieces with no range of their own end at 60 m (≤ 2 m) / 120 m (≤ 8 m),
     cross-faded (900 pieces). From Lumen and the high vantages only the architecture is drawn.
3. **Ranges fade, never pop (step 3):** the city kit's detail classes (220 / 400 m) and City v2's module batches use
   `visibility_range_fade_mode = SELF`. **Verified on the Mobile renderer** (desktop probe on Xvfb): a 100 m / 20 m range
   fades 1.0 → 0.84 → 0.60 → 0.35 → 0 across 95–130 m, even for an unshaded custom shader that never writes ALPHA.
   Without it, it snaps 1 → 0. Godot draws a batch see-through **only while it's inside its fade band**.

## Before → after (colour + shadow draws of the headset view; mean of 3 repeats; V100 GPU ms relative only)

Before = `main` (84ec412) + only the probe instrumentation (worktree `wt_p5_base`). After = `perf-1105`. Both were run on
the same harness, vantages and day phase.

| vantage | before: colour + shadow = total | after | Δ draws | prims before → after | V100 GPU ms before → after |
|---|---|---|---|---|---|
| temple centre, facing N (Lumen) | 386 + 586 = **972** | 156 + 150 = **306** | **−68 %** | 1.75 M → 0.42 M | 3.21 → 2.91 |
| temple centre, facing S | 580 + 839 = **1,420** | 513 + 123 = **636** | **−55 %** | 1.84 M → 0.57 M | 3.33 → 3.18 |
| temple spawn (the old 1,957 vantage) | 629 + 1,424 = **2,054** | 296 + 253 = **549** | **−73 %** | 4.13 M → 0.54 M | 3.86 → 3.08 |
| table | 649 + 1,530 = **2,179** | 346 + 270 = **617** | **−72 %** | 4.49 M → 0.66 M | 3.92 → 3.18 |
| mirror area (mirror OFF) | 242 + 311 = **553** | 218 + 127 = **346** | −37 % | 0.82 M → 0.44 M | 2.30 → 2.34 |
| Lumen avenue (street) | 211 + 124 = **335** | 157 + 48 = **205** | −39 % | 0.12 M → 0.10 M | 1.69 → 1.69 |
| Lumen north street | 136 + 109 = **246** | 107 + 45 = **153** | −38 % | 0.21 M → 0.15 M | 1.80 → 1.78 |
| Lumen high, looking back over everything | 1,738 + 24 = **1,763** | 300 + 10 = **310** | **−82 %** | 2.05 M → 0.18 M | 3.34 → **1.52** |
| NYC side street | 102 + 102 = **205** | 108 + 93 = **201** | −2 % | 0.17 M → 0.22 M | 2.09 → 2.17 |
| NYC avenue | 114 + 88 = **202** | 117 + 72 = **190** | −6 % | 0.11 M → 0.13 M | 1.78 → 1.84 |
| **swing path** (25 m/s, ~470 frames, per frame) | mean **933**, worst **2,465** | mean **174**, worst **283** | **−81 % / −89 %** | — | 2.71 → 1.86 |

The repeats agree to within ±1–3 % on draws and ±0.05 ms on GPU. Notes:
- **NYC barely moves.** Real cities already streamed, and their shadows were already 2 splits. The extra prims and
  +0.06–0.08 ms come from the new **cross-fade band**: City v2 modules stay drawn (see-through) a little past their old
  hard cut. That's the price of "no pop". It can be trimmed by narrowing the margin (8 % of range today).
- **The swing's biggest one-frame jump** is the path's instant 180° turn: 1,621–1,637 draws before → 133–135 after.
  An earlier path flew *through* the Donut tower's base, which put the camera inside an occluder; that's fixed.
- **Process step (scripts) p95 during the swing:** 19.5–23.2 ms → 15.9–16.3 ms on the sim. That's a V100-host number;
  the tower's will be lower.
- The temple's ranges don't change the temple vantages (stations are within 60 m). They matter from Lumen and the high views.

## Zeke's FOV-ring design: what's in, what isn't (Godot 4.7.2)

| his piece | done how | status |
|---|---|---|
| full quality inside the real FOV | engine frustum culling already uses each eye's runtime projection; nothing here lowers quality in view | ✅ (unchanged; foveation is xrfix's) |
| a 10–15 % margin at lower quality, +5 % lower still | **not built.** No engine (Godot/Unreal/Unity) does angle-based LOD; a head turn (300–600 °/s) brings the margin into the centre faster than any swap, and peripheral vision is the most flicker-sensitive. The established "margin" is a *full-quality* cull guard band; the engine has it via the predicted pose | ❌ by design, documented |
| placeholders outside, swapped seamlessly | distance-based HLOD (the Lumen stand-in) + cross-faded visibility ranges + identical shading at the handover, so turning your head never shows a placeholder | ✅ Lumen + temple; real cities already stream |
| behind you ≈ nothing | frustum culling draws nothing behind you; the other destinations aren't drawn at full detail at all | ✅ |
| multiplayer: each client renders only what it sees; the server keeps detail only where players are | client side ✅ (all of the above is per client). Server-side interest management (per-area send lists, sleep for empty areas) is **not** in this branch | ❌ next (vrgame-multiplayer finding #5) |
| the "circle" Zeke saw in NY | VRS_XR foveation (`vrs_min_radius` 20 %) is the prime suspect | xrfix-1104's (merged to main 11-05: foveation off by default + a toggle) |

## What's left to reach 8 ms

Draws and prims at the temple are down 55–73 %, but **the V100's GPU time only fell 5–20 %**. With the culling and
shadow work done, what's left is **per-pixel**. Attribution on the V100 after the changes (hide one subtree at a time,
Δ GPU ms; temple total ≈ 3.0–3.2 ms):

| what | V100 ms | share | note |
|---|---|---|---|
| the floor: sky panorama, fog, clear, resolve (everything hidden) | 0.72–0.88 | ~27 % | can't be culled; scales with render resolution + MSAA |
| **the temple architecture** (yard disc r 110 m, platform, steps, 32 columns + caps, band) | **0.96–1.12** | **~35 %** | very few prims. It's `PBR.mat`: **world-space triplanar, 4 maps (albedo/normal/rough/AO) = 12 anisotropic texture fetches per pixel** over most of the view |
| the overgrowth grass (one MultiMesh, ~7,700 tufts) | 0.31–0.47 | 10–15 % | alpha-tested blades, overdraw |
| the Lumen stand-in (looking north) | 0.12–0.30 | 4–10 % | lit-window shader over the skyline |
| chess set (facing S) | 0.36 | — | 290 colour draws: 32 pieces × ~4 surfaces + 55 Label3Ds |

Ranked, to get from Zeke's 13–14 ms to ≈ 8 ms on the tower (ms are **(est.)**; only the tower can say):
1. **Take this build to the headset first.** If the tower was partly CPU/draw-bound, the 70 % draw cut alone is
   several ms. If it was fill-bound, it's ~1–2 ms. Step 1 below decides which.
2. **Mirror (xrfix-1104):** +8–12 ms while on. Nothing here changes that; the mirror's own render now benefits from the
   cheaper shadows and the stand-in.
3. **Render resolution / foveation / visibility mask (xrfix-1104):** fill-bound ⇒ ms ∝ pixels. Record SteamVR's
   per-app resolution % with every number.
4. **The temple floor and columns' triplanar PBR: probably the biggest per-pixel item left (~35 % of the V100's temple
   frame).** Same look for a fraction of the cost: a **planar (top-down) world mapping** for the flat yard, platform and
   steps. On a horizontal surface, triplanar at sharpness 4 is already ~100 % the Y projection, so it's 4 fetches
   instead of 12. Optionally lower anisotropy 16× → 4×. *Not done:* it's a look decision and asks for a still-image
   A/B in the headset. Zeke's call.
5. **Grass:** ~10–15 %. Shorter `visibility_range_end` (90 m today) or fewer tufts far out.
6. **Chess set:** merge the 55 labels into one mesh and the pieces' surfaces into one each: about −200 colour draws
   facing south.
7. **Multiplayer interest management** (server sends + sim only where players are): the network half of Zeke's design.
   It doesn't affect the headset's ms.

Rough arithmetic (est.): today 13.5 ms → this branch −1.5 to −4 (draws/shadows; tower-dependent) → planar floor −1 to −2
→ grass −0.5 → ≈ 8–10 ms with the mirror off. The rest is render resolution/foveation (xrfix). With the mirror on, 8 ms
needs the mirror's AVATARS mode or a short mirror far plane (the plan doc's conclusion stands).

## Headset checks for Zeke (tower, SteamVR frame timing; build from `perf-1105`)

Build `perf-1105`, SteamVR frame-timing graph open, auto quality OFF (Settings ▸ Graphics), mirror OFF, the same time of
day. Compare against the current deployed build at the same spots; note SteamVR's render resolution %.
1. **GPU ms at the temple:** centre facing north, centre facing south, the table, the spawn. Hold still 10 s at each and
   read app GPU ms + reprojection %. *This is the number the whole branch is for.*
2. **Head-shake test at the table:** shake your head fast left-right for 5 s. Any tearing / late frames?
3. **Walk from the temple into Lumen through the yard** (not the gate), and back. At ~95 m from the centre going out
   (~85 m coming back) Lumen swaps between the stand-in and the full city. **Do you see it change?** Towers should look
   the same; signs, holo ads, kerbs, cars and rooftop clutter appear at the swap. Do it by day **and at night** (lit windows).
4. **Use the Lumen gate** (teleport): the city should be full detail when you arrive.
5. **Swing down the avenue fast:** any building/detail **pop** (it should fade now)? Any building **vanishing in one eye
   only**, especially near corners? If yes: relaunch with `IRIS_NO_LUMEN_OCC=1` (Lumen's occlusion off) and say whether
   it goes away.
6. **Look back at the temple from Lumen's high rooftops:** temple stations fade out past 60/120 m. Anything missing that
   you'd expect to see from there?
7. **Shadows at the table:** small props (cans, cups, chess men, dice) no longer cast sun shadows; your body and hands
   still do. Acceptable? Far shadows now end at 60 m (2 splits): any visible shadow edge on the yard?
8. **Sunset/twilight:** sit through the sun going down. Shadows should fade out with no flicker or step.
9. **New York** side street: the same feel as before (it barely changed). Note any module fade you can see.

## Tests

All on `perf-1105`:
- `vr.sh cc`: 0 failures
- unit (all 32 suites, run headless directly): **476 pass, 0 fail**, incl. the new `unit/render_budget_unit.gd` (17)
- `vr.sh test city_big` **25/0** · `vr.sh mp teleport` **10/0** · `vr.sh bootclean` **7/0**
- `vr.sh xr body` **29/0** (own head shadow) · `xr web` **64/0** (swinging across the stand-in swap) · `xr mirror` **10/0**
- `vr.sh xr renderbudget` (new, in REGRESS_XR): passes on the branch. **Against main's code it fails RB2 (8 vantages over
  their ceiling, e.g. table 2,226 > 720), RB3 (shadow ceilings) and RB5 (Lumen full from the temple)**, so it catches a regression.
- `vr.sh xr renderstress` (new: a private server, **4 bots flying round Lumen's avenue + me in VR** flying the swing ×3):
  **5/5**. Worst frame 305–337 draws, mean 185, biggest step 152–161, script p95 17.7–18.2 ms (solo 16); V100 GPU 1.86 ms
  (unchanged by the players). Ceilings are in `xr/renderbudget_ceilings.json` "stress".
- Final gate on the branch before the merge: `vr.sh xr renderbudget` **6/6** (RB0–RB5, every vantage ×3).
- **After merging main (xrfix-1104: render scale, foveation off by default, mirror diet + `far_layer.gd`)**: cc clean;
  unit **486 pass / 1 fail**. The fail is `fracture_unit` F8, a CPU-time band in code this branch doesn't touch, hit while
  the XR sim was running; it passes 3/3 alone. `xr/renderbudget` **6/6** (same numbers as before the merge),
  `xr/mirror` **10/0**, `xr/body` **29/0**. The stand-in's meshes join xrfix's far layer by themselves (its watcher
  catches every visual added under `City`), so they're in your eyes but never in a mirror.
- Not run: the full `vr.sh regress` (offline + MP ×40 suites). The offline/MP suites run headless, and every change here
  is skipped on a headless game (render diet, skyline/LOD) except the motes MultiMesh (no bodies) and the sun/moon settings.
