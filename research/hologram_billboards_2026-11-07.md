# Hologram billboards: research (2026-11-07, Zeke: "research how to do the hologram billboard, see what's on Sketchfab")

City1 (branch city1-1107) already ships a holo_board template: unshaded, see-through, edge frame, scan lines, sweep band, flicker, one shared material/atlas.

## Technique: what the good ones add (sources below)
- **Fresnel rim glow:** the edges facing away glow brighter. This is the single biggest "hologram" cue, and the Three.js guides weight it highest.
- **Additive blend + grain/noise variation:** cheap, and reads as light instead of a panel.
- **Glitch:** stepped vertex displacement in waves, plus occasional RGB split. Use it sparingly and keep the timing random per board.
- ★ **VR-specific (my reasoning, not from the sources):** screen-space scan lines differ per eye and shimmer/warp in a headset. The Unity VFX thread hit warping even in world space from above. Use object/UV-space scan lines anchored to the board. Keep glitch displacement small, because large vertex jumps in stereo can be uncomfortable.
- **CC0 starting code:** "Simple Hologram Shader" (godotshaders, CC0); "Hologram Effect" (additive, scanlines, grain, stepped displacement, flicker). Check its licence before copying.

## Sketchfab: usable licences only (CC-BY = credit in props/ASSET_SOURCES.md; no NC/ND)
Searched via the public API (downloadable, 5 queries, 37 models, 31 with a usable licence). The picks are in hologram_sketchfab_picks_2026-11-07.jpg (TL, TR, BL, BR):
- **Holo-Puck** (TL), CC-BY, thekaz.2128, 2.6k faces: a metal projector base. It fits the freestanding holograms' puck. uid 4b8be9e0e41d4611a99e9641ee0f83f0
- **HoloBoard** (TR), CC-BY, toomanydemons, 10.8k faces, 953 likes, game-ready PBR: a framed sci-fi screen. Good for Aurelion kiosks or the lobby rather than street billboards (it has a physical frame). uid 0644f1b2450543e2a96d5eae496798b6
- **Holo Screen** (BL), CC-BY, 12 faces: just a textured quad. Skip.
- **Looking Glass Hologram 1** (BR), CC-BY, skyeshark: floating neon text plus ring glyphs, animated. A STYLE reference for logo holograms (text floating in air, no box). uid 8dd9df2a64fc4ee181202d2973350a85

## Proposal
1. Upgrade holo_board's shader: fresnel rim, additive mode, UV-space scan lines, a rare small glitch, and a slight RGB split on the glitch only.
2. Add a "floating logo" variant with no box: text plus ring glyphs, like Looking Glass, for Aurelion and the music boards.
3. Import Holo-Puck as the projector base (credit it). Use HoloBoard in the Aurelion lobby / divisions if Zeke likes it.

Sources: https://godotshaders.com/shader/simple-hologram-shader/ · https://godotshaders.com/shader/hologram-effect/ · https://godotshaders.com/?p=10900 · https://threejs-journey.com/lessons/hologram-shader · https://masterallarts.pages.dev/learn/threejs-from-zero/s0-06-hologram-shader/ · https://realtimevfx.com/t/mikolaj-ptaszynski-sketch-18-hologram-wip/6692
