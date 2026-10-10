# HIT A BUMP — photoreal arcade (Blender 5.2 / EEVEE)

Zeke 2026-10-10: *"super realistic looking"*. This is true 3-D, rendered on the server's V100.
It is not the 2.5-D numpy style of `scripts/lyric_scenes/*.py`. A dark late-night arcade:
patterned blacklight carpet, a row of neon-trimmed upright cabinets, haze catching moving-head
beams. On the hero cabinet's CRT, an **original 8-bit game, "HIT A BUMP"**, plays in sync with
the song. The marquee, side art and title screen carry the Tzeke000 emblem; there are no real
game IPs or company logos.

## Pipeline

| step | script | runs on | output (git-ignored, `assets/scenes/arcade/`) |
|---|---|---|---|
| 1 | `game_frames.py` | venv python, CPU (16 procs, ~1.5 min) | `game/g_####.png` (1024×896 CRT texture), `attract/a{0,1}_####.png`, `screen_stats.npz` |
| 2 | `make_art.py` | venv python, CPU (5 s) | `art/*.png`: marquees, side art, bezel, control panel + wear, carpet, signs |
| 3 | `build_scene.py` | `blender -b`, CPU (~3 s) | `arcade.blend` (+ `arcade_shots.txt`) |
| 4 | `render_frames.py` | `blender -b`, GPU, under the lock | `frames/f_####.png` |
| 5 | `render.sh` | runs 1–4 if needed, renders in chunks (resumable), then muxes | `state/tzeke_songs/hit_a_bump/hit_a_bump_arcade.mp4` |

`layout.py` holds the shared cabinet dimensions, so the printed control-panel art lines up with the 3-D buttons.
Every step is deterministic: fixed seeds, all animation **baked** from
`state/tzeke_songs/hit_a_bump/timeline_30fps.npz`, with no handlers and no Python drivers. Scene frame `f` is
timeline index `f-1`, which is audio time `(f-1)/30`, and game image `g_f`.

```bash
scripts/lyric_scenes/real/arcade/render.sh            # whole song (resume = run it again)
scripts/lyric_scenes/real/arcade/render.sh --start 900 --end 1100 --scale 50   # a test window
```

## Look and engine

- **EEVEE 5.2** at 1920×1080/30 fps with 32 TAA samples. Screen-space ray tracing (half-res, denoised) handles
  the glossy CRT faces and coated laminate. Fast GI makes the emissive neon and screens bounce.
  Virtual shadow maps are on. **Haze** is a room-sized object volume (density 0.012–0.045, modulated by slow 3-D noise,
  anisotropy 0.5, 4 px froxels, volumetric shadows). Camera **DOF** (f/2–4, 7 blades) and camera **motion blur**
  (180° shutter at the START position, so cuts never smear). **AgX – Punchy**.
  The compositor adds a gentle bloom and a breath of lens dispersion; ffmpeg adds light film grain.
- **Staging choice:** the area/wash lights barely scatter into the haze (`volume_factor` 0.08).
  Only the moving-head beams (6×) and the back-wall shafts (14×) light it. The haze therefore
  *catches the beams* instead of milking out the whole frame. The hero cabinet's T-molding runs at 2.5× the others,
  and the other screens are dimmer.
- **CRT:** a curved tube face (48×36 grid, 22 mm sag) behind a printed bezel. The game is rendered at
  256×224, then upscaled 4× with a horizontal beam blur, brightness-dependent Gaussian scanlines,
  halation, rounded tube corners and a vignette baked in. EEVEE mip-maps it, so scanlines never moiré at a
  distance. The image is the emission of a glossy, near-black glass surface, so reflections sit on top of it.
  (A separate dithered glass pane was tried and removed: it speckled the screens.)

## Beat mapping

| element | cause (timeline) |
|---|---|
| game hero jump apex | **on the beat** (`T_bi` changes; every 2nd beat in calm) and **on every snare** |
| BUMP block shatters (+100, coin) | **snare** at the apex (head-bump); snare rolls too tight for a block pop coins |
| score ticks (rolling digits, flash) | **kick** (+50); drone kills +200, boss hits +500 |
| drone explodes / boss hit flash | **kick**; the hero's bolt is fired **6 frames before** the kick so the hit lands *on* it |
| boss jaw opens → fireball | fired 20 frames before a **snare**, reaching the hero exactly at his jump apex (he hops it) |
| "WARNING! / BOSS APPROACHING", red sky, blink rate doubling | **build** `T_pre` |
| TV static | **half-beat blackout** `T_black` (all room power cuts too, so the screen is the only light) |
| white flash, FEVER!/BOSS FIGHT!, palette swap | **drop entry** (`entries`) |
| boss fight, health bar drains to 0 on the last kick, STAGE CLEAR | **BIG drop** `T_big` |
| title screen, INSERT COIN → CREDIT 01 (coin-door lights flare) | intro (index 200) |
| control-panel buttons (top row / bottom row) | **kick** / **snare** (`btn_kick`, `btn_snare`); start buttons blink on the **beat** |
| T-molding, wall neon, neon washes | **sub** (`bass` envelope); dims in the build, off in the blackout |
| coin-door lights | **hi-hats** (`high` minus its smooth) |
| other cabinets' attract screens | **spectrum** (`bars`, 40 bands) |
| carpet blacklight glow | **sub** × drop intensity |
| marquees | power + a fluorescent flicker, pump on kicks in drops |
| screen spill lights | the game frame's **mean linear colour** (`screen_stats.npz`), so the entry white-flash *floods the room* |
| moving-head beams | calm: dim, static, cool; drops: sweep + palette per drop + beat pulse; build: converge **red** on the hero cabinet; **strobe** hits (`T_strobe_age`, budgeted) go white |
| back-wall shafts | **kick** punch in drops; red beat-blink in the build |
| camera | calm: slow dollies/cranes (smoothstep); build: **accelerating push into the CRT** (anticipation), arriving at the glass on the blackout; **entry**: starts inside the screen and is blown back out through the glass (ease-out); drops: 2-bar handheld shots cut on **bar downbeats**, shake on **kicks**; BIG drop: boss-fight cut pattern (screen ↔ cabinet); sung "bump" = punch-in (`punch`) |

## Measured (V100, shared with the voice model + the rocket job, 2026-10-10)

| | 1920×1080, 32 samples |
|---|---|
| calm (frames 301–330) | **1.99 s/frame** steady (max 2.13); first frame of a process ~3.5 s (shader compile) |
| heavy: BIG drop, beams + strobe + kick shake (3200–3229) | **1.99 s/frame** (max 2.18) |
| 30-frame chunk wall time, including blend load | ~63 s, so ~2.1 s/frame effective |
| **full song (3570 frames)** | **~2.1 h of GPU time** (plus any waiting on the shared lock) |
| peak VRAM (blender process) | **2.3 GB** |

`render.sh` uses 120-frame chunks: about 4 min each, then the GPU lock is released.
⚠ After `--rebuild` (or any change to the scripts), **delete `assets/scenes/arcade/frames/`**. Resume skips any
frame already on disk, including stale ones.

## Known weak spots

- When a drop shot puts the camera inside a moving-head cone, the coloured haze lowers the CRT's contrast for a moment.
- In the wide shots the game's sprites are small. The game reads in the medium and close shots, which is where the cut pattern spends the drops.
- Every build uses the same grammar: a push into the CRT, then a pull back out on the entry. This is deliberate as a motif, but it happens 5 times.
- There are no people in the arcade. Cabinet edges are unbevelled, but the T-molding hides most of them.

## Photosensitivity

Full-white frames only on the drop-entry flash (1 frame, then a 5-frame decay) and the budgeted strobe hits
(12, drops only, never 2 in a row). Flag it on publish anyway.

## Generated files (not in git)

`assets/scenes/arcade/` is about 2.5 GB (3570 CRT frames plus 2×3570 attract frames). Rebuild with steps 1–3.
