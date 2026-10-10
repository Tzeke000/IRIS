# lyric_scenes — music-reactive scene plugins for `scripts/lyric_viz.py`

Zeke 2026-10-10: *"make each one of those and we'll do it all to the song Hit a Bump"* —
steampunk, spooky Halloween, robot fight, a car cruising a street. Scene #1 (`--look stage`,
LED cube) and #2 (`--look dystopia`, megacity) live inside `lyric_viz.py` (sections 3c / 3d)
and are the reference implementations. These four are plugins so each can be built alone.

## Contract — `scripts/lyric_scenes/<name>.py`

`<name>` ∈ `lyric_viz.SCENE_NAMES` = steampunk, halloween, robotfight, carstreet.
Loading it makes `<name>` a Style, a `--viz` mode and a `--look` (no lyrics, flat bg, 16:9).

```python
PALETTES: dict[str, dict]      # colour families; FIRST key = default (--scene-palette picks)
STYLE: dict                    # Style kwargs, at least palette=[4 RGB accents]
BLOOM: float                   # default --bloom for --look <name> (~0.3–0.45)

def draw(R, img, i, a, lv) -> None:
    """Paint the WHOLE frame into img (float32 H×W×3, RGB, 0..255, already allocated)."""

def fx(R, img, i, a, lv) -> np.ndarray:   # optional
    """Post-FX (flash, shake, aberration, zoom). Return the image (may be new)."""
```

- `R` = the Renderer: `R.W, R.H, R.fps, R.logo` (PIL RGBA Tzeke000 symbol or None),
  `R.title` ("HIT A BUMP"), `R.style.strobe` (False under --no-strobe — honour it),
  `R.lyric_zoom` (per-frame punch ≥1.0 when the word "bump"/"bomb" is sung, or None),
  `R.font(size, name)` (PIL font from assets/fonts), `R.scene_cache` (dict for YOUR caches).
- `a` = Analysis: per-frame `rms, bass, high` (0..1), `kick, snare, drop, beat` (bool),
  `build` (0..1), `bars` (n×40 spectrum), `bpm`, `beat_i` (int beat count), `beat_ph` (0..1).
- `lv` = the lyric_viz module: `lv.stage_timeline(a, fps, "acid", label=...)` gives sections +
  events (see `dystopia_timeline` for how to extend it): `I` intensity, `hot` (in a drop),
  `big` (the final big drop), `pre` (0..1 pre-drop ramp, last 8 beats), `black` (the half-beat
  power-cut before each drop), `entries` (drop-entry frames, snapped to the kick), `kage`/`sage`
  (frames since kick/snare), `strobe_age` (budgeted strobe hits — ONLY strobe on these),
  `one_age` (frames since a bar downbeat in a drop), `bi`/`ph` (gap-filled beat index/phase),
  `gbeat`, `bar`, `segs`. Helpers: `lv._gauss_smooth`, `lv._age_since`, `lv._hash01`,
  `lv.scene_palette(R, module)` → the chosen palette dict.

## Skills (Zeke 2026-10-10: use them whenever Blender is used)

Load with the Skill tool before writing bpy: `blender-rules`, `blender-headless-batch-scripting`,
`blender-audio-reactive`, `blender-slotted-actions-animation`, `blender-drivers-and-handlers`,
`blender-vse-python` (+ mesh / materials / geometry-nodes as needed), and for the motion
`motion-design` + the fitting `disney-*` skills. Blender = headless `blender -b` only.

## Hard rules

1. **STATELESS.** `draw(i)` is a pure function of `i` (+ cached, seeded precomputation).
   Renders are chunk-parallel: a worker starting at frame 1800 never ran frames 0..1799.
   Motion = cumsums / "frames since event" arrays computed once over the whole song and
   cached in `R.scene_cache`. Randomness seeded by constants or event indices. No
   `self`-style mutation between frames.
2. **Speed:** ≤ ~1 s/frame/process at 1920×1080 (dystopia is 0.9). Draw far/blurry layers
   at half res, cache static geometry, vectorise with numpy/cv2.
3. **The music must be legible.** Every element reacts to a NAMED cause (kick, snare,
   hi-hat sparkle `high` minus its smooth, sub `bass`, build `pre`, drop entry, big drop).
   Big moves on drop entries; calm sections breathe but stay alive.
4. **Photosensitivity:** full-frame flashes only on `strobe_age` hits and only if
   `R.style.strobe`; never 2–3 near-white frames in a row.
5. **No real brands/logos** except Tzeke000's own (`R.logo`) and the title "HIT A BUMP".
6. Don't edit `lyric_viz.py` — if the hook needs a change, say so.

## Testing

`scripts/lyric_scenes/test.sh <scene> <tag> START:END` → `scratch/scenes/<scene>/<tag>.mp4`
at 960×540, then `sheet_at.py` for a contact sheet. **Only `Read` images ≤900 px / <150 KB.**
Song sections ("Hit a Bump", 119 s): 0–32.9 calm (intro build) | 32.9–47.1 drop |
47.1–59.7 calm | 59.7–70.0 drop | 70.0–72.8 calm | 72.8–86.8 drop | 86.8–92.5 calm |
92.5–99.2 drop | 99.2–102.4 calm | 102.4–117.0 BIG drop | 117–119 outro.
Good windows: `28:36` (build → first drop entry), `100:106` (into the big drop), `10:13` (calm).

Full render (do it once, at the end): `scripts/server/lyric_viz_parallel.sh --workers 16
<same args as test.sh, --size 1920x1080, no --window> --out state/tzeke_songs/hit_a_bump/hit_a_bump_<scene>.mp4`

## Generated art (not in git)

`assets/scenes/<scene>/` is git-ignored (~57 MB of sprites/maps; the repo is public and the
art is reproducible). Rebuild with headless Blender 5.2.1 — each generator is deterministic,
and a scene whose assets are missing stops with its own rebuild command:

```
B="$HOME/.local/bin/blender -b --factory-startup --python-exit-code 1 --python"
$B scripts/lyric_scenes/blender/steampunk_assets.py
$B scripts/lyric_scenes/blender/halloween_scene.py
$B scripts/lyric_scenes/blender/robotfight_mechs.py
$B scripts/lyric_scenes/blender/robotfight_arena.py
$B scripts/lyric_scenes/blender/carstreet_car.py
```

## The four scenes (built 2026-10-10, all rendered to "Hit a Bump")

| look | palettes (first = default) | Zeke's verdict |
|---|---|---|
| `carstreet` | midnight · synth · rain | "great" — loved the big-drop tunnel |
| `steampunk` | brass · verdigris · furnace | concept liked, "needs a little work" — parked |
| `robotfight` | neon · industrial · toxic | — |
| `halloween` | classic · blood · spectral | — |
