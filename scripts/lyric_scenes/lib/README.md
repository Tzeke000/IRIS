# scripts/lyric_scenes/lib — the shared scene toolkit (one copy, imported, never forked)

Import inside Blender: `sys.path.insert(0, "/ABS/scripts/lyric_scenes")`, then `from lib import bpyx, song, importer, ...`.
CLIs: absolute paths, PIL/numpy ones with `PY=~/venvs/iris-v100/bin/python -I`. **Every Blender launch goes through `scripts/server/render.sh`.**

| Job | Use | Notes |
|---|---|---|
| Build / verify / stills / frames / parallel / mux / full / Discord parts | `render/scene.sh /ABS/real/<scene>/scene.env VERB` | verbs: `build · verify [ARGS] · stills SPEC [PCT] [SPP] · frames [S] [E] · par [S] [E] [W] · wait · status · mux · full · parts · sheet`; launches its own capped units; spec in the file header |
| Verify the SAVED .blend in a fresh process (images, NaN, camera inside geometry, cut holds, instance counts [+ grounding, direction, NaN render, scene rules]) | `scene.sh ENV verify` · or Blender `qa/verify_saved.py -- --out J [--ground G --grounded P] [--direction Car_:+Y] [--render F,…] [--extra rules.py]` | the arcade scar: in-memory checks passed, the file didn't. JSON always written; exit 1 on a failure. **Save builds with `bpyx.save_blend`** (the default save turns absolute image paths into `//…`) |
| Render frames/stills (Blender side) | `render/frames.py` | atomic tmp→rename; resume skips only **complete** PNGs; `--cams` + `CAM_PROPS` (exposure_offset, hide_collections) + `--hook` |
| Is a PNG complete? missing/torn frames | `python3 render/pngcheck.py missing DIR S E` · `torn DIR` | signature + IEND (the old `-s` / `>1 KB` tests passed torn files) |
| Discord-sized parts | `render/discord_parts.sh MASTER OUT [3] [7.5] [960]` | re-encodes any part over 8 MB |
| Contact sheet / safe-to-Read copy | `$PY qa/sheet.py OUT.jpg FILES… \| --dir D \| --video V --times …` · `qa/sheet.py shrink SRC` | **guaranteed** ≤900 px, <150 KB |
| Timing + A/B variants | `qa/time_ab.py` (Blender) `--cams/--frames --variants JSON --out times.json [--save DIR]` | warm-frame; variants = `render/patch.py` specs; `--bookend` for drift |
| Pixel diff / 1:1 crops | `$PY qa/crops.py diff A B [--heat H]` · `compare OUT A B --box … \| --auto N` | proves a cut is invisible |
| Scene patch spec (hide/show, eevee/render settings, dotted `set`, exposure) | `render/patch.py` | shared by time_ab + frames; restorable; typos raise |
| Song → beat grid, frames, sections, hits, envelopes, curves | `song.Song.load(song_map.json)` · CLI `$PY song.py MAP --marks …` | kick-fitted grid; **`ear.json` next to the map outranks the analysis** |
| Import a model (+ unlit / missing-texture / scale report) | `importer.import_model(path, coll, fix_unlit="report"\|"lit")` | `bake_world`, `normalize`, `crop`, `decimate`, `islands`, `sub_mesh`, `drop_flat_bottom`, `load_img`, `write_log` |
| Keys, node DSL, materials, meshes, lights | `bpyx` | `bake(…, cuts=, thin=)` (reconciled — see its docstring), `show_mask`, `NB`, `gn_tree`, `new_mat`, `MB`, `lathe`, `boxes_mesh`, `points_obj`, `light`, `look_at`, `save_blend`; numpy rotations `track_matrix` / `euler_seq` (= `to_track_quat` / `to_euler(prev)`) |
| Camera: shot table → cut-safe bake, stills, DOF, QA | `camera` | `Shot(start_frame, name, fn(i, u, n) → dict)`, `bake_shots(cam, shots, n, post=shake)` → arrays + `cuts` (every channel CONSTANT on a shot's last frame; `scene["cuts"]` + markers), `still_cam(…, fstop=, **frames_props)`, `dof`, `on_screen(_by_shot)`, `hold_at_cuts` / `cut_violations`, `view_hit` |
| Look: colour, EEVEE block, HDRI world, 5.x compositor | `look` | `color(sc, look="Punchy")` (adds `AgX - `), `eevee_defaults` (strict, motion blur START), `eevee_volumes`, `hdri_world(path, rot_deg, camera=)` + `hdri_key` / `hdri_rotation` (azimuth convention MEASURED — boss v1's was 180° off), `post_chain(mist=, glare=[…], lens=, vignette=, grain=)` or the parts `comp_tree · mist_haze · glare · lens_distortion · vignette · film_grain` |
| Rigid bodies → keys; settle; grounding | `physics` | `settle(pieces, statics=[…])` (rest poses), `with Sim((f0, f1), substeps=) as S: S.add/release/bake/sample/to_keys` (own scene, temp_override, plain keys — no Bullet in render workers), `ground_from`, `drop_to_ground`, `floating`, `lowest_columns` |
| Build-time asserts | `checks` | direction, AABB clearance, inside count, floating, spacing, NaN, BVH camera-inside, GN instance count, bad images |
| Build report JSON (always saved, even on error) | `report.Report(path)` | carries the importer log → feed to credits |
| Scatter / keep-clear zones | `scatter_zones` | see `profiles/blender/scatter_zones.md` |
| Probe a model before using it (bbox m, polys, unlit / missing textures, rigs + actions, scale sanity) | Blender `assets/probe.py -- --out J FILE.glb … [--strict]` via `render.sh` | imports through `importer` (same detection the build gets); bone-display widgets listed apart; facing NOT inferred (use a +Y marker / `checks.direction_check`) |
| Fetch an asset (licence-gated) | `python3 -I assets/fetch.py sketchfab SLUG UID \| polyhaven KIND ID RES \| ambientcg ID RES --root /data/iris_assets/<scene>` | **CC0 + CC-BY only** (`assets/licence.ALLOWED_LICENCES`); `.partial`→rename; safe unzip; 429 backoff |
| Search Sketchfab | `$PY assets/search.py "query" … [--thumbs DIR]` | drops non-CC0/CC-BY; thumbnail sheet |
| ASSETS.md + video-description credits | `python3 -I assets/credits.py --used build_report.json --out real/<scene>/ASSETS.md` · `--audit /data/iris_assets` | from what was LOADED; blocks BY-SA/NC/ND/unknown and CC-BY without author/url |
| Tests | `tests/run_all.sh [--quick]` | 21 suites (11 pure, 8 Blender, 2 end-to-end shell), ~5–8 min; every Blender launch via render.sh |

**Per-scene folder now holds only:** `scene.env` (+ optional `stills_hook.py`), the build (`build_scene.py` or `build.py` + stage modules), scene-only modules (e.g. `vehicles.py`), and the generated `ASSETS.md`. No per-scene render.sh / render_frames.py / sheet / timing / fetch copies. Delivered scenes keep their old files (frozen).
