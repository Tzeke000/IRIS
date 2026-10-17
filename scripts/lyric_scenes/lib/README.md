# scripts/lyric_scenes/lib — the shared scene toolkit (one copy, imported, never forked)

Import inside Blender: `sys.path.insert(0, "/ABS/scripts/lyric_scenes")`, then `from lib import bpyx, song, importer, ...`.
CLIs: absolute paths, PIL/numpy ones with `PY=~/venvs/iris-v100/bin/python -I`. **Every Blender launch goes through `scripts/server/render.sh`.**

| Job | Use | Notes |
|---|---|---|
| Build / stills / frames / parallel / mux / full / Discord parts | `render/scene.sh /ABS/real/<scene>/scene.env VERB` | verbs: `build · stills SPEC [PCT] [SPP] · frames [S] [E] · par [S] [E] [W] · wait · status · mux · full · parts · sheet`; launches its own capped units; spec in the file header |
| Render frames/stills (Blender side) | `render/frames.py` | atomic tmp→rename; resume skips only **complete** PNGs; `--cams` + `CAM_PROPS` (exposure_offset, hide_collections) + `--hook` |
| Is a PNG complete? missing/torn frames | `python3 render/pngcheck.py missing DIR S E` · `torn DIR` | signature + IEND (the old `-s` / `>1 KB` tests passed torn files) |
| Discord-sized parts | `render/discord_parts.sh MASTER OUT [3] [7.5] [960]` | re-encodes any part over 8 MB |
| Contact sheet / safe-to-Read copy | `$PY qa/sheet.py OUT.jpg FILES… \| --dir D \| --video V --times …` · `qa/sheet.py shrink SRC` | **guaranteed** ≤900 px, <150 KB |
| Timing + A/B variants | `qa/time_ab.py` (Blender) `--cams/--frames --variants JSON --out times.json [--save DIR]` | warm-frame; variants = `render/patch.py` specs; `--bookend` for drift |
| Pixel diff / 1:1 crops | `$PY qa/crops.py diff A B [--heat H]` · `compare OUT A B --box … \| --auto N` | proves a cut is invisible |
| Scene patch spec (hide/show, eevee/render settings, dotted `set`, exposure) | `render/patch.py` | shared by time_ab + frames; restorable; typos raise |
| Song → beat grid, frames, sections, hits, envelopes, curves | `song.Song.load(song_map.json)` · CLI `$PY song.py MAP --marks …` | kick-fitted grid; **`ear.json` next to the map outranks the analysis** |
| Import a model (+ unlit / missing-texture / scale report) | `importer.import_model(path, coll, fix_unlit="report"\|"lit")` | `bake_world`, `normalize`, `crop`, `decimate`, `islands`, `sub_mesh`, `drop_flat_bottom`, `load_img`, `write_log` |
| Keys, node DSL, materials, meshes, lights | `bpyx` | `bake(…, cuts=, thin=)` (reconciled — see its docstring), `show_mask`, `NB`, `gn_tree`, `new_mat`, `MB`, `lathe`, `boxes_mesh`, `points_obj`, `light`, `look_at` |
| Build-time asserts | `checks` | direction, AABB clearance, inside count, floating, spacing, NaN, BVH camera-inside, GN instance count, bad images |
| Build report JSON (always saved, even on error) | `report.Report(path)` | carries the importer log → feed to credits |
| Scatter / keep-clear zones | `scatter_zones` | see `profiles/blender/scatter_zones.md` |
| Fetch an asset (licence-gated) | `python3 -I assets/fetch.py sketchfab SLUG UID \| polyhaven KIND ID RES \| ambientcg ID RES --root /data/iris_assets/<scene>` | **CC0 + CC-BY only** (`assets/licence.ALLOWED_LICENCES`); `.partial`→rename; safe unzip; 429 backoff |
| Search Sketchfab | `$PY assets/search.py "query" … [--thumbs DIR]` | drops non-CC0/CC-BY; thumbnail sheet |
| ASSETS.md + video-description credits | `python3 -I assets/credits.py --used build_report.json --out real/<scene>/ASSETS.md` · `--audit /data/iris_assets` | from what was LOADED; blocks BY-SA/NC/ND/unknown and CC-BY without author/url |
| Tests | `tests/run_all.sh [--quick]` | 12 suites, ~1 min; Blender suites via render.sh |

**Per-scene folder now holds only:** `scene.env` (+ optional `stills_hook.py`), the build (`build_scene.py` or `build.py` + stage modules), scene-only modules (e.g. `vehicles.py`), and the generated `ASSETS.md`. No per-scene render.sh / render_frames.py / sheet / timing / fetch copies. Delivered scenes keep their old files (frozen).
