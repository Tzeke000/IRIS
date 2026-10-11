# SAMURAI v2: third-party assets (provenance and credits)

The downloaded models stay on `/data/iris_assets/characters/` and are **never committed**. The build reads them
from there (`figures.CHARS`). The full research log, including the candidates that were rejected, is in
`/data/iris_assets/characters/ASSETS.md`. That file is the master copy; the three used assets are copied here.

All three are **CC BY 4.0** (http://creativecommons.org/licenses/by/4.0/). They were fetched with the Sketchfab
Download API on 2026-10-10 by `scratch/scenes/samurai/models/scripts/sf_download.py`, which refuses any licence other
than CC0, CC-BY or CC-BY-SA and keeps the GLB only. Each directory has a `SOURCE.json` (uid, author, licence, URL, date).

| Role | Asset | Author | Dir (on /data) | sha256 of model.glb |
|---|---|---|---|---|
| Ronin body | [Eric Rigged 001 - Rigged 3D Business Man](https://sketchfab.com/3d-models/eric-rigged-001-rigged-3d-business-man-a46bc9f67aaa415bb4f3241eef900e7f) | Renderpeople ([renderpeople](https://sketchfab.com/renderpeople)); official Renderpeople account | `sf_renderpeople_eric/` | `5fd39ba8…99ba8ae5e` |
| Ronin garments (haori, hakama, jinbei) | [Japanese Clothes](https://sketchfab.com/3d-models/japanese-clothes-a87d993c197749e3a243fac636f23485) | Takeru ([takeru1124](https://sketchfab.com/takeru1124)) | `sf_japanese_clothes/` | `3a795b63…a9aba` |
| Opponent | [Death Samurai Character](https://sketchfab.com/3d-models/death-samurai-character-6e0a7667723747e0972488085f877bd6) | Alexandru Jardan ([AlexandruJardan](https://sketchfab.com/AlexandruJardan)) | `sf_death_samurai/` | `5fef1531…390d81` |

**What we changed (CC BY asks us to say so):**
- *Eric*: rescaled to 1.80 m. Body faces fully covered by the garments are deleted (trunk and legs). The model is re-posed by our solver.
- *Japanese Clothes*: scaled ×0.0105 to fit Eric and re-skinned to his rig (nearest-skin weights plus spring bones for the skirt, hakama legs and sleeve bags). Covered jinbei faces are deleted. The materials are darkened and given sheen and a rain film.
- *Death Samurai*: we fitted our own armature (body joints plus finger chains fitted to the gloves). The plates are bone-parented and the cloth is skinned. The tassets and shoulder plates hang on spring bones. The materials have a lacquer coat and are darkened for rain.

**Ours (no licence involved):** the kasa straw hat, both katana (blades, tsuba, tsuka, habaki), both saya, the whole
street, rain, effects and lights. These are all procedural in `build_scene.py`, carried over from v1.

**Not used:** MPFB2 / MakeHuman CC0 bodies (`/data/iris_assets/characters/mpfb_base/`). They are the documented fallback
if a CC-BY asset has to be dropped.

## CREDITS block (paste into the video description)

```
3D characters (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), modified (re-rigged, re-posed, re-shaded):
- "Eric Rigged 001 - Rigged 3D Business Man" by Renderpeople — https://sketchfab.com/3d-models/eric-rigged-001-rigged-3d-business-man-a46bc9f67aaa415bb4f3241eef900e7f
- "Japanese Clothes" by Takeru (takeru1124) — https://sketchfab.com/3d-models/japanese-clothes-a87d993c197749e3a243fac636f23485
- "Death Samurai Character" by Alexandru Jardan (AlexandruJardan) — https://sketchfab.com/3d-models/death-samurai-character-6e0a7667723747e0972488085f877bd6
Scene, animation, swords, kasa and effects: procedural (Blender), Tzeke000.
```
