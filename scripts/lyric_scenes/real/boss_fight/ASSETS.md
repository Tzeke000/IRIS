# Boss fight (knight vs dragon, night) — arena v1 assets

All files live on `/data/iris_assets/medieval/` (vetted in `profiles/blender/medieval_boss_prep.md`). Licence gate:
CC0 / CC-BY only. Nothing from a download is executed; the build only imports the meshes and images.

| asset | source | author | licence | use in the arena |
|---|---|---|---|---|
| European Dragon | https://sketchfab.com/3d-models/european-dragon-82f393a2e6c048ad80c171ce3b3a7b87 | ReginaCachoa (Regina Cachoa) | CC-BY 4.0 | the dragon (rigged, Idle Stand), ×5 scale; Walk pose baked for the 12-dragon size proof |
| Medieval Knight \| Sculpture \| Game ready | https://sketchfab.com/3d-models/medieval-knight-sculpture-game-ready-6cdd055b4afa41eb9360dbbfe75c7f10 | by__Rx | CC-BY 4.0 | the plate knight (rigged), scaled to 1.85 m |
| Gothic Architectural Structures | https://sketchfab.com/3d-models/gothic-architectural-structures-affc5b590531447788aafc23a011154c | oneironauticus (Samuel F. Angrick-Johanns) | CC-BY 4.0 | the 8 arcade piers + arches (pre-fractured), and the rubble chunks |
| Rosetón 2021 Rose window | https://sketchfab.com/3d-models/roseton-2021-rose-window-9785040d7f5245a38a2c4b1f15912b5c | carlos.moreno.g | CC-BY 4.0 | rose window tracery in the north gable (own stone + stained-glass materials) |
| Pack of old towers in ruins | https://sketchfab.com/3d-models/pack-of-old-towers-in-ruins-f213359c6cbb4c29bf8880764faa0fb8 | taz83 (JB3D) | CC-BY 4.0 | the 4 curtain-wall corner towers |
| Rubble Pile - Photoscan (Optimized) | https://sketchfab.com/3d-models/rubble-pile-photoscan-optimized-5986d1487d9443b883d67b121c2c903c | GameDevNick (Coozy) | CC-BY 4.0 | rubble piles |
| Medieval braziers l Lights l Enviroment | https://sketchfab.com/3d-models/medieval-braziers-l-lights-l-enviroment-fc5c7544dd224365b055febee02bb496 | burnedhrum | CC-BY 4.0 | the braziers (left one of the pair) |
| Medieval Wall Torch | https://sketchfab.com/3d-models/medieval-wall-torch-77db436da2844cbfb4dde0bb9b396835 | Kigha | CC-BY 4.0 | wall torches |
| Poly Haven HDRI `moonlit_golf` (4k) | https://polyhaven.com/a/moonlit_golf | Poly Haven | CC0 | night sky + moonlight (horizon misted out for camera rays: the HDRI's own ground shows a modern clubhouse) |
| Poly Haven textures (2k): `rough_block_wall`, `castle_wall_varriation`, `castle_wall_slates`, `stone_tiles_02`, `cobblestone_floor_08`, `grassy_cobblestone`, `forest_ground_04`, `old_stone_wall`, `roof_slates_02`, `medieval_wood` | https://polyhaven.com/textures | Poly Haven | CC0 | walls, platform, floor, stairs, courtyard, terrain, vault, slates, charred timber |
| Poly Haven models (2k): `treasure_chest`, `kite_shield`, `antique_estoc`, `gothic_statue`, `wooden_barrels_01`, `wooden_crate_01`, `dead_tree_trunk` | https://polyhaven.com/models | Poly Haven | CC0 | hoard chests, a fallen knight's shield + sword, stair statues, courtyard dressing |

Hand-built in `build_arena.py` (no external asset): platform, grand staircase, walls/gable/buttresses/pinnacles,
clerestory, pointed vault + ribs, timber trusses, curtain walls, terrain, gold heap + GN coins, flames, and the
**PLACEHOLDER gold rocket launcher** (swap the `PRIZE_placeholder` collection; keep `PRIZE_anchor`'s transform).

Build-time tool (not an asset): the Cell Fracture extension (GPL-3, extensions.blender.org) — only its Voronoi cell
calculation is used; the intersect runs in the build with the MANIFOLD boolean solver.

## Credits block (paste into the video description)

- "European Dragon" by ReginaCachoa — https://sketchfab.com/3d-models/european-dragon-82f393a2e6c048ad80c171ce3b3a7b87 — CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/)
- "Medieval Knight | Sculpture | Game ready" by by__Rx — https://sketchfab.com/3d-models/medieval-knight-sculpture-game-ready-6cdd055b4afa41eb9360dbbfe75c7f10 — CC-BY 4.0
- "Gothic Architectural Structures" by oneironauticus — https://sketchfab.com/3d-models/gothic-architectural-structures-affc5b590531447788aafc23a011154c — CC-BY 4.0
- "Rosetón 2021 Rose window" by carlos.moreno.g — https://sketchfab.com/3d-models/roseton-2021-rose-window-9785040d7f5245a38a2c4b1f15912b5c — CC-BY 4.0
- "Pack of old towers in ruins" by taz83 — https://sketchfab.com/3d-models/pack-of-old-towers-in-ruins-f213359c6cbb4c29bf8880764faa0fb8 — CC-BY 4.0
- "Rubble Pile - Photoscan (Optimized)" by GameDevNick — https://sketchfab.com/3d-models/rubble-pile-photoscan-optimized-5986d1487d9443b883d67b121c2c903c — CC-BY 4.0
- "Medieval braziers l Lights l Enviroment" by burnedhrum — https://sketchfab.com/3d-models/medieval-braziers-l-lights-l-enviroment-fc5c7544dd224365b055febee02bb496 — CC-BY 4.0
- "Medieval Wall Torch" by Kigha — https://sketchfab.com/3d-models/medieval-wall-torch-77db436da2844cbfb4dde0bb9b396835 — CC-BY 4.0
- HDRI, textures and props: Poly Haven (CC0) — https://polyhaven.com
