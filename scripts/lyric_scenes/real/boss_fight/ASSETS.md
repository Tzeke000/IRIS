# Boss fight (knight vs dragon, night) — arena assets (v1, v2, v3, v3b, v3c)

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

## Arena v2 (`build_arena_v2.py`) — all in `/data/iris_assets/medieval/{r2_props,r3,sketchfab}/` with a `SOURCE.json` each

| asset | source | author | licence | use |
|---|---|---|---|---|
| RPG 7 ( Free Model ) | https://sketchfab.com/3d-models/rpg-7-free-model-99c0af8a1803490a86edc48cd3bfc700 | javadbayat | CC-BY 4.0 | the PRIZE: gilded RPG-7 on its cradle |
| Hl. Benedikt von Nursia | https://sketchfab.com/3d-models/hl-benedikt-von-nursia-96eadf67a4244051a3115c04149d5951 | www.noe-3d.at | CC0 | saint statue inside (west aisle) |
| Hl. Servatius | https://sketchfab.com/3d-models/hl-servatius-c8a2cb0958e84e38a64c03ad57947185 | www.noe-3d.at | CC0 | saint statue inside (east aisle) |
| Hl. Nepomuk | https://sketchfab.com/3d-models/hl-nepomuk-e89f7ee51be4493da0c10aff4861d5c5 | www.noe-3d.at | CC0 | saint statue inside (east aisle) |
| Hl. Ludwig | https://sketchfab.com/3d-models/hl-ludwig-28e0d713def04474916b0f4114acc5ed | www.noe-3d.at | CC0 | the two 26 m colossi at the stair foot (one mirrored) |
| Wooden Knight Effigy (Southwark Cathedral) | https://sketchfab.com/3d-models/wooden-knight-effigy-southwark-cathedral-6f7006f3cf5649f5b5b221ba5c0c9f4a | nate_sid | CC-BY 4.0 | the knight tomb (re-surfaced as stone) |
| Gargoyle Statue (Photogrammetry) | https://sketchfab.com/3d-models/gargoyle-statue-photogrammetry-cb31805592fc4a4a99d9a294cbe50694 | pigfinite | CC-BY 4.0 | gargoyles on the gate piers |
| Gargoyle #3DST27 | https://sketchfab.com/3d-models/gargoyle-3dst27-2581387ab6454f1787e50d5909b325aa | alban | CC-BY 4.0 | gargoyle columns flanking the dais |
| Filialkirche St. Martin | https://sketchfab.com/3d-models/filialkirche-st-martin-7abe6e8db3bf4d28b32e42892d477ad4 | www.noe-3d.at | CC-BY 4.0 | the chapel + bell tower in the courtyard |
| Windleshaw Chantry/'Abbey' (c.1415) - St.Helens | https://sketchfab.com/3d-models/windleshaw-chantryabbey-c1415-sthelens-5b5f40d76f8644188712d8b3595dc947 | Capptis | CC-BY 4.0 | chantry tower ruin in the courtyard |
| St Alphage Church Ruin 2024 | https://sketchfab.com/3d-models/st-alphage-church-ruin-2024-0e4589280ddd40eb8c2307b515a2d995 | artfletch | CC-BY 4.0 | ruined church in the courtyard |
| Abbey Wall Ruins | https://sketchfab.com/3d-models/abbey-wall-ruins-37936ba6a1674f15be60075315469a43 | tonyflanagan | CC-BY 4.0 | abbey wall ruins in the courtyard |
| Medieval Houses (3 Pack) | https://sketchfab.com/3d-models/medieval-houses-3-pack-38443304774c481e9498b3e810389e91 | aagawde | CC-BY 4.0 | half-timbered houses in the courtyard (v3: one re-used outside the south wall) |
| Medieval house \| Generic Textures \| Game ready | https://sketchfab.com/3d-models/medieval-house-generic-textures-game-ready-3eb9e3b600264e2f99100fc43619497e | by__Rx | CC-BY 4.0 | the tall house in the courtyard |
| Day 206: Medieval stone house - 1scanaday | https://sketchfab.com/3d-models/day-206-medieval-stone-house-1scanaday-48233398589f4f4392b87d853ab4464e | edemaistre | CC-BY 4.0 | stone house in the courtyard (v3: one re-used outside the south wall) |
| Dirty stones pile | https://sketchfab.com/3d-models/dirty-stones-pile-893e64e62f414b439672a8770dd2869d | matousekfoto | CC-BY 4.0 | stone-pile cores of the rubble heaps |
| Poly Haven texture `rock_pitted_mossy` | https://polyhaven.com/a/rock_pitted_mossy | Poly Haven | CC0 | close-up stone detail on the scans (v3: the hand-built masonry) |

## Arena v3b (phase 2: hoard + exterior, `build_arena_v3.py`)

| asset | source | author | licence | use |
|---|---|---|---|---|
| Medieval Bow and Arrow | https://sketchfab.com/3d-models/medieval-bow-and-arrow-33209afdfd8f4c52b3bc7764b639177a | cgdude0007 | CC-BY 4.0 | the bow on the west display board (arrows not used); new download, `/data/iris_assets/medieval/r4_v3b/prop_bow_cgdude0007/` |
| Zweihänder (Great Sword/Doppelhänder) | https://sketchfab.com/3d-models/zweihander-great-sworddoppelhander-26ed92ffd2104a95b865529a67404d56 | leeeck | CC-BY 4.0 | planted point-down in the west gold mound |
| The Unparalleled Holy Sword | https://sketchfab.com/3d-models/the-unparalleled-holy-sword-192120774a404d76a5b945aad268d847 | Bl4ckGh0st | CC-BY 4.0 | the gold winged sword planted in the east gold mound |
| Holy Greatsword of the Ancient Kingdom | https://sketchfab.com/3d-models/holy-greatsword-of-the-ancient-kingdom-d3a1f493ed4d4d0aa5ad63da7a16bd33 | Bl4ckGh0st | CC-BY 4.0 | on the west display board, under the bow |
| Gems (21 different shapes) | https://sketchfab.com/3d-models/gems-21-different-shapes-1fb35b1aefae4d819dde9de90162602a | blue-odym | CC-BY 4.0 | gems in the open chests, the coin spills and on the heap (materials made non-refractive for render cost) |
| Poly Haven models `ornate_medieval_mace`, `ornate_war_hammer`, `treasure_chest` | https://polyhaven.com/a/ornate_medieval_mace · https://polyhaven.com/a/ornate_war_hammer · https://polyhaven.com/a/treasure_chest | Poly Haven | CC0 | mace + war hammer crossed on the east board; 3 more chests (2 opened: lid rotated on its hinge) |
| Poly Haven textures `painted_plaster_wall`, `medieval_blocks_02`, `roof_slates_02`, `medieval_wood`, `brown_mud_leaves_01` | https://polyhaven.com/textures | Poly Haven | CC0 | town walls / stone houses / roofs / timber framing / the dirt path and streets |

Hand-built in v3b (no external asset): the open-chest coin domes, coin skirts and spills (v1 `GN_Coins`), oak display
boards + iron pegs, the v3 terrain (valley + hills + sinking rim), the path and streets, all town cottages + the church
and spire (procedural), strip fields + hedges + haystacks, procedural conifer/deciduous/bush templates (GN-instanced),
and the ground-fog volume.

## Arena v3 (interior pass, `build_arena_v3.py`)

No new downloads. Everything v3 adds is hand-built in the script: broken screen walls (coursed ashlar), fallen column
drums, collapsed arch-ring chunks, stone pews (upright as cover, two tipped as dressing), fallen cornice blocks, masonry
fragments and gravel chips. Their stone material (`M_AshlarV3`) uses the Poly Haven texture `rock_pitted_mossy`
(CC0, https://polyhaven.com/a/rock_pitted_mossy, already on disk) re-tinted to the piers' measured albedo. The settled
rubble also re-uses fracture cells of the v1 piers / arches / vault (themselves from the CC-BY "Gothic Architectural
Structures" above). Covered by the existing credit lines.

## Arena v3c (Zeke's v3b notes)

No new downloads. The courtyard grass (4 hand-built blade/weed tuft templates, GN-instanced) is made in the script. The
"white building" fix only rewires existing materials (the glTF KHR_materials_unlit scans — St Alphage ruin, the stone
house, the gate-pier gargoyles, the stone piles — now use a lit Principled BSDF; the Poly Haven `rock_pitted_mossy` normal
adds the close-up detail).

## Credits block (paste into the video description)

- "European Dragon" by ReginaCachoa — https://sketchfab.com/3d-models/european-dragon-82f393a2e6c048ad80c171ce3b3a7b87 — CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/)
- "Medieval Knight | Sculpture | Game ready" by by__Rx — https://sketchfab.com/3d-models/medieval-knight-sculpture-game-ready-6cdd055b4afa41eb9360dbbfe75c7f10 — CC-BY 4.0
- "Gothic Architectural Structures" by oneironauticus — https://sketchfab.com/3d-models/gothic-architectural-structures-affc5b590531447788aafc23a011154c — CC-BY 4.0
- "Rosetón 2021 Rose window" by carlos.moreno.g — https://sketchfab.com/3d-models/roseton-2021-rose-window-9785040d7f5245a38a2c4b1f15912b5c — CC-BY 4.0
- "Pack of old towers in ruins" by taz83 — https://sketchfab.com/3d-models/pack-of-old-towers-in-ruins-f213359c6cbb4c29bf8880764faa0fb8 — CC-BY 4.0
- "Rubble Pile - Photoscan (Optimized)" by GameDevNick — https://sketchfab.com/3d-models/rubble-pile-photoscan-optimized-5986d1487d9443b883d67b121c2c903c — CC-BY 4.0
- "Medieval braziers l Lights l Enviroment" by burnedhrum — https://sketchfab.com/3d-models/medieval-braziers-l-lights-l-enviroment-fc5c7544dd224365b055febee02bb496 — CC-BY 4.0
- "Medieval Wall Torch" by Kigha — https://sketchfab.com/3d-models/medieval-wall-torch-77db436da2844cbfb4dde0bb9b396835 — CC-BY 4.0
- "RPG 7 ( Free Model )" by javadbayat — https://sketchfab.com/3d-models/rpg-7-free-model-99c0af8a1803490a86edc48cd3bfc700 — CC-BY 4.0
- "Wooden Knight Effigy (Southwark Cathedral)" by nate_sid — https://sketchfab.com/3d-models/wooden-knight-effigy-southwark-cathedral-6f7006f3cf5649f5b5b221ba5c0c9f4a — CC-BY 4.0
- "Gargoyle Statue (Photogrammetry)" by pigfinite — https://sketchfab.com/3d-models/gargoyle-statue-photogrammetry-cb31805592fc4a4a99d9a294cbe50694 — CC-BY 4.0
- "Gargoyle #3DST27" by alban — https://sketchfab.com/3d-models/gargoyle-3dst27-2581387ab6454f1787e50d5909b325aa — CC-BY 4.0
- "Filialkirche St. Martin" by www.noe-3d.at — https://sketchfab.com/3d-models/filialkirche-st-martin-7abe6e8db3bf4d28b32e42892d477ad4 — CC-BY 4.0
- "Windleshaw Chantry/'Abbey' (c.1415) - St.Helens" by Capptis — https://sketchfab.com/3d-models/windleshaw-chantryabbey-c1415-sthelens-5b5f40d76f8644188712d8b3595dc947 — CC-BY 4.0
- "St Alphage Church Ruin 2024" by artfletch — https://sketchfab.com/3d-models/st-alphage-church-ruin-2024-0e4589280ddd40eb8c2307b515a2d995 — CC-BY 4.0
- "Abbey Wall Ruins" by tonyflanagan — https://sketchfab.com/3d-models/abbey-wall-ruins-37936ba6a1674f15be60075315469a43 — CC-BY 4.0
- "Medieval Houses (3 Pack)" by aagawde — https://sketchfab.com/3d-models/medieval-houses-3-pack-38443304774c481e9498b3e810389e91 — CC-BY 4.0
- "Medieval house | Generic Textures | Game ready" by by__Rx — https://sketchfab.com/3d-models/medieval-house-generic-textures-game-ready-3eb9e3b600264e2f99100fc43619497e — CC-BY 4.0
- "Day 206: Medieval stone house - 1scanaday" by edemaistre — https://sketchfab.com/3d-models/day-206-medieval-stone-house-1scanaday-48233398589f4f4392b87d853ab4464e — CC-BY 4.0
- "Dirty stones pile" by matousekfoto — https://sketchfab.com/3d-models/dirty-stones-pile-893e64e62f414b439672a8770dd2869d — CC-BY 4.0
- "Medieval Bow and Arrow" by cgdude0007 — https://sketchfab.com/3d-models/medieval-bow-and-arrow-33209afdfd8f4c52b3bc7764b639177a — CC-BY 4.0
- "Zweihänder (Great Sword/Doppelhänder)" by leeeck — https://sketchfab.com/3d-models/zweihander-great-sworddoppelhander-26ed92ffd2104a95b865529a67404d56 — CC-BY 4.0
- "The Unparalleled Holy Sword" by Bl4ckGh0st — https://sketchfab.com/3d-models/the-unparalleled-holy-sword-192120774a404d76a5b945aad268d847 — CC-BY 4.0
- "Holy Greatsword of the Ancient Kingdom" by Bl4ckGh0st — https://sketchfab.com/3d-models/holy-greatsword-of-the-ancient-kingdom-d3a1f493ed4d4d0aa5ad63da7a16bd33 — CC-BY 4.0
- "Gems (21 different shapes)" by blue-odym — https://sketchfab.com/3d-models/gems-21-different-shapes-1fb35b1aefae4d819dde9de90162602a — CC-BY 4.0
- Statues "Hl. Benedikt von Nursia", "Hl. Servatius", "Hl. Nepomuk", "Hl. Ludwig" by www.noe-3d.at — CC0 (credit given as thanks)
- HDRI, textures and props: Poly Haven (CC0) — https://polyhaven.com
