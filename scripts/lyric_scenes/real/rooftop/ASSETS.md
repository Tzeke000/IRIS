# BE MY ONE (3-D cyberpunk rooftop party) — third-party assets

All files live under `/data/iris_assets/bemyone/` (never in git). Each download sits in its own directory with a
`SOURCE.json` (Sketchfab: `sf_download.py`, licence-gated to CC0/CC-BY; Poly Haven: `ph_fetch.py`, md5-checked).

| Asset | Source | Author | Licence | Used for |
|---|---|---|---|---|
| "Hover car" | https://sketchfab.com/3d-models/hover-car-703eba5d2b60405dbeaf53552331ec24 | Igor.Friedman | CC BY 4.0 | the breakdown flying car (nose = model local −X, verified) |
| "Cyberpunk skyscraper" (×3 models; v2 uses 2 — model f0af05… carried a baked "2023" sign and is no longer placed) | https://sketchfab.com/3d-models/cyberpunk-skyscraper-e7c9ef989822425e9a0e09c37b79170f · https://sketchfab.com/3d-models/cyberpunk-skyscraper-f0af05e128284feba38a1611edba3dff · https://sketchfab.com/3d-models/cyberpunk-skyscraper-5392c4e6daf944b3a17b43a58b55fe08 | Artemis26 | CC BY 4.0 | 8 hero towers along the fly-through (own facade shader, model accents kept) |
| "Japanese neon sign set (low poly for background)" | https://sketchfab.com/3d-models/japanese-neon-sign-set-low-poly-for-background-188e5d8dd2cb4f098871b5dd5f6fa990 | ydvisual | CC BY 4.0 | the 12 blade neon signs + billboard imagery |
| exterior_aircon_unit, modular_airduct_rectangular_01, power_box_01, modular_electric_cables (downloaded, unused) | https://polyhaven.com/models | Poly Haven | CC0 | rooftop props |
| concrete_floor_worn_001 (2k) | https://polyhaven.com/a/concrete_floor_worn_001 | Poly Haven | CC0 | wet roof |
| asphalt_02 (shared, `/data/iris_assets/polyhaven/`) | https://polyhaven.com/a/asphalt_02 | Poly Haven | CC0 | wet streets |
| shanghai_bund (2k HDRI) | https://polyhaven.com/a/shanghai_bund | Poly Haven | CC0 | reflection-only world light (camera sees the procedural night sky) |
| B612 Bold font (shared, `/data/iris_assets/fonts/b612/`) | https://github.com/polarsys/b612 | Airbus / Intactile | SIL OFL 1.1 | hologram + drone "BE MY ONE" |
| Zeke's logo (his own artwork) | — | Zeke | his | hologram; alpha mask derived by `prep_logo.py` into `/data/iris_assets/bemyone/derived/` |

Rejected: "Hong Kong signs pack" (its description says some textures came from CGTrader → licence chain unclear);
"Off-World Police Interceptor" (Blade Runner fan design); "Taiwan style Signboard" (real brand logos).

Video description credits (CC BY 4.0):
- "Hover car" by Igor.Friedman — CC BY 4.0
- "Cyberpunk skyscraper" by Artemis26 — CC BY 4.0
- "Japanese neon sign set (low poly for background)" by ydvisual — CC BY 4.0

## v2 (10-13 evening) — real city layout, real towers, more flying cars

| Asset | Source | Author | Licence | Used for |
|---|---|---|---|---|
| OpenStreetMap buildings + roads, Mong Kok / Yau Ma Tei (bbox 22.308,114.160 – 22.328,114.180), fetched via Overpass API (`osm_fetch.py`) | https://www.openstreetmap.org/copyright | OpenStreetMap contributors | **ODbL 1.0** (credit required) | 3,403 real footprints (height / building:levels tags; untagged → plausible Kowloon heights), 1,607 real roads → lit wet road ribbons, 4,265 street lamps, 2,687 left-hand-traffic cars |
| "[FREE] Atlanta, Corperate Office Building" | https://sketchfab.com/3d-models/free-atlanta-corperate-office-building-d96380fb001345cca9a9be121f3e43d5 | 99.Miles | CC BY 4.0 | 3 hero towers (has baked parallax interiors) |
| "Singapore Office Skyscraper [FREE]" | https://sketchfab.com/3d-models/singapore-office-skyscraper-free-2305f0fe03ba44229a1f768dcb48bd8f | 99.Miles | CC BY 4.0 | 2 hero towers |
| "Futuristic Hover Car" | https://sketchfab.com/3d-models/futuristic-hover-car-6243cd7c7f2e4d4cba7e0ad1a3938a35 | VertaScan | CC BY 4.0 | sky-lane hero cars (nose = local −Y, measured) |
| "Futuristic Sci-Fi Car" | https://sketchfab.com/3d-models/futuristic-sci-fi-car-525bb5ad44d34ad1b74aca686ea836a4 | dylanheyes | CC BY 4.0 | sky-lane hero cars (nose = local +Y, measured) |
| "London Office Building" (downloaded, not placed) | https://sketchfab.com/3d-models/london-office-building-8d3ea98595b4403cbf21065e8f9b7bb7 | pukkafilms | CC BY 4.0 | — |
| "LOW POLY - ASIAN TOWER BLOCK" (downloaded, not placed) | https://sketchfab.com/3d-models/low-poly-asian-tower-block-694adf5094fa41f99a245ed822293f47 | Colin.Greenall | CC BY 4.0 | — |
| "cyberpunk hovercar" (downloaded, REJECTED: has wheels + a baked ground plane) | https://sketchfab.com/3d-models/cyberpunk-hovercar-870ec9021974465a8c46a05564b59f44 | albarabhat62 | CC BY 4.0 | — |

No add-ons installed: Blosm/BlenderGIS (GPL) would also import OSM, but a 70-line Overpass fetch + numpy extrusion
was simpler and fully headless; interior mapping (Van Dongen 2008) is hand-built in shader nodes.

Video description credits to ADD for v2:
- Map data © OpenStreetMap contributors (ODbL) — https://www.openstreetmap.org/copyright
- "[FREE] Atlanta, Corperate Office Building" and "Singapore Office Skyscraper [FREE]" by 99.Miles — CC BY 4.0
- "Futuristic Hover Car" by VertaScan — CC BY 4.0
- "Futuristic Sci-Fi Car" by dylanheyes — CC BY 4.0

Research sources (10-13): shell sizes/altitudes https://www.pricey.jp/web/articles/2131 · ascent timing (NRCan)
https://download-telecharger.services.geo.ca/pub/nrcan_rncan/publications/STPublications_PublicationsST/325/325940/gid_325940.pdf ·
shell types https://fireworks.com/safety/glossary · https://montreal-fireworks.com/pyro_glossary.html · colour chemistry
https://livescience.com/65860-why-blue-fireworks-are-rare.html · interior mapping https://80.lv/articles/interior-mapping-rendering-real-rooms-without-geometry ·
ODbL attribution https://osmfoundation.org/wiki/Attribution · Overpass etiquette https://dev.overpass-api.de/overpass-doc/en/preface/commons.html ·
nav-light convention https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43-217.pdf · Spinner lift
https://bladerunner.fandom.com/wiki/Spinner · glTF axes https://docs.godotengine.org/en/4.2/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html

## v2.1 (10-13 night) — real geography to the horizon + a real drone show

| Asset | Source | Author | Licence | Used for |
|---|---|---|---|---|
| Copernicus DEM GLO-30, tile N22 E114 (`dem_cop30/`, window prepared by `dem_prep.py` inside Blender) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N22_00_E114_00_DEM/ | ESA / Copernicus | Copernicus DEM licence — free for any use with attribution | real terrain ±16 km: Kowloon hills, Lion Rock range, HK Island peaks, Victoria Harbour (sea plane to the horizon) |
| OpenStreetMap buildings + roads, urban Kowloon in 4 tiles (lat 22.293–22.346, lon 114.150–114.208) + major roads ±8 km | https://www.openstreetmap.org/copyright | OpenStreetMap contributors | ODbL 1.0 | 12,379 real footprints (natural edges: harbour S/W, hills N; east edge eroded), 19,110 roads → highway light streams |
| "Quadrocopter 2" (camera gimbal removed, LED dome added) | https://sketchfab.com/3d-models/quadrocopter-2-6885c5f1a68148e1b2b3a4f69b8b3d70 | yanix | CC BY 4.0 | the 720 show drones (GN instances) + the hero close-up drone |
| Skybrush Studio for Blender 5.0.4 (evaluated, NOT used in the build) | https://github.com/skybrush-io/studio-blender/releases/download/5.0.4/skybrush-studio-for-blender-5.0.4.zip | CollMot Robotics | GPL-3.0 | enables headless on 5.2 (62 operators, `create_takeoff_grid` works; harmless GPU draw-handler errors), but smart transitions run on Skybrush's server — free community tier capped at 64 drones → emulated offline (`drone_plan.py`: Hungarian matching + synchronised smootherstep moves + Skybrush-style safety report) |

Video description credits to ADD for v2.1:
- Terrain: Copernicus DEM GLO-30 — © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA
- "Quadrocopter 2" by yanix — CC BY 4.0
- Map data © OpenStreetMap contributors (already listed)

Drone-show research (10-13): spacing ≥2–2.5 m https://sphengineering.com/news/how-to-create-a-beautiful-drone-show-presentation-using-nothing-but-a-simple-animation-and-drone-show-software ·
https://stpetecatalyst.com/ever-wonder-how-drone-show-works-so-did-we/ · altitude/airspace https://www.intel.com/content/dam/www/public/us/en/documents/brief/planning-for-your-intel-drone-light-show-brief.pdf ·
transitions/limits https://docs.skybrush.io/public/skybrush-studio-for-blender/latest/concepts.html · https://skybrush.io/blog/2026-07-30-skybrush-studio-5.0/ ·
launch/landing https://docs.skybrush.io/public/skybrush-live-doc/latest/dialogs/adapt_show.html · community-server cap https://skybrush.io/blog/2025-02-04-skybrush-store/ ·
show size https://skyelementsdrones.com/new-york · record https://www.guinnessworldrecords.com/news/2025/7/spectacular-light-show-featuring-11000-drones-creates-record-breaking-image-in-the-sky ·
Intel Shooting Star hardware https://www.engadget.com/2016/11/06/intel-shooting-star-drone/ · camera look https://celestial.show/capturing-drone-light-shows/
