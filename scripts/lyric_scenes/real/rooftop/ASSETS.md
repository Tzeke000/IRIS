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
