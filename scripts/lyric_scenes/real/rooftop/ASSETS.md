# BE MY ONE (3-D cyberpunk rooftop party) — third-party assets

All files live under `/data/iris_assets/bemyone/` (never in git). Each download sits in its own directory with a
`SOURCE.json` (Sketchfab: `sf_download.py`, licence-gated to CC0/CC-BY; Poly Haven: `ph_fetch.py`, md5-checked).

| Asset | Source | Author | Licence | Used for |
|---|---|---|---|---|
| "Hover car" | https://sketchfab.com/3d-models/hover-car-703eba5d2b60405dbeaf53552331ec24 | Igor.Friedman | CC BY 4.0 | the breakdown flying car (nose = model local −X, verified) |
| "Cyberpunk skyscraper" (×3 models) | https://sketchfab.com/3d-models/cyberpunk-skyscraper-e7c9ef989822425e9a0e09c37b79170f · https://sketchfab.com/3d-models/cyberpunk-skyscraper-f0af05e128284feba38a1611edba3dff · https://sketchfab.com/3d-models/cyberpunk-skyscraper-5392c4e6daf944b3a17b43a58b55fe08 | Artemis26 | CC BY 4.0 | 8 hero towers along the fly-through (own facade shader, model accents kept) |
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
