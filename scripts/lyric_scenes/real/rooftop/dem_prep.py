"""Copernicus GLO-30 tile → a local elevation window in scene metres. Runs INSIDE Blender (its libtiff decodes the
COG's floating-point predictor, which tifffile can't without imagecodecs):
    blender -b --factory-startup --python dem_prep.py -- TILE.tif OUT.npz
Tile N22 E114: 3600x3600 px, 1 arc-second, row 0 = lat 23.0 (north), col 0 = lon 114.0.
→ z[ny,nx] (m), x0,y0 (scene coords of z[0,0]), step (m)"""
import math, sys
import numpy as np
import bpy
tif, out = sys.argv[sys.argv.index("--") + 1:][:2]
LAT0, LON0, OFF = 22.318, 114.170, (0.0, 500.0)
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 110574.0
im = bpy.data.images.load(tif)
im.colorspace_settings.name = "Non-Color"
W_, H_ = im.size
px = np.empty(W_ * H_ * 4, np.float32)
im.pixels.foreach_get(px)
z = px.reshape(H_, W_, 4)[::-1, :, 0].copy()      # Blender pixels start bottom-left → flip to row 0 = north
n = z.shape[0]
print("tile", z.shape, "min/max", z.min(), z.max())
R = 16000.0
step_deg = 1.0 / n
lat_hi, lat_lo = LAT0 + (R - OFF[1] + 500) / KY, LAT0 - (R + OFF[1]) / KY
lon_lo, lon_hi = LON0 - R / KX, LON0 + R / KX
r0, r1 = int((23.0 - lat_hi) / step_deg), int((23.0 - lat_lo) / step_deg)
c0, c1 = int((lon_lo - 114.0) / step_deg), int((lon_hi - 114.0) / step_deg)
c0 = max(c0, 0)
w = z[r0:r1, c0:c1][::-1]                         # flip so row index increases northward (scene +y)
lat_s = 23.0 - (r1 - 1) * step_deg
lon_s = 114.0 + c0 * step_deg
x0 = (lon_s - LON0) * KX + OFF[0]
y0 = (lat_s - LAT0) * KY + OFF[1]
np.savez_compressed(out, z=w, x0=x0, y0=y0, sx=step_deg * KX, sy=step_deg * KY)
def at(x, y):
    i, j = int(round((y - y0) / (step_deg * KY))), int(round((x - x0) / (step_deg * KX)))
    return w[i, j]
print("window", w.shape, f"x0 {x0:.0f} y0 {y0:.0f} step {step_deg*KX:.1f}x{step_deg*KY:.1f} m")
for nm, (x, y) in {"home(0,0)": (0, 0), "MongKok(0,500)": (0, 500), "T(80,520)": (80, 520), "W harbour(-2200,200)": (-2200, 200),
                   "Victoria Harbour(0,-2600)": (0, -2600), "Lion Rock(?)(-700,4300)": (-700, 4300), "Peak(-2500,-5200)": (-2500, -5200)}.items():
    print(f"  {nm}: {at(x, y):.1f} m")
