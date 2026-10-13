"""Derive an RGBA hologram mask from Zeke's logo (white background baked in).
alpha = smoothstep(dist-to-white): outlines + plume solid, pale-blue fill semi-transparent.
usage: prep_logo.py SRC.png OUT.png   (librosa venv has PIL+numpy)"""
import sys
import numpy as np
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
im = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32) / 255.0
rgb = im[..., :3] * im[..., 3:4] + (1 - im[..., 3:4])          # composite any transparency onto white
d = np.sqrt(((1.0 - rgb) ** 2).sum(-1)) / np.sqrt(3.0)
a = np.clip((d - 0.04) / (0.30 - 0.04), 0, 1)
a = a * a * (3 - 2 * a)
o = np.dstack([rgb, a])
Image.fromarray((o * 255).astype(np.uint8), "RGBA").save(out)
print(out, o.shape, "alpha mean", a.mean().round(3))
