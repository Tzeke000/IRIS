"""sheet.py OUT.jpg COLS IMG... -> labelled grid, <=900 px wide, JPEG quality stepped down until <150 KB."""
import os, sys
from PIL import Image, ImageDraw
out, cols, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
w = 900 // cols
ims = []
for f in files:
    im = Image.open(f).convert("RGB"); ims.append(im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS))
h = max(i.height for i in ims)
s = Image.new("RGB", (cols * w, ((len(ims) + cols - 1) // cols) * h))
for k, (f, i) in enumerate(zip(files, ims)):
    x, y = (k % cols) * w, (k // cols) * h
    s.paste(i, (x, y)); ImageDraw.Draw(s).text((x + 4, y + 4), os.path.splitext(os.path.basename(f))[0][-26:], fill=(255, 255, 0))
for q in (85, 75, 65, 55, 45, 35):
    s.save(out, quality=q)
    if os.path.getsize(out) < 150_000: break
print(out, s.size, os.path.getsize(out))
