"""thumbsheet.py DIR [W] -> DIR/sheet.jpg (numbered grid of DIR/[0-9]*.jpg, <=900px wide)."""
import glob, os, sys
from PIL import Image, ImageDraw
d = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
fs = sorted(glob.glob(os.path.join(d, "[0-9]*.jpg"))); w = 900 // W
ims = [Image.open(f).convert("RGB") for f in fs]
ims = [i.resize((w, int(i.height * w / i.width))) for i in ims]; h = max(i.height for i in ims)
s = Image.new("RGB", (W * w, ((len(ims) + W - 1) // W) * h))
for k, i in enumerate(ims):
    s.paste(i, ((k % W) * w, (k // W) * h)); ImageDraw.Draw(s).text(((k % W) * w + 4, (k // W) * h + 4), f"{k:02d}", fill=(255, 255, 0))
s.save(os.path.join(d, "sheet.jpg"), quality=75); print(s.size, os.path.getsize(os.path.join(d, "sheet.jpg")))
