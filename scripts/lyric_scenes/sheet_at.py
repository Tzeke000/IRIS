"""Contact sheet at explicit times (seconds, relative to the clip start).
usage: sheet_at.py <video> <out.jpg> <cols> t1,t2,... [label_offset_s]
(<=900px wide; JPEG quality stepped down until <150 KB)"""
import os
import sys
import av
from PIL import Image, ImageDraw
src, out, cols = sys.argv[1], sys.argv[2], int(sys.argv[3])
want = sorted(float(x) for x in sys.argv[4].split(","))
off = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
c = av.open(src); st = c.streams.video[0]
frames, k = [], 0
for fr in c.decode(video=0):
    t = float(fr.pts * st.time_base)
    while k < len(want) and t >= want[k] - 1e-3:
        frames.append((t, fr.to_image())); k += 1
    if k >= len(want):
        break
w, h = frames[0][1].size
tw = 900 // cols; th = int(h * tw / w)
rows = (len(frames) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, th * rows), "black")
for i, (t, im) in enumerate(frames):
    im = im.resize((tw, th), Image.LANCZOS)
    ImageDraw.Draw(im).text((4, 4), f"{t + off:.2f}s", fill=(255, 255, 0))
    sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
q = 82
while True:
    sheet.save(out, quality=q)
    if os.path.getsize(out) < 150_000 or q <= 40:
        break
    q -= 6
print(out, sheet.size, os.path.getsize(out) // 1024, "KB q", q)
