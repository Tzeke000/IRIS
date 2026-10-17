"""Labelled contact sheets and safe-to-Read copies (≤900 px longest side, <150 KB) — the one sheet maker.

Replaces rooftop/sheet.py, rooftop/thumbsheet.py, lyric_scenes/sheet_at.py (+ its 2 byte-identical copies) and ~12 scratch
variants. PIL venv, run isolated:

    PY=~/venvs/iris-v100/bin/python          # PIL + numpy + av
    $PY -I sheet.py OUT.jpg [--cols N] [--label name|index|time|none] [--labels a,b,..] FILES...
    $PY -I sheet.py OUT.jpg --dir DIR [--glob '*.png'] [--every K] [--limit N]
    $PY -I sheet.py OUT.jpg --video V.mp4 (--times 1.5,30,61 | --n 12) [--offset S]   # label = time (s)
    $PY -I sheet.py shrink SRC [OUT]            # one image → safe-to-Read JPEG (default SRC stem + .safe.jpg)
  common: [--width 900] [--max-kb 150] [--title TEXT]

GUARANTEE: the written JPEG is ≤ --width px on its LONGEST side and < --max-kb. The old copies stepped quality down to
q40 and then gave up (a big sheet could still come out >150 KB); this one keeps shrinking the pixels after quality bottoms out.
Python API: make_sheet(sources, out, ...), shrink(src, out), fit_jpeg(img, out), video_frames(path, times), grid(...).
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MAX_PX = 900          # host rule: never Read an image wider/taller than this
MAX_BYTES = 150_000   # host rule: never Read an image bigger than this
QUALITIES = (85, 78, 70, 62, 55, 48, 42, 36)


def _font(size):
    try:
        return ImageFont.load_default(size=size)       # PIL >= 10.1 (scalable built-in font)
    except TypeError:                                  # older PIL: fixed bitmap font
        return ImageFont.load_default()


def fit_jpeg(img, out, max_bytes=MAX_BYTES, max_px=MAX_PX, qualities=QUALITIES):
    """Save `img` as JPEG at `out` so that its longest side ≤ max_px AND file size < max_bytes.
    Steps quality down first, then shrinks the pixels by 15 % and retries. Returns a dict with what it did."""
    img = img.convert("RGB")
    s = min(1.0, max_px / max(img.size))
    while True:
        im = img if s >= 1.0 else img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        for q in qualities:
            im.save(out, "JPEG", quality=q, optimize=True)
            n = os.path.getsize(out)
            if n < max_bytes:
                return {"path": out, "size": im.size, "bytes": n, "quality": q, "scale": round(s, 3)}
        if max(im.size) <= 64:
            raise RuntimeError(f"cannot fit {out} under {max_bytes} bytes")
        s *= 0.85


def shrink(src, out=None, max_bytes=MAX_BYTES, max_px=MAX_PX):
    """One image → a safe-to-Read JPEG copy. Returns fit_jpeg's dict."""
    out = out or os.path.splitext(src)[0] + ".safe.jpg"
    return fit_jpeg(Image.open(src), out, max_bytes, max_px)


def label_box(draw, xy, text, font, fg=(255, 235, 0), bg=(0, 0, 0)):
    """Text on a dark box, readable on any frame."""
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle((l - 3, t - 2, r + 3, b + 2), fill=bg)
    draw.text((x, y), text, fill=fg, font=font)


def grid(images, labels=None, cols=None, width=MAX_PX, title=None, bg=(16, 16, 16)):
    """Tile PIL images into a sheet `width` px wide; each cell keeps aspect (letterboxed to the tallest aspect)."""
    n = len(images)
    if n == 0:
        raise ValueError("no images")
    cols = cols or (n if n <= 3 else 3 if n <= 9 else 4)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cw = width // cols
    aspect = max(im.height / im.width for im in images)
    ch = max(1, round(cw * aspect))
    th = 22 if title else 0
    sheet = Image.new("RGB", (cw * cols, ch * rows + th), bg)
    d = ImageDraw.Draw(sheet)
    font = _font(max(11, min(18, cw // 18)))
    if title:
        label_box(d, (4, 3), title, _font(15), fg=(255, 255, 255))
    for k, im in enumerate(images):
        im = im.convert("RGB")
        s = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        x0 = (k % cols) * cw + (cw - im.width) // 2
        y0 = th + (k // cols) * ch + (ch - im.height) // 2
        sheet.paste(im, (x0, y0))
        if labels and k < len(labels) and labels[k]:
            label_box(d, ((k % cols) * cw + 4, th + (k // cols) * ch + 3), str(labels[k]), font)
    return sheet


def video_frames(path, times):
    """[(t_actual, PIL.Image)] for the first decoded frame at/after each time (seconds). Uses PyAV, else ffmpeg."""
    want = sorted(float(t) for t in times)
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        out = []
        with av.open(path) as c:
            st = c.streams.video[0]
            k = 0
            for fr in c.decode(video=0):
                t = float(fr.pts * st.time_base)
                while k < len(want) and t >= want[k] - 1e-3:
                    out.append((t, fr.to_image()))
                    k += 1
                if k >= len(want):
                    break
        return out
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(want):
            p = os.path.join(td, f"{i}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", p], check=True)
            out.append((t, Image.open(p).copy()))
    return out


def video_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def make_sheet(sources, out, cols=None, labels="name", width=MAX_PX, max_bytes=MAX_BYTES, title=None):
    """sources: list of paths or PIL images. labels: "name" | "index" | "none" | explicit list. Returns fit_jpeg dict."""
    ims, names = [], []
    for s in sources:
        if isinstance(s, Image.Image):
            ims.append(s)
            names.append("")
        else:
            ims.append(Image.open(s))
            names.append(os.path.splitext(os.path.basename(s))[0][-28:])
    if isinstance(labels, (list, tuple)):
        lab = list(labels)
    elif labels == "index":
        lab = [f"{k:02d}" for k in range(len(ims))]
    elif labels == "name":
        lab = names
    else:
        lab = None
    sh = grid(ims, lab, cols, width, title)
    r = fit_jpeg(sh, out, max_bytes, width)
    r["n"] = len(ims)
    return r


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "shrink":
        if len(argv) < 2:
            sys.exit("usage: sheet.py shrink SRC [OUT]")
        r = shrink(argv[1], argv[2] if len(argv) > 2 else None)
        print(f"[sheet] {r['path']} {r['size'][0]}x{r['size'][1]} {r['bytes'] // 1024} KB q{r['quality']}")
        return r
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--cols", type=int)
    ap.add_argument("--label", default=None, choices=["name", "index", "time", "none"])
    ap.add_argument("--labels", help="explicit comma-separated labels")
    ap.add_argument("--dir")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--every", type=int, default=1, help="with --dir: take every K-th file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--times", help="with --video: t1,t2,... seconds")
    ap.add_argument("--n", type=int, default=0, help="with --video: N evenly spaced times")
    ap.add_argument("--offset", type=float, default=0.0, help="added to the time labels (clip start in the song)")
    ap.add_argument("--width", type=int, default=MAX_PX)
    ap.add_argument("--max-kb", type=float, default=MAX_BYTES / 1000)
    ap.add_argument("--title")
    a = ap.parse_intermixed_args(argv)          # OUT --cols 3 FILES... works
    max_bytes = int(a.max_kb * 1000)
    if a.video:
        if a.times:
            ts = [float(x) for x in a.times.split(",") if x]
        else:
            n = a.n or 12
            dur = video_duration(a.video)
            ts = [dur * (k + 0.5) / n for k in range(n)]
        frs = video_frames(a.video, ts)
        lab = [f"{t + a.offset:.2f}s" for t, _ in frs] if (a.label or "time") == "time" else (a.label or "none")
        if a.labels:
            lab = a.labels.split(",")
        r = make_sheet([im for _, im in frs], a.out, a.cols, lab, a.width, max_bytes, a.title)
    else:
        files = list(a.files)
        if a.dir:
            files += sorted(glob.glob(os.path.join(a.dir, a.glob)))
        files = [f for f in files if not os.path.basename(f).startswith(".tmp_")][::max(1, a.every)]
        if a.limit:
            files = files[:a.limit]
        if not files:
            sys.exit("[sheet] no input images")
        lab = a.labels.split(",") if a.labels else (a.label or "name")
        r = make_sheet(files, a.out, a.cols, lab, a.width, max_bytes, a.title)
    print(f"[sheet] {r['path']} {r['n']} imgs {r['size'][0]}x{r['size'][1]} {r['bytes'] // 1024} KB q{r['quality']}"
          f"{'' if r['scale'] >= 1 else ' scale ' + str(r['scale'])}")
    return r


if __name__ == "__main__":
    _main()
