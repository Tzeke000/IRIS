"""1:1 crop comparisons and pixel diffs between renders of the same frame (quality check after a speed/look change).

Replaces .claude/skills/blender-render-optimization/scripts/{crop_compare,pixel_diff}.py (those are now shims).
PIL + numpy venv, run isolated:

    PY=~/venvs/iris-v100/bin/python
    $PY -I crops.py diff A.png B.png [--thresh 8] [--heat OUT.jpg] [--json]
        changed pixels (> THRESH/255 on any channel), %, max/mean diff, bbox of the change, PSNR.
        Be My One: far lights in roof shots 0–3 px → hide; road traffic 65 px in the outro → keep.
    $PY -I crops.py compare OUT.jpg A.png B.png [C.png ...] (--box X0,Y0,X1,Y1 ... | --auto N) [--size 300]
        [--labels before,after]   rows = crops, columns = images, at 1:1 (scaled down only if the sheet would pass
        900 px — then the scale is printed, because 1:1 is the point). --auto N picks the N tiles where A and B
        differ most, so you look where a regression would show first.

Output JPEGs are always safe to Read (≤900 px, <150 KB) via qa/sheet.fit_jpeg.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet import MAX_BYTES, MAX_PX, _font, fit_jpeg, label_box  # noqa: E402


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int16)


def diff_stats(a, b, thresh=8):
    """a, b: HxWx3 int arrays (or paths). Returns dict(changed, pct, max, mean, bbox, psnr)."""
    if isinstance(a, str):
        a = load_rgb(a)
    if isinstance(b, str):
        b = load_rgb(b)
    if a.shape != b.shape:
        raise ValueError(f"size mismatch {a.shape} vs {b.shape}")
    d = np.abs(a - b).max(-1)
    m = d > thresh
    n = int(m.sum())
    bbox = None
    if n:
        ys, xs = np.nonzero(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    mse = float(((a - b).astype(np.float64) ** 2).mean())
    return {"changed": n, "pct": round(100.0 * n / d.size, 4), "max": int(d.max()), "mean": round(float(d.mean()), 4),
            "bbox": bbox, "psnr": None if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 2), "thresh": thresh}


def heatmap(a, b, gain=8):
    """Amplified |A−B| (max over channels) on a dimmed copy of A — where the change is."""
    d = np.clip(np.abs(a - b).max(-1) * gain, 0, 255).astype(np.uint8)
    base = (a.astype(np.float32) * 0.25).astype(np.uint8)
    base[..., 0] = np.maximum(base[..., 0], d)
    base[..., 1] = np.maximum(base[..., 1], d // 3)
    return Image.fromarray(base)


def auto_boxes(a, b, n=2, size=300):
    """The n non-overlapping size×size tiles (on a size/2 stride) with the largest summed difference."""
    d = np.abs(a - b).max(-1).astype(np.float64)
    h, w = d.shape
    size = min(size, h, w)
    step = max(1, size // 2)
    cands = []
    for y in range(0, max(1, h - size + 1), step):
        for x in range(0, max(1, w - size + 1), step):
            cands.append((d[y:y + size, x:x + size].sum(), x, y))
    cands.sort(reverse=True)
    out = []
    for s, x, y in cands:
        if all(abs(x - bx) >= size or abs(y - by) >= size for bx, by, _, _ in out):
            out.append((x, y, x + size, y + size))
        if len(out) == n:
            break
    return out


def compare(out, images, boxes, labels=None, max_bytes=MAX_BYTES, max_px=MAX_PX):
    """Sheet of 1:1 crops: one row per box, one column per image. Returns fit_jpeg dict (+ 'crop_scale')."""
    ims = [Image.open(p).convert("RGB") if isinstance(p, str) else p for p in images]
    labels = labels or [os.path.splitext(os.path.basename(p))[0] if isinstance(p, str) else f"#{i}"
                        for i, p in enumerate(images)]
    cw = max(x1 - x0 for x0, y0, x1, y1 in boxes)
    chs = [y1 - y0 for x0, y0, x1, y1 in boxes]
    sheet = Image.new("RGB", (cw * len(ims), sum(chs)), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    font = _font(14)
    y = 0
    for (x0, y0, x1, y1), ch in zip(boxes, chs):
        for c, im in enumerate(ims):
            sheet.paste(im.crop((x0, y0, x1, y1)), (c * cw, y))
            label_box(d, (c * cw + 4, y + 3), f"{labels[c]}  [{x0},{y0}]", font)
        y += ch
    r = fit_jpeg(sheet, out, max_bytes, max_px)
    r["crop_scale"] = r["scale"]
    return r


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("diff", "compare"):
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"crops.py {cmd}")
    if cmd == "diff":
        ap.add_argument("a")
        ap.add_argument("b")
        ap.add_argument("--thresh", type=int, default=8)
        ap.add_argument("--heat")
        ap.add_argument("--json", action="store_true")
        x = ap.parse_args(rest)
        A, B = load_rgb(x.a), load_rgb(x.b)
        s = diff_stats(A, B, x.thresh)
        if x.heat:
            s["heat"] = fit_jpeg(heatmap(A, B), x.heat)["path"]
        if x.json:
            print(json.dumps(s))
        else:
            print(f"changed pixels > {s['thresh']}/255: {s['changed']} ({s['pct']:.3f} %), max diff {s['max']}, "
                  f"mean {s['mean']}, bbox {s['bbox']}, psnr {s['psnr']}")
        return s
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--box", action="append", default=[], help="X0,Y0,X1,Y1 (repeatable)")
    ap.add_argument("--auto", type=int, default=0, help="pick N most-different tiles between the first two images")
    ap.add_argument("--size", type=int, default=300, help="tile size for --auto")
    ap.add_argument("--labels")
    x = ap.parse_intermixed_args(rest)
    boxes = [tuple(int(v) for v in b.split(",")) for b in x.box]
    if x.auto:
        if len(x.images) < 2:
            sys.exit("--auto needs at least two images")
        boxes += auto_boxes(load_rgb(x.images[0]), load_rgb(x.images[1]), x.auto, x.size)
    if not boxes:
        sys.exit("give --box X0,Y0,X1,Y1 or --auto N")
    r = compare(x.out, x.images, boxes, x.labels.split(",") if x.labels else None)
    note = "" if r["crop_scale"] >= 1 else f"  (NOT 1:1 — scaled {r['crop_scale']}; use fewer/smaller boxes)"
    print(f"[crops] {r['path']} {r['size'][0]}x{r['size'][1]} {r['bytes'] // 1024} KB boxes {boxes}{note}")
    return r


if __name__ == "__main__":
    _main()
