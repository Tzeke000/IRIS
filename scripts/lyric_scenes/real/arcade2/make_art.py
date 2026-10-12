"""Static printed art for the arcade (deterministic, PIL + numpy):

    ~/venvs/iris-v100/bin/python scripts/lyric_scenes/real/arcade/make_art.py \
        [--logo state/tzeke_songs/_assets/tzeke000_symbol.png] [--out assets/scenes/arcade/art]

marquee_hero.png   backlit marquee: Tzeke000 emblem + "HIT A BUMP" (the only real brand here)
marquee_<k>.png    made-up games for the other cabinets (layout.OTHER_GAMES)
side_hero.png      side art (mapped to the side-panel profile bbox), side_generic.png (hue-shifted per cab)
bezel_hero.png     printed bezel around the CRT, cpanel_hero.png + cpanel_wear.png (control panel)
carpet.png         tileable cosmic arcade carpet (+ carpet_glow.png: UV-reactive fibres mask)
sign_rules.png     wall sign, coindoor.png coin-door plate decals
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402

FONTS = Path(__file__).resolve().parents[4] / "assets" / "fonts"


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def keyed_logo(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    white = a.min(-1) > 225
    hh, ww = white.shape
    bg = np.zeros_like(white)
    q = deque([(0, 0), (0, ww - 1), (hh - 1, 0), (hh - 1, ww - 1)])
    while q:
        y, x = q.popleft()
        if 0 <= y < hh and 0 <= x < ww and white[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.MinFilter(3)))  # eat the white fringe
    return Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")


def grad(w, h, top, bot, horiz=False):
    t = np.linspace(0, 1, w if horiz else h)
    c = np.array(top, float)[None] * (1 - t[:, None]) + np.array(bot, float)[None] * t[:, None]
    if horiz:
        a = np.repeat(c[None], h, 0)
    else:
        a = np.repeat(c[:, None], w, 1)
    return Image.fromarray(a.astype(np.uint8), "RGB")


def gradient_text(size_wh, text, fnt, top, bot, stroke=10, stroke_col=(10, 6, 16), shadow=14):
    """Text filled with a vertical gradient, black stroke, drop shadow, white top highlight."""
    w, h = size_wh
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bb = d.textbbox((0, 0), text, font=fnt, stroke_width=stroke)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = (w - tw) // 2 - bb[0], (h - th) // 2 - bb[1]
    sh = Image.new("L", (w, h), 0)
    ImageDraw.Draw(sh).text((x + shadow, y + shadow), text, font=fnt, fill=200, stroke_width=stroke)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    layer.paste((0, 0, 0, 255), (0, 0), sh)
    st = Image.new("L", (w, h), 0)
    ImageDraw.Draw(st).text((x, y), text, font=fnt, fill=255, stroke_width=stroke)
    layer.paste((*stroke_col, 255), (0, 0), st)
    fill = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fill).text((x, y), text, font=fnt, fill=255)
    g = grad(w, h, top, bot).convert("RGBA")
    # chrome-ish horizon line through the middle of the letters
    ga = np.asarray(g).copy()
    mid = y + bb[1] + th // 2
    ga[mid - 2:mid + 2, :, :3] = (255, 255, 255)
    ga[:y + bb[1] + th // 5] = np.minimum(ga[:y + bb[1] + th // 5].astype(int) + 60, 255)
    layer.paste(Image.fromarray(ga), (0, 0), fill)
    return layer


def starburst(w, h, cx, cy, n, c0, c1):
    img = Image.new("RGB", (w, h), c0)
    d = ImageDraw.Draw(img)
    R = max(w, h) * 1.5
    for k in range(n):
        if k % 2:
            continue
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        d.polygon([(cx, cy), (cx + R * math.cos(a0), cy + R * math.sin(a0)),
                   (cx + R * math.cos(a1), cy + R * math.sin(a1))], fill=c1)
    return img


def marquee_hero(logo, out):
    w, h = 2048, 660
    img = starburst(w, h, w // 2, h + 200, 48, (40, 6, 50), (70, 12, 80))
    img = Image.blend(img, grad(w, h, (20, 4, 40), (160, 20, 60)), 0.55)
    # pixel-block border (8-bit nod)
    d = ImageDraw.Draw(img)
    for x in range(0, w, 40):
        c = [(255, 200, 40), (255, 90, 40), (240, 40, 120)][(x // 40) % 3]
        d.rectangle([x + 2, 6, x + 36, 34], fill=c)
        d.rectangle([x + 2, h - 34, x + 36, h - 6], fill=c)
    img = img.convert("RGBA")
    t = gradient_text((1500, 470), "HIT A BUMP", font("Anton-Regular.ttf", 330),
                      (255, 240, 120), (255, 60, 30), stroke=14)
    img.alpha_composite(t, ((w - 1500) // 2 + 120, 70))
    lg = logo.resize((420, 420), Image.LANCZOS)
    glow = Image.new("RGBA", (520, 520), (0, 0, 0, 0))
    gl = Image.new("L", (520, 520), 0)
    gl.paste(lg.split()[3], (50, 50))
    gl = gl.filter(ImageFilter.GaussianBlur(18))
    glow.paste((255, 255, 255, 180), (0, 0), gl)
    img.alpha_composite(glow, (40, (h - 520) // 2))
    img.alpha_composite(lg, (90, (h - 420) // 2))
    d = ImageDraw.Draw(img)
    f2 = font("BebasNeue-Regular.ttf", 64)
    d.text((1180, 548), "TZEKE000", font=f2, fill=(140, 220, 255), anchor="mm")
    img.convert("RGB").save(out / "marquee_hero.png")


def marquee_other(k, name, out):
    w, h = 2048, 660
    rng = np.random.default_rng(100 + k)
    hues = [((20, 30, 120), (0, 200, 255)), ((120, 0, 90), (255, 90, 200)), ((0, 70, 40), (90, 255, 120)),
            ((90, 40, 0), (255, 200, 40)), ((60, 0, 120), (160, 120, 255)), ((100, 0, 0), (255, 120, 60))][k % 6]
    img = grad(w, h, (8, 6, 16), hues[0]).convert("RGBA")
    d = ImageDraw.Draw(img)
    for _ in range(140):                                   # star field
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.random() * 3
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
    for q in range(10):                                    # perspective grid floor
        y = int(h * 0.62 + (q / 10) ** 2 * h * 0.38)
        d.line([(0, y), (w, y)], fill=(*hues[1], 255), width=3)
    for q in range(-12, 13):
        d.line([(w // 2 + q * 30, int(h * 0.62)), (w // 2 + q * 260, h)], fill=(*hues[1], 255), width=3)
    t = gradient_text((1900, 430), name, font(["ArchivoBlack-Regular.ttf", "Anton-Regular.ttf", "BebasNeue-Regular.ttf"][k % 3], 250),
                      (255, 255, 255), hues[1], stroke=12)
    img.alpha_composite(t, ((w - 1900) // 2, 40))
    img.convert("RGB").save(out / f"marquee_{k}.png")


def side_art(logo, out, hero=True):
    # texture maps the side-profile bbox: u = (y - ymin)/(ymax - ymin), v = z / zmax
    ys = [p[0] for p in L.SIDE]
    zs = [p[1] for p in L.SIDE]
    ymin, ymax, zmax = min(ys), max(ys), max(zs)
    w = 1024
    h = int(w * zmax / (ymax - ymin))
    img = Image.new("RGB", (w, h), (10, 10, 14))
    d = ImageDraw.Draw(img)
    cols = [(255, 60, 30), (255, 150, 30), (255, 220, 60)] if hero else [(255, 255, 255), (170, 170, 170), (90, 90, 90)]
    # three bold diagonal swooshes
    for k, c in enumerate(cols):
        off = k * 90
        poly = [(-50, h * 0.95 - off), (w + 50, h * 0.35 - off), (w + 50, h * 0.35 - off - 60), (-50, h * 0.95 - off - 60)]
        d.polygon(poly, fill=c)
    # pixel squares trailing
    rng = np.random.default_rng(4 if hero else 5)
    for _ in range(60):
        x = rng.integers(0, w)
        y = int(h * 0.95 - x * (h * 0.6) / w - rng.integers(-260, 40))
        s = int(rng.integers(10, 34))
        d.rectangle([x, y, x + s, y + s], fill=cols[int(rng.integers(0, 3))])
    img = img.convert("RGBA")
    if hero:
        lg = logo.resize((600, 600), Image.LANCZOS)
        img.alpha_composite(lg, ((w - 600) // 2 + 60, int(h * 0.06)))
        t = gradient_text((1000, 260), "HIT A BUMP", font("Anton-Regular.ttf", 180), (255, 240, 120), (255, 60, 30), stroke=10)
        img.alpha_composite(t, (20, int(h * 0.42)))
    img.convert("RGB").save(out / ("side_hero.png" if hero else "side_generic.png"))


def bezel_art(out):
    w = 1024
    h = int(w * 0.598 / (2 * L.INNER))
    img = Image.new("RGB", (w, h), (8, 8, 10))
    d = ImageDraw.Draw(img)
    sw = L.SCREEN_W / (2 * L.INNER) * w
    sh = L.SCREEN_H / 0.598 * h
    x0, y0 = (w - sw) / 2, (h - sh) / 2
    for k, c in enumerate([(255, 60, 30), (255, 160, 30), (60, 180, 255)]):
        m = 14 + k * 9
        d.rounded_rectangle([x0 - m, y0 - m, x0 + sw + m, y0 + sh + m], radius=26 + m, outline=c, width=4)
    f = font("BebasNeue-Regular.ttf", 34)
    d.text((w / 2, y0 + sh + 52), "HIT A BUMP  -  1 OR 2 PLAYERS  -  JUMP ON THE BEAT", font=f, fill=(220, 220, 230), anchor="mm")
    d.text((w / 2, y0 - 50), "INSERT TOKEN TO PLAY", font=f, fill=(255, 200, 60), anchor="mm")
    img.save(out / "bezel_hero.png")


def cpanel_art(out):
    w = 1024
    h = int(w * L.CP_LEN / (2 * L.INNER))
    img = Image.new("RGB", (w, h), (16, 16, 22))
    d = ImageDraw.Draw(img)
    # u (m) -> px, v (m from the front lip) -> py (image top = BACK of the panel)
    px = lambda u: (u + L.INNER) / (2 * L.INNER) * w           # noqa: E731
    py = lambda v: (1 - v / L.CP_LEN) * h                       # noqa: E731
    # swooshes
    d.polygon([(0, h), (w * 0.55, h * 0.15), (w * 0.62, h * 0.15), (w * 0.07, h)], fill=(200, 40, 30))
    d.polygon([(w * 0.09, h), (w * 0.64, h * 0.15), (w * 0.68, h * 0.15), (w * 0.13, h)], fill=(240, 150, 30))
    d.rectangle([0, 0, w, 10], fill=(60, 60, 70))
    # joystick plate ring + 8-way arrows
    jx, jy = px(L.JOY[0]), py(L.JOY[1])
    r = 70
    d.ellipse([jx - r, jy - r, jx + r, jy + r], outline=(230, 230, 230), width=5)
    for a in range(8):
        ang = a * math.pi / 4
        ax, ay = jx + math.cos(ang) * (r + 18), jy + math.sin(ang) * (r + 18)
        d.regular_polygon((ax, ay, 9), 3, rotation=-math.degrees(ang) - 90, fill=(230, 230, 230))
    f = font("BebasNeue-Regular.ttf", 30)
    col = dict(red=(255, 70, 60), yellow=(255, 220, 60), blue=(80, 160, 255), green=(80, 230, 120),
               white=(240, 240, 240), purple=(190, 110, 255))
    for (u, v, c, lab) in L.BUTTONS:
        bx, by = px(u), py(v)
        rr = L.BTN_R / (2 * L.INNER) * w + 10
        d.ellipse([bx - rr, by - rr, bx + rr, by + rr], outline=col[c], width=4)
        if lab:
            d.text((bx, by + rr + 18), lab, font=f, fill=(240, 240, 240), anchor="mm")
    for (u, v, lab) in L.START:
        d.text((px(u), py(v) + 26), lab + " START", font=font("BebasNeue-Regular.ttf", 22), fill=(220, 220, 220), anchor="mm")
    img.save(out / "cpanel_hero.png")
    # wear: scuffs around the stick + buttons, scratches, the front edge rubbed by wrists
    rng = np.random.default_rng(9)
    wear = Image.new("L", (w, h), 0)
    dw = ImageDraw.Draw(wear)
    for _ in range(900):
        x, y = rng.random() * w, rng.random() * h
        ln = rng.random() * 40 + 4
        a = rng.random() * math.pi
        dw.line([(x, y), (x + math.cos(a) * ln, y + math.sin(a) * ln)], fill=int(30 + rng.random() * 70), width=1)
    for (cx, cy, rr) in [(jx, jy, 120)] + [(px(u), py(v), 50) for (u, v, _, _) in L.BUTTONS]:
        for _ in range(200):
            a = rng.random() * 6.283
            r0 = rr * (0.4 + rng.random() * 0.6)
            dw.arc([cx - r0, cy - r0, cx + r0, cy + r0], math.degrees(a), math.degrees(a) + 20, fill=int(40 + rng.random() * 70))
    edge = np.zeros((h, w), np.float32)
    edge[int(h * 0.82):] = np.linspace(0, 1, h - int(h * 0.82))[:, None]
    wa = np.maximum(np.asarray(wear.filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32), edge * 150 * (0.6 + 0.4 * rng.random((h, w))))
    Image.fromarray(wa.clip(0, 255).astype(np.uint8)).save(out / "cpanel_wear.png")


def carpet(out):
    """Tileable 'cosmic' arcade carpet: deep navy pile with neon shapes (they glow under the
    blacklight). Drawn with wrap-around so the tile repeats seamlessly."""
    s = 2048
    rng = np.random.default_rng(21)
    base = np.asarray(grad(s, s, (14, 10, 34), (16, 12, 40))).astype(np.float32)
    pile = rng.normal(0, 7, (s, s, 1)).astype(np.float32)
    img = Image.fromarray(np.clip(base + pile, 0, 255).astype(np.uint8))
    glow = Image.new("L", (s, s), 0)
    d, dg = ImageDraw.Draw(img), ImageDraw.Draw(glow)
    neon = [(0, 230, 255), (255, 40, 200), (255, 230, 40), (60, 255, 90), (255, 120, 30)]
    for k in range(170):
        c = neon[k % len(neon)]
        x, y = rng.random() * s, rng.random() * s
        kind = rng.integers(0, 5)
        sz = 30 + rng.random() * 70
        for ox in (-s, 0, s):
            for oy in (-s, 0, s):
                X, Y = x + ox, y + oy
                if not (-200 < X < s + 200 and -200 < Y < s + 200):
                    continue
                if kind == 0:                                     # squiggle
                    pts = [(X + t * 14, Y + 18 * math.sin(t * 1.3 + k)) for t in range(10)]
                    d.line(pts, fill=c, width=9)
                    dg.line(pts, fill=255, width=9)
                elif kind == 1:                                   # ring planet
                    d.ellipse([X - sz / 2, Y - sz / 2, X + sz / 2, Y + sz / 2], outline=c, width=8)
                    dg.ellipse([X - sz / 2, Y - sz / 2, X + sz / 2, Y + sz / 2], outline=255, width=8)
                    d.ellipse([X - sz, Y - sz / 6, X + sz, Y + sz / 6], outline=c, width=5)
                    dg.ellipse([X - sz, Y - sz / 6, X + sz, Y + sz / 6], outline=255, width=5)
                elif kind == 2:                                   # triangle
                    a = rng.random() * 6.283 if ox == 0 and oy == 0 else k
                    tri = [(X + sz / 2 * math.cos(a + q * 2.094), Y + sz / 2 * math.sin(a + q * 2.094)) for q in range(3)]
                    d.polygon(tri, outline=c, width=7)
                    dg.polygon(tri, outline=255, width=7)
                elif kind == 3:                                   # star dot cluster
                    for q in range(5):
                        xx, yy = X + (q * 23 % 60) - 30, Y + (q * 37 % 60) - 30
                        d.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=c)
                        dg.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=255)
                else:                                             # zig-zag bolt
                    pts = [(X, Y), (X + 25, Y + 20), (X + 5, Y + 32), (X + 35, Y + 60)]
                    d.line(pts, fill=c, width=8)
                    dg.line(pts, fill=255, width=8)
    img = img.filter(ImageFilter.GaussianBlur(1.2))          # tufted pile softens the print
    img.save(out / "carpet.png")
    glow.filter(ImageFilter.GaussianBlur(1.5)).save(out / "carpet_glow.png")


def sign_rules(out):
    w, h = 1024, 640
    img = Image.new("RGB", (w, h), (236, 232, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, w - 12, h - 12], outline=(200, 30, 30), width=14)
    f1, f2 = font("Anton-Regular.ttf", 110), font("BebasNeue-Regular.ttf", 70)
    d.text((w / 2, 150), "NO FOOD OR DRINK", font=f1, fill=(200, 30, 30), anchor="mm")
    d.text((w / 2, 280), "ON MACHINES", font=f1, fill=(200, 30, 30), anchor="mm")
    d.text((w / 2, 430), "TOKENS ONLY  -  NO REFUNDS", font=f2, fill=(30, 30, 30), anchor="mm")
    d.text((w / 2, 530), "PLEASE PLAY NICE", font=f2, fill=(30, 30, 30), anchor="mm")
    img.save(out / "sign_rules.png")


def coindoor(out):
    w, h = 512, 616
    img = Image.new("RGB", (w, h), (150, 152, 158))
    a = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(3)
    brush = rng.normal(0, 1, (h, 1)) * 10 + rng.normal(0, 1, (h, w)) * 4    # horizontal brushing
    img = Image.fromarray(np.clip(a + brush[..., None], 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = font("BebasNeue-Regular.ttf", 40)
    d.text((w / 2, 60), "INSERT TOKEN", font=f, fill=(40, 40, 46), anchor="mm")
    d.text((w / 2, h - 50), "COIN RETURN", font=font("BebasNeue-Regular.ttf", 28), fill=(40, 40, 46), anchor="mm")
    img.save(out / "coindoor.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo", type=Path, default=Path("state/tzeke_songs/_assets/tzeke000_symbol.png"))
    ap.add_argument("--out", type=Path, default=Path("assets/scenes/arcade/art"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logo = keyed_logo(a.logo)
    logo.save(a.out / "logo_keyed.png")
    marquee_hero(logo, a.out)
    for k, name in enumerate(L.OTHER_GAMES):
        marquee_other(k, name, a.out)
    side_art(logo, a.out, True)
    side_art(logo, a.out, False)
    bezel_art(a.out)
    cpanel_art(a.out)
    carpet(a.out)
    sign_rules(a.out)
    coindoor(a.out)
    print("[make_art] ->", a.out)


if __name__ == "__main__":
    main()
