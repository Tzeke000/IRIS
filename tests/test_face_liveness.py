"""Unit tests for brain/face_liveness.py pure logic — synthetic embeddings/scores only,
no live engine, no models, no camera.

    /home/iris/venvs/iris-v100/bin/python tests/test_face_liveness.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import face_liveness as fl  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: object = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if not cond else ""))
    if not cond:
        FAILS.append(name)


rng = np.random.RandomState(7)


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def jitter(base, amount):
    """A new sample ~`amount` noise away from base (cos stays high for small amount)."""
    return unit(base + amount * unit(rng.randn(512)))


# ── synthetic identities: a helmet, a person (zeke), a stranger ───────────────
helmet = unit(rng.randn(512))
zeke = unit(rng.randn(512))
stranger = unit(rng.randn(512))
decoys = {"helmet": [jitter(helmet, 0.4) for _ in range(6)]}   # several samples, varied "light"
persons = {"zeke": [jitter(zeke, 0.4) for _ in range(6)]}

# 1. the helmet under new light matches the decoy gallery and is called a decoy
h_now = jitter(helmet, 0.5)
dn, ds = fl.best_match(h_now, decoys)
pn, ps = fl.best_match(h_now, persons)
check("helmet matches decoy gallery", dn == "helmet" and ds >= fl.DECOY_SIM_MIN, (dn, ds))
check("helmet decoy decision", fl.decoy_decision(ds, ps), (ds, ps))

# 2. a PERSON standing exactly where the helmet sits does NOT match (different embedding)
for who, base in (("zeke", zeke), ("stranger", stranger)):
    e = jitter(base, 0.5)
    _, ds2 = fl.best_match(e, decoys)
    _, ps2 = fl.best_match(e, persons)
    near = fl.near_position([100, 100, 160, 170], [[100, 100, 160, 170]])
    check(f"{who} at the helmet's spot is near=True", near is True)
    check(f"{who} at the helmet's spot is NOT a decoy (appearance gate)",
          not fl.decoy_decision(ds2, ps2, near=near, require_near=True), (ds2, ps2))

# 3. position alone never makes a decoy; require_near can only remove matches
check("position alone never decoy", not fl.decoy_decision(0.2, 0.0, near=True))
check("require_near blocks a far appearance match", not fl.decoy_decision(0.9, 0.0, near=False, require_near=True))
check("require_near passes a near appearance match", fl.decoy_decision(0.9, 0.0, near=True, require_near=True))
check("margin: decoy-like AND person-like is not a decoy", not fl.decoy_decision(0.6, 0.55))
check("near_position None with no memory", fl.near_position([0, 0, 10, 10], []) is None)

# 4. hysteresis: median + latch
h = fl.Hysteresis(window_s=1.5, hi=0.7, lo=0.35, min_samples=3)
check("starts uncertain", h.state == "uncertain")
h.update(0.0, 0.9); h.update(0.5, 0.9)
check("uncertain below min_samples", h.state == "uncertain")
h.update(1.0, 0.9)
check("goes live on high median", h.state == "live")
h.update(1.5, 0.1)                      # one bad frame: median still high
check("single low outlier does not flip", h.state == "live", h.median)
h.update(2.0, 0.5); h.update(2.5, 0.5)  # mid-band: hold previous state
check("mid band holds state (latch)", h.state == "live", h.median)
for t in (3.0, 3.5, 4.0, 4.5):
    h.update(t, 0.1)
check("sustained low -> spoof", h.state == "spoof", h.median)
h.update(5.0, 0.95)
check("single high outlier does not un-spoof", h.state == "spoof", h.median)
check("window trims old samples", all(5.0 - ts <= 1.5 for ts, _ in h.buf) and len(h.buf) == 4, list(h.buf))

# 5. fusion rule
check("recognised person never demoted by spoof", fl.fuse("zeke", "helmet", True, "spoof") == "person:zeke")
check("recognised person never a decoy", fl.fuse("zeke", None, False, "live") == "person:zeke")
check("unknown + decoy -> decoy", fl.fuse("unknown", "helmet", True, "live") == "decoy:helmet")
check("unknown + spoof -> face_like (never visitor)", fl.fuse("unknown", None, False, "spoof") == "face_like")
check("unknown + live -> visitor", fl.fuse("unknown", None, False, "live") == "visitor")
check("unknown + too few samples -> uncertain", fl.fuse("unknown", None, False, "uncertain") == "uncertain")
check("decoy name but no hit -> not decoy", fl.fuse("unknown", "helmet", False, "live") == "visitor")
check("actual verdict mirrors live path", fl.actual_verdict("unknown") == "visitor" and fl.actual_verdict("zeke") == "person:zeke")

# 6. Silent-Face crop convention
lx, ly, rx, ry = fl.antispoof_box(1280, 720, [600, 300, 80, 100], 2.7)
check("crop centred + scaled", abs((lx + rx) / 2 - 640) <= 1 and abs((rx - lx) - 216) <= 1, (lx, ly, rx, ry))
lx, ly, rx, ry = fl.antispoof_box(1280, 720, [0, 0, 80, 100], 4.0)
check("crop shifted inside image at the corner", lx == 0 and ly == 0 and rx <= 1279 and ry <= 719, (lx, ly, rx, ry))
lx, ly, rx, ry = fl.antispoof_box(1280, 720, [100, 100, 600, 600], 4.0)
check("scale clamped when face is huge", ry - ly <= 719, (lx, ly, rx, ry))

# 7. gallery + log round trip on a temp dir (never the real state/)
with tempfile.TemporaryDirectory() as td:
    gal = fl.DecoyGallery(Path(td) / "decoys")
    img = np.zeros((40, 40, 3), dtype="uint8")
    for s in decoys["helmet"][:3]:
        gal.add("helmet", s, img, {"bbox": [1, 2, 3, 4]})
    gal.load(force=True)
    check("gallery reloads samples", len(gal.samples.get("helmet", [])) == 3)
    check("gallery remembers boxes", gal.remembered_boxes("helmet") == [[1, 2, 3, 4]] * 3)
    log = fl.ShadowLog(Path(td) / "shadow.jsonl", max_bytes=300)
    for i in range(20):
        log.write({"ts": 1000 + i, "source": "live", "actual_verdict": "visitor",
                   "would_verdict": "decoy:helmet", "decoy_sim": 0.8, "antispoof_median": 0.7, "differs": True})
    check("log rotates (one backup)", (Path(td) / "shadow.jsonl.1").exists())
    recs = fl.read_log(Path(td) / "shadow.jsonl", since_ts=0)
    summ = fl.summarize(recs)
    check("summary groups by actual/would", any("would=decoy:helmet" in k for k in summ["groups"]), summ)

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
