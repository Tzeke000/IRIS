"""Offline liveness eval — replay recorded clips / frames through the SAME scoring as the
shadow runner (brain/face_liveness.py) and print ISO/IEC 30107-3-style numbers.

Folder layout (one sub-folder per class; the prefix says which side it is on):
    <root>/real_zeke_still_60s/        clip.mp4 | frames *.jpg/*.png
    <root>/real_zeke_lying_eyes_closed/
    <root>/attack_phone_photo_1m/
    <root>/attack_phone_photo_2m/
    <root>/attack_tv_face/
    <root>/attack_helmet/              (decoy - expected decoy:helmet)
  real_*   = bona fide presentations (a living person)
  attack_* = presentation attacks / decoys (photo, screen, helmet, mask)
  Optional <class>/roi.json = [x1, y1, x2, y2]: only faces whose CENTRE lies inside are
  scored (e.g. the TV while Zeke sits on the couch in the same shot). With a roi every
  face in it is scored; without one, the largest face (or all, --all-faces).

Metrics (per frame AND per clip, clip = the final latched state):
  APCER(class) = attack frames the fused verdict would accept as a LIVING face
                 (visitor or person:*)                                  - lower is better
  BPCER(class) = real frames the fused verdict would REJECT as not-a-person
                 (face_like or decoy:*)                                  - must be ~0
  ACER         = (max APCER over attack classes + mean BPCER) / 2
  no_face      = frames where no face was detected at all (lying down / turned away)
  uncertain    = frames still in the hysteresis dead band
Also printed: anti-spoof-only APCER/BPCER at a few thresholds, and decoy/person sims.

Usage:
  /home/iris/venvs/iris-v100/bin/python scripts/vision/liveness_eval.py <root> \
      [--fps 5] [--det 0.5] [--all-faces] [--out scratch/liveness/eval_<ts>.json]
CPU only (no GPU, never touches the live runtime or the camera).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   # append (not insert): repo modules never shadow stdlib/site

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
CACHE = ROOT / "scratch" / "liveness" / "person_gallery_cache.npz"


def person_gallery(fl, sc) -> dict:
    """Enrolled PEOPLE (faces/<pid>/*, '_' dirs skipped), embedded on CPU; cached by mtimes."""
    import cv2
    import numpy as np
    files = sorted(p for p in (ROOT / "faces").glob("*/*")
                   if p.parent.name[:1] != "_" and p.suffix.lower() in IMG_EXT)
    sig = json.dumps([[str(p), p.stat().st_mtime] for p in files])
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["sig"]) == sig:
            out: dict = {}
            for pid, e in zip(z["pids"], z["embs"]):
                out.setdefault(str(pid), []).append(e)
            return out
    pids, embs = [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < 640:   # same upscale the live loader uses for tight crops
            s = 640 / max(1, min(h, w))
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        dets = sc.embedder.detect_full(img, thresh=0.3)
        if not dets:
            continue
        d = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        e = sc.embedder.embed_at(img, d["bbox"], kps=d["kps"], det_score=d["det_score"])
        if e is not None:
            pids.append(p.parent.name)
            embs.append(e["emb"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if embs:
        np.savez(CACHE, sig=np.array(sig), pids=np.array(pids), embs=np.stack(embs))
    out = {}
    for pid, e in zip(pids, embs):
        out.setdefault(pid, []).append(e)
    return out


def frames_of(path: Path, fps: float):
    """Yield (t_seconds, frame) for an image folder (0.5 s apart, sorted) or a video file."""
    import cv2
    if path.is_dir():
        imgs = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXT)
        for i, p in enumerate(imgs):
            f = cv2.imread(str(p))
            if f is not None:
                yield i * 0.5, f, p.name
        return
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / src_fps, f, f"{path.name}#{i}"
        i += 1
    cap.release()


def units_of(class_dir: Path):
    """A class folder holds clips (videos) and/or loose frames (one sequence)."""
    vids = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VID_EXT)
    imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    subdirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
    return vids + ([class_dir] if imgs else []) + subdirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--det", type=float, default=0.5, help="detector threshold (live engine uses 0.5)")
    ap.add_argument("--all-faces", action="store_true", help="score every face, not just the largest")
    ap.add_argument("--min-face-px", type=int, default=28, help="live engine drops boxes smaller than this")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from brain import face_liveness as fl
    sc = fl.Scorer()
    persons = person_gallery(fl, sc)
    print(f"persons: { {k: len(v) for k, v in persons.items()} }  decoys: { {k: len(v) for k, v in sc.decoys.samples.items()} }")

    root = Path(a.root)
    rows, per_class = [], {}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        side = "real" if cdir.name.startswith("real_") else "attack" if cdir.name.startswith("attack_") else None
        if side is None:
            print(f"skip {cdir.name} (prefix real_ or attack_)")
            continue
        roi = None
        if (cdir / "roi.json").exists():
            roi = json.loads((cdir / "roi.json").read_text())
        c = per_class.setdefault(cdir.name, {"side": side, "frames": 0, "no_face": 0, "accept": 0, "reject": 0,
                                              "uncertain": 0, "clips": [], "as": [], "dsim": [], "psim": [],
                                              "verdicts": {}})
        for unit in units_of(cdir):
            hyst = fl.Hysteresis()
            last = None
            for t, frame, label in frames_of(unit, a.fps):
                c["frames"] += 1
                all_dets = sc.embedder.detect_full(frame, thresh=0.2)   # keypoint source, like the runner
                dets = [d for d in all_dets if d["det_score"] >= a.det]
                dets = [d for d in dets
                        if (d["bbox"][2] - d["bbox"][0]) >= a.min_face_px and (d["bbox"][3] - d["bbox"][1]) >= a.min_face_px]
                if roi:
                    dets = [d for d in dets if roi[0] <= (d["bbox"][0] + d["bbox"][2]) / 2 <= roi[2]
                            and roi[1] <= (d["bbox"][1] + d["bbox"][3]) / 2 <= roi[3]]
                if not dets:
                    c["no_face"] += 1
                    rows.append({"class": cdir.name, "frame": label, "face": False})
                    continue
                if not a.all_faces and not roi:
                    dets = [max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))]
                for d in dets:
                    s = sc.signals(frame, d["bbox"], persons, kps=d["kps"], det_score=d["det_score"], full_dets=all_dets)
                    s.pop("_emb", None)
                    pid = s["person_name"] if (s.get("person_sim") or 0) >= 0.45 else "unknown"
                    state = hyst.update(t, s["antispoof"])
                    v = fl.fuse(pid, s.get("decoy_name"), bool(s.get("decoy_hit")), state)
                    last = v
                    c["verdicts"][v] = c["verdicts"].get(v, 0) + 1
                    if v.startswith("person:") or v == "visitor":
                        c["accept"] += 1
                    elif v == "face_like" or v.startswith("decoy:"):
                        c["reject"] += 1
                    else:
                        c["uncertain"] += 1
                    c["as"].append(s["antispoof"])
                    if s.get("decoy_sim") is not None:
                        c["dsim"].append(s["decoy_sim"])
                    if s.get("person_sim") is not None:
                        c["psim"].append(s["person_sim"])
                    rows.append({"class": cdir.name, "frame": label, "face": True, "bbox": [int(x) for x in d["bbox"]],
                                 "det": round(d["det_score"], 3), **{k: v2 for k, v2 in s.items() if k != "antispoof_per_model"},
                                 "state": state, "verdict": v})
            c["clips"].append({"unit": unit.name, "final": last})

    def med(v):
        return round(statistics.median(v), 3) if v else None

    print("\nclass                              side    frames no_face  accept reject uncert  APCER/BPCER  as_med decoy_med person_med  verdicts")
    apcers, bpcers = [], []
    for name, c in per_class.items():
        scored = c["accept"] + c["reject"] + c["uncertain"]
        if c["side"] == "attack":
            rate = c["accept"] / scored if scored else None
            apcers.append(rate if rate is not None else 0.0)
            lab = f"APCER={rate:.3f}" if rate is not None else "APCER=n/a"
        else:
            rate = c["reject"] / scored if scored else None
            bpcers.append(rate if rate is not None else 0.0)
            lab = f"BPCER={rate:.3f}" if rate is not None else "BPCER=n/a"
        print(f"{name:34s} {c['side']:6s} {c['frames']:7d} {c['no_face']:7d} {c['accept']:7d} {c['reject']:6d} {c['uncertain']:6d}  "
              f"{lab:12s} {med(c['as'])!s:>6} {med(c['dsim'])!s:>9} {med(c['psim'])!s:>10}  {c['verdicts']}")
        for cl in c["clips"]:
            print(f"    clip {cl['unit']}: final={cl['final']}")
    if apcers or bpcers:
        acer = ((max(apcers) if apcers else 0.0) + (statistics.mean(bpcers) if bpcers else 0.0)) / 2
        print(f"\nACER (frame-level, fused) = {acer:.3f}   [max APCER={max(apcers) if apcers else None}, mean BPCER={statistics.mean(bpcers) if bpcers else None}]")
    print("\nanti-spoof ONLY (per frame, no decoy gallery):  thr | APCER(all attack) | BPCER(all real)")
    att = [x for c in per_class.values() if c["side"] == "attack" for x in c["as"]]
    rea = [x for c in per_class.values() if c["side"] == "real" for x in c["as"]]
    for thr in (0.35, 0.5, 0.7, 0.9):
        ap_ = sum(x >= thr for x in att) / len(att) if att else float("nan")
        bp_ = sum(x < thr for x in rea) / len(rea) if rea else float("nan")
        print(f"   {thr:.2f} | {ap_:.3f} | {bp_:.3f}")
    out = a.out or str(ROOT / "scratch" / "liveness" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"classes": {k: {kk: vv for kk, vv in v.items() if kk not in ("as", "dsim", "psim")}
                                                 for k, v in per_class.items()}, "rows": rows}, indent=1, default=float))
    print("rows ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
