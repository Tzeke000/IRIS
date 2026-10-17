"""Is a rendered PNG COMPLETE? — the one definition, used by frames.py (resume) and scene.sh (chunk skip / mux gate).
Stdlib only (system python3 or Blender's python).

The old resume tests were `size > 1 KB` (rocket4) or `[[ -s file ]]` (police/rooftop/arcade): a render killed
mid-write leaves a truncated f_####.png that counts as done and lands in the video. A PNG is complete only if it starts
with the 8-byte signature AND ends with the 12-byte IEND chunk (written last by every encoder), so a torn file never passes.

    python3 pngcheck.py missing DIR S E [--prefix f_]    → "MISSING n" + compact ranges (exit 0)
    python3 pngcheck.py torn DIR [--delete]                → lists incomplete *.png (and .tmp_* leftovers) in DIR
"""
import os
import sys

SIG = b"\x89PNG\r\n\x1a\n"
IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def is_complete_png(path):
    try:
        n = os.path.getsize(path)
        if n < len(SIG) + len(IEND) + 25:          # signature + IHDR (25) + IEND
            return False
        with open(path, "rb") as f:
            if f.read(8) != SIG:
                return False
            f.seek(-12, os.SEEK_END)
            return f.read(12) == IEND
    except OSError:
        return False


def frame_path(d, f, prefix="f_"):
    return os.path.join(d, f"{prefix}{f:04d}.png")


def missing(d, s, e, prefix="f_"):
    return [f for f in range(s, e + 1) if not is_complete_png(frame_path(d, f, prefix))]


def ranges(nums):
    """[1,2,3,7,9,10] → '1-3,7,9-10'"""
    out, run = [], []
    for n in sorted(nums):
        if run and n == run[-1] + 1:
            run.append(n)
        else:
            if run:
                out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [n]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def torn(d):
    """Incomplete PNGs in d (final names) + temp leftovers of killed renders."""
    bad = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.startswith(".tmp_"):
            bad.append(p)
        elif fn.endswith(".png") and not is_complete_png(p):
            bad.append(p)
    return bad


def _main(argv):
    if not argv or argv[0] not in ("missing", "torn"):
        sys.exit(__doc__)
    prefix = "f_"
    if "--prefix" in argv:
        i = argv.index("--prefix")
        prefix = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if argv[0] == "missing":
        d, s, e = argv[1], int(argv[2]), int(argv[3])
        m = missing(d, s, e, prefix) if os.path.isdir(d) else list(range(s, e + 1))
        print(f"MISSING {len(m)}")
        if m:
            print(ranges(m))
        return m
    bad = torn(argv[1])
    for p in bad:
        print(p)
        if "--delete" in argv:
            os.remove(p)
    print(f"TORN {len(bad)}{' (deleted)' if bad and '--delete' in argv else ''}")
    return bad


if __name__ == "__main__":
    _main(sys.argv[1:])
