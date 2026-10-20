"""Cut one finished clip: trim -> reframe -> styled captions -> emoji pops -> hook text ->
loudness-normalised audio -> H.264. Runs ffmpeg on the CPU; launch long jobs through
scripts/server/render.sh (capped slice) like any render.

usage: /home/iris/venvs/iris-v100/bin/python scripts/video_edit/make_clip.py <video> \
         --start 61.5 --end 92 --style tiktok_gaming --out out.mp4 \
         [--layout vertical_blur|vertical_crop|landscape] [--hook "WAIT FOR IT"] [--zoom 75.2,77.0] [--speed 1.0]
Needs <video>.transcript.json (transcribe.py) for captions; without it, the clip has no captions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent))
import captions  # noqa: E402

EMOJI_DIR = Path(__file__).resolve().parents[2] / "scratch" / "video_edit" / "emoji"
EDIT_PY = "/home/iris/venvs/edit-tools/bin/python"   # has cairosvg for crisp emoji


def emoji_png(code: str, px: int = 240) -> Path:
    p = EMOJI_DIR / f"{code}_{px}.png"
    if p.exists():
        return p
    svg = EMOJI_DIR / f"{code}.svg"
    if not svg.exists():
        subprocess.run(["curl", "-sfL", "-o", str(svg),
                        f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/svg/{code}.svg"], check=True)
    subprocess.run([EDIT_PY, "-c", "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[2], "
                    "output_width=int(sys.argv[3]), output_height=int(sys.argv[3]))", str(svg), str(p), str(px)],
                   check=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--style", default="tiktok_gaming", choices=sorted(captions.STYLES))
    ap.add_argument("--layout", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook", default="")
    ap.add_argument("--zoom", default="", help="a,b seconds (clip-relative) for a 1.25x punch-in")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()

    st = captions.STYLES[a.style]
    layout = a.layout or ("landscape" if st["layout"] == "landscape" else "vertical_blur")
    W, H = (1920, 1080) if layout == "landscape" else (1080, 1920)
    video = Path(a.video)
    tp = video.with_suffix(video.suffix + ".transcript.json")
    tr = json.loads(tp.read_text()) if tp.exists() else {"segments": []}
    dur = (a.end - a.start) / a.speed
    tmp = Path(tempfile.mkdtemp(prefix="clip_"))
    ass_text, emo = captions.build(tr, a.start, a.end, a.style, W, H, speed=a.speed)
    if a.hook:   # big top hook for the first 2.2 s (the scroll-stopper)
        ass_text += (f"Dialogue: 1,0:00:00.00,0:00:02.20,Cap,,0,0,0,,{{\\an8\\pos({W // 2},{int(H * 0.12)})"
                     f"\\fs{int(st['size'] * 1.1)}\\c&H0000E5FF&\\fscx70\\fscy70\\t(0,160,\\fscx100\\fscy100)}}"
                     f"{captions._clean(a.hook.upper())}\n")
    ass = tmp / "cap.ass"
    ass.write_text(ass_text, encoding="utf-8")

    sp = f",setpts=PTS/{a.speed}" if a.speed != 1.0 else ""
    if layout == "vertical_blur":   # gameplay: sharp 16:9 band over a blurred fill
        v = (f"[0:v]fps={a.fps}{sp},split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
             f"crop={W}:{H},boxblur=24:2,eq=brightness=-0.08[bg];[b]scale={W}:-2[fg];"
             f"[bg][fg]overlay=(W-w)/2:(H-h)/2-140[v0]")
    elif layout == "vertical_crop":  # face cam / centred action: fill the frame
        v = f"[0:v]fps={a.fps}{sp},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}[v0]"
    else:
        v = f"[0:v]fps={a.fps}{sp},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2[v0]"
    last = "v0"
    if a.zoom:
        z0, z1 = (float(x) for x in a.zoom.split(","))
        v += (f";[{last}]split[z0][z1];[z1]crop=iw/1.25:ih/1.25,scale={W}:{H}[zz];"
              f"[z0][zz]overlay=0:0:enable='between(t,{z0},{z1})'[vz]")
        last = "vz"
    v += f";[{last}]ass='{ass}'[vc]"
    last = "vc"
    inputs = ["-ss", f"{a.start:.3f}", "-t", f"{a.end - a.start:.3f}", "-i", str(video)]
    for i, e in enumerate(emo, start=1):
        inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(emoji_png(e["code"]))]
        y = int(H * st["pos_y"]) - st["size"] - 260
        v += (f";[{i}:v]format=rgba,fade=in:st={e['t']}:d=0.12:alpha=1,"
              f"fade=out:st={e['t'] + e['dur'] - 0.2:.2f}:d=0.2:alpha=1[e{i}];"
              f"[{last}][e{i}]overlay=(W-w)/2:{y}:enable='between(t,{e['t']},{e['t'] + e['dur']:.2f})'[ve{i}]")
        last = f"ve{i}"
    at = f"atempo={a.speed}," if a.speed != 1.0 else ""
    af = f"[0:a]{at}loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.08,afade=t=out:st={max(0, dur - 0.25):.2f}:d=0.25[aout]"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", v + ";" + af, "-map", f"[{last}]",
           "-map", "[aout]", "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"[make_clip] {a.out}: {dur:.1f}s {W}x{H} style={a.style} layout={layout} "
          f"captions={ass_text.count('Dialogue:')} emoji={len(emo)}")


if __name__ == "__main__":
    main()
