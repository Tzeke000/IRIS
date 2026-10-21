"""Split-screen 9:16 short: a DAW screen recording on top, our rendered music video underneath,
locked to the same moment of the song (Zeke 10-21, "format #3").

The two files never start at the same song time, so the offset is found by cross-correlating the
two audio tracks (onset envelopes at 200 Hz), then checked: the peak must stand well above the rest.

usage: /home/iris/venvs/iris-v100/bin/python scripts/video_edit/split_screen.py DAW.mp4 VIDEO.mp4 \
         --start 97.4 --end 128.4 --out out.mp4 [--hook "wait for it..."] [--audio video|daw]
         [--daw-x 0.5]   # horizontal centre of the DAW crop, 0..1 of the frame width (0.5 = centre)
--start/--end are SONG times = times in VIDEO.mp4 (the music video starts at song 0).
Run long jobs through scripts/server/render.sh like every render.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

SR = 8000
HOP = 40  # 200 Hz envelope


def _pcm(path: str, t0: float = 0.0, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", path]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _env(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    e = np.sqrt((x[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1) + 1e-9)
    e = np.log(e)
    d = np.maximum(np.diff(e, prepend=e[0]), 0)          # onset strength
    return (d - d.mean()) / (d.std() + 1e-9)


def find_offset(daw: str, video: str) -> tuple[float, float]:
    """Return (offset, confidence): song_time = daw_time - offset. Confidence = peak / 2nd-peak."""
    a, b = _env(_pcm(daw)), _env(_pcm(video))
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lag = int(np.argmax(c))
    peak = c[lag]
    mask = np.ones_like(c, bool)
    mask[max(0, lag - 100): lag + 100] = False            # ±0.5 s around the peak
    second = c[mask].max()
    if lag > n // 2:
        lag -= n
    return lag / (SR / HOP), float(peak / max(second, 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("daw")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook", default="")
    ap.add_argument("--audio", default="video", choices=["video", "daw"])
    ap.add_argument("--daw-x", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=None, help="skip auto-sync: song = daw - offset")
    a = ap.parse_args()

    if a.offset is None:
        off, conf = find_offset(a.daw, a.video)
        print(f"[split] auto-sync offset {off:+.3f} s (confidence {conf:.2f})")
        if conf < 1.3:
            sys.exit("[split] sync not trustworthy (peak too close to the rest) — pass --offset by ear")
    else:
        off = a.offset
    dur = a.end - a.start
    daw_t0 = a.start + off
    if daw_t0 < 0:
        sys.exit(f"[split] the DAW recording starts {-daw_t0:.2f} s after song {a.start} — pick a later --start")

    W, Hh = 1080, 960
    x = f"(iw-{W})*{a.daw_x:.3f}"
    v = (f"[0:v]fps=30,scale=-2:{Hh},crop={W}:{Hh}:{x}:0[top];"
         f"[1:v]fps=30,scale=-2:{Hh},crop={W}:{Hh}[bot];"
         f"[top][bot]vstack=2[v0];[v0]drawbox=y={Hh - 3}:w=iw:h=6:color=black@1:t=fill[v1]")
    last = "v1"
    if a.hook:
        txt = a.hook.replace("'", "").replace(":", "")
        v += (f";[{last}]drawtext=text='{txt}':fontfile=/home/iris/.local/share/fonts/iris_edit/BebasNeue-Regular.ttf"
              f":fontsize=110:fontcolor=white:borderw=6:bordercolor=black:x=(w-tw)/2:y={Hh}-th/2"
              f":enable='lt(t,2.2)'[v2]")
        last = "v2"
    ai = "1:a" if a.audio == "video" else "0:a"
    af = (f"[{ai}]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.08,"
          f"afade=t=out:st={max(0, dur - 0.4):.2f}:d=0.4[aout]")
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-ss", f"{daw_t0:.3f}", "-t", f"{dur:.3f}", "-i", a.daw,
           "-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}", "-i", a.video,
           "-filter_complex", v + ";" + af, "-map", f"[{last}]", "-map", "[aout]", "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"[split] {a.out}: {dur:.1f}s 1080x1920, DAW from {daw_t0:.2f}s, video from {a.start:.2f}s, audio={a.audio}")


if __name__ == "__main__":
    main()
