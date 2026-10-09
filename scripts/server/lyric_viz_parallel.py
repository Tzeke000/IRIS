"""Chunk-parallel lyric_viz render on the server (Zeke 2026-10-09: "run it on
like 16 of the cores and the GPU and then stitch together if it will work well").

    scripts/server/lyric_viz_parallel.sh [--workers N] [--chunks M]
        [--cv-threads K] [--nice L] [--keep-chunks] [--work-dir DIR]
        <any scripts/lyric_viz.py args, incl. --out and optionally --window>

How it stays seamless (the whole point — a naive split shows SEAMS):
  1. PREP, once: whisper + audio analysis run in ONE process and are cached
     (--words-json + the analysis cache), so the workers share a single
     transcription — one whisper load on the GPU next to the live voice, and
     every chunk agrees on when each word lands.
  2. CHUNKS: [f0, f1) is cut into M contiguous ranges handed to N workers from
     a queue. Each worker runs `lyric_viz.py --frame-range a:b --lossless`,
     which SIMULATES all frame-to-frame state (spin, palette, debris depth,
     shockwaves, decays ...) through frame a-1 without drawing, so its first
     frame equals frame a of a single full render. Output is video-only,
     mathematically lossless RGB.
  3. ASSEMBLE: `lyric_viz.py --assemble manifest.json --assemble-wait` starts
     with the workers and decodes each chunk as soon as it is done, in order,
     through the SAME encode() a single render uses — one real encode running
     alongside the render, audio muxed once. It raises on any gap / missing / short
     chunk, and this driver then checks the final frame count with ffprobe.
Any worker failing kills the rest and exits non-zero. A missing chunk can never
silently produce a short video.

Workers run at nice 15 (and ionice idle) with OpenCV capped at a few threads
each, so the live voice + vision on this box stay responsive.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUNNER = REPO / "scripts" / "server" / "lyric_viz.sh"

# flags the driver owns; passing them through would break the tiling
_OWNED = ("--frame-range", "--lossless", "--assemble", "--prep-only",
          "--threads", "--tiktok")


def _pop(argv: list[str], name: str, has_value: bool = True):
    """Remove `name` (and its value) from argv; return (value, rest)."""
    out, val, k = [], None, 0
    while k < len(argv):
        tok = argv[k]
        if tok == name:
            if has_value:
                if k + 1 >= len(argv):
                    sys.exit(f"[parallel] {name} needs a value")
                val = argv[k + 1]
                k += 2
            else:
                val = True
                k += 1
            continue
        if has_value and tok.startswith(name + "="):
            val = tok.split("=", 1)[1]
            k += 1
            continue
        out.append(tok)
        k += 1
    return val, out


def _repo_path(p: str) -> Path:
    """lyric_viz.sh cds into the repo, so relative paths mean repo-relative."""
    q = Path(p).expanduser()
    return q if q.is_absolute() else (REPO / q)


def _nice_prefix(level: int) -> list[str]:
    pre = ["nice", "-n", str(level)]
    if subprocess.run(["which", "ionice"], capture_output=True).returncode == 0:
        pre = ["ionice", "-c", "3"] + pre
    return pre


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception as e:
        return f"(no log: {e!r})"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="chunk-parallel lyric_viz render",
        usage="lyric_viz_parallel.sh [--workers N] [--chunks M] "
              "[--cv-threads K] [--nice L] [--keep-chunks] [--work-dir DIR] "
              "<lyric_viz args> --out OUT.mp4")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=0,
                    help="number of frame ranges (default 3 x workers: "
                         "smaller pieces balance the heavy drop sections and "
                         "let the streaming encoder start sooner)")
    ap.add_argument("--cv-threads", type=int, default=0,
                    help="OpenCV threads per worker (default: spare cores / "
                         "workers, at least 1)")
    ap.add_argument("--nice", type=int, default=15,
                    help="workers' niceness (default 15). The encoder runs 5 "
                         "steps less nice (but never below 10) so it keeps "
                         "pace with 16 renderers instead of being starved")
    ap.add_argument("--keep-chunks", action="store_true")
    ap.add_argument("--work-dir", type=Path, default=None)
    me, rest = ap.parse_known_args()
    for f in _OWNED:
        if any(t == f or t.startswith(f + "=") for t in rest):
            ap.error(f"{f} is managed by the parallel driver (or unsupported "
                     f"here) - render it with scripts/server/lyric_viz.sh")
    out_s, rest = _pop(rest, "--out")
    if not out_s:
        ap.error("--out is required")
    out = _repo_path(out_s).resolve()
    words_s, rest = _pop(rest, "--words-json")
    window, rest_nowin = _pop(list(rest), "--window")
    workers = max(1, me.workers)
    n_chunks = me.chunks or workers * 3
    cpu = os.cpu_count() or 4
    cvt = me.cv_threads or max(1, (cpu - 4) // workers)
    work = (me.work_dir or out.parent / f".{out.stem}.parts").resolve()
    work.mkdir(parents=True, exist_ok=True)
    words = _repo_path(words_s).resolve() if words_s else work / "words.json"
    nice = _nice_prefix(me.nice)
    nice_enc = _nice_prefix(max(10, me.nice - 5))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    t_all = time.time()

    # -- 1. prep: one transcription + analysis, cached for everyone ----------
    print(f"[parallel] prep (whisper + analysis, once) -> {words.name}",
          flush=True)
    prep_log = work / "prep.log"
    with open(prep_log, "w") as lf:
        rc = subprocess.run(nice + [str(RUNNER)] + rest
                            + ["--words-json", str(words), "--prep-only",
                               "--out", str(out)],
                            stdout=lf, stderr=subprocess.STDOUT, env=env,
                            cwd=REPO).returncode
    info = None
    for ln in prep_log.read_text(errors="replace").splitlines():
        if ln.startswith("PREP "):
            info = json.loads(ln[5:])
    if rc != 0 or info is None:
        print(f"[parallel] ✗ PREP FAILED rc={rc}\n{_tail(prep_log)}",
              file=sys.stderr)
        return 2
    f0, f1, fps = int(info["f0"]), int(info["f1"]), int(info["fps"])
    total = f1 - f0
    n_chunks = max(1, min(n_chunks, total))
    print(f"[parallel] prep ok in {time.time() - t_all:.0f}s: frames "
          f"[{f0}, {f1}) = {total} @ {fps} fps, {info['W']}x{info['H']}; "
          f"{n_chunks} chunks on {workers} workers, {cvt} cv thread(s) each",
          flush=True)

    # -- 2. chunks: contiguous ranges from a queue ---------------------------
    bounds = [f0 + (total * k) // n_chunks for k in range(n_chunks + 1)]
    chunks = [{"k": k, "f0": bounds[k], "f1": bounds[k + 1],
               "path": str(work / f"chunk_{k:03d}.mkv"),
               "log": work / f"chunk_{k:03d}.log"}
              for k in range(n_chunks) if bounds[k + 1] > bounds[k]]
    manifest = work / "manifest.json"
    abort = Path(str(manifest) + ".abort")
    for c in chunks:                       # never stitch a stale file
        Path(c["path"]).unlink(missing_ok=True)
        Path(c["path"] + ".ok").unlink(missing_ok=True)
    abort.unlink(missing_ok=True)
    manifest.write_text(json.dumps(
        [{"path": c["path"], "f0": c["f0"], "f1": c["f1"]} for c in chunks]))
    asm_out = work / ("assembled" + (out.suffix or ".mp4"))
    asm_out.unlink(missing_ok=True)

    # -- 3. assemble, STREAMING: the one real encode runs while chunks are
    #    still rendering (x264 'slow' at 1080x1920 is ~10 fps here, so a
    #    stitch-at-the-end pass would cost as much as the render itself).
    #    Audio is muxed once at the end. Tiling + per-chunk counts checked.
    asm_log = work / "assemble.log"
    asm_lf = open(asm_log, "w")
    asm = subprocess.Popen(nice_enc + [str(RUNNER)] + rest
                           + ["--assemble", str(manifest), "--assemble-wait",
                              "--out", str(asm_out)],
                           stdout=asm_lf, stderr=subprocess.STDOUT, env=env,
                           cwd=REPO, start_new_session=True)
    pending = list(chunks)
    running: dict = {}
    done = 0
    failed = None

    def _stop(p):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            p.wait(timeout=20)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass

    def _kill_all():
        abort.write_text("abort")
        for p, _c, lf in list(running.values()):
            _stop(p)
            lf.close()
        running.clear()
        if asm.poll() is None:
            _stop(asm)
        asm_out.unlink(missing_ok=True)

    def _on_signal(signum, _frm):
        print(f"[parallel] signal {signum}: stopping workers", file=sys.stderr)
        _kill_all()
        sys.exit(130)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    t_render = time.time()
    while pending or running:
        while pending and len(running) < workers and failed is None:
            c = pending.pop(0)
            lf = open(c["log"], "w")
            cmd = (nice + [str(RUNNER)] + rest_nowin
                   + ["--words-json", str(words),
                      "--frame-range", f"{c['f0']}:{c['f1']}", "--lossless",
                      "--threads", str(cvt), "--out", c["path"]])
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 env=env, cwd=REPO, start_new_session=True)
            running[p.pid] = (p, c, lf)
        time.sleep(1.0)
        if asm.poll() not in (None, 0):
            _kill_all()
            print(f"[parallel] ✗ ASSEMBLER DIED rc={asm.returncode} while "
                  f"chunks were rendering - all stopped.\n{_tail(asm_log)}",
                  file=sys.stderr)
            return 4
        for pid in list(running):
            p, c, lf = running[pid]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[pid]
            if rc != 0 or not Path(c["path"]).is_file():
                failed = (c, rc)
                break
            Path(c["path"] + ".ok").write_text("ok")
            done += 1
            el = time.time() - t_render
            print(f"[parallel] chunk {c['k']:03d} [{c['f0']}, {c['f1']}) ok "
                  f"- {done}/{len(chunks)} done, {el:.0f}s elapsed",
                  flush=True)
        if failed is not None:
            _kill_all()
            c, rc = failed
            print(f"[parallel] ✗ CHUNK {c['k']:03d} [{c['f0']}, {c['f1']}) "
                  f"FAILED rc={rc} - all workers + the assembler stopped, "
                  f"nothing written to {out}. Log {c['log']}:\n"
                  f"{_tail(c['log'])}", file=sys.stderr)
            return 3
    t_chunks = time.time() - t_render
    print(f"[parallel] all {len(chunks)} chunks rendered in {t_chunks:.0f}s; "
          f"waiting for the encoder to finish", flush=True)
    rc = asm.wait()
    asm_lf.close()
    if rc != 0 or not asm_out.is_file():
        asm_out.unlink(missing_ok=True)
        print(f"[parallel] ✗ ASSEMBLE FAILED rc={rc}\n{_tail(asm_log)}",
              file=sys.stderr)
        return 4
    t_asm = time.time() - t_render - t_chunks

    # -- 4. verify the deliverable, independently of the assembler -----------
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                         "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames,width,height",
                         "-show_entries", "format=duration", "-of", "json",
                         str(asm_out)], capture_output=True, text=True)
    try:
        j = json.loads(pr.stdout)
        got = int(j["streams"][0]["nb_read_frames"])
        dur = float(j["format"]["duration"])
    except Exception as e:
        print(f"[parallel] ✗ ffprobe failed on {out}: {e!r} {pr.stderr}",
              file=sys.stderr)
        return 5
    if got != total:
        print(f"[parallel] ✗ OUTPUT HAS {got} FRAMES, EXPECTED {total} - "
              f"left at {asm_out}, NOT moved to {out}", file=sys.stderr)
        return 5
    os.replace(asm_out, out)
    if not me.keep_chunks:
        for c in chunks:
            Path(c["path"]).unlink(missing_ok=True)
            Path(c["path"] + ".ok").unlink(missing_ok=True)
    print(f"[parallel] ✓ {out} - {got} frames ({got / fps:.2f}s video, "
          f"container {dur:.2f}s), chunks {t_chunks:.0f}s + encoder tail "
          f"{t_asm:.0f}s, total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
