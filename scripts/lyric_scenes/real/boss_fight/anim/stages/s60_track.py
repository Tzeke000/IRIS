"""s60 — evaluate both rigs on every frame → S.trk key points (world) for FX, cameras, audit; debug npz for plots."""
import os

import numpy as np

from anim import track as TK


def run(S):
    end = S.args.slice_end or S.N
    frames = np.arange(1, end + 1)
    S.trk = TK.sample(S, frames)
    S.trk_frames = frames
    np.savez_compressed(os.path.join(S.debug, "track.npz"), frames=frames, **S.trk)
    S.R.add("track.frames", int(len(frames)))
