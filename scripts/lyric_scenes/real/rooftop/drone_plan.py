"""Drone-show transition matching (what Skybrush's server does for us, offline): minimum total squared distance
assignment between consecutive formations (Hungarian / linear_sum_assignment). Plain venv with scipy, run -I.
    drone_plan.py IN.npz OUT.npz      IN: f0..fK (N,3) formations in order   OUT: f0..fK re-ordered so drone i → row i
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

src, out = sys.argv[1], sys.argv[2]
d = np.load(src)
keys = sorted([k for k in d.files if k.startswith("f")], key=lambda k: int(k[1:]))
forms = [d[k].astype(np.float64) for k in keys]
res = {keys[0]: forms[0]}
prev = forms[0]
for k, F in zip(keys[1:], forms[1:]):
    C = ((prev[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    nxt = np.empty_like(F)
    nxt[r] = F[c]
    dist = np.linalg.norm(nxt - prev, axis=1)
    print(f"[plan] {k}: mean move {dist.mean():.1f} m, max {dist.max():.1f} m")
    res[k] = nxt
    prev = nxt
np.savez(out, **res)
