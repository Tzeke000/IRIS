"""dragon stick dumps (character frame) for offline plots — no rendering."""
import json

import numpy as np

CHAINS = {
    "spine": ["DEF-Spine", "DEF-Spine.001", "DEF-Spine.002", "DEF-Spine.003", "DEF-Spine.004", "DEF-Spine.005"],
    "neck": ["DEF-neck", "DEF-neck.001", "DEF-neck.002", "DEF-neck.003", "DEF-neck.004"],
    "jaw": ["DEF-Bone", "DEF-Bone.002", "DEF-Teeth_Bottom"],
    "tail": ["DEF-tail"] + [f"DEF-tail.{i:03d}" for i in range(1, 17)],
    "armL": ["DEF-collar.L", "DEF-upper_arm.L", "DEF-upper_arm.L.001", "DEF-forearm.L", "DEF-forearm.L.001", "DEF-forearm.L.002", "DEF-palm.L"],
    "armR": ["DEF-collar.R", "DEF-upper_arm.R", "DEF-upper_arm.R.001", "DEF-forearm.R", "DEF-forearm.R.001", "DEF-forearm.R.002", "DEF-palm.R"],
    "legL": ["DEF-Hip.L", "DEF-thigh.L", "DEF-thigh.L.001", "DEF-shin.L", "DEF-shin.L.001", "DEF-foot.L", "DEF-foot.L.001", "DEF-toe.L"],
    "legR": ["DEF-Hip.R", "DEF-thigh.R", "DEF-thigh.R.001", "DEF-shin.R", "DEF-shin.R.001", "DEF-foot.R", "DEF-foot.R.001", "DEF-toe.R"],
    "wingL": ["DEF-Wing_Back_Support.L", "DEF-Shoulder_Blade.L", "DEF-Wing_Base.L", "DEF-Wing_Fold_1.L", "DEF-Wing_Fold_2.L", "DEF-Wing_Fold_2.02.L", "DEF-Wing_Fold_2.03.L"],
    "wingL3": ["DEF-Wing_Fold_2.L", "DEF-Wing_Fold_3.L", "DEF-Wing_Fold_3.02.L", "DEF-Wing_Fold_3.03.L"],
    "wingL4": ["DEF-Wing_Fold_2.L", "DEF-Wing_Fold_4.L", "DEF-Wing_Fold_4.02.L", "DEF-Wing_Fold_4.03.L"],
    "wingR": ["DEF-Wing_Back_Support.R", "DEF-Shoulder_Blade.R", "DEF-Wing_Base.R", "DEF-Wing_Fold_1.R", "DEF-Wing_Fold_2.R", "DEF-Wing_Fold_2.02.R", "DEF-Wing_Fold_2.03.R"],
}


def chains(skel):
    out = {}
    for k, names in CHAINS.items():
        out[k] = [skel.find(n) for n in names]
        missing = [n for n, f in zip(names, out[k]) if f is None]
        assert not missing, f"dragon chain {k} missing {missing}"
    return out


def tails(skel, W, names):
    """bone head positions + the last bone's tail (W: (B,4,4) char-frame pose matrices)"""
    pts = [W[skel.idx[n], :3, 3] for n in names]
    last = skel.idx[names[-1]]
    ln = skel.arm.data.bones[names[-1]].length
    pts.append((W[last] @ np.array([0, ln, 0, 1.0]))[:3])
    return np.array(pts)


def dump(path, skel, poses):
    """poses: {label: (B,4,4) char-frame matrices}"""
    ch = chains(skel)
    out = {}
    for lab, W in poses.items():
        out[lab] = {k: tails(skel, W, v).round(3).tolist() for k, v in ch.items()}
    with open(path, "w") as fh:
        json.dump(out, fh)
