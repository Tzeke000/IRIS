"""NLA placement + root paths shared by the knight and the dragon.

BASE crossfade scheme (two REPLACE tracks, BASE_A bottom / BASE_B top): every strip runs until next.start + next.xf.
Strips on BASE_B blend in (own xf) and out (next xf); strips on BASE_A never blend, so whatever shows through a
fading BASE_B strip is always a full-influence pose (no rest-pose leaks). A clip that ends early gets a HOLD strip
(its last frame) on the same track. Loop clips repeat to fill.
COMBINE layers (donors / hand keys / flinch) go on their own tracks above, greedily packed so strips never overlap.
"""
import math

import numpy as np

from anim import common as C


def hold_action(src_action, owner, frame, name):
    """2-key action that holds src_action's values at `frame` (every F-curve)."""
    from bpy_extras import anim_utils
    cb0 = anim_utils.action_get_channelbag_for_slot(src_action, src_action.slots[0])
    act, slot, cb = C.new_action(name, owner)
    for fc in cb0.fcurves:
        v = fc.evaluate(frame)
        C.write_fc(cb, fc.data_path, fc.array_index, [0.0, 1.0], [v, v], "CONSTANT", group=fc.group.name if fc.group else "")
    return act


def place_base(ob, segs, meta, N, R, prefix, resolve_align=None):
    """segs: [dict(start, clip, end, xf, scale, align, loop)]; meta[clip] = dict(action, n, f0)."""
    segs = [dict(s) for s in segs]
    for s in segs:                                   # resolve aligned starts first (the previous strip's end needs them)
        if s.get("align"):
            loc, at = s["align"]
            if isinstance(loc, str):
                loc = resolve_align(s["clip"], loc)
            s["start"] = at - loc * s.get("scale", 1.0)
        s["start"] = int(round(s["start"]))
    segs.sort(key=lambda s: s["start"])
    for a, b in zip(segs, segs[1:]):
        assert b["start"] > a["start"] + 1, f"{prefix} overlapping base segments {a['clip']}@{a['start']} / {b['clip']}@{b['start']}"
    bad = [f"{segs[i + 1]['clip']}@{segs[i + 1]['start']}+xf{segs[i + 1].get('xf', 6)} vs {segs[i + 2]['clip']}@{segs[i + 2]['start']}"
           for i in range(len(segs) - 2) if not segs[i + 2]["start"] > segs[i + 1]["start"] + segs[i + 1].get("xf", 6)]
    assert not bad, f"{prefix} segments start inside the previous crossfade: {bad}"
    placed = []
    for i, s in enumerate(segs):
        m = meta[s["clip"]]
        act, n = m["action"], m["n"]
        scale = s.get("scale", 1.0)
        start = s["start"]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        end = (nxt["start"] + nxt.get("xf", 6)) if nxt else N + 1
        top = (i % 2 == 1)
        tr = f"{prefix}BASE_B" if top else f"{prefix}BASE_A"
        length = (n - 1) * scale
        last = st_hold = None
        if s.get("loop"):
            rep = max((end - start) / max(length, 1e-6), 0.05)
            st = C.put(ob, tr, act, start, scale=scale, repeat=rep, name=f"{s['clip']}@{start}",
                       extrap="HOLD" if i == 0 else "NOTHING")
        else:
            st = C.put(ob, tr, act, start, scale=scale, name=f"{s['clip']}@{start}", extrap="HOLD" if i == 0 else "NOTHING")
            nat_end = start + length
            hs = int(math.floor(nat_end)) + 1
            if hs < end:
                import bpy
                hname = f"{act.name}__hold"
                ha = bpy.data.actions.get(hname) or hold_action(act, ob, act.frame_range[1], hname)
                st_hold = C.put(ob, tr, ha, hs, scale=float(end - hs), name=f"hold@{hs}")
            elif nat_end > end + 0.5:
                st.frame_end = float(end)          # trims the action range (keeps scale): never overlap
                R.log(f"note {prefix}{s['clip']}@{start}: trimmed {nat_end:.0f} -> {end}")
        if top:
            st.blend_in = float(s.get("xf", 6)) if i > 0 else 0.0
            (st_hold or st).blend_out = float(nxt.get("xf", 6)) if nxt else 0.0
        placed.append(dict(clip=s["clip"], start=start, end=end, track=tr, scale=scale))
    return placed


def pack_tracks(items):
    """greedy interval packing: items [(start, end, payload)] → [(track_index, start, end, payload)]"""
    ends = []
    out = []
    for s, e, p in sorted(items, key=lambda x: x[0]):
        for ti, te in enumerate(ends):
            if te < s:
                ends[ti] = e
                out.append((ti, s, e, p))
                break
        else:
            ends.append(e)
            out.append((len(ends) - 1, s, e, p))
    return out


# ------------------------------------------------------------------------------------------------- root paths
def path_xy(keys, N):
    fr = np.arange(1, N + 1)
    ks = [(k[0], k[1], k[2] if len(k) > 2 else "") for k in keys]
    ks.sort(key=lambda k: k[0])
    # de-duplicate equal frames (keep the last)
    d = {}
    for k in ks:
        d[k[0]] = k
    ks = [d[k] for k in sorted(d)]
    return C.hermite_keys(ks, fr)


def gsmooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, (r, r), mode="edge")
    return np.convolve(xp, k, mode="valid")


def resolve_yaw(keys, own_xy, other_xy, N):
    """keys: (frame, v) with v = number (absolute, nearest representation), None (face the other character),
    ("rel", d) (exactly d degrees from the previous key, spins), or a ("motion", a, b) range marker."""
    exp = []
    for k in keys:
        if isinstance(k[0], str) and k[0] == "motion":
            _, a, b = k
            for fr in range(int(a), int(b) + 1, 8):
                i = fr - 1
                j = min(i + 6, N - 1)
                d = own_xy[j] - own_xy[max(i - 6, 0)]
                if np.linalg.norm(d) > 1e-3:
                    exp.append((fr, ("abs", math.degrees(math.atan2(-d[0], d[1])))))
            continue
        fr, v = k[0], k[1]
        opt = k[2] if len(k) > 2 else ""
        if v is None:
            i = int(fr) - 1
            exp.append((fr, ("abs", C.yaw_to(own_xy[i], other_xy[i])), opt))
        elif isinstance(v, tuple) and v[0] == "rel":
            exp.append((fr, v, opt))
        else:
            exp.append((fr, ("abs", float(v)), opt))
    exp = [e if len(e) == 3 else (e[0], e[1], "") for e in exp]
    exp.sort(key=lambda x: x[0])
    vals = []
    prev = None
    for fr, (kind, v), opt in exp:
        if kind == "rel":
            val = prev + v
        else:
            val = v
            if prev is not None:
                val = prev + ((v - prev + 180.0) % 360.0 - 180.0)
        vals.append((fr, val, opt))
        prev = val
    d = {}
    for k in vals:
        d[k[0]] = k
    vals = [d[k] for k in sorted(d)]
    return C.hermite_keys(vals, np.arange(1, N + 1))[:, 0], vals


def height(keys, xy, ground, N, smooth=4.0):
    """keys (frame, value, 'rel'|'abs') → z (N,). rel = above the (smoothed) ground under the path; abs = world z.
    Between a rel key and an abs key the two are cross-blended by a smooth weight."""
    fr = np.arange(1, N + 1)
    g = np.array([ground(x, y) for x, y in xy])
    g = gsmooth(g, smooth)
    ks = sorted(keys, key=lambda k: k[0])
    mode = [("abs" if (len(k) > 2 and "abs" in k[2]) else "rel") for k in ks]
    opt = [(k[2].replace("abs", "").replace("rel", "").strip() if len(k) > 2 else "") for k in ks]
    # rel keys are poses (stop: no Catmull-Rom overshoot through the floor); abs (flight) keys flow unless told to stop
    opt = [(o if o else ("" if m == "abs" else "stop")).replace("flow", "") for o, m in zip(opt, mode)]
    w_keys = [(k[0], 1.0 if m == "abs" else 0.0, "stop") for k, m in zip(ks, mode)]
    w = C.hermite_keys(w_keys, fr)[:, 0]
    rel_keys = [(k[0], k[1] if m == "rel" else 0.0, o) for k, m, o in zip(ks, mode, opt)]
    abs_keys = [(k[0], k[1] if m == "abs" else (g[min(int(k[0]), N) - 1] + k[1]), o) for k, m, o in zip(ks, mode, opt)]
    zr = C.hermite_keys(rel_keys, fr)[:, 0]
    za = C.hermite_keys(abs_keys, fr)[:, 0]
    return (1 - w) * (g + zr) + w * za, g


def bake_root(root, pos, yaw_deg, N, teleports=()):
    fr = np.arange(1, N + 1, dtype=np.float64)
    C.bake_obj(root, "location", fr, pos)
    rot = np.zeros((N, 3))
    rot[:, 2] = np.radians(yaw_deg)
    C.bake_obj(root, "rotation_euler", fr, rot)
    if teleports:
        from bpy_extras import anim_utils
        cb = anim_utils.animdata_get_channelbag_for_assigned_slot(root.animation_data)
        for fc in cb.fcurves:
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) + 1 in teleports:
                    kp.interpolation = "CONSTANT"
