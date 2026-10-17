"""Boss fight animatic — shared constants, the EVENT TABLE (choreo_v1.json), action/NLA writers, paths and curves.

Frames: Blender frame BF = choreo `frame` + 1 (frame 1 = t 0.000 s; lib/render/scene.sh mux offset (S-1)/FPS).
Character frame convention (both rigs): each character hangs under a ROOT empty; in the root's local frame the
character FACES +Y, +X is its right, +Z up. Root yaw = heading measured counter-clockwise from world +Y, so
forward(yaw) = (-sin yaw, cos yaw).
"""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

REPO = "/home/iris/IRIS"
SCENE_DIR = f"{REPO}/scripts/lyric_scenes/real/boss_fight"
OUT_DIR = f"{REPO}/scratch/scenes/boss_fight"
ARENA = f"{OUT_DIR}/arena_v3.blend"
OUT_BLEND = f"{OUT_DIR}/fight_v2.blend"
CHOREO = f"{SCENE_DIR}/choreo_v2.json"
SONG = "/data/iris_assets/dragon_slayer/dragon_slayer.wav"
SONG_MAP = "/data/iris_assets/dragon_slayer/analysis/song_map.json"
FPS = 30
SONG_S = 254.28
NF = int(math.floor(SONG_S * FPS))          # 7628 frames: 1..7628
MED = "/data/iris_assets/medieval/"
UAL1 = ("/data/iris_assets/library/quaternius/universal-animation-library/Universal Animation Library[Standard]/"
        "Unreal-Godot/UAL1_Standard.glb")
UAL2 = ("/data/iris_assets/library/quaternius/universal-animation-library-2/Universal Animation Library 2[Standard]/"
        "Unreal-Godot/UAL2_Standard.glb")
KNIGHT_GLB = MED + "sketchfab/knight_rx_sculpt/model.glb"
TREX_GLB = MED + "r2_anim/trex_5anim_lasqueti/model.glb"
ESTOC_DIR = MED + "polyhaven/"
G = 9.81
DRAGON_S = 5.0
KNIGHT_H = 1.85


def BF(choreo_frame):
    return int(choreo_frame) + 1


# --------------------------------------------------------------------------------------------- the event table
# Semantic key per row of choreo_v1.json, bound by a substring of its "what" text so a reordered/edited table
# fails loudly instead of silently re-targeting a move.
EVENT_KEYS = [
    ("cam_valley", "valley sweep"), ("cam_colossi", "colossi"), ("dragon_asleep", "asleep"),
    ("knight_foot", "foot of the stairs"), ("climb", "climb begins"), ("kit_sword", "KIT 1"),
    ("kit_fire", "KIT 2"), ("kit_rocks", "KIT 3"), ("doors", "pushes the doors"), ("eye_open", "EYE SNAPS"),
    ("rise_roar", "rises off the hoard"), ("smolder", "SMOLDER"), ("knight_in", "knight steps in"),
    ("circling", "circling standoff"), ("draw", "DRAWS the sword"), ("tail_test", "test TAIL SWIPE"),
    ("dodge_p1", "DODGE"), ("combo", "3-hit sword combo"), ("talon", "TALON SWIPE"),
    ("parry", "PARRY"), ("tail_p2", "pillar 2"), ("fireballs", "FIREBALL x2"), ("ceiling", "ceiling chunk"),
    ("neck_slash", "overhead slash"), ("bite_dodge", "BITE lunge"), ("tail_p3", "pillar 3"),
    ("bridge", "reset: both back off"), ("spikes", "EARTH MAGIC"), ("stumble", "stumbles"),
    ("rocks", "orbiting rocks launched"), ("dfireballs", "FIREBALL spit"), ("bite_miss", "BITE MISSES"),
    ("slam", "BODY SLAM"), ("thrown", "BEHIND COVER"), ("potion", "POTION"), ("gem_out", "pulls the GEM"),
    ("gem_crush", "CRUSHES the gem"), ("launch", "launches upward"), ("roof", "BURSTS UP"),
    ("roof_rain", "roof chunks rain"), ("moon", "MOON"), ("run_out", "runs out to the courtyard"),
    ("dive", "DIVE from the sky"), ("shield_breath", "SHIELD holds the fire"), ("big_fireball", "BIG FIREBALL"),
    ("volley", "fireball volley"), ("dash_dodges", "dash dodges"), ("lands", "LANDS hard"),
    ("standoff", "slow standoff"), ("raise_hand", "raises hand"), ("sword_forming", "begins forming"),
    ("inhale", "inhales"), ("sword_formed", "fully formed"), ("sword_falls", "SWORD FALLS"),
    ("thrash", "thrashes"), ("collapse", "collapses"), ("outro_walk", "music box"),
    ("eye_again", "eye opens"), ("shoulder_rpg", "shoulders the RPG"), ("fire_rpg", "FIRES on the last impact"),
]


@dataclass
class Event:
    key: str
    idx: int
    t: float
    f: int                    # Blender frame
    sec: str
    who: str
    what: str
    contact: list = field(default_factory=list)


def load_events(path=CHOREO):
    with open(path) as fh:
        d = json.load(fh)
    rows = d["events"]
    assert len(rows) == len(EVENT_KEYS), f"choreo has {len(rows)} rows, EVENT_KEYS {len(EVENT_KEYS)}"
    out = {}
    for i, (row, (key, needle)) in enumerate(zip(rows, EVENT_KEYS)):
        assert needle in row["what"], f"event {i}: expected '{needle}' in {row['what']!r} — choreo changed?"
        out[key] = Event(key, i, float(row["t_snap"]), BF(row["frame"]), row["sec"], row["who"], row["what"],
                         list(row.get("contact", [])))
    fr = [e.f for e in out.values()]
    assert fr == sorted(fr), "event table not time-ordered"
    return out


# Contact kinds → allowed overlap window around the planned frame (frames before, frames after).
# A sword passes THROUGH a limb for a few frames (no hit-stop in the blocking pass); a projectile is hidden the frame
# after it stops; the holy sword stays embedded (pinned) until it dissolves.
WINDOWS = {"sword": (1, 6), "talon_parry": (1, 4), "tail_pillar": (1, 8), "projectile": (1, 1), "shield": (1, 2),
           "slam": (1, 4), "body_cover": (3, 8), "spikes": (1, 50), "holy": (1, 392), "roof": (4, 12),
           "rubble": (0, 3), "rocket": (1, 1), "breath": (10, 50)}


@dataclass
class Contact:
    key: str                  # event key (+ suffix for derived sub-hits)
    f: int                    # planned first-contact frame
    a: str                    # audit body name (source)
    b: str                    # audit body name (target)
    kind: str
    derived: bool = False     # True = generated from an event row (combo hit 2/3, 2nd fireball...) or by the sim
    note: str = ""
    optional: bool = False    # an ALLOWANCE only (may touch inside the window, need not)

    @property
    def window(self):
        lo, hi = WINDOWS[self.kind]
        return self.f - lo, self.f + hi


# --------------------------------------------------------------------------------------------- curves
def hermite_keys(keys, frames, stop_ends=True):
    """keys: [(frame, value(s), opt)] opt: "stop" → zero tangent there, "lin" → straight segment to the next key.
    Cardinal (Catmull-Rom on frames) spline through the keys; constant outside. Returns (len(frames), dim)."""
    ks = sorted(keys, key=lambda k: k[0])
    t = np.array([k[0] for k in ks], np.float64)
    v = np.array([np.atleast_1d(np.asarray(k[1], np.float64)) for k in ks])
    opt = [k[2] if len(k) > 2 else "" for k in ks]
    n = len(ks)
    m = np.zeros_like(v)
    for i in range(n):
        if "stop" in opt[i] or n == 1:
            continue
        if i == 0 or i == n - 1:
            if stop_ends:
                continue
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
        else:
            j0, j1 = i - 1, i + 1
        if "lin" in opt[i - 1 if i > 0 else 0] and i > 0:
            m[i] = (v[i] - v[i - 1]) / max(t[i] - t[i - 1], 1e-9)
            continue
        m[i] = (v[j1] - v[j0]) / max(t[j1] - t[j0], 1e-9)
    fr = np.asarray(frames, np.float64)
    out = np.empty((len(fr), v.shape[1]))
    for k, f in enumerate(fr):
        if f <= t[0]:
            out[k] = v[0]
            continue
        if f >= t[-1]:
            out[k] = v[-1]
            continue
        i = int(np.searchsorted(t, f, side="right") - 1)
        h = t[i + 1] - t[i]
        s = (f - t[i]) / h
        if "lin" in opt[i]:
            out[k] = v[i] + (v[i + 1] - v[i]) * s
            continue
        h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        out[k] = h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1]
    return out


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a, np.float64))))


def yaw_to(p, q):
    """heading (deg) for a character at p to face q (yaw 0 = +Y, CCW positive)."""
    d = np.asarray(q, np.float64)[:2] - np.asarray(p, np.float64)[:2]
    return math.degrees(math.atan2(-d[0], d[1]))


def fwd(yaw_deg):
    a = math.radians(yaw_deg)
    return np.array([-math.sin(a), math.cos(a), 0.0])


def right(yaw_deg):
    a = math.radians(yaw_deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------------------------- quaternions (w,x,y,z)
def q_mul(a, b):
    a, b = np.asarray(a), np.asarray(b)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def q_conj(a):
    a = np.asarray(a, np.float64).copy()
    a[..., 1:] *= -1
    return a


def q_axis_angle(axis, ang):
    axis = np.asarray(axis, np.float64)
    ang = np.asarray(ang, np.float64)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.where(n > 1e-12, n, 1.0)
    h = ang[..., None] / 2 if ang.ndim else ang / 2
    return np.concatenate([np.cos(h) * np.ones_like(axis[..., :1]), np.sin(h) * axis], -1)


def q_from_rotvec(v):
    v = np.asarray(v, np.float64)
    a = np.linalg.norm(v, axis=-1)
    ax = v / np.where(a[..., None] > 1e-12, a[..., None], 1.0)
    return np.concatenate([np.cos(a / 2)[..., None], np.sin(a / 2)[..., None] * ax], -1)


def q_to_mat(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_q(m):
    """(…,3,3) rotation → (…,4) quaternion w>=0 (Shepperd)."""
    m = np.asarray(m, np.float64)
    shp = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    out = np.empty((len(m), 4))
    for i, r in enumerate(m):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
        q = np.array(q)
        out[i] = q if q[0] >= 0 else -q
    return out.reshape(shp + (4,))


def q_slerp(a, b, t):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * t
        return r / np.linalg.norm(r)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def q_continuous(q):
    """flip signs so consecutive quaternions stay in one hemisphere (keys interpolate the short way)."""
    q = np.array(q, np.float64)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


# --------------------------------------------------------------------------------------------- bpy: actions + NLA
def _bpy():
    import bpy
    return bpy


def new_action(name, owner, fake_user=True):
    """standalone slotted action for `owner` (an Object); returns (action, slot, channelbag)."""
    bpy = _bpy()
    from bpy_extras import anim_utils
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act.use_fake_user = fake_user
    slot = act.slots.new(id_type="OBJECT", name=owner.name)
    cb = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    return act, slot, cb


_INTERP = {}


def _interp_code(name):
    if not _INTERP:
        items = _bpy().types.Keyframe.bl_rna.properties["interpolation"].enum_items
        for k in ("CONSTANT", "LINEAR", "BEZIER"):
            _INTERP[k] = items[k].value
    return _INTERP[name]


def write_fc(cb, path, index, frames, values, interp="LINEAR", group=""):
    fc = cb.fcurves.find(path, index=index) or cb.fcurves.new(path, index=index, group_name=group)
    kp = fc.keyframe_points
    kp.clear()
    n = len(frames)
    kp.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = np.asarray(frames, np.float32)
    co[1::2] = np.asarray(values, np.float32)
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", np.full(n, _interp_code(interp), np.int32))
    fc.update()
    return fc


def write_bone_quats(cb, bone, frames, Q, interp="LINEAR"):
    Q = q_continuous(Q)
    p = f'pose.bones["{bone}"].rotation_quaternion'
    for i in range(4):
        write_fc(cb, p, i, frames, Q[:, i], interp, group=bone)


def write_bone_loc(cb, bone, frames, L, interp="LINEAR"):
    p = f'pose.bones["{bone}"].location'
    for i in range(3):
        write_fc(cb, p, i, frames, L[:, i], interp, group=bone)


def write_vec(cb, path, frames, arr, interp="LINEAR"):
    arr = np.asarray(arr, np.float64)
    for i in range(arr.shape[1]):
        write_fc(cb, path, i, frames, arr[:, i], interp)


def track(ob, name):
    """get-or-create NLA track `name` on ob (new tracks go on TOP)."""
    ad = ob.animation_data or ob.animation_data_create()
    tr = ad.nla_tracks.get(name)
    if tr is None:
        tr = ad.nla_tracks.new()
        tr.name = name
    return tr


def put(ob, track_name, action, start, blend="REPLACE", bin_=0.0, bout=0.0, extrap="NOTHING", scale=1.0,
        repeat=1.0, name=None, influence=None):
    tr = track(ob, track_name)
    try:
        st = tr.strips.new(name or action.name, int(round(start)), action)
    except RuntimeError:
        raise RuntimeError(f"NLA no space: {ob.name}/{track_name} {name or action.name}@{start} range "
                           f"{tuple(action.frame_range)}; existing {[(x.name, x.frame_start, x.frame_end) for x in tr.strips][-3:]}")
    if action.slots:
        st.action_slot = action.slots[0]
    st.blend_type = blend
    st.extrapolation = extrap
    if scale != 1.0:
        st.scale = scale
    if repeat != 1.0:
        st.repeat = repeat
    st.blend_in, st.blend_out = bin_, bout
    if influence is not None:
        st.use_animated_influence = True
        st.influence = influence
    return st


def key_visibility(ob, pairs, viewport=True):
    """pairs [(frame, hidden)] → CONSTANT keys on hide_render (+ hide_viewport)."""
    ad = ob.animation_data or ob.animation_data_create()
    if ad.action is None:
        act = _bpy().data.actions.new(f"{ob.name}_vis")
        ad.action = act
    for path in (("hide_render", "hide_viewport") if viewport else ("hide_render",)):
        fc = ad.action.fcurve_ensure_for_datablock(ob, path)
        for f, h in pairs:
            fc.keyframe_points.insert(f, 1.0 if h else 0.0, options={"FAST"})
        for k in fc.keyframe_points:
            k.interpolation = "CONSTANT"
        fc.update()
    first = sorted(pairs)[0][1]
    ob.hide_render = bool(first)
    return ob


def obj_fc(ob, path, index=0):
    ad = ob.animation_data or ob.animation_data_create()
    if ad.action is None:
        ad.action = _bpy().data.actions.new(f"{ob.name}_act")
    return ad.action.fcurve_ensure_for_datablock(ob, path, index=index)


def bake_obj(ob, path, frames, arr, interp="LINEAR"):
    """bulk keys on an object's own action (location / rotation_euler / scale / custom)."""
    arr = np.asarray(arr, np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    fcs = []
    for i in range(arr.shape[1]):
        fc = obj_fc(ob, path, i)
        kp = fc.keyframe_points
        kp.clear()
        kp.add(len(frames))
        co = np.empty(2 * len(frames), np.float32)
        co[0::2] = np.asarray(frames, np.float32)
        co[1::2] = arr[:, i]
        kp.foreach_set("co", co)
        kp.foreach_set("interpolation", np.full(len(frames), _interp_code(interp), np.int32))
        fc.update()
        fcs.append(fc)
    return fcs
