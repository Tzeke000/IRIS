"""ROCKET LAUNCH v4 — "Blast Off" (Tzeke000), built on the v3 RAVE cut (and v2 photoreal) below.

v4 (Zeke 10-12/10-13, song map in docs/research/blastoff_v4_*.md + the Blast Off v4 memory note):
  * three boosters at 120 deg; booster 3 sputters after the alarm, separates early, cartwheels STILL BURNING and
    EXPLODES on the 0:56 false drop (JangaFX CC0 Mid-Air Explosion); a real flight model (flight_model.py) drives
    the liftoff, the TVC-limited tumble (gimbal-clamped, saturating, damped), the sag/drift and the recovery;
  * LED-pixel shader on rocket + tower + speaker (rainbow chase, rings boot bar by bar in the build, all RED
    with the alarm, rainbow REBOOT sweep bottom->top on the 1:07 halftime drop), a 32-band spectrum panel;
  * an onboard fisheye feed + a warning HUD (separate compositor pass: post_hud.py, B612 font);
  * ONE big speaker (Sousinho CC-BY cabinet + v3's moving cones, LED matrices, moving heads, strobes),
    lasers in DROP 2 only; asteroid glimpse + trail; it shatters one bar before drop 2 ends and the debris is
    Blender rigid-body physics (no-tunnel recipe + Cell Fracture crumble), baked to keys, gated by an assert.
  * countdown board re-spaced (asserted inside the frame).

(v2 header follows)
ROCKET LAUNCH v2 — photoreal, music-synced Blender scene for "Hit a Bump" (Tzeke000).

    ~/.local/bin/blender -b --factory-startup --python-exit-code 1 \
        --python scripts/lyric_scenes/real/rocket2/build_scene.py -- \
        [--timeline state/tzeke_songs/hit_a_bump/timeline_30fps.npz] \
        [--lib /data/iris_assets/rocket2/rocket2_lib.blend] [--assets /data/iris_assets] \
        [--out scratch/scenes/rocket2/rocket2.blend]

v2 rebuilds v1 (../rocket) for photorealism (docs/research/rocket_photoreal_2026-10-10.md) and
follows Zeke's "look for what is already online first": the vehicle is assembled from NASA's
public-domain SLS Block 1 hardware model + the Shuttle SRB + the Orion capsule, the tower is NASA's
Mobile Launcher, steam/smoke/clouds are JangaFX CC0 VDB simulations, the Earth is NASA Blue/Black
Marble, scrub is Poly Haven CC0. prep_assets.py turns the downloads into one library .blend.

Built DETERMINISTICALLY from the timeline npz: every music reaction is BAKED (camera, lights,
plumes, LED clock, steam, exposure); no handlers, no Python drivers. Frame f = timeline index f-1.

Story (SLS sequence, drop entries from the npz so it re-times itself):
  intro         pre-dawn pad under xenon floods; LED countdown; deluge water starts sheeting off
                the mobile-launcher deck (T-10 s), hydrogen burn-off sparks, the four core engines
                START on the build's bar (anticipation: translucent plumes + Mach disks + first
                steam), pad floods cut on T_black; T-0 = drop entry: SRB IGNITION.
  drop 1        SRB ignition + liftoff: blinding plume, steam blasting out of the flame trench,
                tower clear, long-lens tracking with heat shimmer, onboard booster camera.
  calm          climbing into a cumulus deck lit from inside by the plume.
  drop 2        MAX-Q: condensation (vapour) cone shroud, flickering on the kicks.
  drop 3        BOOSTER SEPARATION (separation-motor puffs, boosters tumble away).
  drop 4        LAUNCH-ABORT-TOWER JETTISON: tower motor fires, ogive + panels fly off, Orion.
  calm          MECO + core separation, coast; the high plume becomes a sunlit "jellyfish".
  BIG drop      upper-stage ignition and the sun breaking Earth's limb (NASA Blue/Black Marble).
  outro         engine cutoff, drift over the Earth.
Music: kick = engine pulse + camera jolt, snare = ice shed / flood flicker, sub-bass = plume and
cloud glow, hi-hat = beacon strobes (no sparkle glints: real ice falls, it doesn't twinkle).
"""
import argparse
import glob
import math
import os
import sys

import bmesh  # noqa: F401
import bpy
import numpy as np
from mathutils import Euler as Euler_, Matrix, Quaternion, Vector  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SONG = os.path.join(REPO, "state", "tzeke_songs", "hit_a_bump")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=os.path.join(SONG, "timeline_30fps.npz"))
    ap.add_argument("--audio", default=os.path.join(SONG, "hit_a_bump_half.wav"))
    ap.add_argument("--logo", default=os.path.join(REPO, "state", "tzeke_songs", "_assets",
                                                   "tzeke000_symbol.png"))
    ap.add_argument("--font", default=os.path.join(REPO, "assets", "fonts", "BebasNeue-Regular.ttf"))
    ap.add_argument("--assets", default="/data/iris_assets")
    ap.add_argument("--lib", default="/data/iris_assets/rocket2/rocket2_lib.blend")
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "scenes", "rocket2", "rocket2.blend"))
    ap.add_argument("--samples", type=int, default=0, help="0 = tuned default")
    ap.add_argument("--res-pct", type=int, default=100)
    # per-song story anchors (seconds). Default = first 5 drop entries / auto MECO (Hit a Bump).
    ap.add_argument("--beats", default="", help="IGN,MAXQ,BSEP,FAIRSEP,SES in seconds (snapped to entries)")
    ap.add_argument("--meco", type=float, default=-1.0, help="MECO time in s (default: calm before SES)")
    ap.add_argument("--e-start", type=float, default=-1.0, help="E_calm shot start in s (default: end of max-Q seg)")
    ap.add_argument("--pad-end", type=float, default=-1.0, help="end of the pad/liftoff act in s (default: end of drop 1)")
    ap.add_argument("--title", default="HIT A BUMP", help="countdown-clock title")
    # v3 'rave' cut (Zeke 10-11): LED neon rocket, kick-driven flames, a giant LED speaker as the
    # payload that blasts an incoming asteroid apart with sound on the last drop
    ap.add_argument("--rave", action="store_true", help="v3 rave cut (LEDs, speaker payload, asteroid act)")
    ap.add_argument("--ign", type=float, default=-1.0, help="liftoff in s, snapped to a bar (NOT a drop entry)")
    ap.add_argument("--ast", type=float, default=-1.0, help="rave: asteroid act starts (s)")
    ap.add_argument("--fire", type=float, default=-1.0, help="rave: the speaker opens fire (s) = the last drop")
    ap.add_argument("--boom", type=float, default=-1.0, help="rave: the asteroid shatters (s)")
    ap.add_argument("--after", type=float, default=-1.0, help="rave: aftermath calm (s)")
    ap.add_argument("--final", type=float, default=-1.0, help="rave: the final hit (s)")
    ap.add_argument("--fade", type=float, default=-1.0, help="rave: fade to black (s)")
    # v4 (Zeke's song map 10-13; every value is snapped to the bar grid)
    ap.add_argument("--v4", action="store_true", help="v4 cut (3 boosters, B3 failure, tumble, HUD, speaker v2)")
    ap.add_argument("--stems", default="/data/iris_assets/blastoff/stems/blastoff_elements_30fps.npz")
    ap.add_argument("--build0", type=float, default=32.87, help="0:33 build: LED rings boot bar by bar")
    ap.add_argument("--b3alarm", type=float, default=43.77, help="2nd half of the build: all LEDs red + HUD")
    ap.add_argument("--b3sep", type=float, default=53.33, help="booster 3 separates (too early)")
    ap.add_argument("--b3boom", type=float, default=56.07, help="booster 3 explodes (0:56 false drop)")
    ap.add_argument("--reboot", type=float, default=66.97, help="1:07 halftime drop: LED rainbow reboot sweep")
    ap.add_argument("--bridge", type=float, default=76.5, help="1:17 bridge")
    ap.add_argument("--bd2", type=float, default=130.53, help="2:10 breakdown 2")
    ap.add_argument("--glimpse", type=float, default=122.5, help="asteroid glimpse (breakdown 1)")
    ap.add_argument("--cellfrac", default="/data/tmp_dl/cellfrac/x", help="Cell Fracture extension (unpacked)")
    ap.add_argument("--speaker-gltf", default="/data/iris_assets/blastoff/sf_stage_speaker_gltf/scene.gltf")
    ap.add_argument("--movinghead", default="/data/iris_assets/blastoff/sf_moving_head/model.glb")
    ap.add_argument("--font-hud", default="/data/iris_assets/fonts/b612/B612-Bold.ttf")
    return ap.parse_args(argv)


ARGS = parse_args()
AS = ARGS.assets

# =====================================================================================
# 1. MUSIC → per-frame control arrays (pure numpy, stateless over the whole song)
# =====================================================================================
TL = np.load(ARGS.timeline)
FPS = int(TL["fps"])
N = len(TL["rms"])
BPM = float(TL["bpm"])
BAR = 4 * 60.0 * FPS / BPM                                   # frames per bar (~48.6)
IDX = np.arange(N)


def gsmooth(x, sigma):
    if sigma <= 0:
        return x.astype(np.float64)
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x.astype(np.float64), r, mode="edge"), k, mode="valid")


def age_env(age, tau):
    a = np.asarray(age, np.float64)
    return np.where(a < 1e5, np.exp(-a / tau), 0.0)


ENT = sorted(int(e) for e in TL["entries"])
assert len(ENT) >= 5, f"need 5 drop entries, got {ENT}"
if ARGS.beats:
    _want = [float(v) * FPS for v in ARGS.beats.split(",")]
    assert len(_want) == 5, "--beats needs 5 values"
    IGN, MAXQ, BSEP, FAIRSEP, SES = [min(ENT, key=lambda e: abs(e - w)) for w in _want]
else:
    IGN, MAXQ, BSEP, FAIRSEP, SES = ENT[:5]
SEGS = [tuple(int(v) for v in s) for s in TL["segs"]]
if ARGS.meco >= 0:
    MECO = min((s0 for s0, s1, h in SEGS), key=lambda s0: abs(s0 - ARGS.meco * FPS))
else:
    MECO = max(s0 for s0, s1, h in SEGS if h == 0 and s1 <= SES)      # calm before the big drop
SECO = min(s0 for s0, s1, h in SEGS if h == 0 and s0 > SES)           # outro start

KICK = age_env(TL["T_kage"], 3.0)
SNARE = age_env(TL["T_sage"], 4.0)
HOT = TL["T_hot"].astype(np.float64)
BIG = TL["T_big"].astype(np.float64)
PRE = TL["T_pre"].astype(np.float64)
BLACK = TL["T_black"].astype(np.float64)
ONE = age_env(TL["T_one_age"], 9.0)
STROBE = (TL["T_strobe_age"] == 0).astype(np.float64)
BASS = gsmooth(TL["bass"], 1.5)
_hh = np.clip(TL["high"] - gsmooth(TL["high"], 12), 0, None)
HAT = np.clip(_hh / (np.percentile(_hh, 98) + 1e-6), 0, 1)
BEAT_PH = TL["T_ph"].astype(np.float64)
KIDX = TL["T_kidx"].astype(np.int64)
SIDX = TL["T_sidx"].astype(np.int64)
SAGE = np.minimum(TL["T_sage"].astype(np.float64), 1e5)
DOWNBEATS = np.unique(TL["T_bar_start"])


def snap(f):
    """Nearest bar downbeat to frame index f (cuts land ON the downbeat)."""
    return int(DOWNBEATS[np.argmin(np.abs(DOWNBEATS - f))])


RAVE = bool(ARGS.rave)
if ARGS.ign >= 0:                       # v3: liftoff on the first sung 'blast off', not a drop entry
    IGN = snap(ARGS.ign * FPS)
if RAVE:
    AST0, FIRE, BOOM, AFTER, FINAL = (snap(v * FPS) for v in (ARGS.ast, ARGS.fire, ARGS.boom, ARGS.after,
                                                               ARGS.final))
    FADE = int(round(ARGS.fade * FPS))
    SECO = AST0                         # the upper stage shuts down as the asteroid act opens
    assert IGN < MAXQ < BSEP < FAIRSEP < MECO < SES < AST0 < FIRE < BOOM < AFTER < FINAL < FADE <= N

# --- v4 events (Zeke's song map, 10-13 — authoritative over the timeline's segs) -----------------
V4 = bool(ARGS.v4)
assert RAVE and V4, "rocket4 builds the v4 cut only: pass --rave --v4 (see scratch/scenes/blastoff4_env.sh)"
BUILD0, B3ALARM, B3SEP, B3BOOM, REBOOT, BRIDGE, BD2, GLIMPSE = (
    snap(v * FPS) for v in (ARGS.build0, ARGS.b3alarm, ARGS.b3sep, ARGS.b3boom, ARGS.reboot, ARGS.bridge,
                            ARGS.bd2, ARGS.glimpse))
assert IGN < BUILD0 < B3ALARM < B3SEP < B3BOOM < REBOOT == MAXQ < BRIDGE < BSEP, \
    (IGN, BUILD0, B3ALARM, B3SEP, B3BOOM, REBOOT, MAXQ, BRIDGE, BSEP)
assert FAIRSEP < GLIMPSE < BD2 < AST0 < FIRE < BOOM < AFTER
# the asteroid shatters on the downbeat ONE BAR before drop 2 ends (AFTER = the drop's end = outro start)
assert abs((AFTER - BOOM) - BAR) < 2, ("BOOM must be one bar before AFTER", BOOM, AFTER, BAR)
print("[rocket4] events (s): build", BUILD0 / FPS, "alarm", B3ALARM / FPS, "b3sep", B3SEP / FPS, "boom",
      B3BOOM / FPS, "reboot", REBOOT / FPS, "bridge", BRIDGE / FPS, "hardsep", BSEP / FPS, "fair", FAIRSEP / FPS,
      "glimpse", GLIMPSE / FPS, "bd2", BD2 / FPS, "ast", AST0 / FPS, "fire", FIRE / FPS, "shatter", BOOM / FPS,
      "outro", AFTER / FPS)
# stems (Demucs htdemucs_ft, research #2): per-element channels with their own attack/release
EL = np.load(ARGS.stems, allow_pickle=True)
assert len(EL["kick"]) == N
_kage = np.full(N, 1e5)                             # age since the last kick (frames), on/off-beat weight
_last, _kw_cur = -100000, 0.0
_kw = np.zeros(N)
for i in range(N):
    if EL["kick"][i] > 0:
        _last = i
        _kw_cur = 1.0 if EL["kick_onbeat"][i] > 0 else 0.6
    _kage[i] = i - _last
    _kw[i] = _kw_cur if _last >= 0 else 0.0
KICK4 = np.where(_kage < 1e4, np.exp(-_kage / 3.6), 0.0) * _kw          # 0 / 120 ms punch
def _pnorm(x, lo=5, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)
LEAD = _pnorm(EL["lead_bright"] * EL["other_env"])                     # screeches / leads
SYNTH = _pnorm(EL["other_env"])                                        # the breakdowns' synth
BASS4 = _pnorm(EL["bass_env"])
HAT4 = _pnorm(EL["hat_env"], 5, 99)
SNARE4 = _pnorm(EL["snare_env"], 5, 99) * (EL["snare_solo"].astype(float) > 0).astype(float)
SNARE4 = np.maximum(SNARE4, gsmooth(EL["snare_solo"].astype(float), 1.0) * 1.5).clip(0, 1)
SPEC32 = np.clip(EL["spec32"].astype(np.float64), 0, 1)                 # (N, 32) log-spaced, per-band normalised
# Zeke's section map as masks: drops = kick + screech, breakdowns = synth
SEC_DROP = (((IDX >= REBOOT) & (IDX < BRIDGE)) | ((IDX >= BSEP) & (IDX < FAIRSEP)) | ((IDX >= FIRE) & (IDX < AFTER))
            ).astype(np.float64)
SEC_FALSE = ((IDX >= B3BOOM) & (IDX < REBOOT)).astype(np.float64)
SEC_BD = (((IDX >= FAIRSEP) & (IDX < FIRE))).astype(np.float64)
SEC_BUILD = ((IDX >= BUILD0) & (IDX < B3BOOM)).astype(np.float64)


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 4294967295.0


def ramp(i, a, b):
    return np.clip((np.asarray(i, np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3.0):
    return 1 - (1 - np.clip(x, 0, 1)) ** p


# --- SLS engine states -------------------------------------------------------------------
# T-0 (SRB ignition) is EXACTLY the first drop entry (IGN, timeline index 987 = frame 988).
# The four core engines start on the build's bar (real: T-6.36 s; here the bar before the build
# peak so the start-up reads as the musical wind-up), the burn-off igniters spark a bar earlier,
# the deluge water starts ~T-10 s.
_pre_ix = np.nonzero(PRE > 0)[0]
_pre_ix = _pre_ix[_pre_ix < IGN]
PRE_START = int(_pre_ix[0]) if len(_pre_ix) else max(int(IGN - 2 * BAR), 1)   # v3: no build before an intro liftoff
RS25_T = int(DOWNBEATS[DOWNBEATS <= PRE_START][-1])          # bar at/just before the build
SPARK_T = int(DOWNBEATS[DOWNBEATS < RS25_T][-1])             # one bar earlier: sparklers
DELUGE_T = snap(IGN - 10.0 * FPS)                             # water on the deck
LIFT = IGN + 4                                                # SRB ignition = liftoff (~0.13 s)
_eng_start = [RS25_T + int(round(k * 0.12 * FPS)) for k in range(4)]   # staggered 120 ms
on_eng = [ramp(IDX, s, s + 14) * (1 - ramp(IDX, MECO, MECO + 4)) for s in _eng_start]
on_core = np.mean(on_eng, axis=0)
on_boost = ramp(IDX, IGN - 1, IGN + 1) * (1 - ramp(IDX, BSEP, BSEP + 2))
on_s2 = ramp(IDX, SES - 1, SES + 2) * (1 - ramp(IDX, SECO, SECO + 8))
IGN_FLASH = np.where((IDX >= IGN) & (IDX < IGN + 5), 1.0 - (IDX - IGN) / 5.0, 0.0)   # SRB light-up pop
SPARKS = ramp(IDX, SPARK_T, SPARK_T + 3) * (1 - ramp(IDX, IGN + 2, IGN + 8))
DELUGE = ramp(IDX, DELUGE_T, DELUGE_T + 40) * (1 - ramp(IDX, IGN + 30 * FPS, IGN + 32 * FPS))
PULSE = 1.0 + HOT * (0.30 * KICK + 0.10 * BASS + 0.12 * SNARE) + BIG * 0.35 * ONE
KH = KICK * HOT                                                # how far a kick reaches into the plume length
if RAVE:                                                       # v3: once it flies, the flames ride EVERY kick
    _flying = (IDX >= IGN).astype(np.float64)
    KH = KICK * np.maximum(HOT, 0.75 * _flying)
    PULSE = PULSE + _flying * (0.40 * KICK * (1 - HOT) + 0.18 * KICK * HOT + 0.06 * BASS)
T_SEC = np.maximum(IDX - LIFT, 0) / FPS
sys.path.insert(0, HERE)
import flight_model as FM                                      # noqa: E402  (v4 real flight model)
ALT, VEL = FM.pad_climb(N, LIFT, FPS)                          # metres above the pad: mass, thrust, g, drag

# --- sets (which world is on screen) ---------------------------------------------------
if ARGS.pad_end >= 0 and RAVE:                                 # v3: the climb starts on a bar
    PAD_END = snap(ARGS.pad_end * FPS)
elif ARGS.pad_end >= 0:                                        # long liftoff acts span several drops
    PAD_END = min((s1 for s0, s1, h in SEGS), key=lambda s1: abs(s1 - ARGS.pad_end * FPS))
else:
    PAD_END = [s1 for s0, s1, h in SEGS if s0 == IGN][0]      # end of drop 1
SET = np.zeros(N, np.int32)                                    # 0 pad, 1 clouds, 2 high, 3 space
SET[PAD_END:] = 1
SET[BSEP:] = 2
SET[SES:] = 3


def in_set(k):
    return (SET == k).astype(np.float64)


# =====================================================================================
# 2. SHOT LIST (cuts on drop entries and bar downbeats). Camera language (research §1.14):
#    long-lens tracking from km away + low wide pad remotes + onboard cameras; chase angles
#    only where an onboard cue (fisheye, vibration, exposure pumping) sells them.
# =====================================================================================
def seg_after(f):
    return [s1 for s0, s1, h in SEGS if s0 == f][0]


A_CUTS = [0, snap(IGN * 0.30), snap(IGN * 0.56), snap(IGN * 0.80)]
B_END = PAD_END
B_CUTS = [IGN, snap(IGN + 3.0 * BAR), snap(IGN + 5.5 * BAR), snap(IGN + 7.0 * BAR)]
if ARGS.pad_end >= 0:                                          # spread the four liftoff shots over the act
    B_CUTS = [IGN] + [snap(IGN + (PAD_END - IGN) * q) for q in (0.22, 0.48, 0.74)]
C_CUTS = [PAD_END, snap(PAD_END + (MAXQ - PAD_END) * 0.52)]
D_END = seg_after(MAXQ)                                        # max-Q vapour cone window
E_START = (min((s0 for s0, s1, h in SEGS), key=lambda s0: abs(s0 - ARGS.e_start * FPS))
           if ARGS.e_start >= 0 else D_END)
D_CUTS = [MAXQ, snap(MAXQ + 2.5 * BAR), snap(MAXQ + 4.5 * BAR)]
F_END = seg_after(BSEP)
F_CUTS = [BSEP, snap(BSEP + 3 * BAR), snap(BSEP + 5.5 * BAR)]
H_END = seg_after(FAIRSEP)
H_CUTS = [FAIRSEP, snap(FAIRSEP + 2 * BAR)]
J_CUTS = [SES, snap(SES + 3 * BAR), snap(SES + 6 * BAR)]
if SECO - SES > 14 * BAR:                                      # long big drop: spread the three shots
    J_CUTS = [SES, snap(SES + (SECO - SES) * 0.34), snap(SES + (SECO - SES) * 0.67)]
SHOTS = [  # (start index, name)
    (A_CUTS[0], "A1_wide"), (A_CUTS[1], "A2_clock"), (A_CUTS[2], "A3_logo"), (A_CUTS[3], "A4_low"),
    (B_CUTS[0], "B1_ign"), (B_CUTS[1], "B2_tower"), (B_CUTS[2], "B3_track"), (B_CUTS[3], "B4_tail"),
    (C_CUTS[0], "C1_cloud"), (C_CUTS[1], "C2_side"),
    (D_CUTS[0], "D1_maxq"), (D_CUTS[1], "D2_under"), (D_CUTS[2], "D3_wide"),
    (E_START, "E_calm"),
    (F_CUTS[0], "F1_sep"), (F_CUTS[1], "F2_tail"), (F_CUTS[2], "F3_wide"),
    (F_END, "G_calm"),
    (H_CUTS[0], "H1_fair"), (H_CUTS[1], "H2_side"),
    (MECO, "I_meco"),
    (J_CUTS[0], "J1_limb"), (J_CUTS[1], "J2_side"), (J_CUTS[2], "J3_wide"),
    (SECO, "K_outro"),
]
if RAVE:
    def _sc(sec):
        return snap(sec * FPS)
    # v4: the climb from 0:28 to the 1:28 drop is re-cut around the booster-3 story (cuts every 2-3 bars);
    # drop 2 gets laser shots; the shatter moves to one bar before the drop ends (no 'let's go' cue).
    SHOTS = [(s, n) for s, n in SHOTS if n not in ("H2_side", "J2_side", "J3_wide", "K_outro", "C2_side",
                                                   "D1_maxq", "D2_under", "D3_wide", "E_calm")]
    SHOTS += [(BUILD0, "V1_rings"), (_sc(38.33), "V2_panel"), (B3ALARM, "V3_feed"), (_sc(47.87), "V4_red"),
              (_sc(50.6), "V5_onb"), (B3SEP, "V6_sep"), (B3BOOM, "V7_boom"), (_sc(57.43), "V7b_lost"),
              (_sc(58.8), "V8_tumble"), (_sc(61.53), "V9_fish"), (_sc(64.27), "V10_track"),
              (REBOOT, "R1_neon"), (snap(REBOOT + 2 * BAR), "D3_wide"), (_sc(72.8), "R2_nose"),
              (BRIDGE, "E_calm"),
              (ENT[3], "R3_orbit"), (_sc(108.4), "R4_count"),
              (H_CUTS[1], "S1_rise"), (_sc(120.0), "S2_hero"), (GLIMPSE, "L0_glimpse"), (_sc(126.5), "S3_wide"),
              (AST0, "L1_ast"), (_sc(144.0), "L2_over"), (_sc(150.2), "L3_charge"), (_sc(152.3), "L4_rock"),
              (FIRE, "M1_fire"), (snap(FIRE + 3 * BAR), "M2_wide"), (snap(FIRE + 5 * BAR), "M3_crack"),
              (snap(FIRE + 8 * BAR), "M7_fan"), (snap(FIRE + 12 * BAR), "M6_orbit"),
              (snap(FIRE + 16 * BAR), "M8_tunnel"), (snap(FIRE + 19 * BAR), "M9_blind"),
              (snap(FIRE + 21 * BAR), "M10_pin"), (BOOM, "M4_boom"), (AFTER, "M5_hero"),
              (snap(AFTER + 3 * BAR), "N1_drift"), (FINAL, "N2_final")]
    assert len({s for s, _ in SHOTS}) == len(SHOTS), sorted(SHOTS)
SHOTS.sort()
CUTS = sorted({s for s, _ in SHOTS if s > 0} | {PAD_END, BSEP, SES})
SHOT_OF = np.zeros(N, np.int32)
for k, (s, _) in enumerate(SHOTS):
    SHOT_OF[s:] = k
ONBOARD = {"B4_tail", "D2_under", "F1_sep", "I_meco", "V5_onb", "V9_fish"}   # lens/vibration cues
FISHEYE = {"V9_fish"}                                     # v4: strong barrel distortion + digital breakups
HUD_SHOTS = {"V3_feed": "TRACKING CAM 07", "V5_onb": "ONBOARD CAM 2", "V7b_lost": "TRACKING CAM 07",
             "V9_fish": "ONBOARD CAM 3"}                  # v4: camera-feed shots get the warning HUD (post pass)
print("[rocket2] entries", ENT, "MECO", MECO, "SECO", SECO, "PAD_END", PAD_END,
      "RS25", RS25_T, "SPARK", SPARK_T, "DELUGE", DELUGE_T)
print("[rocket2] shots", [(s, n) for s, n in SHOTS])

# =====================================================================================
# 3. BLENDER HELPERS
# =====================================================================================
SC = bpy.context.scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
             bpy.data.curves, bpy.data.worlds, bpy.data.node_groups, bpy.data.images, bpy.data.volumes):
    for d in list(coll):
        coll.remove(d)

KEY_ENUM = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
LIN, CONST = KEY_ENUM["LINEAR"].value, KEY_ENUM["CONSTANT"].value
CUT_SET = np.zeros(N, bool)
for c in CUTS:
    if 0 < c < N:
        CUT_SET[c - 1] = True                       # the last frame BEFORE a cut holds (no smear)


def _fcurve(idb, path, index):
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_bake")
    return ad.action.fcurve_ensure_for_datablock(idb, path, index=index)


def bake(idb, path, vals, index=0, const=False):
    """Bake one value per timeline frame onto idb.path[index] (bulk foreach_set)."""
    v = np.asarray(vals, np.float64)
    if v.ndim == 0:
        v = np.full(N, float(v))
    assert len(v) == N, (path, len(v))
    keep = np.ones(N, bool)
    keep[1:-1] = (v[1:-1] != v[:-2]) | (v[1:-1] != v[2:])
    keep |= CUT_SET
    keep[np.minimum(np.nonzero(CUT_SET)[0] + 1, N - 1)] = True
    ix = np.nonzero(keep)[0]
    fc = _fcurve(idb, path, index)
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(ix))
    co = np.empty(2 * len(ix), np.float32)
    co[0::2] = ix + 1
    co[1::2] = v[ix]
    kp.foreach_set("co", co)
    interp = np.full(len(ix), CONST if const else LIN, np.int32)
    interp[CUT_SET[ix]] = CONST
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc


def bake_vec(idb, path, arr, const=False):
    arr = np.asarray(arr, np.float64)
    for k in range(arr.shape[1]):
        bake(idb, path, arr[:, k], index=k, const=const)


def link(ob, parent=None):
    SC.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    return link(e, parent)


def mesh_obj(name, verts, faces, mat=None, parent=None, loc=(0, 0, 0), smooth=False, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        loop_v = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32)[loop_v].ravel())
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    return link(ob, parent)


def lathe(name, prof, segs=64, mat=None, parent=None, loc=(0, 0, 0), a0=0.0, a1=2 * math.pi,
          cap=False, smooth=True):
    """Surface of revolution around +Z. prof = [(r, z), ...]. UV: u = angle/2pi, v = profile t."""
    prof = np.asarray(prof, np.float64)
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    na = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(na) / segs
    seglen = np.r_[0, np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))]
    tv = seglen / max(seglen[-1], 1e-9)
    verts, uvs, faces = [], [], []
    for j, (r, z) in enumerate(prof):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append(((a % (2 * math.pi)) / (2 * math.pi), tv[j]))
    for j in range(len(prof) - 1):
        for k in range(segs):
            k2 = (k + 1) % na
            faces.append((j * na + k, j * na + k2, (j + 1) * na + k2, (j + 1) * na + k))
    if cap:
        c = len(verts)
        verts.append((0, 0, prof[0, 1]))
        uvs.append((0.5, 0))
        for k in range(segs):
            faces.append((c, (k + 1) % na, k))
    # per-vertex UV; seam duplicates avoided by computing per-loop below
    me_uvs = np.asarray(uvs, np.float32)
    ob = mesh_obj(name, verts, faces, mat, parent, loc, smooth=smooth, uvs=None)
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    uv = me_uvs[loop_v].copy()
    # fix the wrap seam: in a face whose u spans > 0.5, lift the small ones by 1
    poly_start = np.empty(len(me.polygons), np.int32)
    poly_tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", poly_start)
    me.polygons.foreach_get("loop_total", poly_tot)
    for ps, pt in zip(poly_start, poly_tot):
        us = uv[ps:ps + pt, 0]
        if us.max() - us.min() > 0.5:
            us[us < 0.5] += 1.0
            uv[ps:ps + pt, 0] = us
    uvl.data.foreach_set("uv", uv.ravel())
    return ob


class MB:
    """Accumulate boxes / polys for ONE material, then emit one mesh object."""

    def __init__(self):
        self.v, self.f = [], []

    def box(self, c, s, rot=None):
        cx, cy, cz = c
        sx, sy, sz = (s[0] / 2, s[1] / 2, s[2] / 2)
        pts = [Vector((x * sx, y * sy, z * sz)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        if rot is not None:
            pts = [rot @ p for p in pts]
        o = len(self.v)
        self.v += [(p.x + cx, p.y + cy, p.z + cz) for p in pts]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(o + i for i in q))

    def beam(self, a, b, t):
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = d.to_track_quat("Z", "Y").to_matrix()
        self.box(tuple((a + b) / 2), (t, t, d.length), rot)

    def poly(self, pts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f += [tuple(o + i for i in f) for f in faces]

    def obj(self, name, mat=None, parent=None, loc=(0, 0, 0)):
        return mesh_obj(name, self.v, self.f, mat, parent, loc)


def uv_sphere(name, r=1.0, seg=32, rings=16, mat=None, parent=None, loc=(0, 0, 0)):
    prof = [(r * math.sin(math.pi * k / rings), -r * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(name, prof, seg, mat, parent, loc)


def cube_obj(name, mat=None, parent=None, loc=(0, 0, 0), scale=(1, 1, 1)):
    m = MB()
    m.box((0, 0, 0), (2, 2, 2))
    ob = m.obj(name, mat, parent, loc)
    ob.scale = scale
    return ob


# ---- node helpers --------------------------------------------------------------------
def nd(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items():
        if k == "inp":
            for name, val in v.items():
                n.inputs[name].default_value = val
        else:
            setattr(n, k, v)
    return n


def lk(nt, a, b):
    nt.links.new(a, b)


def _sock(nt, x, vec=False):
    if isinstance(x, (int, float)):
        v = nd(nt, "ShaderNodeValue")
        v.outputs[0].default_value = float(x)
        return v.outputs[0]
    if isinstance(x, (tuple, list)):
        c = nd(nt, "ShaderNodeCombineXYZ")
        for k in range(3):
            c.inputs[k].default_value = float(x[k])
        return c.outputs[0]
    return x


def M(nt, op, a, b=None, c=None, clamp=False):
    n = nd(nt, "ShaderNodeMath", operation=op, use_clamp=clamp)
    for k, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = float(x)
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[0]


def V(nt, op, a, b=None, scale=None):
    n = nd(nt, "ShaderNodeVectorMath", operation=op)
    for k, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (tuple, list)):
            n.inputs[k].default_value = x
        else:
            lk(nt, x, n.inputs[k])
    if scale is not None:
        if isinstance(scale, (int, float)):
            n.inputs["Scale"].default_value = scale
        else:
            lk(nt, scale, n.inputs["Scale"])
    return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]


def value_node(nt, name, v=0.0):
    n = nd(nt, "ShaderNodeValue", name=name, label=name)
    n.outputs[0].default_value = v
    return n


def value_path(name):
    return f'nodes["{name}"].outputs[0].default_value'


def mix_col(nt, fac, a, b):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    for k, x in ((0, fac), (6, a), (7, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        elif isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.node_tree.nodes.clear()
    return m, m.node_tree


def principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, bump=None):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = tuple(color) + (1,)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = tuple(emit) + (1,)
        b.inputs["Emission Strength"].default_value = emit_strength
    if bump:
        tc = nd(nt, "ShaderNodeTexCoord")
        nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": bump[0], "Detail": 8.0})
        lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
        bp = nd(nt, "ShaderNodeBump", inp={"Strength": bump[1]})
        lk(nt, nz.outputs["Fac"], bp.inputs["Height"])
        lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
        cr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.3, "From Max": 0.7,
                                               "To Min": rough * 0.7, "To Max": min(1, rough * 1.3)})
        lk(nt, nz.outputs["Fac"], cr.inputs["Value"])
        lk(nt, cr.outputs["Result"], b.inputs["Roughness"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])
    return m


def emissive(name, color, strength, objcolor_gain=False):
    """Emission; with objcolor_gain the per-object colour (keyable!) multiplies it."""
    m, nt = new_mat(name)
    e = nd(nt, "ShaderNodeEmission", inp={"Color": tuple(color) + (1,), "Strength": strength})
    if objcolor_gain:
        oi = nd(nt, "ShaderNodeObjectInfo")
        mc = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mc.inputs[0].default_value = 1.0
        mc.inputs[6].default_value = tuple(color) + (1,)
        lk(nt, oi.outputs["Color"], mc.inputs[7])
        lk(nt, mc.outputs[2], e.inputs["Color"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, e.outputs[0], o.inputs["Surface"])
    return m


def set_color_keys(ob, rgb_arr, const=True):
    """Bake obj.color (RGBA) — read in shaders through Object Info > Color."""
    arr = np.asarray(rgb_arr, np.float64)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], 3, 1)
    for k in range(3):
        bake(ob, "color", arr[:, k], index=k, const=const)


def no_shadow(ob):
    ob.visible_shadow = False
    return ob


VIS = {}                                    # object -> per-frame visibility (bool array)


def show(ob, mask):
    VIS[ob.name] = np.asarray(mask, bool)
    return ob


def blackbody(k):
    """Approx sRGB-linear colour of a blackbody at k kelvin (Tanner Helland fit)."""
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 329.698727446 * (t - 60) ** -0.1332047592 / 255))
    g = (min(1, max(0, (99.4708025861 * math.log(t) - 161.1195681661) / 255)) if t <= 66
         else min(1, max(0, 288.1221695283 * (t - 60) ** -0.0755148492 / 255)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1, max(0, (138.5177312231 * math.log(t - 10)
                                                                - 305.0447927307) / 255)))
    return tuple(c ** 2.2 for c in (r, g, b))



# =====================================================================================
# 4. MATERIALS — PBR rules (lookdev §2): paint/foam/concrete are DIELECTRIC, albedo never 1.0,
#    roughness from CC0 scans (Non-Color), box projection in metres; metal only where bare.
# =====================================================================================
def load_img(path, data=False):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "Non-Color" if data else "sRGB"
    return im


def tex_box(nt, path, tile_m, data=False, coord=None, blend=0.25):
    """Box-projected image texture in object space; tile_m = metres per tile. Returns the node."""
    if coord is None:
        tc = nd(nt, "ShaderNodeTexCoord")
        coord = tc.outputs["Object"]
    mp = nd(nt, "ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / tile_m,) * 3
    lk(nt, coord, mp.inputs["Vector"])
    t = nd(nt, "ShaderNodeTexImage", image=load_img(path, data), projection="BOX", projection_blend=blend)
    lk(nt, mp.outputs[0], t.inputs["Vector"])
    return t


def ph(name, kind, res="2k"):
    d = os.path.join(AS, "polyhaven", "tex_" + name)
    return os.path.join(d, f"{name}_{kind}_{res}.jpg")


def acg(name, kind):
    return os.path.join(AS, "ambientcg", name, f"{name}_2K-JPG_{kind}.jpg")


def load_logo(path):
    """The emblem PNG is opaque on white: flood-fill the white that touches the border → alpha."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    white = px[..., :3].min(-1) > 0.86
    reg = np.zeros_like(white)
    reg[0, :] = white[0, :]
    reg[-1, :] = white[-1, :]
    reg[:, 0] = white[:, 0]
    reg[:, -1] = white[:, -1]
    for _ in range(2000):
        g = reg.copy()
        g[1:] |= reg[:-1]
        g[:-1] |= reg[1:]
        g[:, 1:] |= reg[:, :-1]
        g[:, :-1] |= reg[:, 1:]
        g &= white
        if (g == reg).all():
            break
        reg = g
    a = 1.0 - reg.astype(np.float32)
    k = np.array([1, 4, 6, 4, 1], np.float32)
    k /= k.sum()
    for ax in (0, 1):
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), k, "valid"), ax, a)
    out = bpy.data.images.new("TzekeLogoKeyed", w, h, alpha=True, float_buffer=False)
    px[..., 3] = a
    out.pixels.foreach_set(px.ravel())
    out.pack()
    bpy.data.images.remove(img)
    return out


LOGO = load_logo(ARGS.logo)
PAINT_R = acg("PaintedMetal017", "Roughness")
PAINT_N = acg("PaintedMetal017", "NormalGL")
PAINT_C = acg("PaintedMetal017", "Color")


def scan_rough_normal(nt, tile_m, rough_lo, rough_hi, nstr=0.35, coord=None):
    """(roughness socket, normal socket) from the ambientCG painted-metal scan."""
    r = tex_box(nt, PAINT_R, tile_m, True, coord)
    n = tex_box(nt, PAINT_N, tile_m, True, coord)
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": rough_lo, "To Max": rough_hi})
    lk(nt, r.outputs["Color"], rr.inputs["Value"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": nstr})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    return rr.outputs["Result"], nm.outputs["Normal"]


def wear(nt, tile_m, lo=0.80, coord=None):
    """Compressed luminance of the scan's colour map: wear/grime WITHOUT its blue pattern."""
    c = tex_box(nt, PAINT_C, tile_m, False, coord)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, c.outputs["Color"], bw.inputs[0])
    mr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.15, "From Max": 0.6, "To Min": lo, "To Max": 1.0})
    lk(nt, bw.outputs[0], mr.inputs["Value"])
    return mr.outputs["Result"]


def col_mul(nt, col, fac):
    n = nd(nt, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    n.inputs[0].default_value = 1.0
    for k, x in ((6, col), (7, fac)):
        if isinstance(x, (tuple, list)):
            n.inputs[k].default_value = tuple(x) + ((1.0,) if len(x) == 3 else ())
        elif isinstance(x, (int, float)):
            n.inputs[k].default_value = (x, x, x, 1.0)
        else:
            lk(nt, x, n.inputs[k])
    return n.outputs[2]


def finish(nt, b):
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, b.outputs[0], o.inputs["Surface"])


def paint_white(name, albedo=0.74, tile=1.6, soot=None):
    """Vehicle white paint / thermal blanket: dielectric, scan roughness + normal, wear.
    soot = (z0, z1) object-space band that darkens with a keyed 'Soot' value (plume staining)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.08})
    rough, nrm = scan_rough_normal(nt, tile, 0.32, 0.62)
    base = col_mul(nt, (albedo, albedo, albedo * 0.985), wear(nt, tile * 1.7, 0.86))
    if soot:
        sv = value_node(nt, "Soot", 0.0)
        tc = nd(nt, "ShaderNodeTexCoord")
        sz = nd(nt, "ShaderNodeSeparateXYZ")
        lk(nt, tc.outputs["Object"], sz.inputs[0])
        st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
        lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (2.5, 2.5, 0.25)), st.inputs["Vector"])
        band = nd(nt, "ShaderNodeMapRange", inp={"From Min": soot[1], "From Max": soot[0]})
        lk(nt, sz.outputs[2], band.inputs["Value"])
        f = M(nt, "MULTIPLY", M(nt, "MULTIPLY", band.outputs["Result"], sv.outputs[0]),
              M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.6, -0.15), clamp=True)
        base = mix_col(nt, f, base, (0.05, 0.045, 0.04))
        rough = M(nt, "ADD", rough, M(nt, "MULTIPLY", f, 0.25))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def frost_layer(nt, base, rough, nrm, zlo, zhi, amount_node, scale=0.9):
    """Cryogenic frost (lookdev §4): white, rough, bumpy, noise-broken band in object Z,
    scaled by a keyed amount (sheds after liftoff). Returns (base, rough, normal)."""
    tc = nd(nt, "ShaderNodeTexCoord")
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sz.inputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": scale, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, tc.outputs["Object"], nz.inputs["Vector"])
    band = nd(nt, "ShaderNodeMapRange", inp={"From Min": zlo, "From Max": zhi, "To Min": 0.0, "To Max": 1.0})
    lk(nt, sz.outputs[2], band.inputs["Value"])
    tri = M(nt, "SUBTRACT", 1.0, M(nt, "ABSOLUTE", M(nt, "MULTIPLY_ADD", band.outputs["Result"], 2.0, -1.0)))
    mixe = M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 1.2, M(nt, "MULTIPLY", tri, amount_node))
    fr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 1.05, "From Max": 1.3})
    lk(nt, mixe, fr.inputs["Value"])
    f = fr.outputs["Result"]
    fine = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 60.0, "Detail": 4.0})
    lk(nt, tc.outputs["Object"], fine.inputs["Vector"])
    bp = nd(nt, "ShaderNodeBump", inp={"Distance": 0.004})
    lk(nt, fine.outputs["Fac"], bp.inputs["Height"])
    lk(nt, f, bp.inputs["Strength"])
    lk(nt, nrm, bp.inputs["Normal"])
    base = mix_col(nt, f, base, (0.86, 0.89, 0.92))
    rough = M(nt, "ADD", M(nt, "MULTIPLY", rough, M(nt, "SUBTRACT", 1.0, f)), M(nt, "MULTIPLY", f, 0.85))
    return base, rough, bp.outputs["Normal"]


# --- core stage: orange spray-on foam, intertank stringers, emblem decal, soot -----------
CORE_R = 4.2
LOGO_H = 6.2
LOGO_Z0 = 22.0                                   # on the LH2 tank, below the intertank
LOGO_ANG = -math.pi / 2                          # faces -Y (the camera side)


def foam_material(name):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.0})
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sz.inputs[0])
    # spray passes: large soft mottling + horizontal banding of the spray robot
    mot = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.35, "Detail": 5.0, "Roughness": 0.55})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.45)), mot.inputs["Vector"])
    band = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 3.0})
    lk(nt, V(nt, "MULTIPLY", ob, (0.15, 0.15, 2.2)), band.inputs["Vector"])
    v = M(nt, "ADD", M(nt, "MULTIPLY", M(nt, "SUBTRACT", mot.outputs["Fac"], 0.5), 0.55),
          M(nt, "MULTIPLY", M(nt, "SUBTRACT", band.outputs["Fac"], 0.5), 0.25))
    dark = (0.30, 0.105, 0.030)
    light = (0.50, 0.22, 0.075)
    base = mix_col(nt, M(nt, "ADD", 0.5, v, clamp=True), dark, light)
    # close-out patches (hand-sprayed repairs): lighter, smoother voronoi cells
    vo = nd(nt, "ShaderNodeTexVoronoi", inp={"Scale": 0.55, "Randomness": 1.0})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.6)), vo.inputs["Vector"])
    patch = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", vo.outputs["Distance"], 0.0),
              M(nt, "GREATER_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", vo.outputs["Distance"], 1.0)), 0.93))
    pw = nd(nt, "ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    lk(nt, vo.outputs["Color"], pw.inputs["Vector"])
    patch = M(nt, "MULTIPLY", M(nt, "LESS_THAN", pw.outputs["Value"], 0.18),
              M(nt, "LESS_THAN", vo.outputs["Distance"], 0.42))
    base = mix_col(nt, M(nt, "MULTIPLY", patch, 0.6), base, (0.58, 0.29, 0.10))
    # foam surface: orange-peel micro bump + lumpy macro bump; matte
    fine = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 22.0, "Detail": 6.0, "Roughness": 0.7})
    lk(nt, ob, fine.inputs["Vector"])
    lump = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 2.2, "Detail": 3.0})
    lk(nt, ob, lump.inputs["Vector"])
    h = M(nt, "ADD", M(nt, "MULTIPLY", fine.outputs["Fac"], 0.35), lump.outputs["Fac"])
    # intertank stringers (z 46.3..52.9): vertical ribs
    ang = M(nt, "ARCTAN2", sz.outputs[1], sz.outputs[0])
    rib = M(nt, "POWER", M(nt, "ABSOLUTE", M(nt, "SINE", M(nt, "MULTIPLY", ang, 72.0))), 6.0)
    it = M(nt, "MULTIPLY", M(nt, "GREATER_THAN", sz.outputs[2], 46.3), M(nt, "LESS_THAN", sz.outputs[2], 52.9))
    h = M(nt, "ADD", h, M(nt, "MULTIPLY", M(nt, "MULTIPLY", rib, it), 1.4))
    # flange rings between tanks/sections
    for zr in (6.2, 46.3, 52.9, 61.7):
        ring = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", sz.outputs[2], zr)), 0.10)
        h = M(nt, "SUBTRACT", h, M(nt, "MULTIPLY", ring, 0.8))
        base = mix_col(nt, M(nt, "MULTIPLY", ring, 0.35), base, (0.18, 0.07, 0.02))
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.18, "Distance": 0.03})
    lk(nt, h, bp.inputs["Height"])
    rough = M(nt, "ADD", 0.72, M(nt, "MULTIPLY", v, 0.2))
    nrm = bp.outputs["Normal"]
    # the Tzeke000 emblem: cylindrical decal on the LH2 tank facing -Y (painted on the foam)
    u = M(nt, "DIVIDE", M(nt, "SUBTRACT", ang, LOGO_ANG), 1.0)          # radians around from centre
    s_u = M(nt, "MULTIPLY", u, CORE_R)                                  # metres along the arc
    lw = LOGO_H * LOGO.size[0] / LOGO.size[1]
    du = M(nt, "ADD", M(nt, "DIVIDE", s_u, -lw), 0.5)
    dv = M(nt, "DIVIDE", M(nt, "SUBTRACT", sz.outputs[2], LOGO_Z0), LOGO_H)
    cxy = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, du, cxy.inputs[0])
    lk(nt, dv, cxy.inputs[1])
    it_ = nd(nt, "ShaderNodeTexImage", image=LOGO, extension="CLIP", interpolation="Cubic")
    lk(nt, cxy.outputs[0], it_.inputs["Vector"])
    front = M(nt, "LESS_THAN", sz.outputs[1], 0.0)
    la = M(nt, "MULTIPLY", M(nt, "MULTIPLY", it_.outputs["Alpha"], front), 0.96)
    base = mix_col(nt, la, base, col_mul(nt, it_.outputs["Color"], 0.8))
    rough = M(nt, "ADD", rough, M(nt, "MULTIPLY", la, -0.25))
    # frost where the cryogenic feedline brackets and flanges chill the foam (light, sheds)
    fv = value_node(nt, "Frost", 1.0)
    base, rough, nrm = frost_layer(nt, base, rough, nrm, 52.0, 62.5, M(nt, "MULTIPLY", fv.outputs[0], 0.55), 1.2)
    # soot: the booster plume stains the base of the core in flight (keyed 'Soot')
    sv = value_node(nt, "Soot", 0.0)
    st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (2.0, 2.0, 0.2)), st.inputs["Vector"])
    sb = nd(nt, "ShaderNodeMapRange", inp={"From Min": 16.0, "From Max": 2.0})
    lk(nt, sz.outputs[2], sb.inputs["Value"])
    sf = M(nt, "MULTIPLY", M(nt, "MULTIPLY", sb.outputs["Result"], sv.outputs[0]),
           M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.5, -0.1), clamp=True)
    base = mix_col(nt, sf, base, (0.035, 0.03, 0.025))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def srb_material(name, tex_img):
    """Booster: white paint × the NASA SRB texture's LUMINANCE (joint bands, marks — no hue),
    scan roughness, soot keyed at the aft skirt after ignition."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0, "Coat Weight": 0.05})
    tc = nd(nt, "ShaderNodeTexCoord")
    t = nd(nt, "ShaderNodeTexImage", image=tex_img, interpolation="Cubic")
    lk(nt, tc.outputs["UV"], t.inputs["Vector"])
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, t.outputs["Color"], bw.inputs[0])
    mr = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.05, "From Max": 0.8, "To Min": 0.06, "To Max": 1.0})
    lk(nt, bw.outputs[0], mr.inputs["Value"])
    base = col_mul(nt, (0.73, 0.72, 0.70), mr.outputs["Result"])
    base = col_mul(nt, base, wear(nt, 2.4, 0.84))
    rough, nrm = scan_rough_normal(nt, 1.8, 0.35, 0.6, 0.3)
    sz = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sz.inputs[0])
    sv = value_node(nt, "Soot", 0.0)
    sb = nd(nt, "ShaderNodeMapRange", inp={"From Min": 9.0, "From Max": 0.0})
    lk(nt, sz.outputs[2], sb.inputs["Value"])
    st = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (3.0, 3.0, 0.3)), st.inputs["Vector"])
    sf = M(nt, "MULTIPLY", M(nt, "MULTIPLY", sb.outputs["Result"], sv.outputs[0]),
           M(nt, "MULTIPLY_ADD", st.outputs["Fac"], 1.4, 0.0), clamp=True)
    base = mix_col(nt, sf, base, (0.04, 0.035, 0.03))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, rough, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def metal(name, color, rough, tile=1.0, nstr=0.25, heat=False, frost=None):
    """Bare/painted metal with scan roughness variation. heat=True: Object Info colour R = glow
    (nozzle interiors when firing). frost=(zlo, zhi): keyed 'Frost' band (cryo lines)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 1.0})
    r, nrm = scan_rough_normal(nt, tile, rough * 0.75, min(1.0, rough * 1.35), nstr)
    base = col_mul(nt, tuple(color), wear(nt, tile * 1.3, 0.78))
    if frost:
        fv = value_node(nt, "Frost", 1.0)
        base, r, nrm = frost_layer(nt, base, r, nrm, frost[0], frost[1], fv.outputs[0], 2.0)
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, r, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    if heat:
        oi = nd(nt, "ShaderNodeObjectInfo")
        sc_ = nd(nt, "ShaderNodeSeparateColor")
        lk(nt, oi.outputs["Color"], sc_.inputs[0])
        lk(nt, M(nt, "MULTIPLY", sc_.outputs[0], 14.0), b.inputs["Emission Strength"])
        b.inputs["Emission Color"].default_value = blackbody(1900) + (1,)
    finish(nt, b)
    return m


def dielectric(name, color, rough, tile=1.5, nstr=0.25):
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    r, nrm = scan_rough_normal(nt, tile, rough * 0.8, min(1.0, rough * 1.25), nstr)
    lk(nt, col_mul(nt, tuple(color), wear(nt, tile * 1.5, 0.8)), b.inputs["Base Color"])
    lk(nt, r, b.inputs["Roughness"])
    lk(nt, nrm, b.inputs["Normal"])
    finish(nt, b)
    return m


def tower_steel(name, color=(0.20, 0.215, 0.23)):
    """Mobile-launcher steel: grey paint + Poly Haven rusty_painted_metal roughness/normal and
    a desaturated grime multiplier (box-mapped, 2.5 m tiles)."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    d = tex_box(nt, ph("rusty_painted_metal", "diff"), 2.5)
    r = tex_box(nt, ph("rusty_painted_metal", "rough"), 2.5, True)
    n = tex_box(nt, ph("rusty_painted_metal", "nor_gl"), 2.5, True)
    bw = nd(nt, "ShaderNodeRGBToBW")
    lk(nt, d.outputs["Color"], bw.inputs[0])
    g = nd(nt, "ShaderNodeMapRange", inp={"From Min": 0.1, "From Max": 0.6, "To Min": 0.55, "To Max": 1.15})
    lk(nt, bw.outputs[0], g.inputs["Value"])
    lk(nt, col_mul(nt, color, g.outputs["Result"]), b.inputs["Base Color"])
    rr = nd(nt, "ShaderNodeMapRange", inp={"To Min": 0.45, "To Max": 0.85})
    lk(nt, r.outputs["Color"], rr.inputs["Value"])
    wet = value_node(nt, "Wet", 0.0)
    lk(nt, M(nt, "SUBTRACT", rr.outputs["Result"], M(nt, "MULTIPLY", wet.outputs[0], 0.35)), b.inputs["Roughness"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.6})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


def concrete(name, tile=4.0, color_gain=1.0):
    """Pad concrete (Poly Haven concrete_floor_02) + keyed 'Wet' (deluge): darker, glossier."""
    m, nt = new_mat(name)
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    d = tex_box(nt, ph("concrete_floor_02", "diff"), tile)
    r = tex_box(nt, ph("concrete_floor_02", "rough"), tile, True)
    n = tex_box(nt, ph("concrete_floor_02", "nor_gl"), tile, True)
    wet = value_node(nt, "Wet", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    pud = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.08, "Detail": 3.0})
    lk(nt, tc.outputs["Object"], pud.inputs["Vector"])
    wmask = M(nt, "MULTIPLY", wet.outputs[0], M(nt, "MULTIPLY_ADD", pud.outputs["Fac"], 2.2, -0.35), clamp=True)
    base = col_mul(nt, d.outputs["Color"], M(nt, "MULTIPLY", color_gain, M(nt, "SUBTRACT", 1.0,
                                                                              M(nt, "MULTIPLY", wmask, 0.45))))
    lk(nt, base, b.inputs["Base Color"])
    lk(nt, M(nt, "MULTIPLY", r.outputs["Color"], M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", wmask, 0.85))),
       b.inputs["Roughness"])
    nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 0.8})
    lk(nt, n.outputs["Color"], nm.inputs["Color"])
    lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


M_FOAM = foam_material("CoreFoam")
M_WHITE = paint_white("VehicleWhite", 0.74, soot=(0.0, 6.0))
M_BLANKET = paint_white("ThermalBlanket", 0.66, tile=0.8)
M_GRAY = dielectric("PaintedGrey", (0.30, 0.30, 0.31), 0.5)
M_METAL = metal("BareAluminium", (0.80, 0.80, 0.81), 0.33)
M_DARK = dielectric("DarkComposite", (0.045, 0.045, 0.048), 0.55)
M_NOZ_OUT = metal("NozzleTubes", (0.30, 0.27, 0.25), 0.42, tile=0.4, nstr=0.5, frost=(1.0, 4.2))
M_NOZ_IN = metal("NozzleInner", (0.20, 0.13, 0.09), 0.55, tile=0.5, heat=True)
M_GOLD = metal("GoldFoil", (1.0, 0.72, 0.30), 0.5, tile=0.3, nstr=0.9)
M_CM = metal("OrionBackshell", (0.62, 0.62, 0.63), 0.45, tile=0.6, nstr=0.6)
M_ML = tower_steel("MLSteel")
M_ML_RAIL = tower_steel("MLRail", (0.32, 0.30, 0.10))
M_CONCRETE = concrete("PadConcrete", 4.0)
M_CONCRETE_DK = concrete("TrenchConcrete", 3.0, 0.55)
M_LED = emissive("LED", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)
M_LAMP = emissive("Lamp", (1.0, 1.0, 1.0), 1.0, objcolor_gain=True)
SEM_MAT = {"SEM_white": M_WHITE, "SEM_foam": M_FOAM, "SEM_metal": M_METAL, "SEM_gray": M_GRAY,
           "SEM_dark": M_DARK, "SEM_nozzle_out": M_NOZ_OUT, "SEM_nozzle_in": M_NOZ_IN, "SEM_gold": M_GOLD,
           "SEM_cm": M_CM, "SEM_ml": M_ML, "SEM_ml_rail": M_ML_RAIL}


# --- exhaust shaders --------------------------------------------------------------------
def color_ramp(nt, fac, stops):
    cr = nd(nt, "ShaderNodeValToRGB")
    els = cr.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], tuple(stops[0][1]) + (1,)
    els[1].position, els[1].color = stops[-1][0], tuple(stops[-1][1]) + (1,)
    for pos, c in stops[1:-1]:
        e = els.new(pos)
        e.color = tuple(c) + (1,)
    lk(nt, fac, cr.inputs[0])
    return cr.outputs[0]


def plume_shell(name, stops, nscale=0.55, along_pow=1.5, diamonds=0.0, n_diam=6.0, diam_reach=0.35,
                edge_pow=2.4, base_alpha=1.0):
    """Additive exhaust shell (Transparent + Emission). Object colour R = intensity, G = flicker
    phase (4D noise W), B = flow scroll. UV v: 0 at the nozzle → 1 at the tail.
    diamonds > 0: bright Mach disks along the first diam_reach of the plume (sea level)."""
    m, nt = new_mat(name)
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = True
    tc = nd(nt, "ShaderNodeTexCoord")
    sep = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sep.inputs[0])
    v = sep.outputs[1]
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, flick, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, -40.0), off.inputs[2])
    p = V(nt, "MULTIPLY", V(nt, "ADD", tc.outputs["Object"], off.outputs[0]), (1.0, 1.0, 0.28))
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": nscale, "Detail": 7.0, "Roughness": 0.62, "Distortion": 0.4})
    lk(nt, p, nz.inputs["Vector"])
    lk(nt, flick, nz.inputs["W"])
    turb = nz.outputs["Fac"]
    col = color_ramp(nt, M(nt, "ADD", v, M(nt, "MULTIPLY", M(nt, "SUBTRACT", turb, 0.5), 0.25), clamp=True), stops)
    lw = nd(nt, "ShaderNodeLayerWeight", inp={"Blend": 0.5})
    face = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), edge_pow)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), along_pow)
    a = M(nt, "MULTIPLY", M(nt, "MULTIPLY", face, along), M(nt, "MULTIPLY_ADD", turb, 1.6, -0.3, clamp=True))
    a = M(nt, "MULTIPLY", a, base_alpha)
    if diamonds > 0:
        dkey = value_node(nt, "Diamonds", 1.0)           # keyed: sea level only (they fade with altitude)
        # Mach disks: narrow bright bands, strongest at the first one, only near the axis (facing)
        ph_ = M(nt, "FRACT", M(nt, "MULTIPLY", v, n_diam / diam_reach))
        disk = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", ph_, 0.5)), 2.0)), 10.0)
        reach = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", v, diam_reach), clamp=True), 1.2)
        core = M(nt, "POWER", lw.outputs["Facing"], 1.5)
        a = M(nt, "ADD", a, M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", disk, reach), core),
                              M(nt, "MULTIPLY", dkey.outputs[0], diamonds)))
    em = nd(nt, "ShaderNodeEmission")
    lk(nt, col, em.inputs["Color"])
    lk(nt, M(nt, "MULTIPLY", a, gain), em.inputs["Strength"])
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    add = nd(nt, "ShaderNodeAddShader")
    lk(nt, tr.outputs[0], add.inputs[0])
    lk(nt, em.outputs[0], add.inputs[1])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, add.outputs[0], o.inputs["Surface"])
    return m


def plume_volume(name, stops, smoke_col=(0.35, 0.3, 0.27), smoke=0.05, r0=0.16, rpow=0.5, sq=(2.2, 2.2, 7.0),
                 along_pow=1.3):
    """Emissive turbulent VOLUME sheath in a unit cylinder (r<=1, z: 0 nozzle → -1 tail), scaled
    per object. Object colour R = intensity, G = flicker W, B = flow scroll. smoke = scattering
    density toward the tail (soot / Al2O3 smoke lit by the plume lights)."""
    m, nt = new_mat(name)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, flick, scroll = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "MULTIPLY", scroll, 1.2), off.inputs[2])
    q = V(nt, "ADD", V(nt, "MULTIPLY", ob, sq), off.outputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6, "Distortion": 0.3})
    lk(nt, q, nz.inputs["Vector"])
    lk(nt, flick, nz.inputs["W"])
    turb = nz.outputs["Fac"]
    R_ = M(nt, "MULTIPLY", M(nt, "ADD", r0, M(nt, "MULTIPLY", M(nt, "POWER", v, rpow), 1.0 - r0)), 0.62)
    rn = M(nt, "ADD", M(nt, "DIVIDE", r, R_), M(nt, "MULTIPLY", M(nt, "SUBTRACT", turb, 0.5), 0.9))
    mask = nd(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
              inp={"From Min": 1.0, "From Max": 0.25, "To Min": 0.0, "To Max": 1.0})
    lk(nt, rn, mask.inputs["Value"])
    mk = mask.outputs["Result"]
    core = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, rn, clamp=True), 3.0)
    along = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), along_pow)
    attach = M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.03), 1.0, clamp=True)
    e = M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, along), attach)
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", core, 2.5, 0.35))
    e = M(nt, "MULTIPLY", e, M(nt, "MULTIPLY_ADD", turb, 1.2, 0.4))
    e = M(nt, "MULTIPLY", e, gain)
    col = color_ramp(nt, M(nt, "ADD", v, M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, core), 0.25)), stops)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(smoke_col) + (1,), "Anisotropy": 0.3})
    lk(nt, col, pv.inputs["Emission Color"])
    lk(nt, e, pv.inputs["Emission Strength"])
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", mk, M(nt, "POWER", v, 1.2)), smoke), pv.inputs["Density"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


# SRB (aluminised APCP): blinding white-yellow core, yellow → orange edges, Al2O3 smoke
SRB_STOPS = [(0.0, (1.0, 0.97, 0.88)), (0.12, (1.0, 0.88, 0.58)), (0.38, (1.0, 0.66, 0.26)),
             (0.75, (0.92, 0.38, 0.10)), (1.0, (0.45, 0.14, 0.04))]
# RS-25 (LH2/LOX): nearly invisible pale blue-violet sheath, the Mach disks carry the light
RS25_STOPS = [(0.0, (0.75, 0.80, 1.0)), (0.2, (0.62, 0.55, 0.95)), (0.6, (0.85, 0.45, 0.45)), (1.0, (0.4, 0.2, 0.3))]
# RL10 (vacuum, LH2): faint, wide, translucent
VAC_STOPS = [(0.0, (1.0, 0.85, 0.7)), (0.1, (0.85, 0.65, 0.9)), (0.5, (0.45, 0.4, 0.8)), (1.0, (0.12, 0.12, 0.35))]
M_SRB_SHELL = plume_shell("SRBPlume", SRB_STOPS, nscale=0.45, along_pow=2.6, edge_pow=1.3)
M_SRB_VOL = plume_volume("SRBPlumeVol", SRB_STOPS, smoke_col=(0.9, 0.88, 0.85), smoke=0.10, along_pow=3.0)
M_RS25_SHELL = plume_shell("RS25Plume", RS25_STOPS, nscale=0.9, along_pow=1.8, diamonds=6.0, n_diam=5.0,
                           diam_reach=0.45, edge_pow=3.0, base_alpha=0.18)
M_VAC_SHELL = plume_shell("VacPlume", VAC_STOPS, nscale=0.15, along_pow=0.9, edge_pow=2.0, base_alpha=0.5)
M_VAC_VOL = plume_volume("VacPlumeVol", VAC_STOPS, smoke=0.0, r0=0.08, rpow=0.32, sq=(1.4, 1.4, 3.0))


def jellyfish_material(name):
    """Twilight plume ('space jellyfish'): a huge translucent bell of exhaust high enough to be
    SUNLIT while the ground is dark. Faked as low-density scattering + an emission rim (blue-
    white outer shell, warm near the axis). Keyed 'Gain' and 'Evolve'."""
    m, nt = new_mat(name)
    gain = value_node(nt, "Gain", 0.0)
    evo = value_node(nt, "Evolve", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)                 # 0 at the vehicle → 1 far
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    bell = M(nt, "ADD", 0.05, M(nt, "MULTIPLY", M(nt, "POWER", v, 0.45), 0.95))   # bell radius
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 2.5, "Detail": 5.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.5)), nz.inputs["Vector"])
    lk(nt, evo.outputs[0], nz.inputs["W"])
    fil = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 9.0, "Detail": 8.0, "Roughness": 0.7,
                                                                   "Distortion": 0.8})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.08)), fil.inputs["Vector"])          # filaments along the flow
    lk(nt, evo.outputs[0], fil.inputs["W"])
    rn = M(nt, "ADD", M(nt, "DIVIDE", r, bell), M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.5), 0.6))
    shell = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", rn, 0.85)), 3.0), clamp=True), 2.0)
    shell = M(nt, "MULTIPLY", shell, M(nt, "MULTIPLY_ADD", fil.outputs["Fac"], 3.0, -1.0, clamp=True))
    inner = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, rn, clamp=True), 0.12)
    fade = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, v, clamp=True), M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.05), 1.0, clamp=True))
    d = M(nt, "MULTIPLY", M(nt, "ADD", shell, inner), fade)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.85, 0.9, 1.0, 1), "Anisotropy": 0.6})
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.0004)), pv.inputs["Density"])
    lk(nt, mix_col(nt, M(nt, "SUBTRACT", 1.0, rn, clamp=True), (0.55, 0.7, 1.0), (1.0, 0.75, 0.55)), pv.inputs["Emission Color"])
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.004)), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_JELLY = jellyfish_material("Jellyfish")

# =====================================================================================
# 5. THE VEHICLE (metres; rig origin = SLS station 0 = core-engine exit plane ~z 0.8,
#    +Z = flight axis). NASA PD hardware from the library + a plain core tank lathe (the NASA
#    model ships the tank only as a 3ds Max file), Shuttle SRB stretched to 5 segments.
# =====================================================================================
with bpy.data.libraries.load(ARGS.lib, link=False) as (_src, _dst):
    _dst.objects = list(_src.objects)
LIB = {o.name: o for o in _dst.objects if o is not None}
print("[rocket2] library:", sorted(LIB))


def lib_obj(name, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), copy=False, new_name=None):
    src = LIB[name]
    ob = src.copy() if copy else src
    if copy:
        ob.name = new_name or (name + "_copy")
    for k, s in enumerate(ob.material_slots):
        mat = s.material
        if mat is not None and mat.name in SEM_MAT:
            ob.material_slots[k].material = SEM_MAT[mat.name]
        elif mat is not None and mat.name.split(".")[0] in SEM_MAT:
            ob.material_slots[k].material = SEM_MAT[mat.name.split(".")[0]]
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


CORE_TOP = 64.15          # LVSA base
LVSA_TOP = 72.2
ICPS_TOP = 76.8           # MSA top
SAJ_Z0, SAJ_Z1 = 76.7, 81.1
OGIVE_Z0 = 82.1
LAS_TOP = 95.5
SRB_X = 6.2               # booster axes at x = ±6.2 (from the NASA attach hardware)
R = CORE_R

RIG = empty("RocketRig")
# v4: the tumble lives on its own pivot between the rig and the stages (keyed from flight_model.attitude), so
# chase cameras (which follow RIG) don't inherit it; onboard cameras ride RIG @ WOB.
WOB = empty("Wobble", parent=RIG)
S1 = empty("S1", parent=WOB)                 # core stage (+ LVSA, which stays with it)
S2 = empty("S2", parent=WOB)                 # ICPS + Orion
# three boosters at 120 deg (slot empties rotate about the core axis; each booster sits on its slot's +X):
# BoosterL 180 deg (unchanged from v3), BoosterR +60, Booster3 -60 (x+, y-: the failure plays to the south
# cameras). Booster 3's slot hangs on RIG, not WOB: after its early sep it flies in the rig (air) frame.
SLOT_AZ = FM.V.srb_az
SLOTS = []
for _k, _az in enumerate(SLOT_AZ):
    _sl = empty(f"BoosterSlot{_k}", parent=WOB if _k < 2 else RIG)
    _sl.rotation_euler = (0.0, 0.0, _az)
    SLOTS.append(_sl)
BL = empty("BoosterL", loc=(SRB_X, 0, 0), parent=SLOTS[0])
BRt = empty("BoosterR", loc=(SRB_X, 0, 0), parent=SLOTS[1])
B3 = empty("Booster3", loc=(SRB_X, 0, 0), parent=SLOTS[2])
BOOSTERS = [BL, BRt, B3]
LASG = empty("LASGroup", parent=S2)          # launch abort tower + ogive: jettisoned on drop 4
PANELS = [empty(f"SAJPanel{k}", parent=S2) for k in range(3)]

core = lathe("CoreTank", [(3.55, 2.35), (3.9, 3.2), (R, 5.4), (R, CORE_TOP)], 96, M_FOAM, S1)
heatshield = MB()
heatshield.poly([(3.55 * math.cos(a), 3.55 * math.sin(a), 2.35) for a in np.linspace(0, 2 * math.pi, 49)[:-1]],
                [tuple(range(48))[::-1]])
heatshield.obj("CoreHeatShield", M_DARK, S1)
ENG = lib_obj("SLS_Engines", S1)
lib_obj("SLS_CoreHW", S1)
lib_obj("SLS_LVSA", S1)
lib_obj("SLS_ICPS", S2)
lib_obj("SLS_MSA", S2)
# ESM body + folded solar arrays (hidden under the SAJ panels until they jettison)
if not RAVE:                                 # v3: the payload under the fairing is the speaker (10b)
    lathe("ESMBody", [(2.0, ICPS_TOP), (2.0, SAJ_Z1 - 0.2), (2.5, SAJ_Z1 + 0.6), (2.5, OGIVE_Z0)], 48, M_GOLD, S2)
    cm = lib_obj("Orion_CM", S2, loc=(0, 0, OGIVE_Z0 - 0.2))
for k, pn in enumerate(PANELS):
    lib_obj(f"SLS_ESM{k}", pn)
lib_obj("SLS_Ogive", LASG)
lib_obj("SLS_LAS", LASG)
srb_tex = load_img(os.path.join(os.path.dirname(ARGS.lib), "SRB_ALL.png"))
M_SRB = srb_material("BoosterPaint", srb_tex)
SEM_MAT["SEM_srb"] = M_SRB
lib_obj("SRB_Body", BL)
lib_obj("SRB_Body", BRt, copy=True, new_name="SRB_BodyR")
lib_obj("SRB_Body", B3, copy=True, new_name="SRB_Body3")
# the attach hardware is modelled in vehicle coords at x = -6.2 (L) / +6.2 (R): the right-hand one rides each
# rotated slot (offset -SRB_X along the slot's x); the left one keeps its own place (counter-rotated)
lib_obj("SLS_SRBatt_L", BL, loc=(-SRB_X, 0, 0), rot=(0, 0, -SLOT_AZ[0]))
lib_obj("SLS_SRBatt_R", BRt, loc=(-SRB_X, 0, 0))
lib_obj("SLS_SRBatt_R", B3, loc=(-SRB_X, 0, 0), copy=True, new_name="SLS_SRBatt_3")
# RL10 vacuum nozzle under the ICPS (hidden in the LVSA until staging)
s2noz = lathe("S2Nozzle", [(0.35, 71.6), (0.5, 70.9), (0.85, 69.6), (1.05, 68.4)], 40, M_NOZ_IN, S2)

# --- v3 RAVE: neon LED strips + a bright LED nose (Zeke 10-11). Parented to the stage they ride, so
#     the existing staging visibility drops them with the core / boosters / abort tower. Colour and
#     level are keyed per object (Object Info) in section 10b.
def led_pixel_material(name, strength=14.0, dots_per_m=4.0, coord="ring", hue_per_m=0.012):
    """v4 LED-pixel shader (research #1 §2): discrete diodes with dark PCB between them, a rainbow chase that
    scrolls with the BEAT count, a mix to RED by the per-object alarm, a white strobe, and an alternate-halves
    alarm blink. Coordinates without UVs (object space): coord 'ring' = arc length around the object's Z axis
    + z (rings and vertical strips), 'diag' = x + y + z (boxes on the tower / speaker).
    Global keyed Value nodes (one F-curve each for every object using the material):
      Beats (MONOTONIC beat count, wrapped with fract here so motion-blur sub-frames don't smear the wrap),
      Level (music level), Strobe (0..1 white), Hue (palette offset).
    Per object (Object Info colour, keyed): R = gain, G = alarm 0..1 (red), B = phase."""
    m, nt = new_mat(name)
    beats = value_node(nt, "Beats", 0.0)
    level = value_node(nt, "Level", 1.0)
    strobe = value_node(nt, "Strobe", 0.0)
    hueo = value_node(nt, "Hue", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["Object"], sp.inputs[0])
    if coord == "ring":
        ang = M(nt, "ADD", M(nt, "ARCTAN2", sp.outputs[1], sp.outputs[0]), math.pi)
        r = M(nt, "SQRT", M(nt, "ADD", M(nt, "MULTIPLY", sp.outputs[0], sp.outputs[0]),
                            M(nt, "MULTIPLY", sp.outputs[1], sp.outputs[1])))
        posm = M(nt, "ADD", M(nt, "MULTIPLY", ang, r), sp.outputs[2])
    else:
        posm = M(nt, "ADD", M(nt, "ADD", sp.outputs[0], sp.outputs[1]), sp.outputs[2])
    oi = nd(nt, "ShaderNodeObjectInfo")
    oc = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, oi.outputs["Color"], oc.inputs[0])
    gain, alarm, phase = oc.outputs[0], oc.outputs[1], oc.outputs[2]
    hue = M(nt, "FRACT", M(nt, "ADD", M(nt, "ADD", M(nt, "MULTIPLY", posm, hue_per_m),
                                           M(nt, "MULTIPLY", beats.outputs[0], 0.25)),
                           M(nt, "ADD", phase, hueo.outputs[0])))
    hsv = nd(nt, "ShaderNodeCombineColor", mode="HSV")
    lk(nt, hue, hsv.inputs[0])
    hsv.inputs[1].default_value = 1.0
    hsv.inputs[2].default_value = 1.0
    col = mix_col(nt, alarm, hsv.outputs[0], (1.0, 0.0, 0.035))
    col = mix_col(nt, strobe.outputs[0], col, (1.0, 1.0, 1.0))
    # diodes: d = |fract(pos * N) - 0.5| -> dots; far away they alias, so fade to the mean (0.45) by distance
    d = M(nt, "ABSOLUTE", M(nt, "SUBTRACT", M(nt, "FRACT", M(nt, "MULTIPLY", posm, dots_per_m)), 0.5))
    dot = nd(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
             inp={"From Min": 0.34, "From Max": 0.16, "To Min": 0.0, "To Max": 1.0})
    lk(nt, d, dot.inputs["Value"])
    cd_ = nd(nt, "ShaderNodeCameraData")
    far = nd(nt, "ShaderNodeMapRange", inp={"From Min": 150.0, "From Max": 600.0, "To Min": 0.0, "To Max": 1.0})
    lk(nt, cd_.outputs["View Distance"], far.inputs["Value"])
    mask = M(nt, "ADD", M(nt, "MULTIPLY", dot.outputs["Result"], M(nt, "SUBTRACT", 1.0, far.outputs["Result"])),
             M(nt, "MULTIPLY", far.outputs["Result"], 0.45))
    # alarm pattern: alternate halves blink on the half-beat, so red reads as a WARNING, not just a colour
    blink = M(nt, "LESS_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", beats.outputs[0], 0.5)), 0.5)
    half = M(nt, "LESS_THAN", M(nt, "FRACT", M(nt, "MULTIPLY", posm, 0.04)), 0.5)
    pat = M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "MULTIPLY", alarm, 0.65),
                                   M(nt, "ABSOLUTE", M(nt, "SUBTRACT", blink, half))))
    st = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", mask, gain), level.outputs[0]),
           M(nt, "MULTIPLY", pat, strength))
    # the red alarm runs ~40 % dimmer: AgX rolls a hot saturated red toward orange/white (research #1 §2)
    st = M(nt, "MULTIPLY", st, M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", alarm, 0.4)))
    em = nd(nt, "ShaderNodeEmission")
    lk(nt, col, em.inputs["Color"])
    lk(nt, st, em.inputs["Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, em.outputs[0], o.inputs["Surface"])
    return m


LED_OBS = []                                  # (object, kind, phase 0..1 up the stage)
if RAVE:
    M_LED = emissive("LEDNeon", (1.0, 1.0, 1.0), 7.0, objcolor_gain=True)
    M_LEDHOT = emissive("LEDHot", (1.0, 1.0, 1.0), 24.0, objcolor_gain=True)
    M_LEDPIX = led_pixel_material("LEDPixelRocket", 14.0, 4.0, "ring")

    def led_ring(name, r, z, parent, tube=0.11, mat=None):
        prof = [(r + tube * math.cos(a), z + tube * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)]
        return no_shadow(lathe(name, prof, 96, mat or M_LEDPIX, parent))

    for k, z in enumerate((9.0, 19.0, 29.0, 39.0, 49.0, 59.0)):
        LED_OBS.append((led_ring(f"LEDCore{k}", R + 0.05, z, S1), "core", z / 64.0))
    _strip = MB()
    for a in np.radians((45.0, 135.0, 225.0, 315.0)):
        _strip.box(((R + 0.06) * math.cos(a), (R + 0.06) * math.sin(a), 34.0), (0.16, 0.16, 52.0),
                   Matrix.Rotation(float(a), 3, "Z"))
    LED_OBS.append((no_shadow(_strip.obj("LEDCoreStrips", M_LEDPIX, S1)), "strip", 0.0))
    _bb = [Vector(c) for c in LIB["SRB_Body"].bound_box]
    _sr = (max(v.x for v in _bb) - min(v.x for v in _bb)) / 2
    _z0, _z1 = min(v.z for v in _bb), max(v.z for v in _bb)
    for side, bem in (("L", BL), ("R", BRt), ("3", B3)):
        for k, q in enumerate((0.20, 0.42, 0.64)):
            LED_OBS.append((led_ring(f"LEDSrb{side}{k}", _sr + 0.05, _z0 + (_z1 - _z0) * q, bem, 0.09), "srb", q))
    for k, z in enumerate((72.9, 75.7)):
        LED_OBS.append((led_ring(f"LEDS2_{k}", 2.62, z, S2, 0.08), "s2", 0.5 * k))
    NOSE = no_shadow(uv_sphere("LEDNose", 0.55, 24, 12, M_LEDHOT, LASG, loc=(0, 0, LAS_TOP + 0.25)))
    LED_OBS.append((NOSE, "nose", 1.0))
    NOSE_L = bpy.data.lights.new("LEDNoseLight", "POINT")
    NOSE_L.shadow_soft_size = 0.5
    NOSE_L.use_shadow = False
    NOSE_LO = link(bpy.data.objects.new("LEDNoseLight", NOSE_L), LASG)
    NOSE_LO.location = (0, 0, LAS_TOP + 1.4)
    print("[rocket3] SRB ring radius", round(_sr, 2), "z", round(_z0, 1), round(_z1, 1))


# --- v4: the audio-visualizer panel. ONE baked image is the source of truth for every spectrum display
#     (rocket panel, speaker side matrices): 32 log-spaced bands from the Demucs-era analysis (spec32,
#     per-band normalised, attack 0 / release 0.18 s) + a peak-hold cap (hold 6 frames, fall 1.5 rows/frame).
SPEC_ROWS = 16


def spectrum_image():
    lvl = SPEC32 ** 1.25                                     # gamma: the bars mustn't sit high
    peak = np.zeros_like(lvl)
    hold = np.zeros(32)
    cur = np.zeros(32)
    fall = 1.5 / SPEC_ROWS
    for i in range(N):
        up = lvl[i] >= cur
        cur = np.where(up, lvl[i], np.where(hold > 0, cur, np.maximum(cur - fall, lvl[i])))
        hold = np.where(up, 6, np.maximum(hold - 1, 0))
        peak[i] = cur
    img = bpy.data.images.new("Spectrum32", 32, N, alpha=False, float_buffer=False)
    px = np.zeros((N, 32, 4), np.float32)
    px[..., 0] = lvl
    px[..., 1] = peak
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.colorspace_settings.name = "Non-Color"
    img.pack()
    return img


SPEC_IMG = spectrum_image()


def spectrum_material(name, strength=6.0, cols=32, rows=SPEC_ROWS):
    """LED matrix (cols x rows cells) on a UV-mapped surface (u across, v up): a cell lights if its row is under
    the band level; green -> yellow -> red by row, a white peak cap, all red while the keyed 'Alarm' is on.
    The keyed 'Frame' Value (= timeline index, CONSTANT keys) picks the image row, so motion-blur sub-frames
    stay on one row."""
    m, nt = new_mat(name)
    frame = value_node(nt, "Frame", 0.0)
    alarm = value_node(nt, "Alarm", 0.0)
    gainv = value_node(nt, "Gain", 1.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, tc.outputs["UV"], sp.inputs[0])
    u, v = sp.outputs[0], sp.outputs[1]
    colf = M(nt, "FLOOR", M(nt, "MULTIPLY", u, cols))
    band = M(nt, "FLOOR", M(nt, "MULTIPLY", M(nt, "DIVIDE", M(nt, "ADD", colf, 0.5), cols), 32.0))
    uvc = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, M(nt, "DIVIDE", M(nt, "ADD", band, 0.5), 32.0), uvc.inputs[0])
    lk(nt, M(nt, "DIVIDE", M(nt, "ADD", M(nt, "FLOOR", frame.outputs[0]), 0.5), float(N)), uvc.inputs[1])
    ti = nd(nt, "ShaderNodeTexImage", image=SPEC_IMG, interpolation="Closest", extension="EXTEND")
    lk(nt, uvc.outputs[0], ti.inputs["Vector"])
    sc_ = nd(nt, "ShaderNodeSeparateColor")
    lk(nt, ti.outputs["Color"], sc_.inputs[0])
    level, peak = sc_.outputs[0], sc_.outputs[1]
    rowf = M(nt, "FLOOR", M(nt, "MULTIPLY", v, rows))
    rowc = M(nt, "DIVIDE", M(nt, "ADD", rowf, 0.5), rows)
    lit = M(nt, "LESS_THAN", rowc, level)
    cap = M(nt, "LESS_THAN", M(nt, "ABSOLUTE", M(nt, "SUBTRACT", rowc, peak)), 0.5 / rows)
    cu = M(nt, "ABSOLUTE", M(nt, "SUBTRACT", M(nt, "FRACT", M(nt, "MULTIPLY", u, cols)), 0.5))
    cv = M(nt, "ABSOLUTE", M(nt, "SUBTRACT", M(nt, "FRACT", M(nt, "MULTIPLY", v, rows)), 0.5))
    cell = M(nt, "MULTIPLY", M(nt, "LESS_THAN", cu, 0.36), M(nt, "LESS_THAN", cv, 0.34))
    grad = color_ramp(nt, rowc, [(0.0, (0.05, 1.0, 0.12)), (0.55, (0.9, 0.95, 0.05)), (0.8, (1.0, 0.45, 0.02)),
                                 (1.0, (1.0, 0.03, 0.05))])
    c = mix_col(nt, alarm.outputs[0], grad, (1.0, 0.02, 0.04))
    c = mix_col(nt, M(nt, "MULTIPLY", cap, M(nt, "SUBTRACT", 1.0, lit)), c, (1.0, 1.0, 1.0))
    on = M(nt, "MAXIMUM", lit, cap)
    st = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "ADD", M(nt, "MULTIPLY", on, 1.0), 0.015), cell),
           M(nt, "MULTIPLY", gainv.outputs[0], strength))
    em = nd(nt, "ShaderNodeEmission")
    lk(nt, c, em.inputs["Color"])
    lk(nt, st, em.inputs["Strength"])
    bz = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.01, 0.01, 0.012, 1), "Roughness": 0.25})
    ad = nd(nt, "ShaderNodeAddShader")
    lk(nt, bz.outputs[0], ad.inputs[0])
    lk(nt, em.outputs[0], ad.inputs[1])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, ad.outputs[0], o.inputs["Surface"])
    return m


M_SPEC_ROCKET = spectrum_material("SpectrumRocket", 7.0)
# a curved panel on the core facing -Y (the camera side), between two LED strips, above the emblem
SPEC_PANEL = no_shadow(lathe("SpectrumPanel", [(R + 0.10, 30.4), (R + 0.10, 38.2)], 48, M_SPEC_ROCKET, S1,
                             a0=math.radians(-90 - 33), a1=math.radians(-90 + 33), smooth=True))
# lathe UV: u = angle/2pi (only the span), v = profile t → remap u to 0..1 across the panel
_me = SPEC_PANEL.data
_uv = np.empty(len(_me.loops) * 2, np.float32)
_me.uv_layers[0].data.foreach_get("uv", _uv)
_uv = _uv.reshape(-1, 2)
_uv[:, 0] = (_uv[:, 0] - _uv[:, 0].min()) / max(_uv[:, 0].max() - _uv[:, 0].min(), 1e-6)
# u grows with the angle -123 -> -57 deg, i.e. left -> right as seen from -Y: low bands on the left
_me.uv_layers[0].data.foreach_set("uv", _uv.ravel())
_bez = MB()
for _z in (30.15, 38.45):
    for _a in np.linspace(math.radians(-123), math.radians(-57), 23)[:-1]:
        _bez.box(((R + 0.12) * math.cos(_a + 0.05), (R + 0.12) * math.sin(_a + 0.05), _z), (0.42, 0.14, 0.3),
                 Matrix.Rotation(float(_a + 0.05 + math.pi / 2), 3, "Z"))
_bez.obj("SpectrumBezel", principled("Bezel", (0.02, 0.02, 0.022), 0.3, 0.6), S1)


def engine_centres():
    me = ENG.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    low = co[co[:, 2] < 1.3]
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            q = low[(np.sign(low[:, 0]) == sx) & (np.sign(low[:, 1]) == sy)]
            out.append((float((q[:, 0].max() + q[:, 0].min()) / 2), float((q[:, 1].max() + q[:, 1].min()) / 2),
                        float(q[:, 2].min()), float((q[:, 0].max() - q[:, 0].min()) / 2)))
    return out


ENG_POS = engine_centres()
print("[rocket2] RS-25 exits", [tuple(round(v, 2) for v in e) for e in ENG_POS])
SRB_EXIT_Z = -0.6
PLUMES = []          # (object, kind, extra)


def plume_pair(tag, parent, kind, loc, base_r, length, shell_mat, vol_mat=None, vol_len=None):
    inner = lathe(f"Plume_{tag}_in", [(base_r * 0.55, 0.0), (base_r * 0.78, -0.08 * length), (base_r * 0.9, -0.3 * length),
                                      (base_r * 0.75, -0.65 * length), (base_r * 0.3, -length)],
                  48, shell_mat, parent, loc=loc)
    no_shadow(inner)
    PLUMES.append((inner, kind, "shell"))
    outer = None
    if vol_mat is not None:
        outer = lathe(f"Plume_{tag}_vol", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, vol_mat, parent,
                      loc=(loc[0], loc[1], loc[2] + 0.3))
        no_shadow(outer)
        PLUMES.append((outer, kind, "vol"))
    return inner, outer


for k, (ex, ey, ez, er) in enumerate(ENG_POS):
    plume_pair(f"rs{k}", S1, f"core{k}", (ex, ey, ez), er * 0.95, 22.0, M_RS25_SHELL)
# v4 TVC: the four RS-25 bells + their plumes swivel about the gimbal plane (keyed from the flight model's
# commanded gimbal, clamped to +/-8 deg); the SRB plumes bend with their nozzles the same way (bake section).
GIMB_Z = 4.0
GIMB = empty("CoreGimbal", loc=(0.0, 0.0, GIMB_Z), parent=S1)
for _o in [ENG] + [o for o, kind, _ in PLUMES if kind.startswith("core")]:
    _o.parent = GIMB
    _o.location = (_o.location.x, _o.location.y, _o.location.z - GIMB_Z)
plume_pair("bl", BL, "boost", (0, 0, SRB_EXIT_Z), 2.0, 34.0, M_SRB_SHELL, M_SRB_VOL)
plume_pair("br", BRt, "boost", (0, 0, SRB_EXIT_Z), 2.0, 34.0, M_SRB_SHELL, M_SRB_VOL)
plume_pair("b3", B3, "boost3", (0, 0, SRB_EXIT_Z), 2.0, 34.0, M_SRB_SHELL, M_SRB_VOL)
vac = lathe("Plume_s2_vol", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, M_VAC_VOL, S2, loc=(0, 0, 68.2))
no_shadow(vac)
PLUMES.append((vac, "s2", "vol"))
vac_in = lathe("Plume_s2_in", [(1.0, 0.0), (1.5, -3.0), (2.3, -12.0), (1.8, -26.0), (0.4, -34.0)],
               48, M_VAC_SHELL, S2, loc=(0, 0, 68.3))
no_shadow(vac_in)
PLUMES.append((vac_in, "s2", "shell"))
LAS_PLUMES = []
for k in range(4):
    a_ = math.pi / 4 + k * math.pi / 2
    lp = lathe(f"LASPlume{k}", [(0.18, 0.0), (0.35, -0.6), (0.6, -3.0), (0.4, -6.5), (0.1, -9.0)], 24,
               M_SRB_SHELL, LASG, loc=(0.9 * math.cos(a_), 0.9 * math.sin(a_), OGIVE_Z0 + 9.2))
    lp.rotation_euler = (math.radians(32) * -math.sin(a_), math.radians(32) * math.cos(a_), 0.0)
    no_shadow(lp)
    LAS_PLUMES.append(lp)
jelly = lathe("Jellyfish", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 40, M_JELLY, RIG, loc=(0, 0, -10.0))
no_shadow(jelly)

# plume lights: a chain of blackbody points down each plume lights the steam, the pad, the
# tower, the vehicle and the clouds (research fix #2: the plume IS the key light)
PLIGHTS = []
for tag, par, zs in (("bl", BL, (-6.0, -26.0, -60.0)), ("br", BRt, (-6.0, -26.0, -60.0)), ("b3", B3, (-6.0, -30.0)),
                     ("core", S1, (-4.0, -16.0)), ("s2", S2, (60.0,))):
    for j, z in enumerate(zs):
        ld = bpy.data.lights.new(f"PlumeLight_{tag}{j}", "POINT")
        ld.color = blackbody(2600 if tag in ("bl", "br", "b3") else 2200)
        ld.shadow_soft_size = 3.0 + 3.0 * j
        ld.volume_factor = 1.0
        ld.use_shadow = tag in ("bl", "br") or j == 0
        lo = bpy.data.objects.new(f"PlumeLight_{tag}{j}", ld)
        lo.location = (0, 0, z)
        link(lo, par)
        PLIGHTS.append((lo, tag, j))
BOUNCE = bpy.data.lights.new("PlumeBounce", "AREA")
BOUNCE.shape = "DISK"
BOUNCE.size = 40.0
BOUNCE.color = blackbody(2400)
BOUNCE.use_shadow = False
BOUNCE_OB = link(bpy.data.objects.new("PlumeBounce", BOUNCE), S1)
BOUNCE_OB.location = (0.0, -30.0, -40.0)
BOUNCE_OB.rotation_euler = (math.radians(-135), 0, 0)     # faces up and toward the vehicle (+Y,+Z)


# --- GN helpers ------------------------------------------------------------------------
def gn_tree(name, inputs):
    """New GN tree with Geometry in/out + the given [(name, socket_type, default)] inputs."""
    t = bpy.data.node_groups.new(name, "GeometryNodeTree")
    t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    socks = {}
    for nm, ty, dv in inputs:
        s = t.interface.new_socket(name=nm, in_out="INPUT", socket_type=ty)
        if dv is not None:
            s.default_value = dv
        socks[nm] = s
    t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return t, socks, t.nodes.new("NodeGroupInput"), t.nodes.new("NodeGroupOutput")


def g_rand(t, gi, lo, hi, salt, vec=False, seed_sock="Seed", seed_scale=1):
    """5.2 Random Value: ONE Min/Max pair retyped by data_type; seed = Seed input + salt."""
    r = t.nodes.new("FunctionNodeRandomValue")
    r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
    want = "VECTOR" if vec else "VALUE"
    ins = [x for x in r.inputs if x.enabled and x.type == want]
    ins[0].default_value = lo
    ins[1].default_value = hi
    if seed_sock is not None:
        ad = t.nodes.new("FunctionNodeIntegerMath")
        ad.operation = "ADD"
        t.links.new(gi.outputs[seed_sock], ad.inputs[0])
        ad.inputs[1].default_value = salt
        t.links.new(ad.outputs[0], r.inputs["Seed"])
    else:
        r.inputs["Seed"].default_value = salt
    return [x for x in r.outputs if x.enabled and x.type == want][0]


def g_math(t, op, a, b=None, c=None, vec=False):
    n = t.nodes.new("ShaderNodeVectorMath" if vec else "ShaderNodeMath")
    n.operation = op
    for k, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float, tuple)):
            n.inputs[k].default_value = x
        else:
            t.links.new(x, n.inputs[k])
    if vec:
        return n.outputs["Value" if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else "Vector"]
    return n.outputs[0]


def gn_obj(name, tree, parent=None, values=None):
    ob = mesh_obj(name, [], [], None, parent)
    md = ob.modifiers.new(name, "NODES")
    md.node_group = tree
    for nm, v in (values or {}).items():
        it = [i for i in tree.interface.items_tree if i.in_out == "INPUT" and i.name == nm][0]
        getattr(md.properties.inputs, it.identifier).value = v
    return ob, md


def gn_input_path(md, tree, nm):
    it = [i for i in tree.interface.items_tree if i.in_out == "INPUT" and i.name == nm][0]
    return getattr(md.properties.inputs, it.identifier).path_from_id("value")


# --- ice flakes shed from the cryogenic tanks (research fix #7: no point glints) ---------
M_ICE = principled("IceFlake", (0.80, 0.85, 0.90), 0.55, 0.0)
M_ICE.node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.3


def flake_tree():
    """Stateless ice shedding: Count flakes born on the core/ICPS skin (seed = event index),
    position = p0 + v*age + ½g·age², tumbling spin; flat white flakes, not glints."""
    t, s, gi, go = gn_tree("IceFlakes", [("Age", "NodeSocketFloat", 0.0), ("Seed", "NodeSocketInt", 0),
                                         ("Count", "NodeSocketInt", 60), ("Z0", "NodeSocketFloat", 44.0),
                                         ("Z1", "NodeSocketFloat", 63.0),
                                         ("G", "NodeSocketFloat", 4.9)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    ang = g_rand(t, gi, -2.6, -0.5, 11)                               # mostly on the camera side
    zr = g_rand(t, gi, 0.0, 1.0, 23)
    z = g_math(t, "MULTIPLY_ADD", zr, g_math(t, "SUBTRACT", gi.outputs["Z1"], gi.outputs["Z0"]), gi.outputs["Z0"])
    cx = g_math(t, "MULTIPLY", g_math(t, "COSINE", ang), R * 1.02)
    sy = g_math(t, "MULTIPLY", g_math(t, "SINE", ang), R * 1.02)
    p0 = t.nodes.new("ShaderNodeCombineXYZ")
    t.links.new(cx, p0.inputs[0])
    t.links.new(sy, p0.inputs[1])
    t.links.new(z, p0.inputs[2])
    rv = g_rand(t, gi, (-1.0, -1.0, -3.0), (1.0, 1.0, 0.5), 37, vec=True)
    radial = g_math(t, "MULTIPLY", p0.outputs[0], (0.35, 0.35, 0.0), vec=True)
    vel = g_math(t, "ADD", radial, rv, vec=True)
    age = gi.outputs["Age"]
    pos = g_math(t, "ADD", p0.outputs[0], g_math(t, "SCALE", vel, age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -1.0)                    # ½ g (flakes flutter: drag)
    t.links.new(g_math(t, "MULTIPLY", g_math(t, "MULTIPLY", age, age), gi.outputs["G"]), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    sp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], sp.inputs["Geometry"])
    t.links.new(pos, sp.inputs["Position"])
    cube = t.nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (1.0, 0.7, 0.05)
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(sp.outputs[0], inst.inputs["Points"])
    t.links.new(cube.outputs["Mesh"], inst.inputs["Instance"])
    r0 = g_rand(t, gi, (0, 0, 0), (6.3, 6.3, 6.3), 51, vec=True)
    spin = g_rand(t, gi, (-9, -9, -9), (9, 9, 9), 53, vec=True)
    t.links.new(g_math(t, "ADD", r0, g_math(t, "SCALE", spin, age, vec=True), vec=True), inst.inputs["Rotation"])
    t.links.new(g_rand(t, gi, 0.06, 0.45, 63), inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = M_ICE
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


FLAKE_T = flake_tree()
flakes, flakes_md = gn_obj("IceFlakes", FLAKE_T, RIG, {"Count": 70})
flakes2, flakes2_md = gn_obj("IceFlakesLiftoff", FLAKE_T, RIG, {"Count": 160, "Z0": 20.0, "Z1": 63.0, "G": 7.4})
no_shadow(flakes)
no_shadow(flakes2)


# --- hydrogen burn-off igniters: sparks fired radially under the core engines -------------
def spark_tree():
    t, s, gi, go = gn_tree("BurnoffSparks", [("Time", "NodeSocketFloat", 0.0), ("Count", "NodeSocketInt", 500),
                                             ("Amount", "NodeSocketFloat", 1.0)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    ph0 = g_rand(t, gi, 0.0, 1.0, 3, seed_sock=None)
    life = 0.9
    tt = g_math(t, "FRACT", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), ph0))
    age = g_math(t, "MULTIPLY", tt, life)
    cyc = g_math(t, "FLOOR", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), ph0))
    idx = t.nodes.new("GeometryNodeInputIndex")
    sd = g_math(t, "ADD", g_math(t, "MULTIPLY", cyc, 7919.0), idx.outputs[0])
    def rnd(lo, hi, salt, vec=False):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        ins[0].default_value, ins[1].default_value = lo, hi
        r.inputs["Seed"].default_value = salt
        t.links.new(sd, r.inputs["ID"])
        return [x for x in r.outputs if x.enabled and x.type == want][0]
    # emitter = one of the 4 engine exits (by index), launched sideways/down with a fan spread
    which = g_math(t, "FLOOR", rnd(0.0, 3.999, 5))
    exits = ENG_POS
    ex = t.nodes.new("ShaderNodeCombineXYZ")
    # piecewise select of the 4 exits via sums of (which == k) * coord
    for axis in range(2):
        acc = None
        for k in range(4):
            sel = g_math(t, "COMPARE", which, float(k), 0.1)
            term = g_math(t, "MULTIPLY", sel, exits[k][axis])
            acc = term if acc is None else g_math(t, "ADD", acc, term)
        t.links.new(acc, ex.inputs[axis])
    ex.inputs[2].default_value = exits[0][2] - 0.3
    az = rnd(0.0, 6.283, 7)
    sp_ = rnd(6.0, 16.0, 9)
    vx = g_math(t, "MULTIPLY", g_math(t, "COSINE", az), sp_)
    vy = g_math(t, "MULTIPLY", g_math(t, "SINE", az), sp_)
    vz = rnd(-6.0, 2.0, 13)
    vel = t.nodes.new("ShaderNodeCombineXYZ")
    t.links.new(vx, vel.inputs[0])
    t.links.new(vy, vel.inputs[1])
    t.links.new(vz, vel.inputs[2])
    pos = g_math(t, "ADD", ex.outputs[0], g_math(t, "SCALE", vel.outputs[0], age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -4.9)
    t.links.new(g_math(t, "MULTIPLY", age, age), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    setp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], setp.inputs["Geometry"])
    t.links.new(pos, setp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.05
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(setp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    alive = g_math(t, "MULTIPLY", g_math(t, "LESS_THAN", rnd(0.0, 1.0, 17), gi.outputs["Amount"]),
                   g_math(t, "SUBTRACT", 1.0, g_math(t, "POWER", tt, 3.0)))
    t.links.new(alive, inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = emissive("Spark", blackbody(2300), 120.0)
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


SPARK_TREE = spark_tree()
sparks, sparks_md = gn_obj("BurnoffSparks", SPARK_TREE, S1)
no_shadow(sparks)


# --- max-Q condensation cone: a conical shroud of fog hugging the upper stack (fix #8) ----
def vcone_material(name):
    m, nt = new_mat(name)
    gain = value_node(nt, "Gain", 0.0)
    evo = value_node(nt, "Evolve", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]                                     # unit cylinder: r<=1, z 0 (front) → -1 (aft)
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    v = M(nt, "MULTIPLY", sp.outputs[2], -1.0, clamp=True)
    r = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    rs = M(nt, "ADD", 0.36, M(nt, "MULTIPLY", M(nt, "POWER", v, 0.45), 0.62))    # cone surface radius
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 4.0, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.12)), nz.inputs["Vector"])           # streaks along the flow
    lk(nt, evo.outputs[0], nz.inputs["W"])
    rn = M(nt, "ADD", r, M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz.outputs["Fac"], 0.5), 0.12))
    outer = M(nt, "MULTIPLY", M(nt, "SUBTRACT", rs, rn), 7.0, clamp=True)          # sharp shock edge
    inner = M(nt, "MULTIPLY", M(nt, "SUBTRACT", rn, 0.28), 5.0, clamp=True)        # hugs the body, not inside it
    front = M(nt, "MULTIPLY", M(nt, "DIVIDE", v, 0.03), 1.0, clamp=True)
    aft = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, v, clamp=True), 2.2)
    d = M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", outer, inner), M(nt, "MULTIPLY", front, aft)),
          M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 3.2, M(nt, "SUBTRACT", -1.0, M(nt, "MULTIPLY", v, 0.8)), clamp=True))
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.95, 0.96, 1.0, 1), "Anisotropy": 0.55,
                                                    "Emission Color": (0.85, 0.9, 1.0, 1)})
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 14.0)), pv.inputs["Density"])
    lk(nt, M(nt, "MULTIPLY", d, M(nt, "MULTIPLY", gain.outputs[0], 0.6)), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_VCONE = vcone_material("VaporCone")
vcone = lathe("VaporCone", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 48, M_VCONE, RIG, loc=(0, 0, OGIVE_Z0 + 1.0))
vcone.scale = (8.0, 8.0, 34.0)
vcone_b = []
for side, bem in ((-1, BL), (1, BRt)):       # collars on the booster nose cones too
    c = lathe(f"VaporConeB{'LR'[side > 0]}", [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)], 32, M_VCONE, bem, loc=(0, 0, 53.0))
    c.scale = (4.2, 4.2, 14.0)
    vcone_b.append(c)
for o in [vcone] + vcone_b:
    no_shadow(o)


# --- VDB helpers (JangaFX CC0 sims: Z-up, voxel units, frame k = file k) -----------------
GROUND_ANCHOR = Vector((-4.0, 8.0, -160.0))       # the sim's floor/centre in its own voxel units
CHIMNEY_ANCHOR = Vector((-8.0, -80.0, -192.0))   # chimney mouth


def anchored(loc, rot, scale, anchor):
    """Object location that puts the sim's `anchor` point (voxel units) at world `loc`."""
    from mathutils import Euler
    R_ = Euler(rot).to_matrix()
    s_ = Vector(scale) if not isinstance(scale, (int, float)) else Vector((scale,) * 3)
    a = Vector((anchor.x * s_.x, anchor.y * s_.y, anchor.z * s_.z))
    return Vector(loc) - R_ @ a


def vdb_seq(name, folder, scale, loc=(0, 0, 0), rot=(0, 0, 0), parent=None, mat=None, mode="EXTEND"):
    files = sorted(glob.glob(os.path.join(folder, "*.vdb")))
    vol = bpy.data.volumes.new(name)
    vol.filepath = files[0]
    vol.is_sequence = len(files) > 1
    vol.frame_start = 1
    vol.frame_duration = len(files)
    vol.sequence_mode = mode
    if mat is not None:
        vol.materials.append(mat)
    ob = bpy.data.objects.new(name, vol)
    ob.location = loc
    ob.rotation_euler = rot
    ob.scale = (scale,) * 3
    link(ob, parent)
    return ob, len(files)


def vdb_play(ob, nfiles, start, speed=1.0, first=0):
    """Bake frame_offset so file index = first + (f - start) * speed, clamped to the sim.
    MEASURED (5.2.1): the sequence mode clamps the scene frame to [1, duration] BEFORE the offset
    is added (file = clamp(f - frame_start + 1) + offset), so CLIP mode shows nothing past the
    sim length — use EXTEND and offset against the clamped frame."""
    assert ob.data.sequence_mode == "EXTEND"
    f = IDX + 1
    pre = np.clip(f - ob.data.frame_start + 1, 1, ob.data.frame_duration)
    k = np.clip(np.floor(first + (IDX - start) * speed).astype(np.int64), 0, nfiles - 1)
    bake(ob.data, "frame_offset", (k - pre).astype(np.float64), const=True)
    return k


JANGA = os.path.join(AS, "jangafx")
GROUND_X = os.path.join(JANGA, "x_GroundExplosionVDB", "ground_explosion", "ground_explosion_VDB")
CHIMNEY = os.path.join(JANGA, "x_IndustrialChimneySmokeVDB", "industrial_chimney_smoke", "industrial_chimney_smoke_VDB")
CLOUDS = os.path.join(JANGA, "x_CloudPackVDB", "CloudPack", "CloudPackVDB")


def smoke_material(name, color=(0.92, 0.92, 0.93), dens=1.0, aniso=0.35, glow_grid="flames",
                   glow_col=blackbody(1900), plume_lit=True):
    """VDB steam/smoke: density grid × keyed 'Density'; lit FROM INSIDE by the plume:
    keyed 'Glow' × (flames grid + a falloff from the vehicle axis 'PlumeXY'/'PlumeZ' world position),
    so the core of the cloud glows orange and the far side stays grey-blue (research §1.2)."""
    m, nt = new_mat(name)
    dn = value_node(nt, "Density", dens)
    glow = value_node(nt, "Glow", 0.0)
    pz = value_node(nt, "PlumeZ", 0.0)
    at = nd(nt, "ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="density")
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(color) + (1,), "Anisotropy": aniso})
    pv.inputs["Density Attribute"].default_value = ""
    d = M(nt, "MULTIPLY", at.outputs["Fac"], dn.outputs[0])
    lk(nt, d, pv.inputs["Density"])
    e = None
    if glow_grid:
        fl = nd(nt, "ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name=glow_grid)
        e = M(nt, "MULTIPLY", fl.outputs["Fac"], 1.2)
    if plume_lit:
        gp = nd(nt, "ShaderNodeNewGeometry")
        sp = nd(nt, "ShaderNodeSeparateXYZ")
        lk(nt, gp.outputs["Position"], sp.inputs[0])
        rr = M(nt, "SQRT", M(nt, "ADD", M(nt, "MULTIPLY", sp.outputs[0], sp.outputs[0]),
                             M(nt, "MULTIPLY", sp.outputs[1], sp.outputs[1])))
        dz = M(nt, "ABSOLUTE", M(nt, "SUBTRACT", sp.outputs[2], pz.outputs[0]))
        fall = M(nt, "EXPONENT", M(nt, "MULTIPLY", M(nt, "ADD", rr, M(nt, "MULTIPLY", dz, 0.8)), -1.0 / 20.0))
        fall = M(nt, "MULTIPLY", fall, d)
        e = fall if e is None else M(nt, "ADD", e, fall)
    if e is not None:
        lk(nt, M(nt, "MULTIPLY", e, glow.outputs[0]), pv.inputs["Emission Strength"])
        pv.inputs["Emission Color"].default_value = tuple(glow_col) + (1,)
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


# LOX/LH2 boil-off vapour: cold, dense — it SINKS and pours down the vehicle (CC0 chimney sim
# turned upside down: its drift axis +Y → world -Z), white and thin
M_VENT = smoke_material("VentVapour", (0.95, 0.96, 1.0), dens=0.35, aniso=0.5, glow_grid=None, plume_lit=False)
VENTS = []
for k, (z, ang, s_) in enumerate(((CORE_TOP + 1.0, -1.25, 0.075), (LVSA_TOP + 2.5, -1.9, 0.055),
                                  (52.5, 2.6, 0.07))):
    rr_ = (R + 0.6) if z < CORE_TOP + 2 else 3.0
    vrot = (math.radians(-90), 0, ang + math.pi / 2)             # sim drift (+Y) → down (−Z): cold vapour sinks
    vloc = anchored((rr_ * math.cos(ang), rr_ * math.sin(ang), z), vrot, s_, CHIMNEY_ANCHOR)
    o, nfi = vdb_seq(f"Vent{k}", CHIMNEY, s_, loc=vloc, rot=vrot, parent=RIG, mat=M_VENT, mode="REPEAT")
    VENTS.append((o, nfi))


def volume_material(name, color=(0.9, 0.9, 0.92), shape="SPHERE", nscale=2.0, detail=6.0,
                    emis_color=(1.0, 0.45, 0.12), aniso=0.3, low_glow=True, sharp=2.2, namp=0.9):
    """Billowing smoke/steam/cloud volume. Keyed Value nodes: Density, Evolve (4D W),
    Glow (emission gain, applied to the lower half). shape SPHERE = unit-sphere falloff in
    object space; SLAB = cloud deck (box, puffy top); COLUMN = trail (cylinder)."""
    m, nt = new_mat(name)
    dens = value_node(nt, "Density", 1.0)
    evo = value_node(nt, "Evolve", 0.0)
    glow = value_node(nt, "Glow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sepo = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sepo.inputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D",
            inp={"Scale": nscale, "Detail": detail, "Roughness": 0.58, "Distortion": 0.25})
    lk(nt, ob, nz.inputs["Vector"])
    lk(nt, evo.outputs[0], nz.inputs["W"])
    n = nz.outputs["Fac"]
    if shape == "SPHERE":
        r = V(nt, "LENGTH", ob)
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, r), M(nt, "MULTIPLY", n, 0.9)),
                 0.55, clamp=True)
        mask = M(nt, "MULTIPLY", mask, 2.2)
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
    elif shape == "SLAB":
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
        # coverage: big puffy cells; density falls off toward a noisy top
        iso = V(nt, "MULTIPLY", ob, (13.0, 13.0, 0.8))
        lk(nt, iso, nz.inputs["Vector"])
        nz2 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.35, "Detail": 3.0})
        lk(nt, iso, nz2.inputs["Vector"])
        top = M(nt, "MULTIPLY_ADD", nz2.outputs["Fac"], 1.3, -0.35)
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", top, h), M(nt, "MULTIPLY", n, 0.7)),
                 0.3, clamp=True)
        mask = M(nt, "MULTIPLY", mask, 3.0)
        edge = M(nt, "MULTIPLY", M(nt, "SUBTRACT", 1.0, M(nt, "MAXIMUM",
                                                           M(nt, "ABSOLUTE", sepo.outputs[0]),
                                                           M(nt, "ABSOLUTE", sepo.outputs[1])), clamp=True), 6.0)
        mask = M(nt, "MULTIPLY", mask, M(nt, "MINIMUM", edge, 1.0))
    elif shape == "UNIFORM":
        h = M(nt, "MULTIPLY_ADD", sepo.outputs[2], 0.5, 0.5, clamp=True)
        mask = M(nt, "MULTIPLY", M(nt, "POWER", M(nt, "SUBTRACT", 1.0, h), 2.0), M(nt, "MULTIPLY_ADD", n, 0.8, 0.6))
    else:  # COLUMN: unit cylinder r<=1 along z in [0,1]; pyroclastic: two octaves, sharp edge
        rr = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
        h = M(nt, "MAXIMUM", sepo.outputs[2], 0.0)
        nz2 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": nscale * 5.5, "Detail": 4.0})
        lk(nt, ob, nz2.inputs["Vector"])
        lk(nt, evo.outputs[0], nz2.inputs["W"])
        fbm = M(nt, "ADD", M(nt, "SUBTRACT", n, 0.5), M(nt, "MULTIPLY", M(nt, "SUBTRACT", nz2.outputs["Fac"], 0.5), 0.6))
        mask = M(nt, "SUBTRACT", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, rr), M(nt, "MULTIPLY", fbm, namp * 2.0)),
                 0.1, clamp=True)
        mask = M(nt, "MINIMUM", M(nt, "MULTIPLY", mask, sharp), 1.0)
    d = M(nt, "MULTIPLY", mask, dens.outputs[0])
    green = value_node(nt, "Green", 0.0)
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": tuple(color) + (1,), "Anisotropy": aniso})
    lk(nt, mix_col(nt, green.outputs[0], tuple(emis_color), (0.3, 1.0, 0.4)), pv.inputs["Emission Color"])
    lk(nt, d, pv.inputs["Density"])
    low = M(nt, "POWER", M(nt, "SUBTRACT", 1.0, h, clamp=True), 2.0) if low_glow else 1.0
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", low, glow.outputs[0]), mask), pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


# =====================================================================================
# 6. THE PAD + WORLD BUILD (research fix #5): raised hardstand with a flame trench, NASA
#    Mobile Launcher, flame deflector, lightning towers + catenary wires, sphere tanks, water
#    tower, xenon floods; numpy heightfield terrain (scrub flats, coastline + Atlantic to the
#    east, a lagoon between the long-lens camera and the pad); Poly Haven scrub scatter.
# =====================================================================================
PAD_TOP = 14.0                                   # hardstand elevation (m)
ML_DECK = PAD_TOP + 6.98                         # deck top (the NASA ML's legs stand on the pad)
PAD_Z = ML_DECK + 1.2                            # vehicle station 0 above the deck
TRENCH_W = 9.5                                   # half width
_rng = np.random.default_rng(1234)
_lat = _rng.random((257, 257)).astype(np.float64)


def vnoise(x, y, scale):
    """Smooth value noise (0..1) from a tiled 256² lattice — numpy, deterministic."""
    u, v = x / scale, y / scale
    i0, j0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - i0, v - j0
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    i0 %= 256
    j0 %= 256
    a, b = _lat[i0, j0], _lat[i0 + 1, j0]
    c, d = _lat[i0, j0 + 1], _lat[i0 + 1, j0 + 1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, y, scale, octaves=5):
    s, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        s = s + amp * vnoise(x + 37.1 * k, y - 11.7 * k, scale / (2 ** k))
        tot += amp
        amp *= 0.5
    return s / tot


CAM_DIR = Vector((-0.34, -0.94, 0.0)).normalized()       # from the pad toward the long-lens site


def height(x, y):
    """Terrain height (m) + masks: (h, sand, pad, water)."""
    land = 1.6 + 2.2 * (fbm(x, y, 900.0) - 0.5) + 0.7 * (fbm(x, y, 70.0) - 0.5)
    # Atlantic to the east
    cx = 1350.0 + 420.0 * (fbm(0.0 * x, y, 2600.0, 3) - 0.5)
    sea = np.clip((x - cx) / 90.0, 0.0, 1.0)
    # lagoon band between the pad and the long-lens camera (SSW)
    s = x * CAM_DIR.x + y * CAM_DIR.y
    t = x * -CAM_DIR.y + y * CAM_DIR.x
    sm = s + 160.0 * (fbm(t, 0.0 * t, 1100.0, 3) - 0.5)
    lag = np.clip((sm - 1450.0) / 70.0, 0, 1) * np.clip((2950.0 - sm) / 70.0, 0, 1)
    water = np.maximum(sea, lag)
    h = land * (1 - water) + (-3.0) * water
    sand = np.clip(1.0 - np.abs(x - cx + 30) / 70.0, 0, 1) + np.clip(1.0 - (np.minimum(np.abs(sm - 1450), np.abs(sm - 2950))) / 45.0, 0, 1) * 0.7
    # the pad mound (superellipse flat top ±95 m, slopes out to ~±175 m)
    q = ((np.abs(x) / 95.0) ** 6 + (np.abs(y) / 95.0) ** 6) ** (1 / 6)
    mound = PAD_TOP * (1 - smoothstep((q - 1.0) / 0.85))
    padm = (q < 1.0).astype(np.float64)
    h = np.maximum(h, mound)
    # flame trench: |y| < TRENCH_W from under the hole out of the east slope
    tf = 3.0 - 2.0 * np.clip(x / 170.0, 0, 1)
    wall = np.clip((TRENCH_W - np.abs(y)) / 1.2, 0, 1) * np.clip((x + 13.0) / 1.2, 0, 1)
    h = h * (1 - wall) + np.minimum(h, tf) * wall
    return h, np.clip(sand, 0, 1), padm, water


def heightfield(name, size, n, mat, z_bias=0.0, hole=None):
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    H, sand, padm, water = height(X, Y)
    if hole is not None:                                       # sink under the near tile (no z-fight)
        inside = (np.abs(X) < hole) & (np.abs(Y) < hole)
        H = np.where(inside, H - 1.5, H)
    H = H + z_bias
    me = bpy.data.meshes.new(name)
    nv = n * n
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.stack([X.ravel(), Y.ravel(), H.ravel()], 1).astype(np.float32).ravel())
    ii = np.arange(n - 1)
    I, J = np.meshgrid(ii, ii, indexing="ij")
    a = (I * n + J).ravel()
    quads = np.stack([a, a + n, a + n + 1, a + 1], 1)
    me.loops.add(quads.size)
    me.polygons.add(len(quads))
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for nm, arr in (("sand", sand), ("pad", padm), ("water", water)):
        at = me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", arr.ravel().astype(np.float32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


def terrain_material():
    """Scrub flats (procedural greens/browns), sand (Poly Haven coast_sand_01), pad top concrete
    (concrete_floor_02), grassy mound slopes; keyed 'Wet' darkens the pad after the deluge."""
    m, nt = new_mat("Terrain")
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Metallic": 0.0})
    tc = nd(nt, "ShaderNodeTexCoord")
    p = tc.outputs["Object"]
    a_s = nd(nt, "ShaderNodeAttribute", attribute_name="sand")
    a_p = nd(nt, "ShaderNodeAttribute", attribute_name="pad")
    g1 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.03, "Detail": 8.0, "Roughness": 0.65})
    lk(nt, p, g1.inputs["Vector"])
    g2 = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 0.4, "Detail": 6.0, "Roughness": 0.7})
    lk(nt, p, g2.inputs["Vector"])
    scrub = mix_col(nt, g1.outputs["Fac"], (0.030, 0.038, 0.016), (0.085, 0.075, 0.040))
    scrub = mix_col(nt, M(nt, "MULTIPLY", M(nt, "SUBTRACT", g2.outputs["Fac"], 0.35, clamp=True), 1.2), scrub,
                    (0.11, 0.10, 0.065))
    sd = tex_box(nt, ph("coast_sand_01", "diff"), 3.0, coord=p)
    col = mix_col(nt, a_s.outputs["Fac"], scrub, col_mul(nt, sd.outputs["Color"], 0.9))
    cd = tex_box(nt, ph("concrete_floor_02", "diff"), 4.0, coord=p)
    wet = value_node(nt, "Wet", 0.0)
    padc = col_mul(nt, cd.outputs["Color"], M(nt, "SUBTRACT", 0.85, M(nt, "MULTIPLY", wet.outputs[0], 0.4)))
    col = mix_col(nt, a_p.outputs["Fac"], col, padc)
    lk(nt, col, b.inputs["Base Color"])
    rgh = M(nt, "ADD", 0.9, M(nt, "MULTIPLY", a_p.outputs["Fac"], M(nt, "MULTIPLY", wet.outputs[0], -0.75)))
    lk(nt, rgh, b.inputs["Roughness"])
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.25, "Distance": 0.2})
    lk(nt, g2.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


def water_material():
    """Lagoon + Atlantic: dark, low roughness, wind ripples; reflects the plume and the floods."""
    m, nt = new_mat("Water")
    b = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.006, 0.012, 0.016, 1), "Roughness": 0.06,
                                                 "IOR": 1.33})
    tc = nd(nt, "ShaderNodeTexCoord")
    w1 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 0.12, "Detail": 6.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", tc.outputs["Object"], (1.0, 2.5, 1.0)), w1.inputs["Vector"])
    w1.name = "Waves"
    bp = nd(nt, "ShaderNodeBump", inp={"Strength": 0.18, "Distance": 0.25})
    lk(nt, w1.outputs["Fac"], bp.inputs["Height"])
    lk(nt, bp.outputs["Normal"], b.inputs["Normal"])
    finish(nt, b)
    return m


M_TERRAIN = terrain_material()
M_WATER = water_material()
terr_near = heightfield("TerrainNear", 1300.0, 521, M_TERRAIN)
terr_far = heightfield("TerrainFar", 24000.0, 481, M_TERRAIN, hole=640.0)
sea_me = MB()
sea_me.poly([(-30000, -30000, 0.0), (30000, -30000, 0.0), (30000, 30000, 0.0), (-30000, 30000, 0.0)], [(0, 1, 2, 3)])
sea = sea_me.obj("Water", M_WATER)

# Mobile Launcher (NASA, PD): tower to the north (+Y), exhaust hole over the trench
ML = lib_obj("ML", None, loc=(0, 0, ML_DECK), rot=(0, 0, math.radians(90)))
# flame deflector: a wedge under the hole that turns the exhaust east (+X) down the trench
fd = MB()
fd.poly([(-12, -TRENCH_W, 3.0), (-12, TRENCH_W, 3.0), (-12, TRENCH_W, PAD_TOP - 1.0), (-12, -TRENCH_W, PAD_TOP - 1.0),
         (14, -TRENCH_W, 3.0), (14, TRENCH_W, 3.0)],
        [(0, 1, 2, 3), (3, 2, 5, 4), (0, 3, 4), (1, 5, 2), (0, 4, 5, 1)])
fd.obj("FlameDeflector", M_CONCRETE_DK)
tw = MB()                                       # trench walls (dark, sooty concrete) to hide the heightfield slope
for sy in (-1, 1):
    tw.box((75.0, sy * (TRENCH_W + 0.6), 7.0), (180.0, 1.2, 10.0))
tw.obj("TrenchWalls", M_CONCRETE_DK)
# lightning protection: three 180 m lattice towers with catenary wires (kept out of the
# foreground of the key shots: north/west/east of the pad)
LT_POS = [(-150.0, 150.0), (160.0, 140.0), (10.0, 230.0)]
lt = MB()
for (x0, y0) in LT_POS:
    pts = [(x0 + dx, y0 + dy) for dx, dy in ((-2.5, -2.5), (2.5, -2.5), (2.5, 2.5), (-2.5, 2.5))]
    for (x, y) in pts:
        lt.beam((x, y, 0), (x0 + (x - x0) * 0.35, y0 + (y - y0) * 0.35, 180), 0.45)
    for zz in np.arange(6.0, 180.0, 7.0):
        f0, f1 = 1 - 0.65 * zz / 180, 1 - 0.65 * (zz + 7) / 180
        for k in range(4):
            (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % 4]
            lt.beam((x0 + (xa - x0) * f0, y0 + (ya - y0) * f0, zz), (x0 + (xb - x0) * f1, y0 + (yb - y0) * f1, zz + 7), 0.18)
    lt.beam((x0, y0, 180), (x0, y0, 195), 0.35)
lt.obj("LightningTowers", M_ML)
for a_, b_ in ((0, 1), (1, 2), (2, 0)):
    p0, p1 = Vector((*LT_POS[a_], 192.0)), Vector((*LT_POS[b_], 192.0))
    cu = bpy.data.curves.new(f"Catenary{a_}{b_}", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.06
    spl = cu.splines.new("POLY")
    pts_ = [p0.lerp(p1, t) - Vector((0, 0, 38.0 * 4 * t * (1 - t))) for t in np.linspace(0, 1, 24)]
    spl.points.add(len(pts_) - 1)
    for k, pp in enumerate(pts_):
        spl.points[k].co = (pp.x, pp.y, pp.z, 1.0)
    cu.materials.append(M_DARK)
    link(bpy.data.objects.new(f"Catenary{a_}{b_}", cu))
# LH2 + LOX sphere tanks and the water tower (simple shapes, they live at 250+ m)
M_TANKW = dielectric("TankWhite", (0.70, 0.70, 0.69), 0.45, tile=3.0)
uv_sphere("LH2Sphere", 12.5, 48, 24, M_TANKW, loc=(-260.0, 230.0, 14.5))
uv_sphere("LOXSphere", 10.0, 48, 24, M_TANKW, loc=(270.0, 250.0, 12.0))
wt = MB()
for k in range(6):
    a = 2 * math.pi * k / 6
    wt.beam((-330 + 8 * math.cos(a), -60 + 8 * math.sin(a), 0), (-330 + 4 * math.cos(a), -60 + 4 * math.sin(a), 70), 0.8)
wt.obj("WaterTowerLegs", M_ML)
uv_sphere("WaterTowerTank", 11.0, 32, 16, M_TANKW, loc=(-330, -60, 80))
# xenon floods: four banks on 30 m masts ~220 m out, hot white pools on the stack
FLOODS = []
FLOOD_POS = [(-170, -150, 30, 30), (180, -160, 30, 60), (200, 120, 30, 85), (-60, -230, 30, 45),
             (-230, 40, 30, 70), (90, -250, 30, 20)]            # (x, y, mast height, aim height above PAD_Z)
fl = MB()
for (x, y, z, _a) in FLOOD_POS:
    fl.beam((x, y, 0), (x, y, z), 0.9)
fl.obj("FloodMasts", M_ML)
for k, (x, y, z, aim) in enumerate(FLOOD_POS):
    panel = no_shadow(cube_obj(f"FloodPanel{k}", M_LAMP, loc=(x, y, z + 1.5), scale=(2.2, 0.3, 1.2)))
    d = Vector((0, 0, PAD_Z + aim)) - Vector((x, y, z))
    panel.rotation_euler = d.to_track_quat("Y", "Z").to_euler()
    FLOODS.append(panel)
    ld = bpy.data.lights.new(f"Flood{k}", "SPOT")
    ld.spot_size = math.radians(16)
    ld.spot_blend = 0.45
    ld.color = (0.88, 0.92, 1.0)                 # xenon ~6000 K
    ld.shadow_soft_size = 0.8
    ld.volume_factor = 0.6
    lo = link(bpy.data.objects.new(f"Flood{k}", ld))
    lo.location = (x, y, z + 1.5)
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    FLOODS.append(lo)
# sodium work lights up the mobile-launcher tower (every few levels) + on the deck
WORKL = []
M_SODIUM = emissive("SodiumLamp", blackbody(2100), 1.0, objcolor_gain=True)
for k, zz in enumerate(np.arange(ML_DECK + 6.0, ML_DECK + 108.0, 8.5)):
    for sx in (-1, 1):
        x, y = 4.7 + sx * 6.5, 14.7 - 7.2
        o = no_shadow(cube_obj(f"WorkLamp{k}_{sx}", M_SODIUM, loc=(x, y, zz), scale=(0.35, 0.35, 0.2)))
        WORKL.append(o)
    ld = bpy.data.lights.new(f"WorkL{k}", "POINT")
    ld.color = blackbody(2100)
    ld.shadow_soft_size = 1.0
    ld.use_shadow = False
    ld.volume_factor = 0.3
    lo = link(bpy.data.objects.new(f"WorkL{k}", ld))
    lo.location = (4.7, 14.7 - 8.5, zz)
    WORKL.append(lo)
for k, (x, y) in enumerate(((-17.0, -12.0), (17.0, -12.0), (-17.0, 10.0), (17.0, 10.0))):
    ld = bpy.data.lights.new(f"DeckL{k}", "POINT")
    ld.color = blackbody(2300)
    ld.shadow_soft_size = 2.0
    ld.use_shadow = False
    lo = link(bpy.data.objects.new(f"DeckL{k}", ld))
    lo.location = (x, y, ML_DECK + 3.0)
    WORKL.append(lo)
# sky searchlights (the Artemis-style light show) + red aviation beacons on the towers
SEARCH = []
for k, (x, y) in enumerate([(-420, -300), (330, -380), (60, -520)]):
    ld = bpy.data.lights.new(f"Search{k}", "SPOT")
    ld.spot_size = math.radians(1.6)
    ld.spot_blend = 0.15
    ld.color = (0.82, 0.88, 1.0)
    ld.shadow_soft_size = 0.2
    ld.use_shadow = False
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"Search{k}", ld))
    lo.location = (x, y, 2.0)
    SEARCH.append(lo)
BEACONS = []
for k, (x, y) in enumerate(LT_POS + [(4.7 * 1.0, 14.7 * 1.0)]):
    z = 196.0 if k < 3 else ML_DECK + 112.0
    o = no_shadow(uv_sphere(f"Beacon{k}", 0.6, 12, 6, M_LAMP, loc=(x, y, z)))
    BEACONS.append(o)
    ld = bpy.data.lights.new(f"BeaconL{k}", "POINT")
    ld.color = (1.0, 0.05, 0.02)
    ld.shadow_soft_size = 0.5
    ld.use_shadow = False
    ld.volume_factor = 1.0
    lo = link(bpy.data.objects.new(f"BeaconL{k}", ld))
    lo.location = (x, y, z + 1.2)
    BEACONS.append(lo)
# distant town / road lights along the coast and across the lagoon (warm sodium + cool LED)
rng = np.random.default_rng(5)
dl = MB()
for k in range(700):
    a = rng.uniform(-math.pi, math.pi)
    rr = rng.uniform(3200, 9000)
    x, y = rr * math.cos(a), rr * math.sin(a)
    hh, *_ = height(np.array([x]), np.array([y]))
    if hh[0] < 0.2:
        continue
    s = rng.uniform(0.8, 2.4)
    dl.box((x, y, hh[0] + s / 2 + 2.0), (s, s, s))
dl_ob = dl.obj("ShoreLights", emissive("ShoreLamp", (1.0, 0.62, 0.30), 40.0))
no_shadow(dl_ob)

# --- v4: LED lights on the launch tower + pad (research #1 §3). Tower footprint measured from the NASA ML
#     mesh (vertices above the deck, rotated into the scene): x -6.0..6.6, y 13.7..26.2. One object per tower
#     LEVEL (a band around the tower + the four corner-strip segments above it) so each level can light up as
#     the rocket passes it; the deck edge and the lightning towers get rings. Same LED-pixel shader.
M_LEDTOWER = led_pixel_material("LEDPixelTower", 9.0, 3.0, "diag", hue_per_m=0.008)
TW_X0, TW_X1, TW_Y0, TW_Y1 = -6.0 - 0.35, 6.6 + 0.35, 13.7 - 0.35, 26.2 + 0.35
TOWER_LEDS = []                                  # (object, height of the level above the pad deck)
_lv = list(np.arange(ML_DECK + 6.0, ML_DECK + 110.0, 8.5))
for k, zz in enumerate(_lv):
    tb = MB()
    tb.box(((TW_X0 + TW_X1) / 2, TW_Y0, zz), (TW_X1 - TW_X0, 0.3, 0.3))
    tb.box(((TW_X0 + TW_X1) / 2, TW_Y1, zz), (TW_X1 - TW_X0, 0.3, 0.3))
    tb.box((TW_X0, (TW_Y0 + TW_Y1) / 2, zz), (0.3, TW_Y1 - TW_Y0, 0.3))
    tb.box((TW_X1, (TW_Y0 + TW_Y1) / 2, zz), (0.3, TW_Y1 - TW_Y0, 0.3))
    z_top = _lv[k + 1] if k + 1 < len(_lv) else ML_DECK + 110.0
    for (cx, cy) in ((TW_X0, TW_Y0), (TW_X1, TW_Y0), (TW_X0, TW_Y1), (TW_X1, TW_Y1)):
        tb.box((cx, cy, (zz + z_top) / 2), (0.28, 0.28, z_top - zz - 0.4))
    TOWER_LEDS.append((no_shadow(tb.obj(f"TowerLED{k}", M_LEDTOWER)), zz - PAD_Z))
DECK_LED = None
_ltb = MB()
for (x0, y0) in LT_POS:
    for zz in np.arange(20.0, 180.0, 20.0):
        h = 2.5 * (1 - 0.65 * zz / 180.0) + 0.35
        for (ax, ay, bx, by) in ((-h, -h, h, -h), (h, -h, h, h), (h, h, -h, h), (-h, h, -h, -h)):
            _ltb.beam((x0 + ax, y0 + ay, zz), (x0 + bx, y0 + by, zz), 0.3)
LT_LED = no_shadow(_ltb.obj("LightningTowerLEDs", M_LEDTOWER))
TOWER_SPILL = []
for k, (x, y, rot_) in enumerate(((0.3, 8.0, (math.radians(-90), 0.0, 0.0)),     # south face, shines -Y
                                   (14.0, 20.0, (0.0, math.radians(-90), 0.0)))):  # east face, shines +X
    ld = bpy.data.lights.new(f"TowerSpill{k}", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = 14.0, 90.0
    ld.use_shadow = False
    ld.volume_factor = 0.35
    lo = link(bpy.data.objects.new(f"TowerSpill{k}", ld))
    lo.location = (x, y, ML_DECK + 55.0)
    lo.rotation_euler = rot_                              # area lights shine along their local -Z
    TOWER_SPILL.append(lo)


# --- deluge water: curtains pouring off the ML deck edges + 'rainbird' fountains --------
def curtain_material():
    m, nt = new_mat("WaterCurtain")
    m.surface_render_method = "BLENDED"
    m.use_transparency_overlap = True
    gain = value_node(nt, "Gain", 0.0)
    flow = value_node(nt, "Flow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    off = nd(nt, "ShaderNodeCombineXYZ")
    lk(nt, flow.outputs[0], off.inputs[2])
    p = V(nt, "ADD", V(nt, "MULTIPLY", tc.outputs["Object"], (2.2, 2.2, 0.18)), off.outputs[0])
    nz = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 1.0, "Detail": 8.0, "Roughness": 0.7})
    lk(nt, p, nz.inputs["Vector"])
    a = M(nt, "MULTIPLY", M(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 2.4, -0.75, clamp=True), gain.outputs[0], clamp=True)
    df = nd(nt, "ShaderNodeBsdfPrincipled", inp={"Base Color": (0.80, 0.84, 0.88, 1), "Roughness": 0.25,
                                                  "Transmission Weight": 0.4})
    tr = nd(nt, "ShaderNodeBsdfTransparent")
    mx = nd(nt, "ShaderNodeMixShader")
    lk(nt, a, mx.inputs[0])
    lk(nt, tr.outputs[0], mx.inputs[1])
    lk(nt, df.outputs[0], mx.inputs[2])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, mx.outputs[0], o.inputs["Surface"])
    return m


M_CURTAIN = curtain_material()
cur = MB()
ML_X = 20.5                                       # ML base footprint in world (rotated 90°): x ±20.5,
ML_Y0, ML_Y1 = -14.6, 32.4                        # y -14.6 (south, open side) .. 32.4 (tower side)
for (x0, y0, x1, y1, ox, oy) in ((-ML_X, ML_Y0, ML_X, ML_Y0, 0.0, -1.0), (-ML_X, ML_Y0, -ML_X, 12.0, -1.0, 0.0),
                                 (ML_X, ML_Y0, ML_X, 12.0, 1.0, 0.0)):
    cur.poly([(x0 + 0.4 * ox, y0 + 0.4 * oy, ML_DECK - 0.2), (x1 + 0.4 * ox, y1 + 0.4 * oy, ML_DECK - 0.2),
              (x1 + 1.6 * ox, y1 + 1.6 * oy, PAD_TOP), (x0 + 1.6 * ox, y0 + 1.6 * oy, PAD_TOP)], [(0, 1, 2, 3)])
curtains = cur.obj("DelugeCurtains", M_CURTAIN)
# v4: LED edge around the mobile-launcher deck (same tower LED shader)
_dk = MB()
for (x0, y0, x1, y1) in ((-ML_X, ML_Y0, ML_X, ML_Y0), (-ML_X, ML_Y1, ML_X, ML_Y1),
                         (-ML_X, ML_Y0, -ML_X, ML_Y1), (ML_X, ML_Y0, ML_X, ML_Y1)):
    _dk.beam((x0, y0, ML_DECK + 0.35), (x1, y1, ML_DECK + 0.35), 0.3)
DECK_LED = no_shadow(_dk.obj("DeckLED", M_LEDTOWER))
no_shadow(curtains)


def rainbird_tree():
    """Stateless water fountains: each droplet loops on a parabola from one of six deck nozzles."""
    t, s, gi, go = gn_tree("Rainbirds", [("Time", "NodeSocketFloat", 0.0), ("Count", "NodeSocketInt", 2400),
                                         ("Amount", "NodeSocketFloat", 0.0)])
    pts = t.nodes.new("GeometryNodePoints")
    t.links.new(gi.outputs["Count"], pts.inputs["Count"])
    idx = t.nodes.new("GeometryNodeInputIndex")
    def rnd(lo, hi, salt, vec=False):
        r = t.nodes.new("FunctionNodeRandomValue")
        r.data_type = "FLOAT_VECTOR" if vec else "FLOAT"
        want = "VECTOR" if vec else "VALUE"
        ins = [x for x in r.inputs if x.enabled and x.type == want]
        ins[0].default_value, ins[1].default_value = lo, hi
        r.inputs["Seed"].default_value = salt
        t.links.new(idx.outputs[0], r.inputs["ID"])
        return [x for x in r.outputs if x.enabled and x.type == want][0]
    life = 2.2
    tt = g_math(t, "FRACT", g_math(t, "ADD", g_math(t, "DIVIDE", gi.outputs["Time"], life), rnd(0.0, 1.0, 3)))
    age = g_math(t, "MULTIPLY", tt, life)
    nozzles = [(-14, -12), (14, -12), (-14, 12), (14, 12), (0, -16), (0, 16)]
    which = g_math(t, "FLOOR", rnd(0.0, 5.999, 5))
    ex = t.nodes.new("ShaderNodeCombineXYZ")
    for axis in range(2):
        acc = None
        for k in range(6):
            term = g_math(t, "MULTIPLY", g_math(t, "COMPARE", which, float(k), 0.1), nozzles[k][axis])
            acc = term if acc is None else g_math(t, "ADD", acc, term)
        t.links.new(acc, ex.inputs[axis])
    ex.inputs[2].default_value = ML_DECK + 0.5
    # spray toward the vehicle centre + up, with spread
    to_c = g_math(t, "SCALE", g_math(t, "NORMALIZE", g_math(t, "MULTIPLY", ex.outputs[0], (-1.0, -1.0, 0.0), vec=True), vec=True),
                  rnd(5.0, 9.0, 7), vec=True)
    up = rnd((-1.5, -1.5, 8.0), (1.5, 1.5, 13.0), 9, vec=True)
    vel = g_math(t, "ADD", to_c, up, vec=True)
    pos = g_math(t, "ADD", ex.outputs[0], g_math(t, "SCALE", vel, age, vec=True), vec=True)
    fall = t.nodes.new("ShaderNodeVectorMath")
    fall.operation = "SCALE"
    fall.inputs[0].default_value = (0.0, 0.0, -4.9)
    t.links.new(g_math(t, "MULTIPLY", age, age), fall.inputs["Scale"])
    pos = g_math(t, "ADD", pos, fall.outputs["Vector"], vec=True)
    setp = t.nodes.new("GeometryNodeSetPosition")
    t.links.new(pts.outputs[0], setp.inputs["Geometry"])
    t.links.new(pos, setp.inputs["Position"])
    ico = t.nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.22
    ico.inputs["Subdivisions"].default_value = 1
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    t.links.new(setp.outputs[0], inst.inputs["Points"])
    t.links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    alive = g_math(t, "LESS_THAN", rnd(0.0, 1.0, 17), gi.outputs["Amount"])
    sz = g_math(t, "MULTIPLY", alive, rnd(0.5, 1.6, 19))
    t.links.new(sz, inst.inputs["Scale"])
    sm = t.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = principled("Spray", (0.82, 0.86, 0.9), 0.2, 0.0)
    t.links.new(inst.outputs[0], sm.inputs["Geometry"])
    t.links.new(sm.outputs[0], go.inputs["Geometry"])
    return t


RAIN_T = rainbird_tree()
rainbirds, rain_md = gn_obj("Rainbirds", RAIN_T)
no_shadow(rainbirds)


# --- scrub scatter on the near terrain (Poly Haven shrubs, CC0), off the pad and the sand --
def scatter_tree():
    t, s, gi, go = gn_tree("Scrub", [])
    dist = t.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "POISSON"
    dist.inputs["Distance Min"].default_value = 9.0
    dist.inputs["Density Max"].default_value = 0.0016
    t.links.new(gi.outputs["Geometry"], dist.inputs["Mesh"])
    a_p = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_p.data_type = "FLOAT"
    a_p.inputs["Name"].default_value = "pad"
    a_s = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_s.data_type = "FLOAT"
    a_s.inputs["Name"].default_value = "sand"
    a_w = t.nodes.new("GeometryNodeInputNamedAttribute")
    a_w.data_type = "FLOAT"
    a_w.inputs["Name"].default_value = "water"
    pos = t.nodes.new("GeometryNodeInputPosition")
    rad = g_math(t, "LENGTH", pos.outputs[0], vec=True)
    ok = g_math(t, "MULTIPLY", g_math(t, "LESS_THAN", a_p.outputs["Attribute"], 0.5),
                g_math(t, "LESS_THAN", g_math(t, "ADD", a_s.outputs["Attribute"], a_w.outputs["Attribute"]), 0.2))
    ok = g_math(t, "MULTIPLY", ok, g_math(t, "GREATER_THAN", rad, 200.0))
    ok = g_math(t, "MULTIPLY", ok, g_math(t, "LESS_THAN", rad, 620.0))
    t.links.new(ok, dist.inputs["Selection"])
    coll = bpy.data.collections.new("ScrubSources")
    for nm in ("Scatter_shrub_02", "Scatter_shrub_01", "Scatter_shrub_04"):
        if nm in LIB:
            src = LIB[nm]
            tris = sum(len(p_.vertices) - 2 for p_ in src.data.polygons)
            dm = src.modifiers.new("Decimate", "DECIMATE")       # instances at 200+ m: ~2.5k tris is plenty
            dm.ratio = min(1.0, 2500.0 / max(tris, 1))
            coll.objects.link(src)
    ci = t.nodes.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst = t.nodes.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    t.links.new(dist.outputs["Points"], inst.inputs["Points"])
    t.links.new(ci.outputs[0], inst.inputs["Instance"])
    r1 = t.nodes.new("FunctionNodeRandomValue")
    r1.data_type = "INT"
    [x for x in r1.inputs if x.enabled and x.type == "INT"][1].default_value = 3
    t.links.new([x for x in r1.outputs if x.enabled and x.type == "INT"][0], inst.inputs["Instance Index"])
    rr = t.nodes.new("FunctionNodeRandomValue")
    rr.data_type = "FLOAT_VECTOR"
    [x for x in rr.inputs if x.enabled and x.type == "VECTOR"][1].default_value = (0.0, 0.0, 6.28)
    t.links.new([x for x in rr.outputs if x.enabled and x.type == "VECTOR"][0], inst.inputs["Rotation"])
    rs = t.nodes.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    ins = [x for x in rs.inputs if x.enabled and x.type == "VALUE"]
    ins[0].default_value, ins[1].default_value = 0.8, 2.2
    t.links.new([x for x in rs.outputs if x.enabled and x.type == "VALUE"][0], inst.inputs["Scale"])
    jg = t.nodes.new("GeometryNodeJoinGeometry")
    t.links.new(gi.outputs["Geometry"], jg.inputs[0])
    t.links.new(inst.outputs[0], jg.inputs[0])
    t.links.new(jg.outputs[0], go.inputs["Geometry"])
    return t


if any(n in LIB for n in ("Scatter_shrub_02", "Scatter_shrub_01")):
    terr_near.modifiers.new("Scrub", "NODES").node_group = scatter_tree()

# --- steam + smoke: JangaFX CC0 VDB sims (research fix #1: real volumes, not cotton balls) --
M_STEAM = smoke_material("Steam", (0.93, 0.93, 0.94), dens=3.0, aniso=0.35, glow_grid="flames")
M_STEAM_EARLY = smoke_material("SteamEarly", (0.94, 0.95, 0.97), dens=2.4, aniso=0.4, glow_grid=None,
                               glow_col=blackbody(5500))
STEAMV = []        # (object, nfiles, start, speed, base loc, drift (m/s), scale, kind)
_vrng = np.random.default_rng(77)
for k, (x, y, sc_, d0, sp_, rz) in enumerate((
        (125.0, 0.0, 1.35, 0, 0.36, 0.0),           # the trench blast, east
        (185.0, -45.0, 1.15, 5, 0.33, 1.3),
        (170.0, 50.0, 1.05, 9, 0.33, 2.4),
        (-34.0, -38.0, 0.75, 3, 0.40, 0.7),          # rising around the mobile launcher base
        (34.0, -44.0, 0.70, 6, 0.42, 3.9))):
    scl = (sc_ * (1.25 if k < 3 else 1.0), sc_, sc_ * (0.72 if k < 3 else 0.9))
    gz = 1.0 if k < 3 else PAD_TOP
    loc = anchored((x, y, gz), (0, 0, rz), scl, GROUND_ANCHOR)
    o, nfi = vdb_seq(f"SteamVDB{k}", GROUND_X, sc_, loc=tuple(loc), rot=(0, 0, rz), mat=M_STEAM)
    o.scale = scl
    STEAMV.append((o, nfi, IGN + d0, sp_, Vector(loc), Vector((9.0, 1.5 * np.sign(y), 0.8)) if k < 3
                   else Vector((0.0, -1.5, 1.6)), sc_, "ign"))
for k, (x, y, sc_) in enumerate(((118.0, 4.0, 0.42), (-10.0, -30.0, 0.30))):  # first steam: RS-25 start
    loc = anchored((x, y, 1.0 if k == 0 else PAD_TOP), (0, 0, 0.9 * k), sc_, GROUND_ANCHOR)
    o, nfi = vdb_seq(f"SteamEarly{k}", GROUND_X, sc_, loc=tuple(loc), rot=(0, 0, 0.9 * k), mat=M_STEAM_EARLY)
    STEAMV.append((o, nfi, RS25_T + 10 + 6 * k, 0.30, Vector(o.location), Vector((3.0, 0.0, 0.6)), sc_, "early"))
for o, *_ in STEAMV:
    o.data.render.precision = "HALF"
def column_material(name):
    """Exhaust column standing over the pad after liftoff, in a unit cylinder (z 0 ground → 1 at
    the vehicle): pyroclastic edge (two noise octaves, research §4), WIDE at the ground and narrow
    where it leaves the nozzles, soft top so its end never reads as a cap; glows only at the top
    where the plume feeds it (keyed Density / Evolve / Glow)."""
    m, nt = new_mat(name)
    dens = value_node(nt, "Density", 1.0)
    evo = value_node(nt, "Evolve", 0.0)
    glow = value_node(nt, "Glow", 0.0)
    tc = nd(nt, "ShaderNodeTexCoord")
    ob = tc.outputs["Object"]
    sp = nd(nt, "ShaderNodeSeparateXYZ")
    lk(nt, ob, sp.inputs[0])
    h = M(nt, "MINIMUM", M(nt, "MAXIMUM", sp.outputs[2], 0.0), 1.0)
    rr = V(nt, "LENGTH", V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.0)))
    rad = M(nt, "SUBTRACT", 1.0, M(nt, "MULTIPLY", M(nt, "POWER", h, 0.7), 0.72))
    n1 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 0.55, "Detail": 5.0, "Roughness": 0.6})
    lk(nt, V(nt, "MULTIPLY", ob, (1.0, 1.0, 0.35)), n1.inputs["Vector"])
    lk(nt, evo.outputs[0], n1.inputs["W"])
    n2 = nd(nt, "ShaderNodeTexNoise", noise_dimensions="4D", inp={"Scale": 3.2, "Detail": 4.0})
    lk(nt, ob, n2.inputs["Vector"])
    lk(nt, evo.outputs[0], n2.inputs["W"])
    fbm = M(nt, "ADD", M(nt, "SUBTRACT", n1.outputs["Fac"], 0.5), M(nt, "MULTIPLY", M(nt, "SUBTRACT", n2.outputs["Fac"], 0.5), 0.6))
    mask = M(nt, "MULTIPLY", M(nt, "ADD", M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", rr, rad)), M(nt, "MULTIPLY", fbm, 1.6)), 3.5, clamp=True)
    top = M(nt, "SUBTRACT", 1.0, M(nt, "POWER", h, 3.0))
    d = M(nt, "MULTIPLY", M(nt, "MULTIPLY", mask, top), dens.outputs[0])
    pv = nd(nt, "ShaderNodeVolumePrincipled", inp={"Color": (0.90, 0.90, 0.91, 1), "Anisotropy": 0.3,
                                                    "Emission Color": blackbody(1900) + (1,)})
    lk(nt, d, pv.inputs["Density"])
    hot = M(nt, "MULTIPLY", M(nt, "POWER", h, 6.0), top)
    axis = M(nt, "SUBTRACT", 1.0, M(nt, "DIVIDE", rr, M(nt, "MULTIPLY", rad, 0.6)), clamp=True)
    lk(nt, M(nt, "MULTIPLY", M(nt, "MULTIPLY", M(nt, "MULTIPLY", hot, mask), axis), glow.outputs[0]),
       pv.inputs["Emission Strength"])
    o = nd(nt, "ShaderNodeOutputMaterial")
    lk(nt, pv.outputs[0], o.inputs["Volume"])
    return m


M_TRAIL = column_material("Trail")
trail = lathe("TrailPad", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 24, M_TRAIL)
M_HAZE = volume_material("PadHaze", (0.8, 0.85, 0.95), "UNIFORM", nscale=3.0, detail=2.0, aniso=0.6)
haze_box = cube_obj("PadHaze", M_HAZE, loc=(0, 0, 400), scale=(4500, 4500, 400))


# =====================================================================================
# 7. COUNTDOWN CLOCK (7-segment LED board, in front of the A2 camera)
# =====================================================================================
FONT = bpy.data.fonts.load(ARGS.font) if os.path.exists(ARGS.font) else None
CLK = empty("CountdownBoard")
brd = MB()
brd.box((0, 0.06, 0.0), (5.6, 0.12, 2.25))
brd.obj("BoardPanel", principled("BoardBlack", (0.012, 0.012, 0.014), 0.55), CLK)
frm = MB()
for x0, x1, z0, z1 in ((-2.85, 2.85, 1.12, 1.2), (-2.85, 2.85, -1.2, -1.12), (-2.85, -2.77, -1.2, 1.2),
                       (2.77, 2.85, -1.2, 1.2)):
    frm.box(((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (x1 - x0, 0.2, z1 - z0))
for x in (-2.1, 2.1):
    frm.box((x, 0.2, -2.6), (0.18, 0.18, 2.9))
frm.obj("BoardFrame", principled("BoardFrame", (0.35, 0.35, 0.36), 0.35, 1.0), CLK)
SEG_DEF = {"a": (0, 1, 1), "b": (1, 0.5, 0), "c": (1, -0.5, 0), "d": (0, -1, 1), "e": (-1, -0.5, 0),
           "f": (-1, 0.5, 0), "g": (0, 0, 1)}
DIGIT_SEGS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc",
              7: "abc", 8: "abcdefg", 9: "abcdfg"}
DH, DW, ST = 0.95, 0.44, 0.11          # digit height, width, stroke


def seg_poly(cx, cz, horiz, L):
    t = ST / 2
    if horiz:
        pts = [(-L / 2, 0), (-L / 2 + t, t), (L / 2 - t, t), (L / 2, 0), (L / 2 - t, -t), (-L / 2 + t, -t)]
        return [(cx + x, -0.005, cz + z) for x, z in pts]
    pts = [(0, -L / 2), (t, -L / 2 + t), (t, L / 2 - t), (0, L / 2), (-t, L / 2 - t), (-t, -L / 2 + t)]
    return [(cx + x, -0.005, cz + z) for x, z in pts]


LED_SEGS = {}              # (slot, seg) -> object
# v4 fix (Zeke spotted it in v3): the last digit hung past the frame (2.67 + DW/2 > inner edge 2.77).
# Re-spaced so the whole readout (T, sign, HH:MM:SS) sits inside +/-2.6; asserted below.
DIG_X = [-1.10, -0.50, 0.30, 0.90, 1.70, 2.30]
for slot, x0 in enumerate(DIG_X):
    for s, (sx, sz, h) in SEG_DEF.items():
        L = (DW - ST * 0.2) if h else (DH / 2 - ST * 0.2)
        pts = seg_poly(x0 + sx * DW / 2, -0.25 + sz * DH / 2, h, L)
        o = mesh_obj(f"Seg{slot}{s}", pts, [tuple(range(6))[::-1]], M_LED, CLK)
        LED_SEGS[(slot, s)] = o
COLON = []
for x in (0.0 + 0.0, 1.55):
    xx = (DIG_X[1] + DIG_X[2]) / 2 if x == 0 else (DIG_X[3] + DIG_X[4]) / 2
    for z in (-0.25 + 0.2, -0.25 - 0.2):
        c = 0.06
        COLON.append(mesh_obj(f"Colon{len(COLON)}", [(xx - c, -0.005, z - c), (xx + c, -0.005, z - c),
                                                      (xx + c, -0.005, z + c), (xx - c, -0.005, z + c)],
                              [(3, 2, 1, 0)], M_LED, CLK))
SIGN_H = mesh_obj("SignH", seg_poly(-1.78, -0.25, True, 0.44), [tuple(range(6))[::-1]], M_LED, CLK)
SIGN_V = mesh_obj("SignV", seg_poly(-1.78, -0.25, False, 0.40), [tuple(range(6))[::-1]], M_LED, CLK)


def text_obj(name, body, size, loc, mat, parent, align="CENTER", extrude=0.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = extrude
    if FONT is not None:
        cu.font = FONT
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    o.rotation_euler = (math.pi / 2, 0, 0)
    cu.materials.append(mat)
    return link(o, parent)


text_obj("ClockT", "T", 1.15, (-2.36, -0.01, -0.62), emissive("LEDAmber", (1.0, 0.42, 0.05), 9.0), CLK)
text_obj("ClockTitle", ARGS.title.replace("_", " "), 0.62, (0.55, -0.01, 0.62), emissive("LEDWhite", (1.0, 0.95, 0.88), 6.0), CLK)
text_obj("ClockSub", "MISSION", 0.26, (-2.0, -0.01, 0.62), emissive("LEDWhiteDim", (0.9, 0.85, 0.8), 2.5), CLK)
# ASSERT (v4 build gate): every LED segment / colon / sign vertex sits inside the frame, |x| < 2.7 in board space
_seg_x = [abs(v.co.x) for o in list(LED_SEGS.values()) + COLON + [SIGN_H, SIGN_V] for v in o.data.vertices]
assert max(_seg_x) < 2.7, f"countdown board: a segment reaches |x| = {max(_seg_x):.3f} (frame inner edge 2.77)"
bpy.context.view_layer.update()
for _nm in ("ClockT", "ClockTitle", "ClockSub"):
    _o = bpy.data.objects[_nm]
    _xs = [_o.location.x + Vector(c).x * _o.scale.x for c in _o.bound_box]
    assert max(abs(x) for x in _xs) < 2.7, f"countdown board: text {_nm} reaches |x| = {max(abs(x) for x in _xs):.3f}"
print(f"[rocket4] ASSERT countdown board: max segment |x| = {max(_seg_x):.3f} < 2.7 (text inside too)")


# =====================================================================================
# 8. WORLD: 5.x multiple-scattering sky + the Earth as a ray-sphere hit in the world shader
#    textured with NASA Blue Marble NG (day), Black Marble 2016 (night lights) and the Blue
#    Marble cloud layer (research fix #9), KSC under the pad, east = downrange (+Y). Stars.
# =====================================================================================
WORLD = bpy.data.worlds.new("World")
SC.world = WORLD
wn = WORLD.node_tree
wn.nodes.clear()
sky = nd(wn, "ShaderNodeTexSky", name="Sky", sky_type="MULTIPLE_SCATTERING")
sky.sun_rotation = math.radians(0.0)
sky.ground_albedo = 0.3
sky.sun_size = math.radians(0.545)
w_gain = value_node(wn, "SkyGain", 1.0)
w_h = value_node(wn, "EarthH", 1.0157)          # (R+h)/R
w_ov = value_node(wn, "EarthOverlay", 0.0)
w_scroll = value_node(wn, "EarthScroll", 0.0)
w_stars = value_node(wn, "Stars", 1.0)
tcw = nd(wn, "ShaderNodeTexCoord")
dvec = V(wn, "NORMALIZE", tcw.outputs["Generated"])
dsep = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, dvec, dsep.inputs[0])
dz = dsep.outputs[2]
H = w_h.outputs[0]
d2 = M(wn, "MULTIPLY", M(wn, "MULTIPLY", H, H), M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", dz, dz)))
hit = M(wn, "MULTIPLY", M(wn, "LESS_THAN", d2, 1.0), M(wn, "LESS_THAN", dz, 0.0))
thc = M(wn, "SQRT", M(wn, "MAXIMUM", M(wn, "SUBTRACT", 1.0, d2), 0.0))
tt = M(wn, "SUBTRACT", M(wn, "MULTIPLY", M(wn, "MULTIPLY", H, dz), -1.0), thc)
hitp = V(wn, "SCALE", dvec, scale=tt)
hc = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, H, hc.inputs[2])
nrm = V(wn, "ADD", hitp, hc.outputs[0])                          # unit normal at the hit (world)
rot = nd(wn, "ShaderNodeVectorRotate", rotation_type="X_AXIS")   # downrange travel (keyed)
lk(wn, nrm, rot.inputs["Vector"])
lk(wn, w_scroll.outputs[0], rot.inputs["Angle"])
en = rot.outputs[0]
# world normal → Earth-fixed direction: up=+Z → KSC radial, +Y → east, -X → north
LAT0, LON0 = math.radians(28.6), math.radians(-80.6)
UP_E = (math.cos(LAT0) * math.cos(LON0), math.cos(LAT0) * math.sin(LON0), math.sin(LAT0))
EAST_E = (-math.sin(LON0), math.cos(LON0), 0.0)
NORTH_E = (-math.sin(LAT0) * math.cos(LON0), -math.sin(LAT0) * math.sin(LON0), math.cos(LAT0))
ens = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, en, ens.inputs[0])
ece = nd(wn, "ShaderNodeCombineXYZ")
for k in range(3):
    acc = M(wn, "ADD", M(wn, "MULTIPLY", ens.outputs[2], UP_E[k]), M(wn, "MULTIPLY", ens.outputs[1], EAST_E[k]))
    lk(wn, M(wn, "ADD", acc, M(wn, "MULTIPLY", ens.outputs[0], -NORTH_E[k])), ece.inputs[k])
es = nd(wn, "ShaderNodeSeparateXYZ")
lk(wn, ece.outputs[0], es.inputs[0])
lon = M(wn, "ARCTAN2", es.outputs[1], es.outputs[0])
lat = M(wn, "ARCSINE", M(wn, "MINIMUM", M(wn, "MAXIMUM", es.outputs[2], -1.0), 1.0))
uv = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, M(wn, "ADD", M(wn, "DIVIDE", lon, 2 * math.pi), 0.5), uv.inputs[0])
lk(wn, M(wn, "ADD", M(wn, "DIVIDE", lat, math.pi), 0.5), uv.inputs[1])
EARTH_D = os.path.join(AS, "nasa_earth", "derived")


def earth_tex(fn, data=False):
    t = nd(wn, "ShaderNodeTexImage", image=load_img(os.path.join(EARTH_D, fn), data), interpolation="Cubic",
           extension="REPEAT")
    lk(wn, uv.outputs[0], t.inputs["Vector"])
    return t


day = earth_tex("bluemarble_5400.jpg")
night = earth_tex("blackmarble_8192.jpg")
cloudt = earth_tex("clouds_8192.jpg", data=True)
cbw = nd(wn, "ShaderNodeRGBToBW")
lk(wn, cloudt.outputs["Color"], cbw.inputs[0])
cloudm = M(wn, "POWER", cbw.outputs[0], 1.4)
earth_alb = mix_col(wn, cloudm, day.outputs["Color"], (0.80, 0.80, 0.82))
skyc = sky.outputs["Color"]
mu = M(wn, "MULTIPLY", V(wn, "DOT_PRODUCT", nrm, dvec), -1.0)
limb = M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", mu, 0.0), 0.6), 1.0, clamp=True)
wgt = M(wn, "MULTIPLY", M(wn, "MULTIPLY", hit, limb), w_ov.outputs[0])
facc = col_mul(wn, earth_alb, 1.0 / 0.30)                           # relative to the sky's ground albedo
fac_mix = mix_col(wn, wgt, (1.0, 1.0, 1.0), facc)
earth_col = col_mul(wn, skyc, fac_mix)
lum = nd(wn, "ShaderNodeRGBToBW")
lk(wn, skyc, lum.inputs[0])
nightf = M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", lum.outputs[0], 40.0), clamp=True)
nbw = nd(wn, "ShaderNodeRGBToBW")
lk(wn, night.outputs["Color"], nbw.inputs[0])
lights = M(wn, "POWER", M(wn, "SUBTRACT", nbw.outputs[0], 0.12, clamp=True), 1.6)
w_city = value_node(wn, "CityGain", 1.0)            # v4: the night side's city lights breathe with the bass
cl_st = M(wn, "MULTIPLY", M(wn, "MULTIPLY", M(wn, "MULTIPLY", lights, nightf), wgt),
          M(wn, "MULTIPLY", M(wn, "MULTIPLY", M(wn, "SUBTRACT", 1.0, cloudm), 0.5), w_city.outputs[0]))
addc = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, cl_st, addc.inputs[0])
lk(wn, earth_col, addc.inputs[6])
addc.inputs[7].default_value = (1.0, 0.62, 0.28, 1)
earth_col = addc.outputs[2]
# stars (only where the sky is dark and the planet isn't)
sv = nd(wn, "ShaderNodeTexVoronoi", inp={"Scale": 420.0, "Randomness": 1.0})
lk(wn, dvec, sv.inputs["Vector"])
sb = nd(wn, "ShaderNodeSeparateColor")
lk(wn, sv.outputs["Color"], sb.inputs[0])
star = M(wn, "POWER", M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", sv.outputs["Distance"], 9.0), clamp=True), 6.0)
star = M(wn, "MULTIPLY", star, M(wn, "POWER", sb.outputs[0], 6.0))
skydark = M(wn, "SUBTRACT", 1.0, M(wn, "MULTIPLY", lum.outputs[0], 12.0), clamp=True)
star = M(wn, "MULTIPLY", M(wn, "MULTIPLY", star, skydark), M(wn, "SUBTRACT", 1.0, hit))
star = M(wn, "MULTIPLY", M(wn, "MULTIPLY", star, w_stars.outputs[0]), 0.35)
# v4: an aurora on the horizon — a thin shell 100-300 km up seen edge-on just above the limb, hung as
# curtains (noise over the azimuth), green with a violet top, gain keyed to the bass (space sets only)
w_aur = value_node(wn, "AuroraGain", 0.0)
w_aurt = value_node(wn, "AuroraT", 0.0)
dzl = M(wn, "SQRT", M(wn, "MAXIMUM", M(wn, "SUBTRACT", 1.0, M(wn, "DIVIDE", 1.0, M(wn, "MULTIPLY", H, H))), 0.0))
elev = M(wn, "ADD", dz, dzl)                       # ~ angle above the limb (0 = grazing the horizon)
aband = M(wn, "SUBTRACT", 1.0, M(wn, "DIVIDE", M(wn, "ABSOLUTE", M(wn, "SUBTRACT", elev, 0.012)), 0.012), clamp=True)
aztop = M(wn, "SUBTRACT", 1.0, M(wn, "DIVIDE", M(wn, "ABSOLUTE", M(wn, "SUBTRACT", elev, 0.03)), 0.008), clamp=True)
azi = M(wn, "ARCTAN2", dsep.outputs[1], dsep.outputs[0])
acn = nd(wn, "ShaderNodeTexNoise", noise_dimensions="2D", inp={"Scale": 1.0, "Detail": 4.0, "Roughness": 0.55})
acv = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, M(wn, "MULTIPLY", azi, 6.0), acv.inputs[0])
lk(wn, w_aurt.outputs[0], acv.inputs[1])
lk(wn, acv.outputs[0], acn.inputs["Vector"])
curt = M(wn, "POWER", M(wn, "MULTIPLY_ADD", acn.outputs["Fac"], 1.6, -0.45, clamp=True), 1.5)
aur = M(wn, "MULTIPLY", M(wn, "MULTIPLY", aband, curt), w_aur.outputs[0])
aurt = M(wn, "MULTIPLY", M(wn, "MULTIPLY", aztop, curt), M(wn, "MULTIPLY", w_aur.outputs[0], 0.45))
adda = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, aur, adda.inputs[0])
lk(wn, earth_col, adda.inputs[6])
adda.inputs[7].default_value = (0.10, 1.0, 0.35, 1)
addb = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, aurt, addb.inputs[0])
lk(wn, adda.outputs[2], addb.inputs[6])
addb.inputs[7].default_value = (0.55, 0.12, 1.0, 1)
earth_col = addb.outputs[2]
adds = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, star, adds.inputs[0])
lk(wn, earth_col, adds.inputs[6])
adds.inputs[7].default_value = (0.9, 0.93, 1.0, 1)
w_sunel = value_node(wn, "SunEl", 0.0)
w_sung = value_node(wn, "SunDisc", 0.0)
sdir = nd(wn, "ShaderNodeCombineXYZ")
lk(wn, M(wn, "COSINE", w_sunel.outputs[0]), sdir.inputs[1])
lk(wn, M(wn, "SINE", w_sunel.outputs[0]), sdir.inputs[2])
cosang = V(wn, "DOT_PRODUCT", dvec, sdir.outputs[0])
disc = nd(wn, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP",
          inp={"From Min": math.cos(math.radians(0.33)), "From Max": math.cos(math.radians(0.24))})
lk(wn, cosang, disc.inputs["Value"])
cor = M(wn, "ADD", M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", cosang, 0.0), 900.0), 3.0),
        M(wn, "MULTIPLY", M(wn, "POWER", M(wn, "MAXIMUM", cosang, 0.0), 45.0), 0.12))
sun_e = M(wn, "MULTIPLY", M(wn, "ADD", M(wn, "MULTIPLY", disc.outputs["Result"], 400.0), cor),
          M(wn, "MULTIPLY", w_sung.outputs[0], M(wn, "SUBTRACT", 1.0, hit)))
addsun = nd(wn, "ShaderNodeMix", data_type="RGBA", blend_type="ADD")
lk(wn, sun_e, addsun.inputs[0])
lk(wn, adds.outputs[2], addsun.inputs[6])
addsun.inputs[7].default_value = (1.0, 0.93, 0.82, 1)
bgw = nd(wn, "ShaderNodeBackground")
lk(wn, addsun.outputs[2], bgw.inputs["Color"])
lk(wn, w_gain.outputs[0], bgw.inputs["Strength"])
wo = nd(wn, "ShaderNodeOutputWorld")
lk(wn, bgw.outputs[0], wo.inputs["Surface"])

SUN = bpy.data.lights.new("Sun", "SUN")
SUN.angle = math.radians(0.53)
SUN.color = (1.0, 0.80, 0.62)
SUN_OB = link(bpy.data.objects.new("Sun", SUN))
MOON = bpy.data.lights.new("Moonfill", "SUN")
MOON.angle = math.radians(2.0)
MOON.color = (0.55, 0.65, 1.0)
MOON_OB = link(bpy.data.objects.new("Moonfill", MOON))
MOON_OB.rotation_euler = (math.radians(55), 0, math.radians(-150))

# --- the cumulus deck (research fix #10): JangaFX CC0 cloud VDBs, clustered, lit by the low
#     sun and from inside by the plume. Parented to a keyed 'Deck' empty that sinks past the climb.
DECK = empty("CloudDeck")
M_CLOUD = smoke_material("Cumulus", (0.96, 0.96, 0.97), dens=0.45, aniso=0.55, glow_grid=None, plume_lit=False)
_cfiles = sorted(glob.glob(os.path.join(CLOUDS, "*.vdb")))
_crng = np.random.default_rng(31)
CLOUD_OBS = []
_cvols = []
for f in _cfiles:
    v = bpy.data.volumes.new(os.path.basename(f))
    v.filepath = f
    v.materials.append(M_CLOUD)
    v.render.precision = "HALF"
    _cvols.append(v)
_slots = []
while len(_slots) < 22:
    x, y = _crng.uniform(-3800, 3800), _crng.uniform(-3000, 4200)
    if math.hypot(x, y) < 330:                      # keep a clear shaft around the climb line
        continue
    if all(math.hypot(x - a, y - b) > 650 for a, b in _slots):
        _slots.append((x, y))
for k, (x, y) in enumerate(_slots):
    v = _cvols[k % len(_cvols)]
    o = bpy.data.objects.new(f"Cloud{k}", v)
    s_ = float(_crng.uniform(1.4, 2.4))
    o.scale = (s_, s_, s_ * float(_crng.uniform(0.75, 1.0)))
    o.rotation_euler = (0, 0, float(_crng.uniform(0, 2 * math.pi)))
    o.location = (x, y, float(_crng.uniform(-60, 60)))
    link(o, DECK)
    CLOUD_OBS.append(o)
M_DECKHAZE = volume_material("DeckHaze", (0.92, 0.93, 0.96), "SLAB", nscale=2.6, detail=7.0,
                             emis_color=(1.0, 0.45, 0.15), aniso=0.5)
deck_haze = cube_obj("DeckHaze", M_DECKHAZE, loc=(0, 0, -260), scale=(9000, 9000, 120), parent=DECK)
M_TRAILF = volume_material("TrailFlight", (0.86, 0.86, 0.88), "COLUMN", nscale=0.5, detail=6.0,
                           emis_color=blackbody(1900), aniso=0.3, sharp=5.0, namp=1.0)
trail_f = lathe("TrailFlight", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 24, M_TRAILF)


# =====================================================================================
# 9. CAMERA + SHOTS
# =====================================================================================
CAM_D = bpy.data.cameras.new("Cam")
CAM_D.sensor_width = 36.0
CAM_D.clip_start = 0.3
CAM_D.clip_end = 60000.0
CAM_D.dof.use_dof = True
CAM_D.dof.aperture_blades = 7
CAM = link(bpy.data.objects.new("Cam", CAM_D))
SC.camera = CAM

PITCH = np.zeros(N)
PITCH[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [7, 18, 26])
PITCH[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [30, 52, 66])
PITCH[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [74, 82])
PITCH[(SET == 0)] = np.clip((ALT[SET == 0] - 150) / 300, 0, 1) * 3.0          # roll/pitch program
RIG_POS = np.zeros((N, 3))
RIG_POS[:, 2] = np.where(SET == 0, PAD_Z + ALT, 0.0)
RIG_POS[:, 0] += 0.04 * KICK * np.maximum(on_core, on_boost)
# tower clear: a tiny lateral drift away from the tower as it rises (SLS does a slight yaw)
RIG_POS[:, 1] += np.where(SET == 0, -np.clip(ALT - 20, 0, None) * 0.03, 0.0)


# --- v4 FLIGHT MODEL (flight_model.py): booster 3 sputters after the alarm, separates early, keeps burning, then
#     explodes; the TVC-limited attitude response (saturation, overshoot, damped swings), the sag + drift of the
#     trajectory, the gimbal angles and the RCS roll jets — all integrated at 10 substeps/frame, baked below.
COM_Z = 40.0                                         # centre of mass above station 0 (the wobble pivot)
B3FRAC = FM.b3_thrust_frac(N, B3ALARM, B3SEP, SNARE4)
_az3 = SLOT_AZ[2]
_att0 = FM.attitude(N, FPS, PAD_END, BSEP, LIFT, B3SEP, B3BOOM, B3FRAC, (-math.cos(_az3), -math.sin(_az3)))
B3_LOC, B3_ANG, B3_AXIS = FM.booster3_path(N, FPS, LIFT, B3SEP, B3BOOM, _att0["tw"] + 2.0)
B3_MID = B3_LOC + B3_AXIS * 22.0                     # the case bursts at mid-body
BOOM_P = Vector(B3_MID[B3BOOM])                      # explosion centre (vehicle / rig frame)
_bd = -np.array(BOOM_P[:2]) / max(np.linalg.norm(BOOM_P[:2]), 1e-6)
ATT = FM.attitude(N, FPS, PAD_END, BSEP, LIFT, B3SEP, B3BOOM, B3FRAC, _bd, blast_below=COM_Z - BOOM_P.z)
_b3_off = math.hypot(BOOM_P.x, BOOM_P.y)
_tilt = np.degrees(np.linalg.norm(ATT["phi"], axis=1))
_pk = int(np.argmax(_tilt))
print(f"[rocket4] FLIGHT: liftoff a0 {(3 * FM.V.srb_T + 4 * FM.V.rs25_T) / FM.V.m0 - 9.81:.2f} m/s^2, alt at "
      f"PAD_END {ALT[PAD_END]:.0f} m; booster 3 bursts {_b3_off:.0f} m off the core axis (z {BOOM_P.z:.0f} m); "
      f"peak tilt {_tilt[_pk]:.1f} deg at {_pk / FPS:.2f} s; tilt at 65.6 s {_tilt[int(65.6 * FPS)]:.1f} deg; "
      f"max gimbal {np.degrees(np.abs(ATT['gimbal'])).max():.2f} deg; drift max {np.abs(ATT['pos']).max(0).round(0)}")
assert 50.0 <= _b3_off <= 80.0, f"booster 3 must burst 50-80 m from the core (got {_b3_off:.1f})"
assert 25.0 <= _tilt[_pk] <= 35.0, f"tumble overshoot must be 25-35 deg (got {_tilt[_pk]:.1f})"
assert _tilt[int(66.0 * FPS)] < 5.0, f"must be recovered (<5 deg) by ~1:06 (got {_tilt[int(66.0 * FPS)]:.1f})"
assert np.degrees(np.abs(ATT["gimbal"])).max() <= 8.0 + 1e-6
# guidance reels the trajectory back onto the (story-paced) nominal before the 1:28 booster sep
WFADE = 1.0 - smoothstep(ramp(IDX, REBOOT, BSEP - BAR))
WOB_PHI = ATT["phi"] * WFADE[:, None]
WOB_ROLL = ATT["roll"] * WFADE
WOB_POS = ATT["pos"] * WFADE[:, None]
GIMBAL = ATT["gimbal"] * WFADE[:, None]
DRIFT_LP = np.zeros_like(WOB_POS)                    # what a chase pilot / tracking mount follows: lagged
_a = 1.0 - math.exp(-1.0 / (0.8 * FPS))
for i in range(1, N):
    DRIFT_LP[i] = DRIFT_LP[i - 1] + _a * (WOB_POS[i] - DRIFT_LP[i - 1])
_P = Vector((0.0, 0.0, COM_Z))


def wob_matrix(i):
    from mathutils import Euler
    Rm = Euler((WOB_PHI[i, 0], WOB_PHI[i, 1], WOB_ROLL[i]), "XYZ").to_matrix().to_4x4()
    return Matrix.Translation(Vector(WOB_POS[i])) @ Matrix.Translation(_P) @ Rm @ Matrix.Translation(-_P)


def rig_matrix(i):
    return Matrix.Translation(Vector(RIG_POS[i])) @ Matrix.Rotation(math.radians(-PITCH[i]), 4, "X")


def wl(i, p):
    """rocket-local → world, for CHASE / tracking cameras: follows the rig plus the LAGGED drift of the tumble,
    never the wobble rotation (so the tumble reads on screen)."""
    return rig_matrix(i) @ (Vector(p) + Vector(DRIFT_LP[i]))


def wl_v(i, p):
    """rocket-local → world for things BOLTED to the stack (onboard cameras): rig @ wobble."""
    return rig_matrix(i) @ wob_matrix(i) @ Vector(p)


def wl_air(i, p):
    """the rig (air) frame without any drift: where booster 3 and its explosion live."""
    return rig_matrix(i) @ Vector(p)


def wl_lag(i, p, lag=6, salt=0, wobble=0.0):
    """Tracking-mount target: the vehicle point as it was `lag` frames ago, plus a slow operator
    wobble (metres at the subject) — real tracking cameras trail the vehicle slightly."""
    j = max(i - lag, 0)
    w = Vector((math.sin(i * 0.071 + salt) + 0.4 * math.sin(i * 0.173 + 2 * salt),
                0.0, math.sin(i * 0.053 + 1.3 + salt) + 0.4 * math.sin(i * 0.191 + salt)))
    return wl(j, p) + w * wobble


def shake_vec(i, amp, salt=0, hf=False):
    """Kick jolts (hashed per kick) + rumble (sum of sines; hf=True: onboard vibration)."""
    k = KIDX[i]
    j = Vector((hash01(k, 1, salt) - 0.5, hash01(k, 2, salt) - 0.5, hash01(k, 3, salt) - 0.5)) * 2 * KICK[i]
    x = i + 977 * salt
    f = 3.7 if hf else 1.0
    r = Vector((math.sin(x * 1.71 * f + 0.3) + 0.5 * math.sin(x * 3.13 * f), math.sin(x * 1.37 * f + 1.1) +
                0.5 * math.sin(x * 2.89 * f + 2.0), math.sin(x * 1.93 * f + 2.2)))
    return (j * 1.0 + r * (0.18 if not hf else 0.45)) * amp


RAVE_SHAKE = np.zeros(N)
# --- v3 RAVE world plan. After booster sep the rig sits at the world origin (only its pitch moves), so
#     the speaker's station, the asteroid and the sound waves are planned in fixed world space.
if RAVE:
    SPK_SC0, SPK_SC1 = 0.30, 2.40                        # stowed in the fairing → gigantic (38 m tall)
    SPK_Z0 = 79.4                                        # stowed centre, rocket-local (SAJ panel bay)
    SPK_HALF = Vector((3.33, 3.36, 8.0))                 # v4 Sousinho cabinet half-extents at scale 1 (front = −Y)
    SPK_TOP = 1.75                                       # moving heads stand this high on the lid (scale-1 units)
    DEP0 = FAIRSEP + int(2.0 * FPS)                      # rises once the panels are clear
    DEP1 = snap(DEP0 + 6 * BAR)                          # fully grown, on station
    D_HAT = Vector((0.16, 1.0, 0.10)).normalized()       # where the threat comes from
    SIDE = D_HAT.cross(Vector((0.0, 0.0, 1.0))).normalized()
    UPV = SIDE.cross(D_HAT).normalized()
    SPK_P = wl(DEP1, (0.0, 0.0, 175.0))
    Q_FACE = D_HAT.to_track_quat("-Y", "Z")              # cabinet front (−Y) → the threat, top up
    KF = np.nonzero(TL["T_kage"] == 0)[0]                # every kick onset (frame index)
    AST_R = 110.0
    A_STOP = SPK_P + D_HAT * 960.0                       # where the sound pins it
    A_START = A_STOP + D_HAT * 11000.0 + UPV * 2600.0 + SIDE * 900.0
    V_IN = (A_STOP - A_START).normalized()
    A_HIT = A_STOP - V_IN * 160.0                        # still moving when the first wave lands
    WAVE_V = 720.0 / FPS                                 # sound-wave rings, metres per frame
    FACE_OFF = SPK_HALF.y * SPK_SC1 + 3.0                # woofer rims sit proud of the cabinet
    T_HIT = int(round((960.0 - FACE_OFF - AST_R * 0.8) / WAVE_V))
    WAVE_K = KF[(KF >= snap(150.2 * FPS)) & (KF < FADE)]  # test pulses on the wind-up, then the drop
    IMP = np.zeros(N)                                    # wave impacts on the rock
    for kf in WAVE_K[(WAVE_K >= FIRE) & (WAVE_K < BOOM)]:
        IMP += np.exp(-(((IDX - (kf + T_HIT)) / 2.5) ** 2))
    IMP = np.clip(IMP, 0.0, 1.5)
    FIREWIN = ((IDX >= FIRE) & (IDX < AFTER)).astype(np.float64)
    RAVE_SHAKE = FIREWIN * (0.9 * KICK + 0.7 * np.clip(IMP, 0, 1)) + (IDX >= FINAL) * 0.9 * KICK
    RAVE_SHAKE = RAVE_SHAKE + (IDX >= BOOM) * 2.5 * np.exp(-np.maximum(IDX - BOOM, 0) / 10.0)
    # v4: booster 3's blast (impulse + rumble) and the saturated-TVC shudder while the stack fights back
    _tb3 = np.maximum(IDX - B3BOOM, 0) / FPS
    RAVE_SHAKE = RAVE_SHAKE + (IDX >= B3BOOM) * (2.5 * np.exp(-_tb3 / 0.35) + 0.6 * np.exp(-_tb3 / 3.0)) \
        + 0.35 * ATT["sat"] + 0.25 * (B3FRAC < 0.9) * (IDX < B3SEP) * (IDX >= B3ALARM)
    spk_loc = np.zeros((N, 3))
    spk_rot = np.zeros((N, 3))
    spk_scl = np.zeros(N)
    _prev = None
    for i in range(N):
        l0, q0, _s0 = (rig_matrix(i) @ Matrix.Translation((0.0, 0.0, SPK_Z0))).decompose()
        u = float(smoothstep(ramp(i, DEP0, DEP1)))
        loc = l0.lerp(SPK_P, u)
        q = q0.slerp(Q_FACE, u)
        # it rides each kick on station (recoil back from the waves), harder while firing
        loc = loc - D_HAT * (u * (0.6 + 2.2 * FIREWIN[i]) * KICK[i])
        e = q.to_euler("XYZ", _prev) if _prev is not None else q.to_euler("XYZ")
        _prev = e
        spk_loc[i] = loc
        spk_rot[i] = (e.x, e.y, e.z)
        spk_scl[i] = SPK_SC0 + (SPK_SC1 - SPK_SC0) * float(ease_out(ramp(i, DEP0 + 10, DEP1), 2.2))
    ast_pos = np.zeros((N, 3))
    AST_SPEED = (A_HIT - A_START).length / ((FIRE - AST0) / FPS)       # ~750 m/s on the run-in
    for i in range(N):
        if i < AST0:                                                   # v4: the glimpse — same line, further out
            a = A_START - V_IN * (AST_SPEED * (AST0 - i) / FPS)
        elif i < FIRE:
            a = A_START.lerp(A_HIT, float(ramp(i, AST0, FIRE)))
        else:
            a = A_HIT + V_IN * (160.0 * (1.0 - math.exp(-(i - FIRE) / 10.0)))
        jit = Vector((hash01(i, 11) - 0.5, hash01(i, 12) - 0.5, hash01(i, 13) - 0.5)) * 2.0
        ast_pos[i] = a + jit * (7.0 * min(IMP[i], 1.0))
    print("[rocket3] speaker DEP0", DEP0, "DEP1", DEP1, "station", tuple(round(v) for v in SPK_P),
          "T_HIT", T_HIT, "waves", len(WAVE_K))


SHOT_FN = {}


def shot(name):
    def deco(fn):
        SHOT_FN[name] = fn
        return fn
    return deco


STACK_C = PAD_Z + 50.0                             # mid-stack height on the pad


def ground_z(x, y):
    return float(max(height(np.array([x]), np.array([y]))[0][0], 0.0))


LAGOON_CAM = CAM_DIR * 3250.0
LAGOON_CAM.z = ground_z(LAGOON_CAM.x, LAGOON_CAM.y) + 1.8
CLOCK_CAM = Vector((-250.0, -560.0, ground_z(-250.0, -560.0) + 1.7))


@shot("A1_wide")             # 400 mm across the lagoon from 3.2 km: compression, haze, shimmer
def _(i, u, n):
    t = u / n
    s = float(smoothstep(t))
    cam = LAGOON_CAM + Vector((14.0 * s, 0.0, 0.0))
    tgt = Vector((0.0, 4.0, STACK_C + 6.0 - 4 * s))
    return dict(cam=cam, tgt=tgt + Vector((0.6 * math.sin(i * 0.031), 0, 0.4 * math.sin(i * 0.047))),
                lens=380 + 70 * s, fstop=8, shake=0.0, haze=1.0)


@shot("A2_clock")            # press-site countdown clock, rack focus to the pad
def _(i, u, n):
    t = u / n
    cam = CLOCK_CAM + Vector((-2.0 * t, 1.0 * t, 0.15 * t))
    tgt = Vector((10.0, 0.0, STACK_C + 2 - 6 * t))
    rack = float(smoothstep((t - 0.45) / 0.3))
    fdist = 13.5 * (1 - rack) + (Vector((0, 0, STACK_C)) - cam).length * rack
    return dict(cam=cam, tgt=tgt, lens=35, fstop=2.8, focus=fdist, shake=0.0, haze=0.4)


@shot("A3_logo")             # 300 mm tilt up the core to the emblem; tower lattice behind
def _(i, u, n):
    t = u / n
    s = float(smoothstep(t))
    cam = Vector((60.0 - 10 * s, -540.0, 16.0))
    tgt = Vector((0.0, 0.0, PAD_Z + 8.0 + (LOGO_Z0 + LOGO_H * 0.5 - 8.0) * s))
    return dict(cam=cam, tgt=tgt, lens=300, fstop=5.6, shake=0.0, haze=0.6)


@shot("A4_low")              # low remote on the hardstand: deluge, sparks, engine start
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    cam = Vector((44.0 - 5 * t - 4 * creep, -52.0 + 6 * t + 4 * creep, PAD_TOP + 1.1))
    tgt = Vector((-2.0, 0.0, PAD_Z + 30 + 6 * t))
    return dict(cam=cam, tgt=tgt, lens=17 + 2 * creep, fstop=8, shake=0.05 * PRE[i] + 0.4 * on_core[i])


@shot("B1_ign")              # 1.3 km due south, 135 mm: the trench blast rolls to the right
def _(i, u, n):
    t = u / n
    cam = Vector((25.0, -1300.0, 6.0))
    tgt = Vector((70.0, 0.0, STACK_C + 12 + ALT[i] * 0.55))
    return dict(cam=cam, tgt=tgt, lens=135 - 20 * t, fstop=8, shake=0.6, haze=1.0)


@shot("B2_tower")            # tower remote at mid height looking UP as the stack clears the tower
def _(i, u, n):
    t = u / n
    cam = Vector((-7.5, 9.5, ML_DECK + 62.0))
    tgt = wl(i, (0.0, 0.0, 60.0))
    tgt.z = max(tgt.z, cam.z + 20.0)
    return dict(cam=cam, tgt=tgt, lens=16, fstop=8, shake=0.7)


@shot("B3_track")            # 5 km tracking mount, 800 mm, trailing the vehicle
def _(i, u, n):
    cam = Vector((-1500.0, -4800.0, 6.0))
    tgt = wl_lag(i, (0, 0, 30.0), lag=5, wobble=2.5)
    return dict(cam=cam, tgt=tgt, lens=800, fstop=11, shake=0.15, haze=1.6)


@shot("B4_tail")             # onboard: booster-mounted camera looking aft at the pad
def _(i, u, n):
    cam = wl(i, (-SRB_X - 2.2, -1.0, 51.0))
    tgt = wl(i, (-SRB_X - 1.2, -2.6, -40.0))
    return dict(cam=cam, tgt=tgt, lens=11, fstop=11, shake=0.8, up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))


DECK_TOP = np.full(N, -5000.0)
_c = (IDX >= C_CUTS[0]) & (IDX < MAXQ)
DECK_TOP[_c] = 120.0 - (IDX[_c] - C_CUTS[0]) / FPS * 34.0
_d = (IDX >= MAXQ) & (IDX < BSEP)
DECK_TOP[_d] = -1100.0 - (IDX[_d] - MAXQ) / FPS * 15.0


@shot("C1_cloud")            # air-to-air long lens: the vehicle climbing out of the deck
def _(i, u, n):
    t = u / n
    cam = Vector((1100.0, -2500.0, DECK_TOP[i] + 140.0 + 90 * t))
    tgt = wl_lag(i, (0, 0, 10.0), lag=4, wobble=1.5)
    return dict(cam=cam, tgt=tgt, lens=420, fstop=8, shake=0.1, haze=0.8)


@shot("C2_side")             # chase angle (exposure pumping on the kicks sells it as footage)
def _(i, u, n):
    t = u / n
    creep = float(PRE[i])
    a = math.radians(-70 + 25 * t)
    d = 150 - 40 * creep
    cam = wl(i, (d * math.cos(a), d * math.sin(a), 30 - 10 * t))
    tgt = wl(i, (0, 0, 45.0))
    return dict(cam=cam, tgt=tgt, lens=55, fstop=8, shake=0.08 + 0.1 * creep)


@shot("D1_maxq")             # close chase in profile: the condensation collar wrapping the upper stack
def _(i, u, n):
    t = u / n
    cam = wl(i, (16.0 - 5 * t, -34.0, OGIVE_Z0 - 8.0 - 6 * t))
    tgt = wl(i, (0.0, 0.0, LVSA_TOP - 4.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=5.6, shake=1.0)


@shot("D2_under")            # onboard: looking aft past the booster to the plume and clouds
def _(i, u, n):
    cam = wl(i, (2.4, -R - 0.6, 56.0))
    tgt = wl(i, (SRB_X - 1.0, -2.0, -60.0))
    return dict(cam=cam, tgt=tgt, lens=12, fstop=11, shake=0.9, up=wl(i, (0, 1, 0)) - wl(i, (0, 0, 0)))


@shot("D3_wide")             # long-lens air-to-air: the condensation collar
def _(i, u, n):
    t = u / n
    cam = wl(i, (700.0, -1250.0, 260.0 - 90 * t))
    tgt = wl_lag(i, (0, 0, 55.0), lag=3, wobble=0.8)
    return dict(cam=cam, tgt=tgt, lens=240, fstop=11, shake=0.3, haze=0.5)


@shot("E_calm")
def _(i, u, n):
    t = u / n
    cam = wl(i, (80.0, -110.0, -20.0 - 10 * t))
    tgt = wl(i, (0, 0, 20.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=11, shake=0.08)


def booster_pos(i, side):
    """v4: side +1 = BoosterR (slot +60 deg), -1 = BoosterL (slot 180 deg); radially out in its slot."""
    tb_ = max(i - BSEP, 0) / FPS
    az = SLOT_AZ[1] if side > 0 else SLOT_AZ[0]
    r_ = SRB_X + 2.2 * tb_ + 0.5 * tb_ ** 2
    return (r_ * math.cos(az), r_ * math.sin(az), -(3.0 * tb_ ** 2) - 2.0 * tb_)


@shot("F1_sep")              # onboard on the core: the right booster peels away
def _(i, u, n):
    _a = SLOT_AZ[1] - math.radians(55)                # onboard on the core, beside BoosterR's slot
    cam = wl(i, ((R + 0.6) * math.cos(_a), (R + 0.6) * math.sin(_a), 47.0))
    b = booster_pos(i, 1)
    tgt = wl(i, (b[0], b[1], b[2] + 30.0))
    return dict(cam=cam, tgt=tgt, lens=13, fstop=11, shake=0.9, up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))


@shot("F2_tail")
def _(i, u, n):
    b = booster_pos(i, -1)
    cam = wl(i, (-9.0, -18.0, 40.0))
    tgt = wl(i, (b[0], 0, b[2] + 25)) * 0.8 + wl(i, (0, 0, 0)) * 0.2
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.35)


@shot("F3_wide")             # far long lens: two bright boosters falling away, the plume balloons
def _(i, u, n):
    t = u / n
    cam = wl(i, (1400.0, -2500.0, -500.0 + 120 * t))
    tgt = wl_lag(i, (0, 0, -40.0), lag=3, wobble=2.0)
    return dict(cam=cam, tgt=tgt, lens=180, fstop=11, shake=0.25)


@shot("G_calm")              # the sunlit 'jellyfish' over a dark Earth
def _(i, u, n):
    t = u / n
    cam = wl(i, (2600.0, -3600.0, -2200.0 - 400 * t))
    tgt = wl(i, (0, 0, -700.0))
    return dict(cam=cam, tgt=tgt, lens=42, fstop=11, shake=0.04)


@shot("H1_fair")             # launch-abort-tower jettison, close
def _(i, u, n):
    t = u / n
    cam = wl(i, (10.0, -20.0 - 6 * t, OGIVE_Z0 + 6.0))
    tgt = wl(i, (0, 0, OGIVE_Z0 + 2.0 + 6 * t))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=8, shake=0.7)


@shot("H2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (34.0, -52.0 + 8 * t, SAJ_Z0 - 4))
    tgt = wl(i, (0, 0, SAJ_Z0 - 2))
    return dict(cam=cam, tgt=tgt, lens=35, fstop=11, shake=0.5)


@shot("I_meco")              # onboard (upper stage) looking aft: the core stage falls away
def _(i, u, n):
    cam = wl(i, (2.9, -2.9, LVSA_TOP + 1.5))
    tgt = wl(i, (0.0, 0.0, 10.0))
    return dict(cam=cam, tgt=tgt, lens=12, fstop=11, shake=0.15, up=wl(i, (0, 1, 0)) - wl(i, (0, 0, 0)))


@shot("J1_limb")
def _(i, u, n):
    t = u / n
    cam = wl(i, (-30.0 + 4 * t, -16.0, 48.0 + 4 * t))
    tgt = wl(i, (0.0, 6.0, 125.0))
    return dict(cam=cam, tgt=tgt, lens=20, fstop=16, shake=0.25)


@shot("J2_side")
def _(i, u, n):
    t = u / n
    cam = wl(i, (24.0 - 6 * t, -10.0, 80.0))
    tgt = wl(i, (0, 0, 79.0))
    return dict(cam=cam, tgt=tgt, lens=28, fstop=8, shake=0.25)


@shot("J3_wide")
def _(i, u, n):
    t = u / n
    s = float(ease_out(t, 2))
    cam = wl(i, (40.0 + 60 * s, -20.0 - 40 * s, 70.0 - 60 * s))
    tgt = wl(i, (0, 0, 80.0))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.2)


@shot("K_outro")
def _(i, u, n):
    t = u / n
    cam = wl(i, (90.0 + 20 * t, -60.0, 15.0))
    tgt = wl(i, (0, 0, 92.0 + 10 * t))
    return dict(cam=cam, tgt=tgt, lens=24, fstop=16, shake=0.0)


if RAVE:
    def _orbit(c, r, a_deg, h):
        a = math.radians(a_deg)
        return c + (SIDE * math.cos(a) + D_HAT * math.sin(a)) * r + UPV * h

    def _spk(i):
        return Vector(spk_loc[i])

    def _ast(i):
        return Vector(ast_pos[i])

    @shot("R1_neon")             # close chase sweeping round the neon stack
    def _(i, u, n):
        t = u / n
        a = math.radians(-120 + 80 * float(smoothstep(t)))
        cam = wl(i, (52 * math.cos(a), 52 * math.sin(a), 22 + 10 * t))
        return dict(cam=cam, tgt=wl(i, (0, 0, 46.0)), lens=24, fstop=8, shake=0.5)

    @shot("R2_nose")             # tight on the nose LED
    def _(i, u, n):
        t = u / n
        cam = wl(i, (17.0 - 4 * t, -26.0, LAS_TOP - 16.0))
        return dict(cam=cam, tgt=wl(i, (0, 0, LAS_TOP - 9.0)), lens=24, fstop=5.6, shake=0.6)

    @shot("R3_orbit")            # high-altitude chase: the neon core over the dark planet
    def _(i, u, n):
        t = u / n
        a = math.radians(-35 + 70 * t)
        cam = wl(i, (95 * math.cos(a), 95 * math.sin(a), 8.0))
        return dict(cam=cam, tgt=wl(i, (0, 0, 52.0)), lens=30, fstop=11, shake=0.3)

    @shot("R4_count")            # the run-up to the 1:50 '3, 2, 1, blast off': the fairing and the nose LED
    def _(i, u, n):
        t = u / n
        cam = wl(i, (8.0, -13.0 + 3 * t, SAJ_Z0 + 1.0))
        return dict(cam=cam, tgt=wl(i, (0, 0, OGIVE_Z0 + 4.0)), lens=30, fstop=5.6, shake=0.4)

    # ---------------- v4: the booster-3 story (0:33 build → 1:07 halftime drop) ----------------
    _B3C = Vector((SRB_X * math.cos(SLOT_AZ[2]), SRB_X * math.sin(SLOT_AZ[2]), 0.0))   # booster 3 axis (local)

    @shot("V1_rings")            # the build: a low chase sweeping round the stack as the rings boot bar by bar
    def _(i, u, n):
        t = u / n
        a = math.radians(-150 + 85 * float(smoothstep(t)))
        cam = wl(i, (44 * math.cos(a), 44 * math.sin(a), 6 + 22 * t))
        return dict(cam=cam, tgt=wl(i, (0, 0, 38.0 + 6 * t)), lens=24, fstop=8, shake=0.35)

    @shot("V2_panel")            # the spectrum panel up close (camera side of the core)
    def _(i, u, n):
        t = u / n
        cam = wl(i, (7.0 - 9.0 * t, -23.0 + 3 * t, 31.0 + 4 * t))
        return dict(cam=cam, tgt=wl(i, (0.0, -R, 34.3)), lens=32, fstop=5.6, shake=0.35)

    @shot("V3_feed")             # ALARM: a tracking feed from a chase plane (gets the warning HUD in post)
    def _(i, u, n):
        t = u / n
        cam = wl(i, (520.0, -1150.0, -40.0 + 50 * t))
        return dict(cam=cam, tgt=wl_lag(i, (0, 0, 40.0), lag=4, wobble=1.4), lens=260, fstop=8, shake=0.25, haze=0.5)

    @shot("V4_red")              # close on the booster-3 side: every LED red, pulsing; B3's plume sputters
    def _(i, u, n):
        t = u / n
        cam = wl(i, (40.0 - 6 * t, -46.0, 18.0 + 10 * t))
        return dict(cam=cam, tgt=wl(i, tuple(_B3C + Vector((0, 0, 24.0)))), lens=26, fstop=8, shake=0.5)

    @shot("V5_onb")              # onboard (core): looking down at booster 3's aft attach (HUD)
    def _(i, u, n):
        a = SLOT_AZ[2] + math.radians(28)
        cam = wl_v(i, ((R + 0.7) * math.cos(a), (R + 0.7) * math.sin(a), 50.0))
        tgt = wl_v(i, tuple(_B3C * 1.25 + Vector((0, 0, -12.0))))
        return dict(cam=cam, tgt=tgt, lens=12, fstop=11, shake=0.9, up=wl_v(i, (0, 0, 1)) - wl_v(i, (0, 0, 0)))

    @shot("V6_sep")              # booster 3 lets go: wide enough to see it pivot, rip free and cartwheel
    def _(i, u, n):
        t = u / n
        a = math.radians(-150 + 12 * t)                  # side-on to booster 3's slot (-60 deg): it peels to the RIGHT
        cam = wl(i, (130.0 * math.cos(a), 130.0 * math.sin(a), 22.0 - 8 * t))
        mid = (Vector(B3_MID[i]) + Vector((0, 0, 34.0))) * 0.5
        return dict(cam=cam, tgt=wl(i, tuple(mid)), lens=30 - 4 * t, fstop=11, shake=0.4)

    @shot("V7_boom")             # 0:56 false drop: it blows ~60 m from the core; stack + fireball in one frame
    def _(i, u, n):
        t = u / n
        tgt = (wl_air(i, BOOM_P) + wl(i, (0, 0, 40.0))) * 0.5
        a = math.radians(-150)
        cam = wl(i, (300.0 * math.cos(a), 300.0 * math.sin(a), 10.0)) + Vector((0, 0, 6 * t))
        return dict(cam=cam, tgt=tgt, lens=24, fstop=11, shake=1.0, up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))

    @shot("V7b_lost")            # the tracking feed drops out (SIGNAL LOST in post), then comes back mid-tumble
    def _(i, u, n):
        cam = wl(i, (430.0, -980.0, -60.0))
        return dict(cam=cam, tgt=wl_lag(i, (0, 0, 40.0), lag=3, wobble=2.0), lens=170, fstop=8, shake=0.6, haze=0.4)

    @shot("V8_tumble")           # outside, close chase: the stack swings off vertical and drifts; the camera lags
    def _(i, u, n):
        t = u / n
        a = math.radians(-152 + 10 * t)                  # perpendicular to the tipping plane: the swing reads
        cam = wl(i, (125.0 * math.cos(a), 125.0 * math.sin(a), 30.0))
        # a chase plane flies the NOMINAL climb (banked with it): up = the rig's axis, so the stack's swing away
        # from its flight path reads on screen (in world terms the tumble and the pitch programme partly cancel)
        return dict(cam=cam, tgt=wl(i, (0, 0, 40.0)), lens=24, fstop=8, shake=0.7,
                    up=wl(i, (0, 0, 1)) - wl(i, (0, 0, 0)))

    @shot("V9_fish")             # onboard FISHEYE (HUD + digital breakups in post): horizon swinging, plume flapping
    def _(i, u, n):
        a = math.radians(-112)
        cam = wl_v(i, ((R + 0.9) * math.cos(a), (R + 0.9) * math.sin(a), 61.0))
        tgt = wl_v(i, ((R + 2.2) * math.cos(a), (R + 2.2) * math.sin(a), -30.0))
        return dict(cam=cam, tgt=tgt, lens=8, fstop=11, shake=1.0, up=wl_v(i, (math.cos(a), math.sin(a), 0)) - wl_v(i, (0, 0, 0)))

    @shot("V10_track")           # long lens, the swings dying out — "barely made it"
    def _(i, u, n):
        t = u / n
        cam = wl(i, (-700.0, -1500.0, 60.0 - 30 * t))
        return dict(cam=cam, tgt=wl_lag(i, (0, 0, 42.0), lag=5, wobble=1.0), lens=380, fstop=11, shake=0.2, haze=0.6)

    @shot("D3_wide")             # long-lens air-to-air: the condensation collar on the halftime drop
    def _(i, u, n):
        t = u / n
        cam = wl(i, (700.0, -1250.0, 260.0 - 90 * t))
        tgt = wl_lag(i, (0, 0, 55.0), lag=3, wobble=0.8)
        return dict(cam=cam, tgt=tgt, lens=240, fstop=11, shake=0.3, haze=0.5)

    @shot("L0_glimpse")          # breakdown 1: a glimpse of something far away, coming (300 mm over the speaker)
    def _(i, u, n):
        t = u / n
        cam = _spk(i) + UPV * 60 - D_HAT * 40 + SIDE * 700
        return dict(cam=cam, tgt=_ast(i) - V_IN * 900, lens=200, fstop=11, shake=0.05)

    @shot("M7_fan")              # drop 2: behind and above the speaker, the laser fans sweeping out at the rock
    def _(i, u, n):
        t = u / n
        cam = _spk(i) - D_HAT * (150 - 20 * t) + SIDE * 95 + UPV * 45
        return dict(cam=cam, tgt=_spk(i) + D_HAT * 320 + UPV * 20, lens=24, fstop=11, shake=0.6)

    @shot("M8_tunnel")           # over the speaker's shoulder down the laser tunnel to the pinned rock
    def _(i, u, n):
        t = u / n
        cam = _spk(i) - D_HAT * 140 + SIDE * (70 - 15 * t) + UPV * 75
        return dict(cam=cam, tgt=_ast(i), lens=50, fstop=11, shake=0.7)

    @shot("M9_blind")            # front, close: moving heads, blinders, LED matrices, the woofers slamming
    def _(i, u, n):
        t = u / n
        cam = _spk(i) + D_HAT * (30 - 6 * t) + SIDE * 72 + UPV * 18
        return dict(cam=cam, tgt=_spk(i) + UPV * 8 + D_HAT * 6, lens=24, fstop=8, shake=0.9)

    @shot("M10_pin")             # side-on at the rock: lasers + rings pin it, the cracks glow
    def _(i, u, n):
        t = u / n
        cam = A_STOP - D_HAT * 380 + SIDE * (270 - 30 * t) + UPV * 70
        return dict(cam=cam, tgt=_ast(i) - D_HAT * 60, lens=32, fstop=11, shake=0.8)

    @shot("S1_rise")             # the speaker climbs out of the stack and grows to full size
    def _(i, u, n):
        t = u / n
        cam = wl(DEP0, (70.0, -110.0, 60.0)) + UPV * (40 * t)
        return dict(cam=cam, tgt=_spk(i), lens=24 - 6 * t, fstop=11, shake=0.15)

    @shot("S2_hero")             # 3/4 front: woofers pumping, LEDs flat out
    def _(i, u, n):
        t = u / n
        a = math.radians(35 - 50 * t)
        cam = _spk(i) + (D_HAT * math.cos(a) + SIDE * math.sin(a)) * 115 + UPV * 6
        return dict(cam=cam, tgt=_spk(i) + UPV * 2, lens=28, fstop=11, shake=0.35)

    @shot("S3_wide")             # the speaker, the spent stack and the limb
    def _(i, u, n):
        t = u / n
        mid = (_spk(i) + wl(i, (0, 0, 60.0))) * 0.5
        cam = mid + SIDE * (320 + 40 * t) - D_HAT * 120 + UPV * 70
        return dict(cam=cam, tgt=mid, lens=24, fstop=16, shake=0.1)

    @shot("L1_ast")              # the threat: a long lens on the incoming rock
    def _(i, u, n):
        t = u / n
        cam = A_START.lerp(A_HIT, 0.62) + SIDE * 5200 - UPV * 900
        return dict(cam=cam, tgt=_ast(i) - V_IN * (500 + 600 * t), lens=55 - 10 * t, fstop=11, shake=0.05)

    @shot("L2_over")             # over the speaker's shoulder: the rock grows in the distance
    def _(i, u, n):
        cam = _spk(i) - D_HAT * 75 + SIDE * 38 + UPV * 22
        return dict(cam=cam, tgt=_spk(i).lerp(_ast(i), 0.08) + UPV * 5, lens=24, fstop=11, shake=0.12)

    @shot("L3_charge")           # front, close: the LEDs charge up
    def _(i, u, n):
        t = u / n
        cam = _spk(i) + D_HAT * (70 - 12 * t) + SIDE * 30 - UPV * 6
        return dict(cam=cam, tgt=_spk(i), lens=26, fstop=8, shake=0.4)

    @shot("L4_rock")             # the asteroid up close, tumbling in
    def _(i, u, n):
        cam = A_STOP - D_HAT * 380 + SIDE * 160 + UPV * 60
        return dict(cam=cam, tgt=_ast(i), lens=45, fstop=11, shake=0.1)

    @shot("M1_fire")             # THE DROP: 3/4 front, waves bursting off the woofers
    def _(i, u, n):
        t = u / n
        cam = _spk(i) + D_HAT * 95 + SIDE * 75 + UPV * (14 - 6 * t)
        return dict(cam=cam, tgt=_spk(i) + D_HAT * 25, lens=22, fstop=11, shake=1.0)

    @shot("M2_wide")             # side-on: speaker, rings marching out, the rock pinned, Earth shaking
    def _(i, u, n):
        t = u / n
        mid = (SPK_P + A_STOP) * 0.5
        cam = mid + SIDE * (1050 - 60 * t) + UPV * 90 - D_HAT * 60
        return dict(cam=cam, tgt=mid - UPV * 40, lens=24, fstop=16, shake=0.6)

    @shot("M3_crack")            # on the rock: every wave lands, the cracks glow
    def _(i, u, n):
        t = u / n
        cam = A_STOP - D_HAT * (300 - 40 * t) + SIDE * 110 + UPV * 35
        return dict(cam=cam, tgt=_ast(i), lens=40, fstop=11, shake=0.8)

    @shot("M4_boom")             # it goes: wide, the speaker in the foreground
    def _(i, u, n):
        t = u / n
        cam = A_STOP - D_HAT * 1450 + SIDE * 520 + UPV * 160
        return dict(cam=cam, tgt=A_STOP - D_HAT * 150 + UPV * 20, lens=28 - 4 * t, fstop=16, shake=1.0)

    @shot("M5_hero")             # the speaker keeps raving as debris streaks past
    def _(i, u, n):
        cam = _spk(i) + D_HAT * 62 - SIDE * 52 - UPV * 10
        return dict(cam=cam, tgt=_spk(i) + UPV * 4, lens=20, fstop=8, shake=0.8)

    @shot("M6_orbit")            # slow orbit: speaker, debris cloud, sunrise on the limb
    def _(i, u, n):
        t = u / n
        cam = _orbit(_spk(i), 190, 60 + 90 * float(smoothstep(t)), 30)
        return dict(cam=cam, tgt=_spk(i) + D_HAT * 60, lens=24, fstop=16, shake=0.4)

    @shot("N1_drift")            # aftermath: the speaker small against the planet
    def _(i, u, n):
        t = u / n
        cam = _spk(i) - D_HAT * 520 + SIDE * 260 + UPV * (140 + 30 * t)
        return dict(cam=cam, tgt=_spk(i) + D_HAT * 200, lens=35, fstop=16, shake=0.05)

    @shot("N2_final")            # the last hit: front, close, flat out — then black
    def _(i, u, n):
        t = u / n
        cam = _spk(i) + D_HAT * (80 - 30 * t) - UPV * 4
        return dict(cam=cam, tgt=_spk(i), lens=24, fstop=8, shake=0.9)


# --- evaluate the shot list per frame ---------------------------------------------------
cam_loc = np.zeros((N, 3))
cam_rot = np.zeros((N, 3))
cam_lens = np.zeros(N)
cam_focus = np.zeros(N)
cam_fstop = np.zeros(N)
SHOT_HAZE = np.zeros(N)
SHOT_ONB = np.zeros(N)
SHOT_FISH = np.zeros(N)
SHOT_BOUNDS = [s for s, _ in SHOTS] + [N]
prev_eul = None
for k, (s0, name) in enumerate(SHOTS):
    s1 = SHOT_BOUNDS[k + 1]
    fn = SHOT_FN[name]
    onb = name in ONBOARD
    for i in range(s0, s1):
        d = fn(i, i - s0, max(s1 - s0 - 1, 1))
        thr = max(on_core[i], on_boost[i], on_s2[i])
        amp = d.get("shake", 0.0) * (0.15 * thr + 0.85 * KICK[i] * HOT[i] + 0.6 * IGN_FLASH[i] + 0.5 * onb * thr
                                     + RAVE_SHAKE[i])
        if SET[i] == 3:
            amp *= 0.5
        if d["lens"] > 150:                                     # long lens: jolts read as metres at
            amp *= 150.0 / d["lens"]                            # the subject, not at the camera
        cam = d["cam"] + shake_vec(i, amp * (0.06 if onb else 0.35), hf=onb)
        if BIG[i] > 0:
            cam = cam + (d["tgt"] - cam) * (0.07 * ONE[i])
        tgt = d["tgt"] + shake_vec(i, amp * (0.25 if onb else 0.9), 1, hf=onb)
        fwd = (tgt - cam).normalized()
        q = fwd.to_track_quat("-Z", "Y")
        if "up" in d:
            up = Vector(d["up"]).normalized()
            right = fwd.cross(up).normalized()
            up2 = right.cross(fwd).normalized()
            q = Matrix((right, up2, -fwd)).transposed().to_quaternion()
        e = q.to_euler("XYZ", prev_eul) if (prev_eul is not None and i != s0) else q.to_euler("XYZ")
        prev_eul = e
        cam_loc[i] = cam
        cam_rot[i] = (e.x, e.y, e.z)
        cam_lens[i] = d["lens"]
        cam_fstop[i] = d["fstop"]
        cam_focus[i] = d.get("focus", (tgt - cam).length)
        SHOT_HAZE[i] = d.get("haze", 0.0)
        SHOT_ONB[i] = 1.0 if onb else 0.0
        SHOT_FISH[i] = 1.0 if name in FISHEYE else 0.0
bake_vec(CAM, "location", cam_loc)
bake_vec(CAM, "rotation_euler", cam_rot)
bake(CAM_D, "lens", cam_lens)
bake(CAM_D, "dof.focus_distance", cam_focus)
bake(CAM_D, "dof.aperture_fstop", cam_fstop)

# countdown board in front of the A2 camera, angled toward it
_a2 = [s for s, n in SHOTS if n == "A2_clock"][0]
_cpos = Vector(cam_loc[_a2])
_cdir = Vector((10.0, 0.0, STACK_C)) - _cpos
_cdir.z = 0
_cdir.normalize()
_left = Vector((-_cdir.y, _cdir.x, 0))
_gz = float(height(np.array([_cpos.x]), np.array([_cpos.y]))[0][0])
CLK.location = _cpos + _cdir * 13.5 + _left * 4.6 + Vector((0, 0, -0.6))
CLK.location.z = max(CLK.location.z, _gz + 2.65)
_face = (-_cdir).copy()
CLK.rotation_euler = (0, 0, math.atan2(_face.x, -_face.y) - math.radians(24))


# =====================================================================================
# 10. ANIMATION BAKES (all music reactions live here)
# =====================================================================================
# --- SAJ panels: pivot each at its own centroid so it tumbles in place as it flies off ---
PANEL_DIR = []
for pn in PANELS:
    ch = pn.children[0]
    ws = [Vector(c) for c in ch.bound_box]
    cen = sum(ws, Vector()) / 8.0
    pn.location = cen
    ch.location = -cen
    dxy = Vector((cen.x, cen.y, 0.0))
    PANEL_DIR.append(dxy.normalized() if dxy.length > 0.1 else Vector((1, 0, 0)))

# --- sets / visibility ---
PAD_ONLY = ["TerrainNear", "TerrainFar", "Water", "ML", "FlameDeflector", "TrenchWalls", "LightningTowers",
            "LH2Sphere", "LOXSphere", "WaterTowerLegs", "WaterTowerTank", "FloodMasts", "ShoreLights",
            "PadHaze", "TrailPad", "DelugeCurtains", "Rainbirds"]
for o in bpy.data.objects:
    if o.name.startswith(("Catenary", "Beacon", "Flood", "Search", "WorkL", "WorkLamp", "DeckL", "Seg", "Colon", "Sign", "Board", "Clock",
                          "SteamVDB", "SteamEarly")) or o.name in PAD_ONLY:
        show(o, SET == 0)
show(bpy.data.objects["CountdownBoard"], IDX < IGN + 2 * FPS)
for o in bpy.data.objects:
    if o.parent == CLK:
        show(o, IDX < IGN + 2 * FPS)
show(SIGN_V, (IDX >= IGN) & (IDX < IGN + 2 * FPS))
for o in CLOUD_OBS + [deck_haze]:
    show(o, (SET == 1) & (DECK_TOP > -3000))
show(trail_f, (SET == 1) | ((SET == 2) & (IDX < FAIRSEP)))
show(trail, (SET == 0) & (IDX >= LIFT))
show(DECK, np.zeros(N, bool))
for o, nfi in VENTS:
    show(o, IDX < LIFT + 20)
for o, nfi, st0, *_ in STEAMV:
    show(o, (SET == 0) & (IDX >= st0 - 1))
show(curtains, (SET == 0) & (DELUGE > 0.01))
show(rainbirds, (SET == 0) & (DELUGE > 0.01))
show(sparks, (SET == 0) & (SPARKS > 0.001))
show(vcone, (IDX >= MAXQ - 2) & (IDX < D_END))
for c in vcone_b:
    show(c, (IDX >= MAXQ - 2) & (IDX < D_END))
SEP = MECO + int(round(BAR / 2))
s1_vis = IDX < SEP + 4 * FPS


def descendants(ob):
    out = []
    for c in ob.children:
        out.append(c)
        out += descendants(c)
    return out


for o in descendants(S1):
    if o.type != "EMPTY" and o.name not in VIS:
        show(o, s1_vis)
for bem in (BL, BRt):
    for o in descendants(bem):
        if o.name not in VIS:
            show(o, IDX < BSEP + 8 * FPS)
B3_ALIVE = IDX < B3BOOM                          # booster 3 is gone (in pieces) from the blast frame on
for o in descendants(B3):
    if o.name not in VIS:
        show(o, B3_ALIVE)
for o in [c for pn in PANELS for c in pn.children] + list(LASG.children):
    show(o, IDX < FAIRSEP + 6 * FPS)
show(flakes, (SAGE < 36) & (SET == 0) & (IDX >= RS25_T))
show(flakes2, (IDX >= LIFT) & (IDX < LIFT + 4 * FPS) & (SET == 0))
_far = np.zeros(N, bool)
for k_, (s0_, nm_) in enumerate(SHOTS):
    if nm_ in ("F3_wide", "G_calm"):
        _far[s0_:SHOT_BOUNDS[k_ + 1]] = True
show(jelly, (SET == 2) & (IDX >= BSEP + 2 * FPS) & _far)

# --- rig + stage motions ---
bake_vec(RIG, "location", RIG_POS)
bake_vec(RIG, "rotation_euler", np.stack([np.radians(-PITCH), np.zeros(N), np.zeros(N)], 1))
# v4 tumble: the wobble pivot (WOB) rotates the stack about its centre of mass and carries the sag/drift;
# the core gimbal swivels the RS-25 bells + plumes by the commanded TVC angle
_wl = np.zeros((N, 3))
for i in range(N):
    _wl[i] = np.array(wob_matrix(i).to_translation())
bake_vec(WOB, "location", _wl)
bake_vec(WOB, "rotation_euler", np.stack([WOB_PHI[:, 0], WOB_PHI[:, 1], WOB_ROLL], 1))
bake_vec(GIMB, "rotation_euler", np.stack([GIMBAL[:, 0], GIMBAL[:, 1], np.zeros(N)], 1))
# boosters 1+2 (the 1:28 hardstyle drop): radially out in their slot frames, nose outward
tb = np.maximum(IDX - BSEP, 0) / FPS
for bem in (BL, BRt):
    ox = SRB_X + 2.2 * tb + 0.5 * tb ** 2
    oz = -(3.0 * tb ** 2) - 2.0 * tb
    ry = np.radians(5.5 * tb ** 1.3)
    bake_vec(bem, "location", np.stack([ox, np.zeros(N), oz], 1))
    bake_vec(bem, "rotation_euler", np.stack([np.zeros(N), ry, np.zeros(N)], 1))
# booster 3: the flight model's early-sep path (rig frame) → its slot frame (rotate by -azimuth)
_c3, _s3 = math.cos(-SLOT_AZ[2]), math.sin(-SLOT_AZ[2])
_b3l = np.stack([_c3 * B3_LOC[:, 0] - _s3 * B3_LOC[:, 1], _s3 * B3_LOC[:, 0] + _c3 * B3_LOC[:, 1], B3_LOC[:, 2]], 1)
bake_vec(B3, "location", _b3l)
bake_vec(B3, "rotation_euler", np.stack([np.zeros(N), B3_ANG, np.zeros(N)], 1))
# booster 3 burns on: sputtering on the snares after the alarm, then (free) at full thrust until it bursts
B3_BURN = on_boost * np.where(IDX < B3SEP, B3FRAC, 1.0) * B3_ALIVE
# launch abort system: the jettison motor yanks tower + ogive up and away (keyed 'LASMotor')
tf = np.maximum(IDX - FAIRSEP - 6, 0) / FPS
bake_vec(LASG, "location", np.stack([np.zeros(N), -0.8 * tf ** 2, 26.0 * tf ** 2 + 3.0 * tf], 1))
bake_vec(LASG, "rotation_euler", np.stack([np.radians(-4.0 * tf ** 1.5), np.zeros(N), np.zeros(N)], 1))
tpn = np.maximum(IDX - FAIRSEP, 0) / FPS
for pn, dvec_ in zip(PANELS, PANEL_DIR):
    base = Vector(pn.location)
    off = (1.4 * tpn + 2.2 * tpn ** 2)
    loc = np.stack([base.x + dvec_.x * off, base.y + dvec_.y * off, base.z - 1.6 * tpn ** 2], 1)
    bake_vec(pn, "location", loc)
    tang = Vector((-dvec_.y, dvec_.x, 0.0))
    ang = np.radians(np.minimum(70 * tpn, 200))
    eul = [Matrix.Rotation(float(a), 3, tang).to_euler() for a in ang]
    bake_vec(pn, "rotation_euler", np.array([(e.x, e.y, e.z) for e in eul]))
# core stage falls back after MECO (stage sep half a bar later)
ts = np.maximum(IDX - SEP, 0) / FPS
bake_vec(S1, "location", np.stack([np.zeros(N), -0.25 * ts ** 2, -(1.4 * ts + 0.9 * ts ** 2)], 1))
bake_vec(S1, "rotation_euler", np.stack([np.radians(1.4 * ts ** 1.2), np.zeros(N), np.zeros(N)], 1))

# --- plumes ---
alt_expand = np.ones(N)
alt_expand[SET == 0] = 1.0 + np.clip(ALT[SET == 0] / 2500.0, 0, 0.12)
alt_expand[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [1.15, 1.35, 1.6])
alt_expand[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, SES], [1.8, 2.4, 3.0])
phase = np.cumsum(0.9 + 0.6 * HOT) / FPS                             # flow scroll
flick = np.cumsum(1.4 + 1.2 * HOT + 0.8 * KICK) / FPS               # 4D noise W: flicker
for o, kind, part in PLUMES:
    vol = part == "vol"
    if kind.startswith("core"):
        on = on_eng[int(kind[4:])]
        lng = on * (1.0 + 0.10 * KH) * np.where(SET == 0, 1.0, alt_expand ** 0.6)
        gain = on * PULSE * np.where(SET == 0, 22.0, 4.0 / alt_expand) * np.where(IDX >= IGN, 0.6, 1.0)
        sx = np.maximum(alt_expand ** 0.8 * (0.6 + 0.4 * on), 1e-3)
        sz = np.maximum(lng, 1e-3)
    elif kind in ("boost", "boost3"):
        on = on_boost if kind == "boost" else B3_BURN
        ex = alt_expand
        lng = on * (1.0 + 0.16 * KH)
        if vol:
            gain = on * PULSE * np.where(SET == 0, 0.45, np.where(SET == 1, 0.30, 0.12)) / ex ** 1.2
            sx = np.where(SET == 0, 10.0, 16.0) * ex * (0.5 + 0.5 * on)
            sz = np.maximum(np.where(SET == 0, 110.0, 150.0) * lng * ex ** 0.7, 1e-3)
        else:
            gain = on * PULSE * 30.0 * np.where(SET == 0, 1.0, 0.8) / ex ** 0.6
            sx = np.maximum(np.where(SET == 0, 1.0, 1.8) * ex ** 0.6 * (0.55 + 0.45 * on), 1e-3)
            sz = np.maximum(np.where(SET == 0, 1.0, 1.9) * lng * ex ** 0.7, 1e-3)
        gain = gain * (1.0 + 3.0 * IGN_FLASH)
    else:  # s2
        on = on_s2
        pulse2 = 1.0 + 0.45 * BIG * ONE + 0.25 * KH
        lng = on * (1.0 + 0.16 * KH + 0.12 * BIG * ONE)
        gain = on * pulse2 * (14.0 if not vol else 0.05)
        sx = np.full(N, 22.0) * (0.4 + 0.6 * on) if vol else np.ones(N) * (0.6 + 0.4 * on)
        sz = np.maximum((np.full(N, 190.0) if vol else np.ones(N)) * lng, 1e-3)
    set_color_keys(o, np.stack([gain, flick, phase], 1), const=False)
    bake_vec(o, "scale", np.stack([sx, sx, sz], 1))
    if kind == "boost":                                   # v4: SRB nozzles gimbal too -> the plume bends
        _az = SLOT_AZ[0] if o.parent == BL else SLOT_AZ[1]
        _ca, _sa = math.cos(-_az), math.sin(-_az)
        bake_vec(o, "rotation_euler", np.stack([_ca * GIMBAL[:, 0] - _sa * GIMBAL[:, 1],
                                                _sa * GIMBAL[:, 0] + _ca * GIMBAL[:, 1], np.zeros(N)], 1))
    show(o, on > 0.001)
for lo, tag, j in PLIGHTS:
    if tag in ("bl", "br", "b3"):
        on = on_boost if tag != "b3" else B3_BURN
        e = on * PULSE * (1.0 + 2.5 * IGN_FLASH) * (2.4e6, 1.4e6, 9.0e5)[j]
        e = e * np.where(SET == 0, 1.0, 0.7)
    elif tag == "core":
        e = on_core * PULSE * (5.0e4, 3.0e4)[j] * np.where(IDX >= IGN, 1.0, 1.0)
        on = on_core
    else:
        e = on_s2 * 8.0e4 * (1 + 0.5 * BIG * ONE)
        on = on_s2
    bake(lo.data, "energy", e)
    show(lo, on > 0.001)
bake(BOUNCE, "energy", on_boost * PULSE * np.where(SET == 0, 2.0e6, np.where(SET == 1, 8.0e5, 3.0e5)))
show(BOUNCE_OB, on_boost > 0.001)
bake(M_RS25_SHELL.node_tree, value_path("Diamonds"), np.where(SET == 0, 1.0, np.where(SET == 1, 0.25, 0.0)))
las_on = np.where((IDX >= FAIRSEP + 6) & (IDX < FAIRSEP + 6 + 36), 1.0, 0.0) * gsmooth(
    ((IDX >= FAIRSEP + 6) & (IDX < FAIRSEP + 42)).astype(float), 1.5)
for lp in LAS_PLUMES:
    set_color_keys(lp, np.stack([las_on * 40.0, flick, phase], 1), const=False)
    show(lp, las_on > 0.01)
# nozzle interiors glow (Object Info colour = heat)
set_color_keys(ENG, np.stack([on_core, on_core, on_core], 1), const=False)
s2heat = np.clip(gsmooth(on_s2, 6) * 1.2, 0, 1)
set_color_keys(s2noz, np.stack([s2heat] * 3, 1), const=False)
# the twilight jellyfish: a sunlit bell trailing kilometres behind in SET 2
jg = np.clip(ramp(IDX, BSEP + 2 * FPS, BSEP + 8 * FPS) * (1 - ramp(IDX, SES - 20, SES)), 0, 1) * (SET == 2)
bake(M_JELLY.node_tree, value_path("Gain"), jg * (1.0 + 0.3 * BASS))
bake(M_JELLY.node_tree, value_path("Evolve"), np.cumsum(np.full(N, 0.02)))
jr = np.interp(IDX, [BSEP, FAIRSEP, SES], [250.0, 700.0, 1100.0])
jl = np.interp(IDX, [BSEP, FAIRSEP, SES], [900.0, 2600.0, 3800.0])
bake_vec(jelly, "scale", np.stack([jr, jr, jl], 1))

# --- steam (VDB playback + drift), vents, trail, deluge water, wetness ---
evo = np.cumsum(np.full(N, 1.0 / FPS))
for o, nfi, st0, speed, base, drift, sc0, kind in STEAMV:
    vdb_play(o, nfi, st0, speed)
    tt_ = np.maximum(IDX - st0, 0) / FPS
    ease = 1 - np.exp(-tt_ / 6.0)
    pos = np.array(base)[None, :] + np.array(drift)[None, :] * (6.0 * ease + 0.35 * tt_)[:, None]
    bake_vec(o, "location", pos)
plume_base_z = RIG_POS[:, 2] - 2.0
steam_glow = on_boost * PULSE * np.clip(1 - ALT / 420, 0.05, 1) * (1 + 1.5 * IGN_FLASH)
bake(M_STEAM.node_tree, value_path("Glow"), steam_glow * 0.35)
bake(M_STEAM.node_tree, value_path("PlumeZ"), np.minimum(plume_base_z, PAD_TOP + 30.0))
bake(M_STEAM_EARLY.node_tree, value_path("Glow"), on_core * 0.10 * (IDX < IGN) + steam_glow * 0.25)
bake(M_STEAM_EARLY.node_tree, value_path("PlumeZ"), np.full(N, PAD_TOP))
bake(M_VENT.node_tree, value_path("Density"), (0.25 + 0.5 * PRE + 0.2 * BASS * (IDX < IGN)) * (1 - ramp(IDX, LIFT, LIFT + 20)))
tr_len = np.maximum(RIG_POS[:, 2] - PAD_TOP - 6.0, 0.1)
tr_r = 13.0 + 0.06 * ALT
bake_vec(trail, "location", np.stack([np.zeros(N), RIG_POS[:, 1] * 0.5, np.full(N, PAD_TOP)], 1))
bake_vec(trail, "scale", np.stack([tr_r, tr_r, tr_len], 1))

def vkeys(mat, density, evolve, glow=None):
    nt = mat.node_tree
    bake(nt, value_path("Density"), density)
    bake(nt, value_path("Evolve"), evolve)
    if glow is not None:
        bake(nt, value_path("Glow"), glow)


vkeys(M_TRAIL, np.full(N, 0.10) * ramp(IDX, LIFT, LIFT + 15), evo * 0.15, 0.25 * on_boost * PULSE)
wet = ramp(IDX, DELUGE_T + 20, DELUGE_T + 120)
for mat in (M_TERRAIN, M_CONCRETE, M_ML, M_ML_RAIL):
    bake(mat.node_tree, value_path("Wet"), wet)
bake(M_CURTAIN.node_tree, value_path("Gain"), DELUGE * 0.95)
bake(M_CURTAIN.node_tree, value_path("Flow"), np.cumsum(DELUGE) / FPS * 2.6)
bake(rainbirds, gn_input_path(rain_md, RAIN_T, "Time"), IDX / FPS)
bake(rainbirds, gn_input_path(rain_md, RAIN_T, "Amount"), DELUGE * (1 - 0.5 * ramp(IDX, IGN, IGN + 60)))
bake(sparks, gn_input_path(sparks_md, SPARK_TREE, "Time"), IDX / FPS)
bake(sparks, gn_input_path(sparks_md, SPARK_TREE, "Amount"), SPARKS * (0.7 + 0.3 * SNARE))
# frost sheds after liftoff; plume soot builds in flight
fr_amt = 1.0 - ramp(IDX, LIFT, LIFT + 90)
for mat in (M_FOAM, M_NOZ_OUT):
    bake(mat.node_tree, value_path("Frost"), fr_amt)
bake(M_SRB.node_tree, value_path("Soot"), 0.9 * ramp(IDX, IGN, IGN + 40))
bake(M_FOAM.node_tree, value_path("Soot"), 0.7 * ramp(IDX, IGN + 30, PAD_END))
bake(M_WHITE.node_tree, value_path("Soot"), 0.8 * ramp(IDX, RS25_T + 10, IGN + 60))
# flight smoke trail: hangs behind the vehicle along the flight path (SET 1 → early SET 2)
tf_loc = np.array([wl(i, (0, 0, -8.0)) for i in range(N)])
bake_vec(trail_f, "location", tf_loc)
bake_vec(trail_f, "rotation_euler", np.stack([np.radians(-PITCH) + math.pi, np.zeros(N), np.zeros(N)], 1))
fr_r = np.where(SET == 1, 14.0, 30.0) * alt_expand
bake_vec(trail_f, "scale", np.stack([fr_r, fr_r, np.where(SET == 1, 2400.0, 1800.0)], 1))
vkeys(M_TRAILF, np.where(SET == 1, 0.06, 0.025) * on_boost, evo * 0.15 + np.cumsum(np.full(N, 0.05)),
      0.15 * on_boost * PULSE)
# cloud deck sinks past the climb
bake_vec(DECK, "location", np.stack([np.zeros(N), np.zeros(N), DECK_TOP - 120.0], 1))
vkeys(M_DECKHAZE, np.full(N, 0.012), evo * 0.05, np.zeros(N))

# --- vapour cone (max-Q): bursts on the drop entry, then flickers with the kicks ---
vc_gain = np.clip(ramp(IDX, MAXQ - 2, MAXQ + 2) * (1 - ramp(IDX, D_END - 30, D_END)), 0, 1)
vc_gain = vc_gain * (0.45 + 0.9 * KICK * HOT + 0.25 * BASS)
vc_gain = vc_gain + np.where(IDX >= MAXQ, 1.2 * np.exp(-np.maximum(IDX - MAXQ, 0) / 6.0), 0.0)
vc_gain = vc_gain * (IDX >= MAXQ - 2)
bake(M_VCONE.node_tree, value_path("Gain"), vc_gain)
bake(M_VCONE.node_tree, value_path("Evolve"), evo * 2.5)
vcs = 1.0 + 0.05 * KICK
bake_vec(vcone, "scale", np.stack([8.0 * vcs, 8.0 * vcs, 34.0 * (1.0 + 0.1 * KICK)], 1))

# --- ice: shed on the snares (stateless bursts) + a sheet of it at liftoff ---
bake(flakes, gn_input_path(flakes_md, FLAKE_T, "Age"), np.minimum(SAGE, 60) / FPS)
bake(flakes, gn_input_path(flakes_md, FLAKE_T, "Seed"), np.maximum(SIDX, 0).astype(np.float64) * 7.0, const=True)
bake(flakes2, gn_input_path(flakes2_md, FLAKE_T, "Age"), np.maximum(IDX - LIFT, 0) / FPS)

# --- pad lights: floods, beacons, searchlights; blackout on T_black before ignition ---
pad_black = ((BLACK > 0) & (IDX < IGN + 2)).astype(np.float64)
pad_cut = np.clip(pad_black + ramp(IDX, IGN - 3, IGN), 0, 1) * (IDX < IGN + 1) + (IDX >= IGN)
pad_cut = np.clip(pad_cut, 0, 1)
flood_on = (1 - pad_cut) + (IDX >= IGN + 6) * 0.5
snare_flash = SNARE * (IDX < IGN) * 0.3
for o in FLOODS:
    if o.type == "LIGHT":
        bake(o.data, "energy", 1.8e6 * flood_on * (1 + snare_flash))
    else:
        set_color_keys(o, np.stack([flood_on * 80.0 * (1 + snare_flash)] * 3, 1) * np.array([0.9, 0.95, 1.0]),
                       const=False)
for o in WORKL:
    if o.type == "LIGHT":
        bake(o.data, "energy", (2.5e3 if o.name.startswith("WorkL") else 5.0e3) * (1 - pad_cut * (IDX < IGN + 6)))
    else:
        set_color_keys(o, np.stack([(1 - pad_cut * (IDX < IGN + 6)) * 40.0] * 3, 1))
rate = np.where(PRE > 0.66, 4.0, np.where(PRE > 0.2, 2.0, 1.0))
blink = (np.mod(BEAT_PH * rate + (TL["T_bi"] % 2) * (rate < 2), 1.0) < 0.35).astype(np.float64)
blink = np.maximum(blink, (HAT > 0.55) * (IDX < IGN)) * (1 - pad_black)        # hi-hats strobe the beacons
for o in BEACONS:
    if o.type == "LIGHT":
        bake(o.data, "energy", blink * 2.0e4, const=True)
    else:
        set_color_keys(o, np.stack([blink * 300.0, blink * 12.0, blink * 6.0], 1))
for k, o in enumerate(SEARCH):
    sw_ = np.sin(IDX / FPS * (0.21 + 0.05 * k) + k * 2.1)
    tilt = np.radians(14 + 7 * np.sin(IDX / FPS * 0.13 + k))
    rx = math.pi - tilt
    rz = np.radians(25 * sw_ + 120 * k)
    bake_vec(o, "rotation_euler", np.stack([rx, np.zeros(N), rz], 1))
    bake(o.data, "energy", 2.4e7 * (1 - pad_black) * (1 - 0.7 * ramp(IDX, IGN, IGN + 60)))

# --- countdown clock (LED segments) ---
rem = np.ceil((IGN - IDX) / FPS).astype(int)
ela = np.floor((IDX - IGN) / FPS).astype(int)
secs = np.where(IDX < IGN, rem, ela)
plus = IDX >= IGN
hh, mm, ss = secs // 3600, (secs // 60) % 60, secs % 60
digs = np.stack([hh // 10, hh % 10, mm // 10, mm % 10, ss // 10, ss % 10], 1)
LED_ON = np.array([1.0, 0.36, 0.035]) * 7.0
LED_OFF = np.array([1.0, 0.36, 0.035]) * 0.05
for (slot, s), o in LED_SEGS.items():
    on = np.array([s in DIGIT_SEGS[int(d)] for d in digs[:, slot]], float)
    set_color_keys(o, LED_OFF[None, :] + on[:, None] * (LED_ON - LED_OFF)[None, :])
colon_on = (BEAT_PH < 0.5).astype(float)
for o in COLON:
    set_color_keys(o, LED_OFF[None, :] + colon_on[:, None] * (LED_ON - LED_OFF)[None, :])
set_color_keys(SIGN_H, np.repeat(LED_ON[None, :], N, 0))
set_color_keys(SIGN_V, plus[:, None] * LED_ON[None, :])

# --- world / sky / exposure ---
nt = WORLD.node_tree
alt_m = np.zeros(N)
alt_m[SET == 0] = 30 + ALT[SET == 0]
alt_m[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, MAXQ, BSEP], [3500, 9000, 16000])
alt_m[SET == 2] = np.interp(IDX[SET == 2], [BSEP, FAIRSEP, MECO, SES], [32000, 58000, 76000, 85000])
alt_m[SET == 3] = np.interp(IDX[SET == 3], [SES, N], [92000, 100000])
bake(nt, 'nodes["Sky"].altitude', alt_m)
RE = 6360e3
dip = np.degrees(np.arccos(RE / (RE + alt_m)))
sun_el = np.full(N, -6.0)                                           # nautical twilight on the pad
sun_el[SET == 1] = np.interp(IDX[SET == 1], [PAD_END, BSEP], [-5.0, -3.0])
sun_el[SET >= 2] = -dip[SET >= 2] - np.interp(IDX[SET >= 2], [BSEP, SES - 20, SES, SES + 3 * BAR, N],
                                              [2.6, 1.1, 0.25, -0.9, -2.6])
bake(nt, 'nodes["Sky"].sun_elevation', np.radians(sun_el))
bake(nt, value_path("EarthH"), (RE + alt_m) / RE)
bake(nt, value_path("SunEl"), np.radians(sun_el))
sun_flare = 1.0 + 2.5 * np.where(IDX >= SES, np.exp(-(IDX - SES) / 12.0), 0.0) + 0.6 * BIG * ONE
bake(nt, value_path("SunDisc"), (SET >= 2) * sun_flare)
bake(nt, value_path("EarthOverlay"), np.clip((alt_m - 8000) / 20000, 0, 1))
bake(nt, value_path("EarthScroll"), -np.cumsum(np.where(SET >= 2, 0.00012, 0.00002)))
show(haze_box, SET == 0)
vkeys(M_HAZE, np.full(N, 0.00022) * (1 - 0.3 * ramp(IDX, IGN, PAD_END)), evo * 0.01)
bake(nt, value_path("Stars"), np.where(SET == 0, 0.5, np.where(SET == 1, 0.8, 1.2)))
# the sun lamp follows the sky's sun (azimuth +Y); grazing light appears once it clears the
# local horizon at altitude (cloud tops at the end of the calm, the jellyfish, the limb)
sun_vis = np.clip((sun_el + dip + 0.3) / 0.6, 0, 1) * (SET >= 1)
el_r = np.radians(sun_el)
bake_vec(SUN_OB, "rotation_euler", np.stack([el_r - math.pi / 2, np.zeros(N), np.zeros(N)], 1))
bake(SUN, "energy", sun_vis * 6.0)
show(SUN_OB, sun_vis > 0.001)
bake(MOON, "energy", np.where(SET == 0, 0.04, np.where(SET == 1, 0.2, np.where(SET == 2, 1.2, 0.05))))
moon_rx = np.where(SET >= 2, math.radians(30) - math.pi / 2, math.radians(55))
moon_rz = np.where(SET >= 2, 0.0, math.radians(-150))
bake_vec(MOON_OB, "rotation_euler", np.stack([moon_rx, np.zeros(N), moon_rz], 1), const=True)
bake_vec(MOON, "color", np.stack([np.where(SET >= 2, 1.0, 0.55), np.where(SET >= 2, 0.62, 0.65),
                                  np.where(SET >= 2, 0.55, 1.0)], 1), const=True)
# exposure ride (research fix #2): night pad exposed for the floods; the RS-25 start lifts the
# scene a little; SRB ignition CLIPS (2-frame overexposure) and the camera stops down ~3 EV
expo = np.where(SET == 0, 0.7, np.where(SET == 1, 0.6, np.where(SET == 2, 1.0, -0.2)))
expo = expo - 0.3 * on_core * (SET == 0) * (IDX < IGN)
expo = expo - 2.2 * smoothstep(ramp(IDX, IGN + 2, IGN + 26)) * (SET == 0)
expo = expo + 0.8 * IGN_FLASH                                    # the overexposed pop (clips the highlights)
expo = expo + 0.35 * STROBE * HOT
pump = SHOT_ONB * (0.25 * KICK * HOT + 0.08 * np.sin(IDX * 0.9))  # onboard auto-exposure pumping
expo = expo + pump
blk = (BLACK > 0) & (IDX > IGN + 10)
expo = np.where(blk, expo - 5.0, expo)
expo = expo - 0.4 * ((IDX >= MECO) & (IDX < SES)) * ramp(IDX, MECO, MECO + 12) * (1 - ramp(IDX, SES - 2, SES))
bake(SC, "view_settings.exposure", expo)

# =====================================================================================
# 10b. v3 RAVE (Zeke 10-11): neon colour + level, the speaker, sound-wave rings, the asteroid,
#      the boom, the Earth shaking. All keyed per frame like everything above.
# =====================================================================================
if RAVE:
    PAL = np.array([(1.0, 0.04, 0.80), (0.04, 0.80, 1.0), (0.55, 0.08, 1.0), (0.30, 1.0, 0.12), (1.0, 0.32, 0.02)])
    bar_i = np.floor((IDX - DOWNBEATS[0]) / BAR).astype(int)
    HUE = PAL[(bar_i // 2) % len(PAL)]
    HUE2 = PAL[(bar_i // 2 + 2) % len(PAL)]
    flying = (IDX >= LIFT).astype(np.float64)
    # --- v4 LED keys (LED-pixel shader): global Beats / Level / Strobe / Hue on each LED material, and per
    #     object (gain, alarm, phase). Story: dim on the pad, strips + nose chase from liftoff, the rings BOOT
    #     one per bar through the 0:33 build (bottom -> top), ALL RED pulsing with the alarm, a flicker when
    #     booster 3 blows, red through the tumble, then a rainbow REBOOT sweep bottom -> top on the 1:07 drop.
    BEATS = np.maximum.accumulate(TL["T_bi"].astype(np.float64) + TL["T_ph"].astype(np.float64))
    _alarm_on = smoothstep(ramp(IDX, B3ALARM, B3ALARM + BAR / 2))
    _sec = dict(
        intro=((IDX >= LIFT) & (IDX < BUILD0)), build=((IDX >= BUILD0) & (IDX < B3ALARM)),
        alarm=((IDX >= B3ALARM) & (IDX < REBOOT)), drop=SEC_DROP > 0, bridge=((IDX >= BRIDGE) & (IDX < BSEP)),
        bd=SEC_BD > 0, outro=IDX >= AFTER)
    LVL = np.full(N, 0.05)
    LVL = np.where(_sec["intro"], 0.30 + 1.0 * KICK4 + 0.40 * BASS4 + 0.45 * HAT4, LVL)
    LVL = np.where(_sec["build"], 0.35 + 1.2 * KICK4 + 0.50 * HAT4 + 0.4 * ramp(IDX, BUILD0, B3ALARM), LVL)
    LVL = np.where(_sec["alarm"], 0.32 + 1.5 * KICK4, LVL)
    LVL = np.where(_sec["drop"], 0.35 + 1.6 * KICK4 + 0.9 * LEAD, LVL)
    LVL = np.where(_sec["bridge"], 0.30 + 1.0 * KICK4 + 0.5 * SYNTH, LVL)
    LVL = np.where(_sec["bd"], 0.30 + 1.3 * SYNTH + 0.3 * KICK4 + 0.3 * HAT4, LVL)
    LVL = np.where(_sec["outro"], 0.25 + 0.8 * SYNTH + 0.6 * KICK4, LVL)
    _tbm = (IDX - B3BOOM) / FPS                          # blast: the LED bus browns out and flickers ~1.2 s
    _flk = np.array([hash01(i, 77) for i in range(N)])
    LVL = LVL * np.where((_tbm >= 0) & (_tbm < 1.2), np.where(_flk < 0.45, 0.08, 1.0) * (0.3 + 0.7 * ramp(_tbm, 0, 1.2)), 1.0)
    IMPACT = gsmooth(EL["impact"].astype(np.float64), 1.5) * 2.0
    STRB = np.clip(0.7 * STROBE * HOT * (1 - _alarm_on * (IDX < REBOOT)) + 0.8 * IMPACT, 0, 1)
    bar_i = np.floor((IDX - DOWNBEATS[0]) / BAR).astype(int)
    HUEOFF = 0.07 * bar_i + 0.15 * (BEATS * 0.0)
    for nt_ in (M_LEDPIX.node_tree,):
        bake(nt_, value_path("Beats"), BEATS)
        bake(nt_, value_path("Level"), LVL)
        bake(nt_, value_path("Strobe"), STRB)
        bake(nt_, value_path("Hue"), HUEOFF, const=True)

    def _zof(o):
        if o.type != "MESH":
            return 0.0
        co_ = np.empty(len(o.data.vertices) * 3, np.float32)
        o.data.vertices.foreach_get("co", co_)
        return float(co_.reshape(-1, 3)[:, 2].mean()) + (o.parent.location.z if o.parent in BOOSTERS else 0.0)

    _rings = [(o, kind, _zof(o)) for o, kind, _ in LED_OBS if kind in ("core", "srb", "s2")]
    _order = sorted(range(len(_rings)), key=lambda k: _rings[k][2])
    _nb = int(round((B3ALARM - BUILD0) / BAR))           # bars in the first half of the build (8)
    REBOOT_LEN = BAR                                     # the sweep climbs the stack in one bar
    for rank, k in enumerate(_order):
        o, kind, z = _rings[k]
        boot = snap(BUILD0 + BAR * min(int(rank * _nb / len(_rings)), _nb - 1))
        g = np.where(IDX < boot, np.where(IDX >= LIFT, 0.10, 0.05), 1.0)
        g = g + 2.5 * np.exp(-np.maximum(IDX - boot, 0) / 4.0) * (IDX >= boot)          # boot flash
        t_sw = REBOOT + int(round(REBOOT_LEN * np.clip(z / 80.0, 0, 1)))
        g = g + 3.0 * np.exp(-np.maximum(IDX - t_sw, 0) / 5.0) * (IDX >= t_sw)          # reboot flash
        al = _alarm_on * (IDX < t_sw)
        set_color_keys(o, np.stack([g, al, np.full(N, (z / 64.0) * 0.5)], 1), const=False)
    for o, kind, ph_ in LED_OBS:
        if kind == "strip":
            t_sw = REBOOT + int(round(REBOOT_LEN * 0.5))
            g = np.where(IDX >= LIFT, 1.0, 0.05) + 3.0 * np.exp(-np.maximum(IDX - t_sw, 0) / 5.0) * (IDX >= t_sw)
            set_color_keys(o, np.stack([g, _alarm_on * (IDX < t_sw), np.zeros(N)], 1), const=False)
        elif kind == "nose":
            t_sw = REBOOT + REBOOT_LEN
            redn = (_alarm_on * (IDX < t_sw))[:, None]
            lv = 0.06 * (1 - flying) + flying * (1.0 + 3.0 * KICK4 + 4.0 * STRB + 1.2 * HAT4) * (LVL / np.maximum(LVL.mean(), 1e-3)) ** 0.5
            col = (0.45 + 0.55 * HUE) * (1 - redn) + np.array([1.0, 0.03, 0.03])[None, :] * redn
            set_color_keys(o, col * lv[:, None], const=False)
            bake(NOSE_L, "energy", lv * 3000.0)
            bake_vec(NOSE_L, "color", col, const=True)
    # --- tower + pad LEDs: breathe/strobe with the intro before liftoff (hi-hats), a white strobe at
    #     ignition, then each tower LEVEL lights as the rocket passes it and stays pulsing with the kicks
    TLV = np.where(IDX < IGN, 0.35 + 0.9 * KICK4 + 0.7 * HAT4 + 0.15 * np.sin(IDX / FPS * 2.4),
                   0.55 + 1.1 * KICK4 + 0.4 * HAT4)
    TSTR = np.where((IDX >= IGN) & (IDX < IGN + 4), 1.0, 0.0) + np.where(IDX < IGN, 0.6 * (HAT4 > 0.7), 0.0)
    nt_ = M_LEDTOWER.node_tree
    bake(nt_, value_path("Beats"), BEATS * np.where(IDX < IGN, 1.0, 2.0))
    bake(nt_, value_path("Level"), TLV)
    bake(nt_, value_path("Strobe"), np.clip(TSTR, 0, 1))
    bake(nt_, value_path("Hue"), 0.11 * bar_i, const=True)
    _rocket_mid = ALT + 45.0                             # the stack's middle, metres above the pad deck station
    for o, zl in TOWER_LEDS:
        passed = np.nonzero((_rocket_mid >= zl) & (IDX >= LIFT))[0]
        tp = int(passed[0]) if len(passed) else N
        g = np.where(IDX < tp, 0.35, 1.3) + 3.0 * np.exp(-np.maximum(IDX - tp, 0) / 5.0) * (IDX >= tp)
        set_color_keys(o, np.stack([g, np.zeros(N), np.full(N, zl * 0.004)], 1), const=False)
        show(o, SET == 0)
    for o in (DECK_LED, LT_LED):
        set_color_keys(o, np.stack([np.full(N, 0.9), np.zeros(N), np.full(N, 0.3)], 1), const=False)
        show(o, SET == 0)
    for k, lo in enumerate(TOWER_SPILL):
        bake(lo.data, "energy", 2.5e5 * TLV * (1 + 2.0 * TSTR))
        bake_vec(lo.data, "color", 0.35 + 0.65 * (HUE if k == 0 else HUE2), const=True)
        show(lo, SET == 0)

    # --- v4 THE SPEAKER (Zeke: ONE BIG speaker, more realistic, more LEDs, more reactive). The cabinet is
    #     "Stage Speaker - black" by Sousinho (CC-BY 4.0, 4K PBR), normalised to 16 m tall at scale 1, turned so
    #     its front (+Y in the asset, verified with a +Y marker render AND numerically: the woofer islands sit at
    #     max Y) faces local -Y as v3 expects; its static woofer cones are cut out and replaced by v3's animated
    #     lathe cones. Plus: LED-pixel rings round the woofers + an LED edge frame, 32-band LED matrices down both
    #     sides (the same spectrum image), a blinder bank, four moving heads (CC-BY Lord Henry) on the lid, and
    #     lasers (emissive geometry + the compositor glare) that fire in DROP 2 only.
    M_CONE = principled("SpeakerCone", (0.05, 0.05, 0.055), rough=0.35, bump=(6.0, 0.05))
    M_LEDSPK = led_pixel_material("LEDPixelSpeaker", 16.0, 7.0, "ring", hue_per_m=0.05)
    M_LEDSPKD = led_pixel_material("LEDPixelSpeakerEdge", 14.0, 6.0, "diag", hue_per_m=0.05)
    SPK = empty("Speaker")
    hx, hy, hz = SPK_HALF

    def import_gltf(path):
        before = set(bpy.data.objects.keys())
        bpy.ops.import_scene.gltf(filepath=path)
        new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
        bpy.context.view_layer.update()
        return new

    def baked_mesh(src, name, xf):
        """copy src's mesh with its WORLD matrix and the 4x4 numpy xf applied to the vertices (no object
        transforms left to validate: the import scale/rotation is baked in, rules: validate-imported-mesh-scale)"""
        me = src.data.copy()
        me.name = name
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = np.c_[co.reshape(-1, 3), np.ones(len(me.vertices))]
        mw = np.array(src.matrix_world)
        co = (xf @ (mw @ co.T)).T[:, :3]
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.update()
        return me

    _new = import_gltf(ARGS.speaker_gltf)
    _src = [o for o in _new if o.type == "MESH"]
    assert len(_src) == 1, [o.name for o in _new]
    _src = _src[0]
    _pts = np.array([list(_src.matrix_world @ Vector(c)) for c in _src.bound_box])
    _lo, _hi = _pts.min(0), _pts.max(0)
    _cen = (_lo + _hi) / 2
    SPK_S = 2 * hz / (_hi[2] - _lo[2])                  # 16 m tall at scale 1 (= 0.0618 per asset unit)
    _xf = np.diag([-SPK_S, -SPK_S, SPK_S, 1.0])         # rotate 180 deg about Z (front +Y -> -Y) and scale
    _xf[:3, 3] = [SPK_S * _cen[0], SPK_S * _cen[1], -SPK_S * _cen[2]]
    _me = baked_mesh(_src, "SpeakerCabinet", _xf)
    # the woofer centres + radius, measured from the asset's loose islands (raw units, front face +Y):
    # cones at z 4.2 and 91.9, x 0.9, radius ~17.7, rim at y 54.7 — cut those islands out (bmesh)
    _woof_raw = [(0.9, 54.7, 4.15), (0.9, 54.7, 91.9)]
    _woof = [tuple((_xf @ np.array([*w, 1.0]))[:3]) for w in _woof_raw]
    WOOF_R = 17.7 * SPK_S
    import bmesh as _bm_
    bm = _bm_.new()
    try:
        bm.from_mesh(_me)
        bm.verts.ensure_lookup_table()
        seen, kill = set(), []
        for v in bm.verts:
            if v.index in seen:
                continue
            stack, comp = [v], []
            seen.add(v.index)
            while stack:
                x = stack.pop()
                comp.append(x)
                for e in x.link_edges:
                    y = e.other_vert(x)
                    if y.index not in seen:
                        seen.add(y.index)
                        stack.append(y)
            P = np.array([list(x.co) for x in comp])
            for (wx, wy, wz) in _woof:
                rr = np.hypot(P[:, 0] - wx, P[:, 2] - wz)
                if rr.max() < WOOF_R * 1.06 and P[:, 1].max() < wy + 0.9 and P[:, 1].min() > wy - 1.0:
                    kill += comp
                    break
        _bm_.ops.delete(bm, geom=list(set(kill)), context="VERTS")
        bm.to_mesh(_me)
    finally:
        bm.free()
    print(f"[rocket4] speaker: asset scale {SPK_S:.4f}, cut {len(set(kill))} woofer-cone verts, woofers at",
          [tuple(round(c, 2) for c in w) for w in _woof])
    CAB = link(bpy.data.objects.new("SpeakerCabinet", _me), SPK)
    for o in _new:
        bpy.data.objects.remove(o, do_unlink=True)
    _cab_pts = np.array([list(v.co) for v in _me.vertices])
    assert _cab_pts[:, 1].min() < -0.9 * hy, "speaker front must face local -Y"
    CONES, SPK_LEDS, SPK_LIGHTS = [], [], []
    for k, (wx, wy, wz) in enumerate(_woof):
        sc_ = WOOF_R / 3.45
        prof = [(3.45 * sc_, 0.0), (3.2 * sc_, -0.15 * sc_), (1.0 * sc_, -1.15 * sc_), (0.95 * sc_, -1.05 * sc_),
                (0.6 * sc_, -0.75 * sc_), (0.0, -0.62 * sc_)]
        cone = lathe(f"SpeakerCone{k}", prof, 64, M_CONE, SPK, loc=(wx, wy - 0.02, wz))
        cone.rotation_euler = (math.radians(90), 0, 0)
        CONES.append(cone)
        ring = led_ring(f"SpeakerLEDRing{k}", WOOF_R + 0.16, 0.0, SPK, 0.07, M_LEDSPK)
        ring.location = (wx, wy - 0.12, wz)
        ring.rotation_euler = (math.radians(90), 0, 0)
        SPK_LEDS.append((ring, 0, k * 0.5))
    edge = MB()
    for zc in (-hz + 0.08, hz - 0.08):
        edge.box((0, -hy - 0.06, zc), (2 * hx - 0.1, 0.1, 0.12))
    for xc in (-hx + 0.06, hx - 0.06):
        edge.box((xc, -hy - 0.06, 0), (0.12, 0.1, 2 * hz - 0.1))
    for xc in (-hx - 0.04, hx + 0.04):                   # rear vertical edges too (seen in the orbit shots)
        for yc in (-hy + 0.05, hy - 0.05):
            edge.box((xc, yc, 0), (0.08, 0.08, 2 * hz - 0.1))
    SPK_LEDS.append((no_shadow(edge.obj("SpeakerLEDEdge", M_LEDSPKD, SPK)), 1, 0.25))
    # side LED matrices: 32 bands x 24 rows, a plane on each side (UV u = front -> back, v = up)
    M_SPEC_SPK = spectrum_material("SpectrumSpeaker", 9.0, 32, 24)
    SPK_SPEC = []
    for sx in (-1, 1):
        x = sx * (hx + 0.03)
        y0, y1, z0, z1 = -hy + 0.35, hy - 0.35, -hz + 0.8, hz - 0.8
        verts = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
        face = [(0, 1, 2, 3)] if sx > 0 else [(3, 2, 1, 0)]
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        o = mesh_obj(f"SpeakerSpectrum{'LR'[sx > 0]}", verts, face, M_SPEC_SPK, SPK, uvs=uvs)
        SPK_SPEC.append(no_shadow(o))
    # blinder bank: 2 x 4 white cells across the top-front edge
    M_BLIND = emissive("Blinder", (1.0, 0.96, 0.9), 40.0, objcolor_gain=True)
    bl_ = MB()
    for r_ in range(2):
        for c_ in range(4):
            bl_.box(((c_ - 1.5) * 1.45, -hy - 0.12, hz - 0.55 - r_ * 0.75), (1.1, 0.12, 0.55))
    BLINDER = no_shadow(bl_.obj("SpeakerBlinders", M_BLIND, SPK))
    for k, zc in enumerate((-4.0, 4.0)):
        ld = bpy.data.lights.new(f"SpeakerLight{k}", "POINT")
        ld.shadow_soft_size = 2.0
        ld.use_shadow = False
        lo = link(bpy.data.objects.new(f"SpeakerLight{k}", ld), SPK)
        lo.location = (0, -hy - 9.0, zc)
        SPK_LIGHTS.append(ld)
    STROBE_L = bpy.data.lights.new("SpeakerStrobe", "POINT")
    STROBE_L.shadow_soft_size = 3.0
    STROBE_L.use_shadow = False
    STROBE_L.color = (1.0, 0.97, 0.92)
    STROBE_LO = link(bpy.data.objects.new("SpeakerStrobe", STROBE_L), SPK)
    STROBE_LO.location = (0, -hy - 6.0, hz - 1.0)

    # --- moving heads on the lid (CC-BY "Moving Head Beam (high Poly)" by Lord Henry). Rest pose: lens UP (+Z),
    #     checked on a marker render (top view shows the lens). pan = Z on the yoke, tilt = X through the yoke.
    _mh = import_gltf(ARGS.movinghead)
    _parts = {o.name.split(".")[0]: o for o in _mh if o.type == "MESH"}
    assert {"Base_0", "Head_0", "Support_0"} <= set(_parts), sorted(_parts)
    MH_S = SPK_TOP / 0.484                               # 0.48 m fixture -> 1.75 scale-1 units (4.2 m at full size)
    MH_TILT_Z = 0.33 * MH_S
    _mhxf = np.diag([MH_S, MH_S, MH_S, 1.0])
    MH_MESH = {k: baked_mesh(o, f"MH_{k}", _mhxf) for k, o in _parts.items()}
    for _im in bpy.data.images:                         # render workers get the asset textures inside the .blend
        if _im.packed_file is None and _im.source == "FILE" and _im.filepath:
            try:
                _im.pack()
            except RuntimeError as e_:
                print("[rocket4] WARN could not pack", _im.name, e_)
    _hco = np.empty(len(MH_MESH["Head_0"].vertices) * 3, np.float32)
    MH_MESH["Head_0"].vertices.foreach_get("co", _hco)
    _hco = _hco.reshape(-1, 3)
    _hco[:, 2] -= MH_TILT_Z
    MH_MESH["Head_0"].vertices.foreach_set("co", _hco.ravel())
    MH_MESH["Head_0"].update()
    _lens_z = float(_hco[:, 2].max())
    for o in _mh:
        bpy.data.objects.remove(o, do_unlink=True)
    M_MHBEAM = emissive("MovingHeadBeam", (1.0, 1.0, 1.0), 6.0, objcolor_gain=True)
    M_MHLENS = emissive("MovingHeadLens", (1.0, 1.0, 1.0), 30.0, objcolor_gain=True)
    MHEADS = []
    for k, (mx, my) in enumerate(((-2.2, -1.6), (2.2, -1.6), (-2.2, 1.6), (2.2, 1.6))):
        root = empty(f"MH{k}", loc=(mx, my, hz), parent=SPK)
        link(bpy.data.objects.new(f"MH{k}_base", MH_MESH["Base_0"]), root)
        pan = empty(f"MH{k}_pan", parent=root)
        link(bpy.data.objects.new(f"MH{k}_yoke", MH_MESH["Support_0"]), pan)
        tilt = empty(f"MH{k}_tilt", loc=(0, 0, MH_TILT_Z), parent=pan)
        link(bpy.data.objects.new(f"MH{k}_head", MH_MESH["Head_0"]), tilt)
        lens = no_shadow(lathe(f"MH{k}_lens", [(0.0, _lens_z + 0.01), (0.33, _lens_z + 0.01)], 24, M_MHLENS, tilt))
        beam = no_shadow(lathe(f"MH{k}_beam", [(0.25, _lens_z), (0.6, _lens_z + 40.0), (2.6, _lens_z + 260.0)],
                               16, M_MHBEAM, tilt))
        MHEADS.append((pan, tilt, lens, beam))

    # --- lasers (DROP 2 only): 4 GN fans of opaque emissive beams (research #1 §8: ~+12 % render vs +170 %
    #     for blended beams; the halo comes from the existing Fog Glow) + a 12-beam "tunnel" aimed at the rock
    def laser_gn_mat(name):
        m, nt = new_mat(name)
        at = nd(nt, "ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="beam_col")
        em = nd(nt, "ShaderNodeEmission", inp={"Strength": 40.0})
        lk(nt, at.outputs["Color"], em.inputs["Color"])
        o = nd(nt, "ShaderNodeOutputMaterial")
        lk(nt, em.outputs[0], o.inputs["Surface"])
        return m

    M_LASER = laser_gn_mat("LaserBeam")
    M_LASER_OBJ = emissive("LaserTunnel", (1.0, 1.0, 1.0), 40.0, objcolor_gain=True)

    def laser_fan_tree():
        t, s_, gi, go = gn_tree("LaserFan", [("Count", "NodeSocketInt", 16), ("Spread", "NodeSocketFloat", 1.0),
                                             ("Sweep", "NodeSocketFloat", 0.0), ("Length", "NodeSocketFloat", 400.0),
                                             ("Radius", "NodeSocketFloat", 0.15),
                                             ("Color", "NodeSocketVector", (0.1, 1.0, 0.2))])
        pts = t.nodes.new("GeometryNodePoints")
        t.links.new(gi.outputs["Count"], pts.inputs["Count"])
        cyl = t.nodes.new("GeometryNodeMeshCylinder")
        cyl.inputs["Vertices"].default_value = 6
        cyl.inputs["Radius"].default_value = 1.0
        cyl.inputs["Depth"].default_value = 1.0
        tr_ = t.nodes.new("GeometryNodeTransform")
        t.links.new(cyl.outputs["Mesh"], tr_.inputs["Geometry"])
        tr_.inputs["Translation"].default_value = (0, 0, 0.5)          # base at the aperture, along +Z
        idx = t.nodes.new("GeometryNodeInputIndex")
        cm1 = g_math(t, "MAXIMUM", g_math(t, "SUBTRACT", gi.outputs["Count"], 1.0), 1.0)
        fr = g_math(t, "SUBTRACT", g_math(t, "DIVIDE", idx.outputs[0], cm1), 0.5)
        ang = g_math(t, "MULTIPLY_ADD", fr, gi.outputs["Spread"], gi.outputs["Sweep"])
        rot = t.nodes.new("ShaderNodeCombineXYZ")
        t.links.new(ang, rot.inputs["Y"])
        scl = t.nodes.new("ShaderNodeCombineXYZ")
        t.links.new(gi.outputs["Radius"], scl.inputs["X"])
        t.links.new(gi.outputs["Radius"], scl.inputs["Y"])
        t.links.new(gi.outputs["Length"], scl.inputs["Z"])
        iop = t.nodes.new("GeometryNodeInstanceOnPoints")
        t.links.new(pts.outputs[0], iop.inputs["Points"])
        t.links.new(tr_.outputs[0], iop.inputs["Instance"])
        t.links.new(rot.outputs[0], iop.inputs["Rotation"])
        t.links.new(scl.outputs[0], iop.inputs["Scale"])
        sna = t.nodes.new("GeometryNodeStoreNamedAttribute")
        sna.data_type = "FLOAT_COLOR"
        sna.domain = "INSTANCE"
        sna.inputs["Name"].default_value = "beam_col"
        t.links.new(iop.outputs[0], sna.inputs["Geometry"])
        t.links.new(gi.outputs["Color"], sna.inputs["Value"])
        sm = t.nodes.new("GeometryNodeSetMaterial")
        sm.inputs["Material"].default_value = M_LASER
        t.links.new(sna.outputs[0], sm.inputs["Geometry"])
        t.links.new(sm.outputs[0], go.inputs["Geometry"])
        return t

    FAN_T = laser_fan_tree()
    FANS = []
    for k, (fx, fz, el) in enumerate(((-hx + 0.4, hz - 0.3, 14.0), (hx - 0.4, hz - 0.3, 14.0),
                                      (-hx + 0.4, 0.0, 2.0), (hx - 0.4, 0.0, 2.0))):
        o, md = gn_obj(f"LaserFan{k}", FAN_T, SPK)
        o.location = (fx, -hy - 0.2, fz)
        o.rotation_euler = (math.radians(90 - el), 0.0, 0.0)            # +Z -> front (-Y), tilted up `el`
        FANS.append((o, md))
    # the tunnel: 12 beams from round the front face converge on the pinned rock (aimed at FIRE, in the
    # speaker's local frame; the cabinet only jitters a few metres with the recoil, so the aim holds)
    _Mf = (Matrix.Translation(Vector(spk_loc[FIRE])) @ Euler_(spk_rot[FIRE]).to_matrix().to_4x4()
           @ Matrix.Scale(float(spk_scl[FIRE]), 4))
    _tgt = _Mf.inverted() @ A_STOP
    TUNNEL = []
    for k in range(12):
        a = 2 * math.pi * k / 12
        p0 = Vector((0.85 * hx * math.cos(a), -hy - 0.2, 0.8 * hz * math.sin(a)))
        d_ = _tgt - p0
        o = no_shadow(lathe(f"LaserTunnel{k}", [(1.0, 0.0), (1.0, d_.length)], 6, M_LASER_OBJ, SPK))
        o.location = p0
        o.rotation_euler = d_.to_track_quat("Z", "Y").to_euler()
        TUNNEL.append(o)

    bake_vec(SPK, "location", spk_loc)
    bake_vec(SPK, "rotation_euler", spk_rot)
    bake_vec(SPK, "scale", np.repeat(spk_scl[:, None], 3, 1))
    spk_vis = IDX >= FAIRSEP - 2
    for o in [SPK] + descendants(SPK):
        show(o, spk_vis)
    # power-up after it clears the stack; then the music: breakdowns = the SYNTH leads, drop 2 = kick + screech
    boot = smoothstep(ramp(IDX, DEP0 + 20, DEP1))
    charge = ramp(IDX, snap(144.0 * FPS), FIRE)                  # the wind-up before the drop
    FIREWIN_ = ((IDX >= FIRE) & (IDX < AFTER)).astype(np.float64)
    spk_lvl = boot * (0.30 + 0.9 * KICK4 + 0.35 * BASS4 + 0.4 * HAT4 + SEC_BD * 1.2 * SYNTH
                      + FIREWIN_ * (1.4 * KICK4 + 0.9 * LEAD) + 0.8 * charge + 0.6 * (IDX >= FINAL) * KICK4
                      + (IDX >= AFTER) * 0.6 * SYNTH)
    spk_lvl = spk_lvl * (1.0 - smoothstep(ramp(IDX, FADE - 6, FADE)))
    spk_strb = np.clip(FIREWIN_ * (KICK4 > 0.85) + 0.6 * STROBE * HOT * SEC_BD + 0.8 * IMPACT * (IDX >= FAIRSEP), 0, 1)
    for nt_ in (M_LEDSPK.node_tree, M_LEDSPKD.node_tree):
        bake(nt_, value_path("Beats"), BEATS * (1.0 + 1.0 * (IDX >= FIRE)))
        bake(nt_, value_path("Level"), spk_lvl)
        bake(nt_, value_path("Strobe"), spk_strb)
        bake(nt_, value_path("Hue"), 0.09 * bar_i, const=True)
    for o, which, ph_ in SPK_LEDS:
        set_color_keys(o, np.stack([np.ones(N), np.zeros(N), np.full(N, ph_)], 1), const=False)
    nts = M_SPEC_SPK.node_tree
    bake(nts, value_path("Frame"), IDX.astype(np.float64), const=True)
    bake(nts, value_path("Gain"), boot * (0.6 + 0.8 * KICK4 * (SEC_DROP + 0.3)) * (1.0 - smoothstep(ramp(IDX, FADE - 6, FADE))))
    bake(nts, value_path("Alarm"), np.zeros(N), const=True)
    blind = boot * np.clip(FIREWIN_ * (KICK4 > 0.85) * KICK4 * 1.2 + 0.7 * STROBE * HOT * SEC_BD + IMPACT * (IDX >= FAIRSEP), 0, 1.5)
    set_color_keys(BLINDER, np.stack([blind] * 3, 1), const=False)
    bake(STROBE_L, "energy", blind * 4.0e5)
    for k, ld in enumerate(SPK_LIGHTS):
        bake(ld, "energy", spk_lvl * 2.5e4)
        bake_vec(ld, "color", HUE if k == 0 else HUE2, const=True)
    # cones: breathe with the bass, punch out on every kick, slam in drop 2; wound back on the charge
    exc = boot * (0.25 * KICK4 + 0.12 * BASS4) + FIREWIN_ * 0.55 * KICK4 + (IDX >= FINAL) * 0.4 * KICK4 \
        - 0.35 * charge * (IDX < FIRE)
    CONE_EXC = 0.55 * exc                                        # scale-1 units (x2.4 on screen)
    for k, cone in enumerate(CONES):
        y0 = cone.location.y
        bake_vec(cone, "location", np.stack([np.full(N, cone.location.x), y0 - CONE_EXC,
                                             np.full(N, cone.location.z)], 1))
    # moving heads: lens glow follows the music; pan/tilt sweep on the bars (faster in drop 2); BEAMS in drop 2 only
    mh_rate = np.where(IDX >= FIRE, 2.0, 1.0)
    _mhph = np.cumsum(mh_rate) / FPS
    for k, (pan, tilt, lens, beam) in enumerate(MHEADS):
        sgn = 1 if k % 2 == 0 else -1
        pz = np.radians(35.0) * np.sin(_mhph * 1.3 + k * 1.1) * sgn + (np.pi if k >= 2 else 0.0)
        tx = np.radians(55.0 + 25.0 * np.sin(_mhph * 0.9 + k)) * boot
        bake_vec(pan, "rotation_euler", np.stack([np.zeros(N), np.zeros(N), pz], 1))
        bake_vec(tilt, "rotation_euler", np.stack([tx, np.zeros(N), np.zeros(N)], 1))
        hcol = HUE if k % 2 == 0 else HUE2
        lv = boot * (0.25 + 1.2 * KICK4 * SEC_DROP + 0.9 * SYNTH * SEC_BD + 0.4 * HAT4)
        set_color_keys(lens, (0.4 + 0.6 * hcol) * lv[:, None], const=False)
        set_color_keys(beam, (0.4 + 0.6 * hcol) * (FIREWIN_ * (0.35 + 1.1 * KICK4))[:, None], const=False)
        show(beam, spk_vis & (FIREWIN_ > 0))
    # lasers: keyed per frame; fan / sweep / sheet modes change every 2 bars, beams blank on hi-hat flicks
    LASER_COLS = np.array([(0.1, 1.0, 0.15), (1.0, 0.05, 0.08), (0.15, 0.35, 1.0), (0.1, 1.0, 1.0),
                           (1.0, 0.85, 0.05), (1.0, 0.08, 0.9)])
    mode = (bar_i // 2) % 3
    blank = np.array([1.0 if (hash01(i, 91) > 0.25 or HAT4[i] < 0.6) else 0.0 for i in range(N)])
    las_int = FIREWIN_ * (0.45 + 1.3 * KICK4 + 0.8 * LEAD) * blank
    # beam radius: >= 1.2 px at the camera's distance so the beams never shimmer away under TAA
    _cdist = np.linalg.norm(cam_loc - spk_loc, axis=1)
    _px = _cdist * (36.0 / np.maximum(cam_lens, 1.0)) / 1920.0
    las_r = np.maximum(0.3, 1.2 * _px) / np.maximum(spk_scl, 1e-3)
    for k, (o, md) in enumerate(FANS):
        cnt = np.where(mode == 2, 18, 10).astype(np.float64)
        spread = np.where(mode == 2, 1.4, np.where(mode == 1, 0.5, 0.9))
        sweep = np.where(mode == 1, 0.6 * np.sin(BEATS * math.pi / 2 + k), 0.15 * np.sin(BEATS * math.pi / 4 + k))
        col = LASER_COLS[((bar_i // 2) + k) % len(LASER_COLS)] * (las_int * (0.5 if False else 1.0))[:, None]
        col = col * np.where(mode == 2, 0.3, 1.0)[:, None]
        bake(o, gn_input_path(md, FAN_T, "Count"), cnt, const=True)
        bake(o, gn_input_path(md, FAN_T, "Spread"), spread)
        bake(o, gn_input_path(md, FAN_T, "Sweep"), sweep)
        bake(o, gn_input_path(md, FAN_T, "Radius"), las_r)
        bake(o, gn_input_path(md, FAN_T, "Length"), np.full(N, 420.0))
        for c_ in range(3):
            bake(o, gn_input_path(md, FAN_T, "Color"), col[:, c_], index=c_)
        show(o, spk_vis & (las_int > 0.01))
    tun_on = FIREWIN_ * ((bar_i // 2) % 2 == 1) * (0.3 + 1.5 * KICK4) * blank
    for k, o in enumerate(TUNNEL):
        tcol = LASER_COLS[(bar_i // 2 + 3) % len(LASER_COLS)] * tun_on[:, None]
        set_color_keys(o, tcol, const=False)
        bake_vec(o, "scale", np.stack([las_r, las_r, np.ones(N)], 1))
        show(o, spk_vis & (tun_on > 0.01))

    # --- SOUND WAVES: a pool of neon rings, one launched off the woofers on every kick from the
    #     wind-up on; they travel at 720 m/s and spread to the rock's size by the time they land.
    M_WAVE = emissive("SoundWave", (1.0, 1.0, 1.0), 5.0, objcolor_gain=True)
    NR = 10
    q_w = D_HAT.to_track_quat("Z", "Y").to_euler()
    for r_ in range(NR):
        prof = [(1.0 + 0.022 * math.cos(a), 0.06 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)]
        w = no_shadow(lathe(f"SoundWave{r_}", prof, 128, M_WAVE))
        w.rotation_euler = q_w
        mine = WAVE_K[r_::NR]
        loc = np.zeros((N, 3))
        scl = np.full((N, 3), 1e-3)
        col = np.zeros((N, 3))
        vis = np.zeros(N, bool)
        for j, kf in enumerate(mine):
            end = mine[j + 1] if j + 1 < len(mine) else N
            ii = np.arange(kf, end)
            age = ii - kf
            dist = FACE_OFF + WAVE_V * age
            amp = np.where(kf < FIRE, 0.35, np.where(kf < AFTER, 1.0, np.where(kf >= FINAL, 1.0, 0.55)))
            life = np.clip(age / 2.0, 0, 1) * np.exp(-age / 40.0) * amp
            rad = 16.0 + 0.135 * (dist - FACE_OFF)
            base = SPK_P + D_HAT * 0.0
            loc[ii] = np.array(base)[None, :] + np.array(D_HAT)[None, :] * dist[:, None]
            scl[ii] = np.stack([rad, rad, rad], 1)
            col[ii] = HUE[kf][None, :] * life[:, None] * 3.0
            vis[ii] = life > 0.01
        bake_vec(w, "location", loc)
        bake_vec(w, "scale", scl)
        set_color_keys(w, col, const=False)
        show(w, vis)

    # --- THE ASTEROID: NASA's PD model of Bennu (1999 RQ36), normalised to a 220 m rock, a CC0 rock
    #     scan on it, glowing veins keyed through Object Info as the waves land.
    _before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=os.path.join(AS, "nasa3d", "bennu", "bennu.glb"))
    _new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - _before]
    _meshes = [o for o in _new if o.type == "MESH"]
    assert _meshes, "Bennu import produced no mesh"
    _src = max(_meshes, key=lambda o: len(o.data.vertices))
    me_ = _src.data.copy()
    _co = np.empty(len(me_.vertices) * 3, np.float32)
    me_.vertices.foreach_get("co", _co)
    _co = _co.reshape(-1, 3).astype(np.float64)
    _mw = np.array(_src.matrix_world)
    _co = (_co @ _mw[:3, :3].T) + _mw[:3, 3]
    _co -= _co.mean(0)
    _co /= np.linalg.norm(_co, axis=1).max()
    me_.vertices.foreach_set("co", _co.astype(np.float32).ravel())
    me_.update()
    for o in _new:
        bpy.data.objects.remove(o, do_unlink=True)

    def rock_material(name):
        m, nt = new_mat(name)
        rk = lambda kind: os.path.join(AS, "ambientcg", "Rock030", f"Rock030_2K-JPG_{kind}.jpg")
        col = tex_box(nt, rk("Color"), 0.16)
        rgh = tex_box(nt, rk("Roughness"), 0.16, True)
        nrm = tex_box(nt, rk("NormalGL"), 0.16, True)
        b = nd(nt, "ShaderNodeBsdfPrincipled")
        lk(nt, col_mul(nt, col.outputs["Color"], 0.30), b.inputs["Base Color"])   # carbonaceous: dark
        lk(nt, rgh.outputs["Color"], b.inputs["Roughness"])
        nm = nd(nt, "ShaderNodeNormalMap", inp={"Strength": 1.3})
        lk(nt, nrm.outputs["Color"], nm.inputs["Color"])
        lk(nt, nm.outputs["Normal"], b.inputs["Normal"])
        tc = nd(nt, "ShaderNodeTexCoord")
        wn_ = nd(nt, "ShaderNodeTexNoise", inp={"Scale": 2.5, "Detail": 6.0})
        warp = V(nt, "ADD", tc.outputs["Object"], V(nt, "SCALE", V(nt, "SUBTRACT", wn_.outputs["Color"], (0.5, 0.5, 0.5)), scale=0.7))
        vor = nd(nt, "ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE", inp={"Scale": 3.0, "Randomness": 1.0})
        lk(nt, warp, vor.inputs["Vector"])
        oi0 = nd(nt, "ShaderNodeObjectInfo")
        # vein width grows with the damage (Object Info alpha-free: use the colour's G as width)
        sepc = nd(nt, "ShaderNodeSeparateColor")
        lk(nt, oi0.outputs["Color"], sepc.inputs[0])
        width = M(nt, "ADD", 0.0015, M(nt, "MULTIPLY", sepc.outputs[1], 0.007, clamp=True))
        veins = M(nt, "LESS_THAN", vor.outputs["Distance"], width)
        # glow colour from R (heat) and B (tint) only — G carries the vein width
        hotc = nd(nt, "ShaderNodeCombineColor")
        lk(nt, sepc.outputs[0], hotc.inputs[0])
        lk(nt, M(nt, "MULTIPLY", sepc.outputs[0], 0.35), hotc.inputs[1])
        lk(nt, sepc.outputs[2], hotc.inputs[2])
        lk(nt, col_mul(nt, hotc.outputs[0], veins), b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = 12.0
        o = nd(nt, "ShaderNodeOutputMaterial")
        lk(nt, b.outputs[0], o.inputs["Surface"])
        return m

    M_ROCK = rock_material("AsteroidRock")
    me_.materials.clear()
    me_.materials.append(M_ROCK)
    AST = link(bpy.data.objects.new("Asteroid", me_))
    me_.polygons.foreach_set("use_smooth", np.ones(len(me_.polygons), bool))
    AST.scale = (AST_R,) * 3
    bake_vec(AST, "location", ast_pos)
    spin = np.cumsum(np.where(IDX < FIRE, 1.0, 0.35) / FPS)
    bake_vec(AST, "rotation_euler", np.stack([0.11 * spin, 0.07 * spin + 0.6, 0.05 * spin], 1))
    crack = np.clip(ramp(IDX, FIRE, BOOM) ** 2.0 * (0.4 + 1.6 * IMP) + 0.5 * IMP, 0, 3.0)
    dmg = ramp(IDX, FIRE, BOOM) ** 1.5                           # vein width (G channel, see rock_material)
    set_color_keys(AST, np.stack([crack * 1.0, np.maximum(dmg, 0.0), crack * 0.9], 1), const=False)
    show(AST, (IDX >= GLIMPSE) & (IDX < BOOM))
    # a soft fill so the night side still reads (the sun is behind it from most cameras)
    AFILL = bpy.data.lights.new("AsteroidFill", "AREA")
    AFILL.size = 400.0
    AFILL.use_shadow = False
    AFILL.color = (0.55, 0.65, 1.0)
    AFO = link(bpy.data.objects.new("AsteroidFill", AFILL))
    AFO.location = A_STOP - D_HAT * 700 + UPV * 300
    AFO.rotation_euler = (A_STOP - AFO.location).to_track_quat("-Z", "Y").to_euler()
    bake(AFILL, "energy", np.full(N, 6.0e7))
    show(AFO, (IDX >= AST0) & (IDX < BOOM + 3 * FPS))

    # (v4: the shatter debris is Blender rigid-body physics — section 10c, after the speaker keys)
    tB = np.maximum(IDX - BOOM, 0) / FPS
    M_FIRE = smoke_material("AsteroidFire", (0.30, 0.26, 0.24), dens=2.2, aniso=0.3, glow_grid="flames",
                            glow_col=blackbody(2300), plume_lit=False)
    q_up = D_HAT.to_track_quat("Z", "Y").to_euler()
    for k, (sc_, twist) in enumerate(((3.0, 0.0), (2.4, 2.1))):
        rot_ = (q_up.x, q_up.y, q_up.z + twist)
        fo, nfi = vdb_seq(f"AstFire{k}", GROUND_X, sc_, loc=tuple(anchored(A_STOP, rot_, sc_, GROUND_ANCHOR)),
                          rot=rot_, mat=M_FIRE)
        vdb_play(fo, nfi, BOOM - 4, speed=1.1 + 0.2 * k)
        fo.data.render.precision = "HALF"
        show(fo, IDX >= BOOM - 1)
    fglow = (IDX >= BOOM) * (0.4 + 6.0 * np.exp(-tB / 1.2))
    bake(M_FIRE.node_tree, value_path("Glow"), fglow)
    FLASH = bpy.data.lights.new("BoomFlash", "POINT")
    FLASH.shadow_soft_size = 60.0
    FLASH.color = blackbody(3200)
    FLO = link(bpy.data.objects.new("BoomFlash", FLASH))
    FLO.location = A_STOP
    bake(FLASH, "energy", (IDX >= BOOM) * 4.0e9 * np.exp(-tB / 0.25))
    show(FLO, (IDX >= BOOM) & (IDX < BOOM + 3 * FPS))
    blast = no_shadow(lathe("BlastRing", [(1.0 + 0.03 * math.cos(a), 0.04 * math.sin(a))
                                          for a in np.linspace(0, 2 * math.pi, 9)], 160, M_WAVE))
    blast.location = A_STOP
    blast.rotation_euler = q_w
    br_ = 30.0 + 900.0 * tB ** 0.7
    bake_vec(blast, "scale", np.stack([br_, br_, br_], 1))
    set_color_keys(blast, np.stack([np.full(N, 1.0), np.full(N, 0.55), np.full(N, 0.85)], 1)
                   * (4.0 * np.exp(-tB / 0.9) * (IDX >= BOOM))[:, None], const=False)
    show(blast, (IDX >= BOOM) & (IDX < BOOM + 4 * FPS))

    # --- the Earth rocks: the planet's surface jitters and the horizon jumps on every kick
    EQ = FIREWIN * (0.6 * KICK + 0.8 * np.clip(IMP, 0, 1)) + (IDX >= BOOM) * 1.6 * np.exp(-tB / 0.7) \
        + (IDX >= FINAL) * 0.6 * KICK
    jit = 0.5 * np.sin(IDX * 2.7) + 0.5 * np.sin(IDX * 5.3 + 1.0)
    bake(WORLD.node_tree, value_path("EarthScroll"), -np.cumsum(np.where(SET >= 2, 0.00012, 0.00002)) + 0.0011 * EQ * jit)
    bake(WORLD.node_tree, value_path("EarthH"), (RE + alt_m * (1.0 + 0.015 * EQ * np.sin(IDX * 3.9 + 0.5))) / RE)
    # exposure: the flash pops, a held breath before the drop, black at the very end
    expo = expo + 1.1 * (IDX >= BOOM) * np.exp(-tB / 0.2) - 0.5 * ((IDX >= snap(152.3 * FPS)) & (IDX < FIRE))
    expo = expo - 9.0 * smoothstep(ramp(IDX, FADE - 6, FADE))
    bake(SC, "view_settings.exposure", expo)

# =====================================================================================
# 10c. v4 SET PIECES: booster-3's explosion (JangaFX CC0 Mid-Air Explosion), casing debris, the RCS jets,
#      the rocket's spectrum panel, the space glow (city lights + aurora on the bass), the asteroid's trail,
#      and the shatter debris as Blender RIGID-BODY physics (Bullet, no-tunnel recipe, Cell Fracture crumble),
#      baked to keys and gated by the no-vertex-inside-the-speaker assert.
# =====================================================================================
if V4:
    # --- the mid-air explosion VDB: its files are numbered without padding (a_0 .. a_125), so give Blender a
    #     padded symlink sequence it can step through (frame k = file k)
    MIDAIR_SRC = os.path.join(JANGA, "midair_explosion", "ex", "Midair_Explosion_01")
    MIDAIR = os.path.join(JANGA, "midair_explosion", "seq")
    os.makedirs(MIDAIR, exist_ok=True)
    for k in range(126):
        dst = os.path.join(MIDAIR, f"midair_{k:04d}.vdb")
        if not os.path.exists(dst):
            os.symlink(os.path.join(MIDAIR_SRC, f"embergen_midair_explosion_a_{k}.vdb"), dst)
    MIDAIR_ANCHOR = Vector((-1.0, -150.0, -41.0))     # where the blast starts in the sim (voxels; measured bboxes)
    tb3 = np.maximum(IDX - B3BOOM, 0) / FPS
    _out = Vector((BOOM_P.x, BOOM_P.y, 0.0)).normalized()
    _dexp = (_out * 0.75 + Vector((0, 0, -0.66))).normalized()  # it carries the booster's momentum out + falls back
    _rot = _dexp.to_track_quat("Y", "Z").to_euler()
    M_B3FIRE = smoke_material("B3Fireball", (0.88, 0.87, 0.85), dens=1.6, aniso=0.35, glow_grid="flames",
                              glow_col=blackbody(3000), plume_lit=False)   # aluminised APCP: white Al2O3 smoke
    _vb = Vector(B3_MID[B3BOOM] - B3_MID[B3BOOM - 1]) * FPS
    _vb = _vb * min(1.0, 40.0 / max(_vb.length, 1e-3))
    B3X = []
    for k, (sc_, tw_, sp_) in enumerate(((0.5, 0.0, 1.0), (0.36, 2.4, 1.25))):
        rot_ = (_rot.x, _rot.y, _rot.z + tw_)
        fo, nfi = vdb_seq(f"B3Explosion{k}", MIDAIR, sc_, loc=(0, 0, 0), rot=rot_, mat=M_B3FIRE, parent=RIG)
        fo.data.render.precision = "HALF"
        vdb_play(fo, nfi, B3BOOM - 1, speed=sp_, first=1)
        off = anchored((0, 0, 0), rot_, sc_, MIDAIR_ANCHOR)
        pos = np.array(BOOM_P + off)[None, :] + np.array(_vb)[None, :] * (1.2 * (1 - np.exp(-tb3 / 1.2)))[:, None] \
            + np.array([0, 0, -1.0])[None, :] * (14.0 * tb3)[:, None]
        bake_vec(fo, "location", pos)
        show(fo, (IDX >= B3BOOM - 1) & (IDX < B3BOOM + 12 * FPS))
        B3X.append(fo)
    vkeys(M_B3FIRE, np.full(N, 1.6) * (1 - 0.5 * ramp(IDX, B3BOOM + 4 * FPS, B3BOOM + 12 * FPS)), evo,
          (IDX >= B3BOOM) * (0.3 + 12.0 * np.exp(-tb3 / 0.7)))
    B3FLASH = bpy.data.lights.new("B3Flash", "POINT")
    B3FLASH.shadow_soft_size = 25.0
    B3FLASH.color = blackbody(3000)
    B3FLO = link(bpy.data.objects.new("B3Flash", B3FLASH), RIG)
    B3FLO.location = BOOM_P
    bake(B3FLASH, "energy", (IDX >= B3BOOM) * 3.0e8 * np.exp(-tb3 / 0.2))
    show(B3FLO, (IDX >= B3BOOM) & (IDX < B3BOOM + 3 * FPS))
    # the condensation shock shell: what makes an explosion read as CLOSE (a Wilson cloud, ~300 m/s, 6 frames)
    m_sh, nt_sh = new_mat("ShockShell")
    m_sh.surface_render_method = "BLENDED"
    lw_ = nd(nt_sh, "ShaderNodeLayerWeight", inp={"Blend": 0.35})
    oi_ = nd(nt_sh, "ShaderNodeObjectInfo")
    sc__ = nd(nt_sh, "ShaderNodeSeparateColor")
    lk(nt_sh, oi_.outputs["Color"], sc__.inputs[0])
    em_ = nd(nt_sh, "ShaderNodeEmission", inp={"Color": (0.92, 0.95, 1.0, 1)})
    lk(nt_sh, M(nt_sh, "MULTIPLY", M(nt_sh, "POWER", lw_.outputs["Fresnel"], 2.0), sc__.outputs[0]), em_.inputs["Strength"])
    tr__ = nd(nt_sh, "ShaderNodeBsdfTransparent")
    ad__ = nd(nt_sh, "ShaderNodeAddShader")
    lk(nt_sh, tr__.outputs[0], ad__.inputs[0])
    lk(nt_sh, em_.outputs[0], ad__.inputs[1])
    lk(nt_sh, ad__.outputs[0], nd(nt_sh, "ShaderNodeOutputMaterial").inputs["Surface"])
    shell = no_shadow(uv_sphere("B3ShockShell", 1.0, 48, 24, m_sh, RIG, loc=tuple(BOOM_P)))
    rsh = 8.0 + 300.0 * tb3
    bake_vec(shell, "scale", np.stack([rsh, rsh, rsh * 0.85], 1))
    set_color_keys(shell, np.stack([(IDX >= B3BOOM) * 0.7 * np.exp(-tb3 / 0.06), np.zeros(N), np.zeros(N)], 1), const=False)
    show(shell, (IDX >= B3BOOM) & (IDX < B3BOOM + 5))
    # casing: curved panels of the burst booster case + burning propellant chunks, flung away from the core
    _crng = np.random.default_rng(56)
    M_HOTCHUNK = emissive("HotChunk", blackbody(2600), 1.0, objcolor_gain=True)
    for k in range(18):
        panel = k < 6
        if panel:
            a0 = _crng.uniform(0, 2 * math.pi)
            ob = lathe(f"B3Casing{k}", [(1.86, -3.0), (1.86, 3.0)], 10, M_SRB, RIG,
                       a0=a0, a1=a0 + _crng.uniform(0.8, 1.6), smooth=True)
        else:
            ob = no_shadow(uv_sphere(f"B3Ember{k}", _crng.uniform(0.4, 1.1), 8, 5, M_HOTCHUNK, RIG))
        d = Vector(_crng.normal(size=3)).normalized()
        if d.dot(-Vector((BOOM_P.x, BOOM_P.y, 0)).normalized()) > 0.35:     # nothing flies INTO the stack
            d = (d + _out * 1.2).normalized()
        v0 = _crng.uniform(35.0, 95.0)
        p = np.array(BOOM_P)[None, :] + (np.array(d) * v0 + np.array(_vb) * 0.6)[None, :] * tb3[:, None] \
            + np.array([0, 0, -9.0])[None, :] * (tb3 ** 2)[:, None]
        bake_vec(ob, "location", p)
        w_ = _crng.normal(size=3) * 2.5
        bake_vec(ob, "rotation_euler", np.stack([w_[0] * tb3, w_[1] * tb3, w_[2] * tb3], 1))
        if not panel:
            hot = (IDX >= B3BOOM) * 20.0 * np.exp(-tb3 / 2.5)
            set_color_keys(ob, np.stack([hot, hot, hot], 1), const=False)
        show(ob, (IDX >= B3BOOM) & (IDX < B3BOOM + 7 * FPS))
    # exposure pop on the blast
    expo = expo + 0.5 * (IDX >= B3BOOM) * np.exp(-tb3 / 0.15)
    # --- RCS roll jets: white puffs near the top of the core when the flight model fires them
    M_RCS = plume_shell("RCSPuff", [(0.0, (1.0, 1.0, 1.0)), (1.0, (0.7, 0.75, 0.8))], nscale=2.0, along_pow=1.2,
                        edge_pow=1.5, base_alpha=0.8)
    RCS = []
    for k in range(4):
        a = k * math.pi / 2
        sgn = 1 if k % 2 == 0 else -1
        ob = no_shadow(lathe(f"RCSPuff{k}", [(0.15, 0.0), (0.5, -0.8), (1.2, -3.0), (0.3, -5.0)], 16, M_RCS, S1,
                             loc=((R + 0.2) * math.cos(a), (R + 0.2) * math.sin(a), 61.5)))
        tang = Vector((-math.sin(a), math.cos(a), 0.0)) * sgn
        ob.rotation_euler = (-tang).to_track_quat("Z", "Y").to_euler()   # the plume runs along local -Z
        fire = (ATT["rcs"] * sgn > 0.5) * WFADE * (IDX >= B3BOOM)
        set_color_keys(ob, np.stack([3.0 * fire, np.cumsum(np.full(N, 0.4)), np.cumsum(np.full(N, 0.3))], 1), const=False)
        show(ob, fire > 0.01)
        RCS.append(ob)
    # --- the rocket's spectrum panel keys (alarm red until the reboot sweep reaches it)
    nts = M_SPEC_ROCKET.node_tree
    bake(nts, value_path("Frame"), IDX.astype(np.float64), const=True)
    bake(nts, value_path("Alarm"), _alarm_on * (IDX < REBOOT + int(round(REBOOT_LEN * 0.42))))
    bake(nts, value_path("Gain"), np.where(IDX >= LIFT, 1.0, 0.12) * (LVL / np.maximum(LVL, 1e-3)) *
         np.where((_tbm >= 0) & (_tbm < 1.2), np.where(_flk < 0.45, 0.05, 1.0), 1.0))
    # --- space glow: the night side's city lights + an aurora on the limb breathe with the bass
    wn_ = WORLD.node_tree
    bake(wn_, value_path("CityGain"), 1.0 + (SET >= 2) * (0.4 + 1.6 * BASS4))
    bake(wn_, value_path("AuroraGain"), (SET >= 2) * (0.45 + 1.1 * BASS4 + 0.3 * KICK4) * (1 - smoothstep(ramp(IDX, FADE - 6, FADE))))
    bake(wn_, value_path("AuroraT"), np.cumsum(np.full(N, 0.012)))

    # --- the asteroid's TRAIL (research #1 §7): a static tube along the (straight) path + a keyed head distance;
    #     a white-hot core fading behind the head + a wider green/cyan halo; nothing ahead of the head
    A_FAR = Vector(ast_pos[GLIMPSE])
    _plen = (A_HIT - A_FAR).length
    m_tr, nt_tr = new_mat("AsteroidTrail")
    m_tr.surface_render_method = "BLENDED"
    m_tr.use_transparency_overlap = True
    head = value_node(nt_tr, "Head", 0.0)
    fade_v = value_node(nt_tr, "Fade", 1.0)
    tc_ = nd(nt_tr, "ShaderNodeTexCoord")
    sp_ = nd(nt_tr, "ShaderNodeSeparateXYZ")
    lk(nt_tr, tc_.outputs["Object"], sp_.inputs[0])
    dd = M(nt_tr, "SUBTRACT", head.outputs[0], sp_.outputs[2])            # metres behind the head
    behind = M(nt_tr, "GREATER_THAN", dd, 0.0)
    core_ = M(nt_tr, "EXPONENT", M(nt_tr, "DIVIDE", M(nt_tr, "MAXIMUM", dd, 0.0), -900.0))
    halo_ = M(nt_tr, "EXPONENT", M(nt_tr, "DIVIDE", M(nt_tr, "MAXIMUM", dd, 0.0), -3500.0))
    # a camera-facing RIBBON (keyed roll about the path, see below): soft across its width (object x / width),
    # and it fades IN over the first ~250 m behind the head so the head isn't capped by a hard edge
    wid = value_node(nt_tr, "Width", 1.0)
    acr = M(nt_tr, "DIVIDE", M(nt_tr, "ABSOLUTE", sp_.outputs[0]), wid.outputs[0])
    face_ = M(nt_tr, "POWER", M(nt_tr, "SUBTRACT", 1.0, M(nt_tr, "MULTIPLY", acr, acr), clamp=True), 2.0)
    face_ = M(nt_tr, "MULTIPLY", face_, M(nt_tr, "MINIMUM", M(nt_tr, "DIVIDE", M(nt_tr, "MAXIMUM", dd, 0.0), 250.0), 1.0))
    tnz = nd(nt_tr, "ShaderNodeTexNoise", inp={"Scale": 0.004, "Detail": 3.0})
    lk(nt_tr, tc_.outputs["Object"], tnz.inputs["Vector"])
    st_ = M(nt_tr, "MULTIPLY", M(nt_tr, "MULTIPLY", M(nt_tr, "ADD", M(nt_tr, "MULTIPLY", core_, 3.0), halo_), behind),
            M(nt_tr, "MULTIPLY", M(nt_tr, "MULTIPLY", face_, fade_v.outputs[0]), M(nt_tr, "MULTIPLY_ADD", tnz.outputs["Fac"], 1.2, 0.4)))
    tcol = mix_col(nt_tr, M(nt_tr, "MINIMUM", M(nt_tr, "MULTIPLY", core_, 1.3), 1.0), (0.25, 1.0, 0.75), (1.0, 0.97, 0.9))
    em_t = nd(nt_tr, "ShaderNodeEmission")
    lk(nt_tr, tcol, em_t.inputs["Color"])
    lk(nt_tr, M(nt_tr, "MULTIPLY", st_, 2.2), em_t.inputs["Strength"])
    trt = nd(nt_tr, "ShaderNodeBsdfTransparent")
    adt = nd(nt_tr, "ShaderNodeAddShader")
    lk(nt_tr, trt.outputs[0], adt.inputs[0])
    lk(nt_tr, em_t.outputs[0], adt.inputs[1])
    lk(nt_tr, adt.outputs[0], nd(nt_tr, "ShaderNodeOutputMaterial").inputs["Surface"])
    TRAILS = []
    _Ralign = V_IN.to_track_quat("Z", "Y").to_matrix()
    _RalT = _Ralign.transposed()
    for k, (rad, gain_) in enumerate(((AST_R * 0.9, 1.0), (AST_R * 2.6, 0.35))):
        m_k = m_tr.copy() if k else m_tr
        m_k.name = f"AsteroidTrail{k}"
        m_k.node_tree.nodes["Width"].outputs[0].default_value = rad
        tro = no_shadow(mesh_obj(f"AsteroidTrail{k}", [(-rad, 0, 0), (rad, 0, 0), (rad, 0, _plen), (-rad, 0, _plen)],
                                 [(0, 1, 2, 3)], m_k))
        tro.location = A_FAR
        rots = np.zeros((N, 3))
        _prev = None
        for i in range(N):                                     # roll the ribbon about the path to face the camera
            c = _RalT @ (Vector(cam_loc[i]) - A_FAR)
            th = math.atan2(c.y, c.x) - math.pi / 2
            e = (_Ralign @ Matrix.Rotation(th, 3, "Z")).to_euler("XYZ", _prev) if _prev else \
                (_Ralign @ Matrix.Rotation(th, 3, "Z")).to_euler("XYZ")
            _prev = e
            rots[i] = (e.x, e.y, e.z)
        bake_vec(tro, "rotation_euler", rots)
        TRAILS.append((tro, m_k))
        show(tro, (IDX >= GLIMPSE) & (IDX < FIRE + 4 * FPS))
    _s_head = np.array([max((Vector(ast_pos[i]) - A_FAR).dot(V_IN), 0.0) for i in range(N)])
    for tro, m_k in TRAILS:
        bake(m_k.node_tree, value_path("Head"), _s_head)
        bake(m_k.node_tree, value_path("Fade"), (1.0 - smoothstep(ramp(IDX, FIRE, FIRE + 3 * FPS))) * (IDX >= GLIMPSE)
             * (1.0 if m_k is m_tr else 0.35))
    # a bow-shock glow round the head while it flies in, so the glimpse reads at 20+ km
    m_bow, nt_bow = new_mat("AsteroidBow")
    m_bow.surface_render_method = "BLENDED"
    lwb = nd(nt_bow, "ShaderNodeLayerWeight", inp={"Blend": 0.3})
    bg_ = value_node(nt_bow, "Gain", 0.0)
    emb = nd(nt_bow, "ShaderNodeEmission", inp={"Color": (0.6, 1.0, 0.85, 1)})
    lk(nt_bow, M(nt_bow, "MULTIPLY", M(nt_bow, "POWER", lwb.outputs["Facing"], 3.0), bg_.outputs[0]), emb.inputs["Strength"])
    trb = nd(nt_bow, "ShaderNodeBsdfTransparent")
    adb = nd(nt_bow, "ShaderNodeAddShader")
    lk(nt_bow, trb.outputs[0], adb.inputs[0])
    lk(nt_bow, emb.outputs[0], adb.inputs[1])
    lk(nt_bow, adb.outputs[0], nd(nt_bow, "ShaderNodeOutputMaterial").inputs["Surface"])
    bow = no_shadow(uv_sphere("AsteroidBow", 1.0, 32, 16, m_bow, None))
    bow.scale = (AST_R * 1.3, AST_R * 1.3, AST_R * 1.3)
    bake_vec(bow, "location", ast_pos)
    bake(nt_bow, value_path("Gain"), 2.0 * (IDX >= GLIMPSE) * (1.0 - smoothstep(ramp(IDX, FIRE - 10, FIRE + 20))))
    show(bow, np.zeros(N, bool))                       # (looked like an outline ring on the stills: off; the trail carries it)
    show(AFO, (IDX >= GLIMPSE) & (IDX < BOOM + 3 * FPS))

    # --- THE SHATTER DEBRIS: Blender's own rigid bodies (profiles/blender/rigid-body + cell-fracture cards):
    #     30 substeps, 20 solver iterations, split impulse, a hidden proxy box 0.5 m bigger than the speaker,
    #     chunks keyed on LINEAR (extrapolated) keys then released to dynamic; Cell Fracture (seeded) for the big
    #     shatter + the crumbling rocks (breakable FIXED glue). Simulated in a separate scene, baked to plain keys
    #     (render workers never touch Bullet), then the BUILD GATE: no debris vertex inside the speaker, ever.
    import importlib.util
    import random
    import mathutils.noise
    _spec = importlib.util.spec_from_file_location("cell_fracture", os.path.join(ARGS.cellfrac, "__init__.py"),
                                                   submodule_search_locations=[ARGS.cellfrac])
    _cf = importlib.util.module_from_spec(_spec)
    sys.modules["cell_fracture"] = _cf
    _spec.loader.exec_module(_cf)
    _cf.register()
    F0, F1 = BOOM - 2, N                                   # sim frames (1-based = index + 1 below)
    REL = BOOM + 2                                         # release frame index (kinematic -> dynamic)
    SIM = bpy.data.scenes.new("DebrisSim")
    SIM.render.fps, SIM.render.fps_base = FPS, 1.0
    SIM.frame_start, SIM.frame_end = F0 + 1, F1
    _win = bpy.context.window_manager.windows[0]
    _main_scene = _win.scene
    _win.scene = SIM
    with bpy.context.temp_override(window=_win, scene=SIM):
        bpy.ops.rigidbody.world_add()
    rbw = SIM.rigidbody_world
    rbw.substeps_per_frame, rbw.solver_iterations, rbw.use_split_impulse = 60, 20, True   # card: 30 min; 60 = margin
    # (measured on this build: at 30 substeps a 470 m/s crumbling rock's 1-2 m fragments sank 0.24 m into the
    #  visual box — the gate caught it; 60 substeps + slower crumble rocks clear it)
    rbw.effector_weights.gravity = 0.0
    rbw.point_cache.frame_start, rbw.point_cache.frame_end = F0 + 1, F1
    rbc = bpy.data.collections.new("DebrisRB")
    SIM.collection.children.link(rbc)
    rbw.collection = rbc

    def make_rb(ob, typ, **kw):
        if ob.name not in rbc.objects:
            rbc.objects.link(ob)
        with bpy.context.temp_override(window=_win, scene=SIM, object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_add(type=typ)
        for k_, v_ in kw.items():
            setattr(ob.rigid_body, k_, v_)

    def key_lin(ob, path, frames_vals, index=None):
        for f_, v_ in frames_vals:
            setattr(ob, path, v_)
            ob.keyframe_insert(path, frame=f_)
        ad = ob.animation_data
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        for fc in cb.fcurves:
            if fc.data_path == path:
                for kp_ in fc.keyframe_points:
                    kp_.interpolation = "LINEAR"
                fc.extrapolation = "LINEAR"

    # speaker collider: a kinematic PASSIVE box following the speaker's keyed WORLD transform, 0.5 m bigger than
    # the visual envelope (cabinet + cones at full excursion + blinders/fans in front + moving heads on the lid)
    _sc_sim = spk_scl[F0:F1]
    assert np.ptp(_sc_sim) < 1e-6, "the speaker must be at full size during the debris sim"
    SCL = float(_sc_sim[0])
    VIS_LO = np.array([-(hx + 0.12), -(hy + max(0.32, float(CONE_EXC.max()) + 0.12)), -(hz + 0.05)]) * SCL
    VIS_HI = np.array([hx + 0.12, hy + 0.1, hz + SPK_TOP + 0.05]) * SCL
    PAD = 1.0                                              # card: 0.5 m; the recoiling speaker needs 1.0 (see note)
    _bc = (VIS_LO + VIS_HI) / 2
    _bh = (VIS_HI - VIS_LO) / 2 + PAD
    _pm = bpy.data.meshes.new("SpeakerProxy")
    _pm.from_pydata([tuple(_bc + np.array([sx, sy, sz]) * _bh) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], [],
                    [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    _pm.update()
    PROXY = bpy.data.objects.new("SpeakerProxy", _pm)
    SIM.collection.objects.link(PROXY)
    for f in range(F0, F1):
        PROXY.location = Vector(spk_loc[f])
        PROXY.rotation_euler = tuple(spk_rot[f])
        PROXY.keyframe_insert("location", frame=f + 1)
        PROXY.keyframe_insert("rotation_euler", frame=f + 1)
    make_rb(PROXY, "PASSIVE", kinematic=True, collision_shape="BOX", restitution=0.55, friction=0.5,
            use_margin=True, collision_margin=0.0)

    # (a) the big shatter: Cell Fracture of (a decimated copy of) the Bennu rock, every cell flung OUTWARD and
    #     AWAY from the speaker side (the speaker-bound debris is the smaller stuff below)
    random.seed(1964)
    mathutils.noise.seed_set(1964)
    _rk = bpy.data.objects.new("RockToFracture", me_.copy())
    SIM.collection.objects.link(_rk)
    _rco = np.empty(len(_rk.data.vertices) * 3, np.float32)
    _rk.data.vertices.foreach_get("co", _rco)
    _rk.data.vertices.foreach_set("co", (_rco * AST_R).astype(np.float32))
    _rk.data.update()
    _dm = _rk.modifiers.new("Dec", "DECIMATE")
    _dm.ratio = 0.12
    with bpy.context.temp_override(window=_win, scene=SIM, view_layer=SIM.view_layers[0]):
        _dg = bpy.context.evaluated_depsgraph_get()
        _dg.update()
        _dmesh = bpy.data.meshes.new_from_object(_rk.evaluated_get(_dg))
    _rk.modifiers.remove(_dm)
    _rk.data = _dmesh
    _rk.location = Vector(ast_pos[BOOM])
    _rk.rotation_euler = tuple(np.array([0.11, 0.07, 0.05]) * spin[BOOM] + np.array([0, 0.6, 0]))
    for o in SIM.objects:
        o.select_set(False)
    _rk.select_set(True)
    SIM.view_layers[0].objects.active = _rk
    with bpy.context.temp_override(window=_win, scene=SIM, view_layer=SIM.view_layers[0], active_object=_rk,
                                   object=_rk, selected_objects=[_rk], selected_editable_objects=[_rk]):
        _r = bpy.ops.object.add_fracture_cell_objects(source={'VERT_OWN'}, source_limit=30, source_noise=0.6,
                                                      margin=0.002, use_recenter=True, use_remove_original=True,
                                                      collection_name="ShatterCells", use_sharp_edges=True)
    CELLS = sorted(bpy.data.collections["ShatterCells"].objects, key=lambda o: o.name)
    print("[rocket4] Cell Fracture (big shatter):", _r, len(CELLS), "cells")
    assert len(CELLS) >= 12, "the shatter needs cells"
    DEBRIS = []                                            # (object, kind)
    _drng = np.random.default_rng(1964)
    rc = Vector(ast_pos[BOOM])
    for c in CELLS:
        d = (c.matrix_world.translation - rc)
        d = d.normalized() if d.length > 1e-3 else Vector(_drng.normal(size=3)).normalized()
        if d.dot(-D_HAT) > -0.2:                           # never toward the speaker: bend it downrange
            d = (d + D_HAT * 1.4).normalized()
        v = d * float(_drng.uniform(40.0, 130.0))
        p0 = Vector(c.location)
        key_lin(c, "location", [(F0 + 1, p0 - v * (2.0 / FPS)), (REL + 1, p0 + v * ((REL - BOOM) / FPS))])
        e0 = Vector(c.rotation_euler)
        wv = Vector(_drng.normal(size=3)) * 0.6
        key_lin(c, "rotation_euler", [(F0 + 1, tuple(e0 - wv * (2.0 / FPS))), (REL + 1, tuple(e0 + wv * ((REL - BOOM) / FPS)))])
        make_rb(c, "ACTIVE", collision_shape="CONVEX_HULL", restitution=0.4, friction=0.6, linear_damping=0.0,
                angular_damping=0.0, use_margin=True, collision_margin=0.0)
        c.rigid_body.mass = 2600.0 * max(c.dimensions.x * c.dimensions.y * c.dimensions.z * 0.5, 1.0)
        c.rigid_body.kinematic = True
        c.rigid_body.keyframe_insert("kinematic", frame=REL + 1)
        c.rigid_body.kinematic = False
        c.rigid_body.keyframe_insert("kinematic", frame=REL + 2)
        DEBRIS.append((c, "cell"))

    # (b) speaker-bound rocks: launched off the rock's near face, timed to land on the speaker in the outro
    #     (AFTER = drop 2's end; the M5_hero shot is on them). Some bounce whole, some are pre-fractured and glued
    #     (breakable FIXED, ~1e6) so they CRUMBLE into boulders on impact.
    def lumpy(name, r, seed):
        g = np.random.default_rng(seed)
        me = bpy.data.meshes.new(name)
        bmx = __import__("bmesh").new()
        try:
            __import__("bmesh").ops.create_icosphere(bmx, subdivisions=2, radius=1.0)
            bmx.to_mesh(me)
        finally:
            bmx.free()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co *= np.array([g.uniform(0.65, 1.0), g.uniform(0.65, 1.0), g.uniform(0.65, 1.0)], np.float32)
        co *= (1.0 + 0.16 * g.standard_normal(len(co)))[:, None].astype(np.float32)
        co *= r / np.linalg.norm(co, axis=1).max()
        me.vertices.foreach_set("co", co.ravel())
        me.update()
        me.materials.append(M_ROCK)
        ob = bpy.data.objects.new(name, me)
        SIM.collection.objects.link(ob)
        return ob

    _arr = [6, 14, 24, 34, 46, 62, 76, 90, 102, 116]           # contact frames after AFTER (aimed, then measured)
    _crumble = {5, 6, 8, 9}                                   # the crumbling ones arrive slower (<= ~260 m/s):
    # a fragment must not move more than a fraction of its size per substep (rigid-body card)
    _glue_coll = bpy.data.collections.new("DebrisGlue")
    SIM.collection.children.link(_glue_coll)
    rbw.constraints = _glue_coll
    _start_c = Vector(ast_pos[BOOM])
    for k, da in enumerate(_arr):
        r = float(_drng.uniform(3.0, 6.0))
        t_hit = AFTER + da                                   # index
        # aim point on the speaker at that frame: the front face or the lid, in its local frame -> world
        Mh = (Matrix.Translation(Vector(spk_loc[t_hit])) @ Euler_(tuple(spk_rot[t_hit])).to_matrix().to_4x4())
        if k % 3 == 2:
            loc_aim = Vector((float(_drng.uniform(-0.6, 0.6)) * hx * SCL, float(_drng.uniform(-0.5, 0.5)) * hy * SCL,
                              (hz + SPK_TOP) * SCL + PAD + r))
        else:
            loc_aim = Vector((float(_drng.uniform(-0.7, 0.7)) * hx * SCL, VIS_LO[1] - PAD - r,
                              float(_drng.uniform(-0.75, 0.75)) * hz * SCL))
        aim = Mh @ loc_aim
        off = Vector(_drng.normal(size=3)) * 0.15
        p_start = _start_c - D_HAT * (AST_R * 1.25 + r) + (SIDE * off.x + UPV * off.y) * AST_R
        d = (aim - p_start)
        v = d / ((t_hit - BOOM) / FPS)
        assert v.length < 600.0, ("debris speed too high for the no-tunnel recipe", v.length)
        ob = lumpy(f"Debris{k}", r, 4000 + k)
        ob.location = p_start
        pieces = [ob]
        if k in _crumble:
            for o in SIM.objects:
                o.select_set(False)
            ob.select_set(True)
            SIM.view_layers[0].objects.active = ob
            random.seed(77 + k)
            mathutils.noise.seed_set(77 + k)
            with bpy.context.temp_override(window=_win, scene=SIM, view_layer=SIM.view_layers[0], active_object=ob,
                                           object=ob, selected_objects=[ob], selected_editable_objects=[ob]):
                bpy.ops.object.add_fracture_cell_objects(source={'VERT_OWN'}, source_limit=14, source_noise=0.6,
                                                         margin=0.001, use_recenter=True, use_remove_original=True,
                                                         collection_name=f"Crumble{k}", use_sharp_edges=True)
            pieces = sorted(bpy.data.collections[f"Crumble{k}"].objects, key=lambda o: o.name)
        for pc in pieces:
            p0 = Vector(pc.location)
            key_lin(pc, "location", [(F0 + 1, p0 - v * (2.0 / FPS)), (REL + 1, p0 + v * ((REL - BOOM) / FPS))])
            make_rb(pc, "ACTIVE", collision_shape="CONVEX_HULL", restitution=0.45, friction=0.6, linear_damping=0.0,
                    angular_damping=0.05, use_margin=True, collision_margin=0.0)
            pc.rigid_body.mass = 2600.0 * max(pc.dimensions.x * pc.dimensions.y * pc.dimensions.z * 0.5, 0.01)
            pc.rigid_body.kinematic = True
            pc.rigid_body.keyframe_insert("kinematic", frame=REL + 1)
            pc.rigid_body.kinematic = False
            pc.rigid_body.keyframe_insert("kinematic", frame=REL + 2)
            DEBRIS.append((pc, "crumble" if k in _crumble else "rock"))
        if k in _crumble:                                    # glue neighbouring cells (breakable)
            P_ = [pc.location.copy() for pc in pieces]
            size_ = float(np.median([max(pc.dimensions) for pc in pieces]))
            for a_ in range(len(pieces)):
                for b_ in range(a_ + 1, len(pieces)):
                    if (P_[a_] - P_[b_]).length < 1.1 * size_:
                        e_ = bpy.data.objects.new(f"Glue{k}_{a_}_{b_}", None)
                        _glue_coll.objects.link(e_)
                        e_.location = (P_[a_] + P_[b_]) / 2
                        e_.parent = pieces[a_]
                        e_.matrix_parent_inverse = pieces[a_].matrix_world.inverted()
                        if e_.rigid_body_constraint is None:
                            with bpy.context.temp_override(window=_win, scene=SIM, object=e_, active_object=e_):
                                bpy.ops.rigidbody.constraint_add(type="FIXED")
                        c_ = e_.rigid_body_constraint
                        c_.type = "FIXED"
                        c_.object1, c_.object2 = pieces[a_], pieces[b_]
                        c_.use_breaking, c_.breaking_threshold = True, 1.0e6
    import time as _time
    _t0 = _time.time()
    with bpy.context.temp_override(window=_win, scene=SIM):
        _res = bpy.ops.ptcache.bake_all(bake=True)
    print(f"[rocket4] Bullet bake {_res} {len(DEBRIS)} bodies x {F1 - F0} frames in {_time.time() - _t0:.1f} s")
    # sample the sim once per frame (numpy bake: exact, 7.7x faster than the stock operator)
    obs_ = [o for o, _ in DEBRIS]
    nF = F1 - F0
    L_ = np.zeros((len(obs_), nF, 3))
    Rq = np.zeros((len(obs_), nF, 3))
    MW = np.zeros((len(obs_), nF, 4, 4))
    for fi in range(nF):
        SIM.frame_set(F0 + 1 + fi)
        for j, o in enumerate(obs_):
            m_ = o.matrix_world
            MW[j, fi] = np.array(m_)
            L_[j, fi] = m_.to_translation()
            Rq[j, fi] = m_.to_euler("XYZ", Euler_(Rq[j, fi - 1])) if fi else m_.to_euler("XYZ")
    # BUILD GATE: every vertex of every debris piece, every frame, in the speaker's moving local frame (the keyed
    # transform the render uses: spk_loc / spk_rot / spk_scl) vs the VISUAL envelope -> nothing may be inside
    _inside, _worst = 0, -1e9
    _vloc = []
    for o in obs_:
        a_ = np.empty(len(o.data.vertices) * 3, np.float32)
        o.data.vertices.foreach_get("co", a_)
        _vloc.append(np.c_[a_.reshape(-1, 3), np.ones(len(o.data.vertices))])
    _clear = np.full(len(obs_), 1e9)
    for fi in range(nF):
        f = F0 + fi
        Ms = (Matrix.Translation(Vector(spk_loc[f])) @ Euler_(tuple(spk_rot[f])).to_matrix().to_4x4())
        inv = np.array(Ms.inverted())
        for j in range(len(obs_)):
            p = (inv @ (MW[j, fi] @ _vloc[j].T)).T[:, :3]
            dep = np.min(np.minimum(p - VIS_LO, VIS_HI - p), axis=1)        # > 0 = inside the envelope
            dmax = float(dep.max())
            _worst = max(_worst, dmax)
            _clear[j] = min(_clear[j], -dmax)
            if dmax > 0:
                _inside += 1
    # contacts: a speaker-bound piece whose velocity changed by >5 % (read off the bake, not predicted)
    _vel = np.diff(L_, axis=1) * FPS
    _hits = []
    for j, (o, kind) in enumerate(DEBRIS):
        if kind == "cell":
            continue
        dv = np.linalg.norm(np.diff(_vel[j], axis=0), axis=1) / np.maximum(np.linalg.norm(_vel[j, :-1], axis=1), 1e-3)
        ch = np.nonzero(dv[REL - F0:] > 0.05)[0]
        if len(ch) and _clear[j] < 3.0:
            _hits.append((o.name, int(F0 + REL - F0 + ch[0] + 2), round(float(_clear[j]), 2)))
    print(f"[rocket4] ASSERT debris-not-inside-speaker: {_inside} vertex-inside events over {nF} frames x "
          f"{len(obs_)} pieces; worst depth {_worst:.3f} m (negative = clear); speaker contacts (piece, frame, "
          f"min clearance m): {_hits[:14]} ... {len(_hits)} total")
    assert _inside == 0, f"DEBRIS INSIDE THE SPEAKER: {_inside} events, worst depth {_worst:.3f} m"
    assert len(_hits) >= 4, f"debris should visibly hit the speaker (got {len(_hits)} contacts)"
    SC["debris_hits"] = ";".join(f"{n}:{f}" for n, f, _ in _hits)
    # free the cache, drop the rigid bodies + glue, move the pieces into the film with plain keys
    with bpy.context.temp_override(window=_win, scene=SIM):
        bpy.ops.ptcache.free_bake_all()
    for e_ in list(_glue_coll.objects):
        bpy.data.objects.remove(e_, do_unlink=True)
    for o in obs_ + [PROXY]:
        if o.rigid_body is not None:
            with bpy.context.temp_override(window=_win, scene=SIM, object=o, active_object=o, selected_objects=[o]):
                bpy.ops.rigidbody.object_remove()
    _win.scene = _main_scene
    for j, o in enumerate(obs_):
        o.animation_data_clear()
        for col_ in list(o.users_collection):
            col_.objects.unlink(o)
        SC.collection.objects.link(o)
        Lf = np.repeat(L_[j, :1], N, 0)
        Rf = np.repeat(Rq[j, :1], N, 0)
        Lf[F0:F1] = L_[j]
        Rf[F0:F1] = Rq[j]
        Lf[F1:] = L_[j, -1]
        Rf[F1:] = Rq[j, -1]
        bake_vec(o, "location", Lf)
        bake_vec(o, "rotation_euler", Rf)
        o.scale = (1.0, 1.0, 1.0)
        hot = np.clip(np.exp(-tB / 2.2) * 2.5, 0, 2.5) * (IDX >= BOOM)
        set_color_keys(o, np.stack([hot, 0.6 * hot, hot * 0.15], 1), const=False)
        show(o, IDX >= BOOM)
    bpy.data.objects.remove(PROXY, do_unlink=True)
    bpy.data.scenes.remove(SIM)
    _cf.unregister()
    bake(SC, "view_settings.exposure", expo)
    # HUD plan for the post pass (post_hud.py): per frame — feed label, mode (1 warning / 2 signal lost /
    # 3 attitude warning), fisheye flag, digital-breakup amount
    _hud = {}
    for k_, (s0_, nm_) in enumerate(SHOTS):
        if nm_ not in HUD_SHOTS:
            continue
        for i in range(s0_, SHOT_BOUNDS[k_ + 1]):
            if i < B3BOOM:
                mode_ = 1
            elif BLACK[i] > 0 or (i < snap(57.43 * FPS) + 5 and nm_ == "V7b_lost"):
                mode_ = 2
            else:
                mode_ = 3
            brk = 0.0
            if nm_ == "V9_fish" and hash01(i // 3, 5) < 0.22:
                brk = float(hash01(i, 6))
            if nm_ == "V7b_lost" and mode_ != 2 and i < snap(57.43 * FPS) + 18:
                brk = 0.8
            _hud[i + 1] = [HUD_SHOTS[nm_], mode_, int(nm_ in FISHEYE), round(brk, 3), round(float(BEATS[i]), 4),
                           round(float(KICK4[i]), 3)]
    import json as _json
    with open(os.path.join(os.path.dirname(os.path.abspath(ARGS.out)), "hud_plan.json"), "w") as fh:
        _json.dump({"font": ARGS.font_hud, "fps": FPS, "frames": _hud}, fh)
    print(f"[rocket4] HUD plan: {len(_hud)} frames ({min(_hud)}..{max(_hud)})")

# --- bake visibility (hide_render, CONSTANT) ---
for name, mask in VIS.items():
    ob = bpy.data.objects[name]
    bake(ob, "hide_render", (~mask).astype(np.float64), const=True)


# =====================================================================================
# 11. RENDER SETTINGS + COMPOSITOR (research fix #4: haze, heat shimmer, grain, dispersion;
#     onboard shots get barrel distortion; highlights clip like a real sensor)
# =====================================================================================
SC.frame_start, SC.frame_end = 1, N
SC.render.fps, SC.render.fps_base = FPS, 1.0
SC.render.resolution_x, SC.render.resolution_y = 1920, 1080
SC.render.resolution_percentage = ARGS.res_pct
SC.view_settings.view_transform = "AgX"
SC.view_settings.look = "AgX - Medium High Contrast"
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.5
SC.render.motion_blur_position = "START"
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_mode = "RGB"
SC.render.image_settings.color_depth = "8"
SC.render.use_overwrite = False
SC.render.use_placeholder = False
SC.render.engine = "BLENDER_EEVEE"
ee = SC.eevee
ee.taa_render_samples = ARGS.samples or 32
ee.use_raytracing = True
ee.ray_tracing_options.resolution_scale = "2"
ee.use_volume_custom_range = True
ee.volumetric_start = 0.5
ee.volumetric_end = 4000.0
ee.volumetric_tile_size = "2"
ee.volumetric_samples = 96
ee.volumetric_sample_distribution = 0.8
ee.use_volumetric_shadows = True
ee.volumetric_shadow_samples = 12
ee.volumetric_ray_depth = 8
ee.motion_blur_steps = 1
ee.use_shadows = True
ee.shadow_pool_size = "1024"
ee.shadow_ray_count = 1
ee.shadow_step_count = 6
ee.use_fast_gi = True
ee.clamp_volume_indirect = 10.0
vl = SC.view_layers[0]
vl.use_pass_mist = True
WORLD.mist_settings.start = 40.0
WORLD.mist_settings.depth = 5000.0
WORLD.mist_settings.falloff = "QUADRATIC"

cg = bpy.data.node_groups.new("Post", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
CN = cg.nodes.new
CL = cg.links.new
rl = CN("CompositorNodeRLayers")
go = CN("NodeGroupOutput")


def cval(name, v):
    n = CN("ShaderNodeValue")
    n.name = n.label = name
    n.outputs[0].default_value = v
    return n.outputs[0]


def cmath(op, a, b):
    n = CN("ShaderNodeMath")
    n.operation = op
    for k, x in enumerate((a, b)):
        if isinstance(x, (int, float)):
            n.inputs[k].default_value = x
        else:
            CL(x, n.inputs[k])
    return n.outputs[0]


# atmospheric haze between camera and subject (mist pass × keyed amount → keyed haze colour)
haze_amt = cval("HazeAmt", 0.0)
hr, hg, hb = cval("HazeR", 0.02), cval("HazeG", 0.025), cval("HazeB", 0.04)
hcol = CN("CompositorNodeCombineColor")
for k, s_ in enumerate((hr, hg, hb)):
    CL(s_, hcol.inputs[k])
hmix = CN("ShaderNodeMix")
hmix.data_type = "RGBA"
CL(cmath("MINIMUM", cmath("MULTIPLY", rl.outputs["Mist"], haze_amt), 0.85), hmix.inputs[0])
CL(rl.outputs["Image"], hmix.inputs[6])
CL(hcol.outputs[0], hmix.inputs[7])
# heat shimmer: animated 4D noise → Displace, strength ∝ distance (mist) × keyed amount
hn = CN("ShaderNodeTexNoise")
hn.noise_dimensions = "4D"
hn.name = "HeatNoise"
hn.inputs["Scale"].default_value = 16.0
hn.inputs["Detail"].default_value = 2.0
sub = CN("ShaderNodeVectorMath")
sub.operation = "SUBTRACT"
CL(hn.outputs["Color"], sub.inputs[0])
sub.inputs[1].default_value = (0.5, 0.5, 0.5)
heat_amt = cval("HeatAmt", 0.0)
hstr = cmath("MULTIPLY", heat_amt, cmath("POWER", rl.outputs["Mist"], 0.5))
scl = CN("ShaderNodeVectorMath")
scl.operation = "MULTIPLY"
CL(sub.outputs[0], scl.inputs[0])
scl.inputs[1].default_value = (2.2, 0.7, 0.0)
scl2 = CN("ShaderNodeVectorMath")
scl2.operation = "SCALE"
CL(scl.outputs[0], scl2.inputs[0])
CL(hstr, scl2.inputs["Scale"])
dsp = CN("CompositorNodeDisplace")
CL(hmix.outputs[2], dsp.inputs["Image"])
CL(scl2.outputs["Vector"], dsp.inputs[1])
# glare: streaks only on true highlights (sun disc, SRB core), then a soft bloom
gs = CN("CompositorNodeGlare")
gs.inputs["Type"].default_value = "Streaks"
gs.inputs["Quality"].default_value = "Medium"
gs.inputs["Threshold"].default_value = 60.0
gs.inputs["Strength"].default_value = 0.5
gs.inputs["Streaks"].default_value = 6
gs.inputs["Streaks Angle"].default_value = math.radians(15)
gs.inputs["Fade"].default_value = 0.9
CL(dsp.outputs["Image"], gs.inputs["Image"])
gl = CN("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Fog Glow"
gl.inputs["Quality"].default_value = "Medium"
gl.inputs["Threshold"].default_value = 1.4
gl.inputs["Strength"].default_value = 0.35
gl.inputs["Size"].default_value = 0.8
CL(gs.outputs["Image"], gl.inputs["Image"])
ld_ = CN("CompositorNodeLensdist")
ld_.name = "Lens"
ld_.inputs["Distortion"].default_value = -0.006
ld_.inputs["Dispersion"].default_value = 0.012
ld_.inputs["Fit"].default_value = True
CL(gl.outputs["Image"], ld_.inputs["Image"])
vg = CN("CompositorNodeEllipseMask")
vg.inputs["Size"].default_value = (1.25, 1.25)
vb = CN("CompositorNodeBlur") if hasattr(bpy.types, "CompositorNodeBlur") else None
mr = CN("ShaderNodeMapRange")
mr.inputs["To Min"].default_value = 0.70
mr.inputs["To Max"].default_value = 1.0
if vb is not None:
    try:
        vb.inputs["Size"].default_value = (300, 300)
    except Exception:
        pass
    CL(vg.outputs["Mask"], vb.inputs["Image"])
    CL(vb.outputs["Image"], mr.inputs["Value"])
else:
    CL(vg.outputs["Mask"], mr.inputs["Value"])
mul = CN("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
mul.inputs[0].default_value = 1.0
CL(ld_.outputs["Image"], mul.inputs[6])
CL(mr.outputs["Result"], mul.inputs[7])
# film grain: per-frame 4D white noise, multiplicative (±~2.5 %), a touch stronger onboard
gn_ = CN("ShaderNodeTexWhiteNoise")
gn_.noise_dimensions = "4D"
gn_.name = "Grain"
gamt = cval("GrainAmt", 0.05)
gsub = cmath("MULTIPLY", cmath("SUBTRACT", gn_.outputs["Value"], 0.5), gamt)
gmul = CN("ShaderNodeMix")
gmul.data_type = "RGBA"
gmul.blend_type = "MULTIPLY"
gmul.inputs[0].default_value = 1.0
CL(mul.outputs[2], gmul.inputs[6])
gcol = CN("CompositorNodeCombineColor")
gadd = cmath("ADD", gsub, 1.0)
for k in range(3):
    CL(gadd, gcol.inputs[k])
CL(gcol.outputs[0], gmul.inputs[7])
CL(gmul.outputs[2], go.inputs["Image"])
SC.compositing_node_group = cg
SC.render.compositor_device = "GPU"

# compositor keys: haze per set/shot, heat shimmer near the ground, onboard barrel distortion
haze = np.where(SET == 0, 0.9 * (0.35 + SHOT_HAZE), np.where(SET == 1, 0.35, 0.0))
bake(cg, 'nodes["HazeAmt"].outputs[0].default_value', haze)
hz_col = np.where((SET == 0)[:, None], np.array([0.008, 0.013, 0.030])[None, :],
                  np.array([0.05, 0.065, 0.10])[None, :])
for k, nm in enumerate(("HazeR", "HazeG", "HazeB")):
    bake(cg, f'nodes["{nm}"].outputs[0].default_value', hz_col[:, k])
heat = np.where(SET == 0, SHOT_HAZE * 0.8 + 0.8 * on_boost * np.clip(1 - ALT / 300, 0, 1), 0.0)
bake(cg, 'nodes["HeatAmt"].outputs[0].default_value', heat)
bake(cg, 'nodes["HeatNoise"].inputs["W"].default_value', IDX / FPS * 1.6)
bake(cg, 'nodes["Grain"].inputs["W"].default_value', IDX * 1.0)
bake(cg, 'nodes["GrainAmt"].outputs[0].default_value', 0.045 + 0.03 * SHOT_ONB)
bake(cg, 'nodes["Lens"].inputs["Distortion"].default_value', -0.006 - 0.075 * SHOT_ONB - 0.22 * SHOT_FISH)

# sound strip (for scrubbing in the GUI; the MP4 mux happens in render.sh)
if os.path.exists(ARGS.audio):
    se = SC.sequence_editor_create()
    se.strips.new_sound("Music", ARGS.audio, channel=1, frame_start=1)
SC.render.use_sequencer = False

# bookkeeping for render_frames.py / tests: per-frame EEVEE volume range wrapped around the
# subject (volumetric_start/end are not animatable): long-lens shots get a tight slab at km range
SC["shots"] = ";".join(f"{s}:{n}" for s, n in SHOTS)
SC["set_of_frame"] = "".join(str(int(k)) for k in SET)
subj = np.array([wl(i, (0, 0, 40.0)) for i in range(N)])
subj[SET == 0] = np.stack([np.full((SET == 0).sum(), 40.0), np.zeros((SET == 0).sum()),
                           np.minimum(subj[SET == 0, 2], PAD_Z + 80)], 1)
dist = np.linalg.norm(cam_loc - subj, axis=1)
vstart = np.full(N, 0.5)
vend = np.clip(1.8 * dist + 300.0, 600.0, 4000.0)
far = (SET == 0) & (dist > 700)
vstart[far] = np.maximum(dist[far] - 520.0, 0.5)
vend[far] = dist[far] + 420.0
vdist = np.full(N, 0.8)
s1 = SET == 1
close1 = s1 & (dist < 400)                                 # chase/onboard on the climb: fine slices near
vend[close1], vdist[close1] = 2500.0, 0.95
far1 = s1 & (dist >= 400)                                  # air-to-air long lens: a slab around the subject
vstart[far1] = np.maximum(dist[far1] - 1500.0, 0.5)
vend[far1] = dist[far1] + 1800.0
vdist[far1] = 0.5
vend = np.where(SET == 2, np.maximum(vend, 4.0 * dist + 1500.0), vend)
if RAVE:
    _sp = IDX >= AST0
    vstart[_sp], vend[_sp], vdist[_sp] = 0.5, 6000.0, 0.6
SC["vol_start"] = [float(round(v, 1)) for v in vstart]
SC["vol_end"] = [float(round(v, 1)) for v in vend]
SC["vol_dist"] = [float(round(v, 3)) for v in vdist]
# per-frame render quality (render_frames.py applies it; EEVEE settings are not animatable).
# MEASURED at 1080 (full res): volume lighting cost ∝ TAA samples; tile 2 → 4 saves ~30 %.
# Fine froxels only where volumes are close to the lens; fewer samples where volumes dominate.
FINE_TILE = {"A4_low", "B2_tower", "B4_tail", "D1_maxq", "D2_under", "C2_side", "V7_boom", "V8_tumble"}
vtile = np.full(N, 4)
spp = np.full(N, 32)
for k_, (s0_, nm_) in enumerate(SHOTS):
    a_, b_ = s0_, SHOT_BOUNDS[k_ + 1]
    if nm_ in FINE_TILE:
        vtile[a_:b_] = 2
    if nm_.startswith(("B", "A4")):
        spp[a_:b_] = 20                                     # steam shots: the volumes carry the frame
    elif nm_.startswith(("C", "D", "F", "G", "V")):
        spp[a_:b_] = 24
SC["vol_tile"] = [int(v) for v in vtile]
SC["spp"] = [int(v) for v in spp]
SC["entries"] = list(ENT)
os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out), compress=True)
print(f"[rocket3] saved {ARGS.out}  frames 1..{N}  engine {SC.render.engine}")
