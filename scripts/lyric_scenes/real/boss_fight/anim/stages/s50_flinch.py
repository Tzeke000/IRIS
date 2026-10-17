"""s50 — procedural FLINCH: one closed-form damped spring per contact on the victim's bone chain (COMBINE layer on the
top NLA track) + a body shove on the root for heavy hits. The spring is exactly 0 on and before the contact frame,
so nothing upstream (projectile endpoints, contact pins) moves. Direction = blade-tip velocity / projectile path /
slam normal, converted into the victim's character frame."""
import math

import numpy as np

from anim import common as C
from anim import posekit as PK
from anim import track as TK
from anim.stages.s40_contacts import bump, rebake

# victim region → (chain builder, preset, strength, shove metres)
def _dragon_regions(D):
    ch = D.chains
    return {
        "foreleg_L": (lambda: [ch["armL"][4], ch["armL"][3], ch["armL"][2], ch["armL"][1], ch["spine"][5]], "dragon_light", 1.0, 0.05),
        "neck": (lambda: [ch["neck"][1], ch["neck"][2], ch["neck"][3], ch["neck"][4], ch["neck"][0]], "dragon_heavy", 1.0, 0.15),
        "chest": (lambda: [ch["spine"][5], ch["spine"][4], ch["neck"][0], ch["neck"][1], ch["neck"][2], ch["neck"][3]], "dragon_heavy", 0.8, 0.25),
        "belly": (lambda: [ch["spine"][2], ch["spine"][3], ch["spine"][4], ch["spine"][5], ch["neck"][0], ch["neck"][1]], "dragon_heavy", 1.2, 0.4),
        "head": (lambda: [ch["neck"][4], ch["neck"][3], ch["neck"][2], ch["neck"][1]], "dragon_heavy", 0.9, 0.1),
        "back": (lambda: [ch["spine"][3], ch["spine"][2], ch["spine"][4], ch["spine"][5], ch["neck"][0], ch["tail"][0], ch["tail"][1]], "dragon_heavy", 1.6, 0.0),
        "tail": (lambda: list(reversed(ch["tail"][9:])), "dragon_tail", 0.8, 0.0),
        "wing": (lambda: [ch["wingL"][3], ch["wingL"][4], ch["wingR"][3], ch["wingR"][4]], "dragon_light", 0.7, 0.0),
    }


def _knight_chain(K):
    f = K.skel.find
    return [f("mixamorig:Spine2"), f("mixamorig:Spine1"), f("mixamorig:Neck"), f("mixamorig:Head"), f("mixamorig:Spine")]


REGION = {"combo": "foreleg_L", "neck_slash": "neck", "fireballs": "chest", "big_fireball": "chest", "spikes": "belly",
          "rocks": "head", "sword_falls": "back", "fire_rpg": "head", "dodge_p1": "tail", "tail_p2": "tail",
          "tail_p3": "tail", "parry": None}
ANCHOR = {"foreleg_L": "D.forearmL", "neck": "D.neck", "chest": "D.chest", "belly": "D.belly", "head": "D.head",
          "back": "D.back", "tail": "D.tail_tip"}


def to_char(v, yaw_deg):
    a = -math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def run(S):
    D, K, N = S.D, S.K, S.N
    reg = _dragon_regions(D)
    d_ev, k_ev = [], []
    log = []
    for c in S.contacts:
        base = c.key.split("#")[0]
        f = c.f
        if c.b == "D.body" or c.a == "D.body" and base in REGION:
            region = REGION.get(base)
            if region is None:
                continue
            P0, P1 = TK.at(S, f - 1), TK.at(S, f)
            anc = P1[ANCHOR[region]]
            if c.a == "K.sword":
                dw = P1["K.tip"] - P0["K.tip"]
            elif region == "tail":
                dw = -(P1["D.tail_tip"] - P0["D.tail_tip"])
            elif base in ("fireballs", "big_fireball", "rocks"):
                dw = anc - P1["K.rhand"]
            elif base == "sword_falls":
                dw = np.array([0, 0, -1.0])
            elif base == "spikes":
                dw = np.array([0, 0, 1.0]) + 0.3 * (anc - P1["K.root"]) / max(np.linalg.norm(anc - P1["K.root"]), 1e-6)
            else:
                dw = anc - P1["K.chest"]
            if np.linalg.norm(dw) < 1e-6:
                dw = np.array([0, 1.0, 0])
            dw = dw / np.linalg.norm(dw)
            chain_fn, preset, strength, shove = reg[region]
            yaw = float((D.base_yaw + D.yoff)[f - 1])
            d_ev.append(dict(frame=f, chain=chain_fn(), dir=to_char(dw, yaw), strength=strength, preset=preset))
            if shove > 0 and any(0 < c2.f - f <= 20 for c2 in S.contacts if c2.a == c.a and c2.b == c.b):
                shove = 0.0          # v2: a combo's next hit is solved for THIS position — no shove between sub-hits
            if shove > 0:
                sh = np.zeros((N, 3))
                fr = np.arange(N, dtype=np.float64)
                s = PK.spring(N, f - 1, shove, 0.35, 0.5)
                sh += np.outer(s, dw * np.array([1, 1, 0.3]))
                D.off += sh
            log.append(dict(key=c.key, f=f, victim="dragon", region=region, dir=dw.round(2).tolist()))
        if c.b in ("K.body", "K.shield") or c.a == "K.body":
            if base == "thrown":
                continue
            P1 = TK.at(S, f)
            src = P1["D.chest"] if base == "slam" else (P1["D.jaw"] if base in ("dfireballs", "shield_breath") else P1["K.head"] + np.array([0, 0, 3.0]))
            dw = P1["K.chest"] - src
            dw = dw / max(np.linalg.norm(dw), 1e-6)
            yaw = float((K.base_yaw + K.yoff)[f - 1])
            heavy = base == "slam"
            k_ev.append(dict(frame=f, chain=_knight_chain(K), dir=to_char(dw, yaw), strength=1.0 if heavy else 0.45,
                             preset="knight_heavy" if heavy else "knight"))
            log.append(dict(key=c.key, f=f, victim="knight", dir=dw.round(2).tolist()))
    for X, evs, name in ((D, d_ev, "D_FLINCH"), (K, k_ev, "K_FLINCH")):
        if not evs:
            continue
        lay = PK.flinch_layers(X.kit, evs, N, f0=1)
        act = PK.write_layer_action(name, X.arm, lay, N, step=1)
        C.put(X.arm, name, act, 1, blend="COMBINE")
    rebake(S, "D")
    rebake(S, "K")
    S.R.add("flinch.events", log)
    S.R.add("flinch.count", dict(dragon=len(d_ev), knight=len(k_ev)))
