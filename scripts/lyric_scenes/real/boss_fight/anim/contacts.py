"""Contact registry: every `contact` row of choreo_v1.json → Contact(s) (+ the sub-hits a row implies: combo hits 2/3,
the 2nd fireball, rocks 2/3, the 2nd spit, more roof chunks, the breath held on the shield). Audit body names:
  K.body (knight proxy) · K.sword (blade) · K.shield (bubble) · D.body (dragon proxy) · Pier_<id> · cover_A ·
  vault · <projectile object name> · colossus_arm.
"""
from anim.common import Contact

KIND = {"dodge_p1": "tail_pillar", "tail_p2": "tail_pillar", "tail_p3": "tail_pillar", "combo": "sword",
        "neck_slash": "sword", "parry": "talon_parry", "fireballs": "projectile", "big_fireball": "projectile",
        "rocks": "projectile", "volley": "projectile", "ceiling": "shield", "dfireballs": "shield",
        "roof_rain": "shield", "slam": "slam", "thrown": "body_cover", "spikes": "spikes", "roof": "roof",
        "sword_falls": "holy", "fire_rpg": "rocket"}
PILLAR = {"pillar_1": "Pier_W2", "pillar_2": "Pier_E2", "pillar_3": "Pier_W3"}


def body_of(token):
    """choreo contact token → audit body name"""
    if token.startswith("dragon."):
        return "D.body"
    if token == "knight.sword":
        return "K.sword"
    if token == "knight.shield":
        return "K.shield"
    if token == "knight.body":
        return "K.body"
    if token in PILLAR:
        return PILLAR[token]
    if token == "cover_column_A":
        return "cover_A"
    if token == "roof_vault":
        return "vault"
    if token == "colossus_E.arm":
        return "colossus_arm"
    return token


def build(E, extra_hits=None):
    """extra_hits: {event_key: [frames of the derived sub-hits]} filled by the knight/FX stages."""
    out = []
    for key, e in E.items():
        if not e.contact:
            continue
        a, b = e.contact
        kind = KIND[key]
        a0 = "roof_chunk_1" if a == "roof_chunks" else body_of(a)
        out.append(Contact(key, e.f, a0, body_of(b), kind, note=f"{a} -> {b}"))
        for j, fr in enumerate((extra_hits or {}).get(key, [])):
            sa = a0
            if sa.endswith("_1"):
                sa = sa[:-2] + f"_{j + 2}"
            out.append(Contact(f"{key}#{j + 2}", fr, sa, body_of(b), kind, derived=True, note=f"sub-hit {j + 2} of {key}"))
    sl = E["slam"]
    out.append(Contact("slam#sword", sl.f, "D.body", "K.sword", "slam", derived=True,
                       note="the knight's blade is crushed with him under the slam", optional=True))
    # the dive breath is held by the shield for ~2 s (event 'shield_breath' has no contact field but IS one)
    sb = E["shield_breath"]
    out.append(Contact("shield_breath", sb.f, "breath", "K.shield", "breath", derived=True,
                       note="fire stream stopped by the shield (window = whole breath)"))
    return sorted(out, key=lambda c: c.f)
