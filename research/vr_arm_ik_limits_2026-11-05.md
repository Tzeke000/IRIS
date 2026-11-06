# VR arm IK: human joint limits + proven elbow/twist methods (armik-1105, 2026-11-05)

Zeke in the headset (v26.11.5.1, r_spartan + Bruno), 21:11–21:27: arms overhead with the palms turned back ⇒ the elbow
"twists like a rag being wrung"; palms toward his face ⇒ avatar elbows stick out (his sit low on his chest); hand flat
with the index pointing left ⇒ the avatar hand points at the sky; he can't point at himself; "it thinks my wrist can't
bend". He said it was worse than v26.11.4.10, and asked me to use existing data and code rather than derive limits by hand.

## 1. Human range of motion: the table the suite judges against

Active ROM, adult, in degrees. Sources: the AAOS "Joint Motion" norms as tabulated in clinical goniometry references
(Norkin & White), cross-checked against Soucie et al. 2011 (674 healthy subjects aged 2–69; their means are mostly a little
*below* the AAOS figures, and women are more mobile, most of all in forearm pronation).

| joint | motion | AAOS norm | used in harness/armik.py (+ tolerance) |
|---|---|---|---|
| shoulder | internal rotation | 70 | −85 (IR) |
| shoulder | external rotation | 90 | +105 (ER) |
| shoulder | flexion / abduction | 180 / 180 | (not limited: the clavicle assist + swing) |
| elbow | flexion | 150 (some tables 145) | 0 … 155 |
| forearm | pronation / supination | 80 / 80 (≈ 170° in total) | each skin-twist piece ≤ 100, total ≤ 185 |
| wrist | flexion / extension | 80 / 70 | −80 … +90 |
| wrist | radial / ulnar deviation | 20 / 30 | −38 … +28 |
| knee | flexion | 135–150 | 0 … 150; **standing straight ≤ 10°** |
| hip | flexion | 120 | ≤ 125 |

The forearm row matters most. A VRM has no forearm twist bones, so the forearm's ~170° shows up as skin twist at two
places: at the **elbow** (the forearm bone's roll against the upper arm) and at the **wrist** (the hand against the
forearm). Either one carrying all of it looks wrong: all at the wrist gives the candy-wrapper wrist; all at the elbow gives
the "rag being wrung" elbow that Zeke saw. The standard answer, below, is to split it.

## 2. Existing implementations used

- **VRArmIK**: Mathias Parger, MIT licence, github.com/dabeschte/VRArmIK. It is the code for Parger et al., "Human
  Upper-Body Inverse Kinematics for Increased Embodiment in Consumer-Grade Virtual Reality", VRST 2018 (head + hands only,
  no trackers). What we took from it:
  - **The hand bone always takes the controller's rotation** (`setHandRotation(target.rotation)`). There is no wrist
    clamp: the player's real hand is by definition a human hand, so anything that pulls the avatar hand off it is an IK
    error showing up somewhere else.
  - **The elbow swivel follows the hand's roll about the forearm** (`rotateElbowWithHandRight/Forward`, power curve 1.5,
    ±45° offset, 0.08 s smoothing). That is what lets the shoulder and elbow absorb a palm turn instead of the forearm.
  - **Twist is distributed**: forearm roll reaches the twist bones at 0.3 / 0.8 of the hand's, and the rest stays at the
    wrist.
  - The README's honest limit applies to us too: rotating the elbow up and down with the hands and head still can't be
    solved without an elbow tracker.
- **Unity Humanoid `upperArmTwist` / `lowerArmTwist`** (Unity scripting reference): twist distribution as a 0–1 weight,
  **default 0.5**, split evenly between the two bones. Final IK's TwistRelaxer does the same job; its author recommends
  stacking relaxers to even the angles out further.
- **Godot 4.6+ `BoneTwistDisperser3D`** can only spread twist over bones that exist. Bundled VRMs have no forearm twist
  bones, so on our rigs it would reduce to the two-bone 50/50 split. Adding real twist bones (re-weighting the skin) is
  the real long-term fix.

## 3. What was wrong (measured by harness/armik.py on Zeke's take)

1. **Palm-pose tilt.** The OpenXR `palm_pose` (grip_surface) on Zeke's runtime has its finger axis about 30° toward the
   thumb from the real hand's. With his wrists relaxed, walking with arms hanging (take t = 180–200 s), every avatar wrist
   read 30–39° *radial* deviation. Before avatarfix-1104 that was harmless (the hand just looked slightly bent).
2. **avatarfix-1104 (`cddabdb`, `8606d4b`) added a 20° radial-deviation wall to the elbow-swivel cost and a hard wrist
   clamp.** Faced with a phantom 30° deviation, the swivel contorted the elbow and humerus to "fix" it, and the clamp
   rotated the hand off the controller: 92–103° p95 off in the take, 14–25 % of frames more than 12° off. That gives "hand
   points at the sky", "can't point at myself" and "wrist can't bend".
3. **`cddabdb` also moved 100 % of forearm twist into the forearm bone** (`FOREARM_SHARE` 0.55 → 1.0, `ROLL_MAX` 85° →
   178°): elbow-bone roll up to 180° in the take. That is the wrung rag.

## 4. What changed (branch armik-1105)

- Wrist clamp off by default (`IRIS_WRIST_CLAMP=1` brings it back): the hand is on the controller, as in VRArmIK.
- Twist split 50/50 forearm/wrist, forearm roll ≤ 95° (Unity's default; VRArmIK's distribution).
- Palm pose → hand rotated 30° about the palm normal (`IRIS_PALM_TILT`, degrees; 0 = the old mapping), so a relaxed real
  wrist is a relaxed avatar wrist and the ROM-aware swivel cost reasons about the true wrist.

## Sources

- VRArmIK, MIT: https://github.com/dabeschte/VRArmIK · write-up https://dabeschte.github.io/paper/2018/12/01/vrarmik.html
- Soucie JM et al. "Range of motion measurements: reference values and a database for comparison studies." Haemophilia
  2011;17:500–507. doi:10.1111/j.1365-2516.2010.02399.x
- AAOS normal ROM values: shoulder ER 90 / IR 70, elbow flexion 150, pronation / supination 80 / 80, wrist flexion /
  extension 80 / 70, radial / ulnar deviation 20 / 30. Tabulated e.g. in a research supplement attributed to the AAOS
  (https://cdn-links.lww.com/permalink/prsgo/b/prsgo_8_6_2020_04_17_hendriks_gox-d-20-00155r2_sdc1.pdf) and in clinical
  ROM charts; some tables give elbow 145 and wrist 60/60.
- Unity HumanDescription.upperArmTwist (default 0.5): https://docs.unity3d.com/ru//ScriptReference/HumanDescription-upperArmTwist.html
- Final IK TwistRelaxer (stacking relaxers): https://discussions.unity.com/t/final-ik-full-body-ik-aim-look-at-fabrik-ccd-ik-1-0-released/525769?page=219
- Earlier research that drove avatarfix-1104: research/vr_avatar_ik_2026-11-04.md (wrist ellipse from Physiotutors /
  PMC3016043). The *limits* in it were right; applying them as a hard clamp to an uncalibrated palm pose was the bug.

## 5. Legs ("standing straight, my knees are bent; straight only on tiptoe")

The geometry is brutal: a hip 2 cm under the leg's full length on a 0.9 m leg is already ~24° of knee bend. Two causes
were measured on the take:
- the typed height put the avatar's standing eye 2.4 cm above Zeke's real one;
- every small head dip went 1:1 into the knees.

Fixed on armik-1105 with VRChat-style auto-calibration plus a knee-lock zone (`avatar_body.stand_fix`):
- The body tracks the player's observed standing eye (rises over ~2 s, settles over ~10 s, only near standing) and lifts
  its head reference by the shortfall (≤ 8 cm).
- The first 3 cm of head drop below standing is absorbed, blending back to 1:1 by 6 cm.

Standing-straight knee p90: 34–35° → 16°. The rest of a real head dip should go into the neck and spine first, which the
crouch solver doesn't do yet. That's the next step.

## 6. armik2-1106 follow-up (2026-11-06): bee_warrior, the wrist instrument, spine-first legs

**bee_warrior.** Its rig rests in the *anatomical position* (arms down, palms FORWARD, a 31° carrying-angle flare at
the elbow), every other rig in a palm-down T-pose. Three effects followed, two of them measurement:
- The anatomical hinge (upper arm × forward) sits 24° off perpendicular to the Bee's bent rest forearm. Swing–twist about
  the rest forearm then reads a pure hinge bend as roll: 2·atan(0.41·tan(θ/2)), i.e. 82° at 130° of flexion. That was the
  "elbow-skin twist 96–117°". The roll is now measured against the virtually straightened rest forearm, the same
  reference arm_ik bends from.
- Fingerless hands assumed "palm = skeleton DOWN, fingers = the hand bone's +Y". All 11 fingered rigs follow the humanoid
  hand convention (fingers ≈ +Y, palm normal = +Z, radial = ±X, within ~10°). On the Bee the old assumption was 90° off
  about the fingers (flexion read as deviation) and 27° off along them.
- The twist cost's anatomical range is centred on the T-pose's palm-down. The Bee's zero is 94° supinated, so its rest
  rotation is now added to the cost.

**The wrist instrument (all rigs).** Two corrections:
- The palm frame must be orthonormal to the rest fingers. The old thumb axis (across the knuckles) leaned along them and
  read ~6° too radial.
- Flexion/deviation are read on the RADIUS. The wrist's axes ride on the radius, which carries all the pronation; the
  forearm bone carries only FOREARM_SHARE of it. On the half-rolled bone a pronated hand's flexion still leaked into
  deviation (the same mechanism armik-1105 fixed, at half strength).

PALM_TILT was re-measured on the corrected reading: 24° (was 30°).

**Legs: spine first.** A real body takes a small head drop in the neck and the trunk (flexion), and the knees only for
a real crouch. Limits used:
- **Cervical flexion 45°** (AAOS, *Joint Motion: Method of Measuring and Recording*, 1965). Youdas JW et al.,
  "Normal range of motion of the cervical spine", *Phys Ther* 1992;72:770–780, measured 50–60° active flexion in
  20–40-year-olds.
- **Thoracolumbar flexion 80°** (AAOS).

The posture lean (pivot = the Spine bone):
- takes only the drop the neck's own pitch doesn't produce;
- is capped at **35°**, under half the trunk's range;
- reaches that cap once the head pitches 45° down (the neck's range): a level-head drop is a crouch, a looking-down drop
  is posture.

Two measured causes of bent "standing" knees were not posture at all:
- Looking UP swings the avatar's eyes up about its low Neck pivot, and the eyes-on-headset pull dropped the hips.
- The standing-height reference was pinned 7 cm high by a jump, which switched the stand fix off for the session.

A hip 2 mm low is already ~15° of knee (two-bone geometry), which is why millimetres matter here.
