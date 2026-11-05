# VR avatar IK: crouch, hands, shoulders and wrists, the Bee's chest, and avatar height (2026-11-04)

Research helper for Zeke's 11-04 headset test of every avatar. The game code was **read only** and nothing was edited or committed.
Every number below comes from a **headless probe of the real IK**: Godot 4.7.2, `avatar_body.gd` + `arm_ik.gd` driven exactly as `rig_xr` drives them, with the pose read inside `skeleton_updated`.
Probe scripts and raw output are in `research/vr_avatar_ik_2026-11-04_probes/`. To re-run one:
`godot --headless --path /data/projects/vr_iris/project -s <probe.gd>`

## Summary (5 lines)

1. **Crouch and legs.** Feet are always placed from fixed constants (`±0.11·scale` wide, ankle at `0.08·scale`) instead of from each avatar's own rest ankle. As a result, 5 of 12 avatars stand with bent knees (10–34°), anna's and the Bee's toes sink 3.6 and 7.6 cm into the floor when they crouch, and jennifer's feet are twice her hip width. In a crouch the knees open out 30° while `FootFlat` holds the feet pointing straight ahead. The ankle absorbs that mismatch as 18–57° of inversion/eversion; a real ankle allows about 20–30° inversion and 10–15° eversion.
2. **Thumb and fingers.** The thumb curls about the *finger* axis (bone direction × palm-down). When the hand closes, the thumb tip moves **away** from the index finger (anna 4.7 → 6.3 cm, kyle 11.3 → 14.1 cm) instead of wrapping onto it. On 4 avatars the finger curl axes are also skewed 7–12°, because the code assumes palms face exactly −Y.
3. **Shoulders and wrists.** The upper arm's roll is set entirely by the elbow-hinge plane, with no limit on humeral rotation, and it reaches 150–159° on 5 avatars. The wrist cost treats all bend directions the same, so radial/ulnar deviation reaches 60–84°; a real wrist allows about 20° radial and 30° ulnar.
4. **The Bee.** The camera sits **11.5 cm behind the Bee's eyes and 18.5 cm behind its thorax front**, so the view is inside its torso. The cause is that the Bee's Hips rest pose is 30 cm forward of the model origin, while `_eyes_on_headset` may slide the hips at most 20 cm. The same offset drags its feet back, giving 87° shin tilt at a full crouch.
5. **Height.** `setup()` scales every avatar to the *player's* height (`_scale = height·0.935 / rest_eye`), so the avatar's own size is thrown away. The fix that keeps head motion 1:1 is a vertical camera offset (XR Tools `player_height_offset`, already used for floor calibration) equal to (avatar eye height − real eye height), with the avatar kept at its own size. Do **not** use world scale, because it scales motion.

---

## Issue 1: Crouch. Knees and ankles bend in ways a real body can't

### What the probe measured (left leg, 1.74 m player, hands in front)

| avatar | rest ankle vs IK target (m) | knee flex standing | ankle inv/ev at full crouch | knee vs toe direction, full crouch | toes at crouch (m) |
|---|---|---|---|---|---|
| anna | 0.135 vs **0.093** | 0° (leg overstretched 4 cm) | −31° | −30° | **−0.036** (in floor) |
| base_male | 0.123 vs 0.081 | 0° | −19° | −30° | 0.003 |
| bee_warrior | 0.198 vs **0.084** | 0° | **−57°** (shin 87° fwd, 85° out) | −23° | **−0.076** |
| jennifer | 0.138 vs 0.138; **feet 0.19 apart vs hips 0.094** | 0° but shin tilted 11° inward | IK collapses at full depth (inv/ev 167°, shin 139°) | +60° (knee caves in) | 0.003 |
| jimmy | 0.093 vs 0.093 | **17°** | −33° | −26° | 0.002 |
| kyle | **0.066** vs 0.087 | **23.5°** | −20° | −27° | 0.023 |
| oldmoustache | 0.093 vs 0.090 | **15°** | −38° | −32° | 0.0 |
| olivia | **0.075** vs 0.090 | **34°** | −25° | −27° | 0.017 |
| seed_san | 0.087 vs 0.093 | **11°** | −22° | −30° | 0.023 |

At full crouch, ankle dorsiflexion is 26–50° on most avatars, but **68° on the Bee**. The measured deep-squat reference is **38.5 ± 5.9°** (Hemmerich et al. 2006, cited in Kim et al. 2015).

The standing knee bend is a length mismatch. The hip-to-ankle-target distance is 0.8–3.6 cm shorter than thigh + shin; for example jimmy 0.722 vs 0.731 m and olivia 0.803 vs 0.839 m. A 1% shortfall already gives about 16° of knee flex.

### Likely causes in our code

- **`scripts/avatar_body.gd:939-940`** puts the foot targets at `(±0.11·_scale, 0.08·_scale, 0)`. Those are constants, not this avatar's rest ankle. `_rest_ankle` is measured at `:119` but only used by the FBT path (`:951`).
  - If the rest ankle is higher than the target (anna, the Bee, base_male, shino, bruno), the crouch pulls the ankle down and the flat foot's toes sink.
  - If it is lower (kyle, olivia), or the rest leg isn't straight (jimmy, oldmoustache), the knees stay bent while standing.
  - The width is wrong too: jennifer is a child-proportioned avatar scaled ×1.725, so `0.11·scale` puts her feet at 2× her hip width (A-frame legs and knee valgus). The Bee's feet are narrower than its hips.
- The crouch model at `:351` and `:375` also hard-codes `ah = 0.08·_scale`, so its knee/shin predictions are made for an ankle height the avatar doesn't have.
- **`FootFlat` (`:399-406`)** restores each foot to its **global rest basis**: toes straight ahead, sole level.
  - Meanwhile the crouch opens the knees up to `KNEE_OUT = 30°` (`:168`, knee poles at `:942-944`). The ankle then has to absorb the shin's outward tilt and axial rotation.
  - A real squatter turns the toes out so that **the knee tracks over the 2nd toe**, keeping knee and foot in the same plane. Ours keeps them 23–32° apart (and +60° on jennifer).
  - There is also no ankle limit: dorsiflexion is `SHIN_MAX = 45°` (`:157`), which sits above the measured 38.5 ± 5.9° heels-down squat. Inversion/eversion has no limit at all.
- **Hip hinge.** The lean is applied to Spine/Chest/UpperChest (`_drive_upper`, `q_ch`), while the pelvis (Hips bone) stays upright. In a real squat much of the trunk angle comes from the **pelvis tilting forward at the hip** (hip flexion 95 ± 27°, Hemmerich 2006). An upright pelvis with a bent spine reads as a hunched back on top of vertical thighs.
- **The Bee** also inherits the Hips-forward offset from Issue 4: its hips sit 30 cm ahead of the feet' reference point, so the legs rake backward.
- **Godot's `TwoBoneIK3D`** has no joint limits. The hinge plane is just hip–target–pole, so whatever the targets ask for, the knee does.

### How others solve it

- **Final IK VRIK** keeps feet planted with a procedural (or animated) locomotion layer and solves each leg with the knee bending toward a *swivel* built from the foot's own orientation. Feet are rotated with the body, so the knee and foot stay in one plane. Ready Player Me's VRIK setup recommends **"Animated" locomotion with an idle animation driving the lower body**: crouch and idle legs come from authored or mocap poses, and IK only plants the feet. (Ready Player Me Final IK setup docs; Resonite VRIK wiki.)
- **Meta Movement SDK** uses *Generative Legs*, ML-generated leg poses from the 3-point upper body, plus a light IK "hip pinning" processor that grounds the legs for seated poses. In other words, legs are a **learned or authored pose**, and IK only fixes contacts. (Meta developer blog; Movement advanced samples.)
- **VRChat and avatar-tool authors** note that after limb changes the standing height comes from the humanoid rest, and that "feet float or sink" whenever the ankle target isn't the avatar's own (head-ratio tool, Booth).
- **Biomechanics references:**
  - Deep squat: ankle dorsiflexion 38.5 ± 5.9°, hip flexion 95 ± 27°, knee flexion up to about 157° with heels raised (Hemmerich 2006, via Kim 2015, CC BY 3.0).
  - Restricting dorsiflexion *increases knee valgus and medial knee displacement* (Macrum et al. 2012, PubMed 22100617). Our clamped/flat foot plus open knees reproduces that visual pathology.

### Fix plan, in order

1. **Feet from the avatar's own rest.** Foot target = the rest ankle position of THIS avatar, in body space. That means x = rest `LeftFoot.x`, y = `_rest_ankle`, both scaled, and the same values passed into `_knee`/`_feet_back`. The invariant is *standing pose = rest legs*: 0° knee flex, toes on the floor.
2. **Toe-out that follows the knee.**
   - In `FootFlat`, yaw each foot about world-up by the knee's opening, so the foot's forward axis points along the knee direction projected onto the floor. Start with a 1:1 coupling to `_splay` and cap it at about 30°, which is a common squat toe-out but my estimate, not a measured value.
   - Keep the sole flat (pitch/roll from rest).
   - Widen the stance slightly with depth (feet move out as the knees open), so the shin stays near-vertical in the frontal plane.
3. **Clamp the ankle.** Use dorsiflexion ≤ 40° heels-down. Past that, either lift the heel (rotate the foot about the toes, as a heels-up squat does, which allows knee flexion to about 157°) or move the feet back. Clamp inversion/eversion to ≤ 10°. The search code at `:375-397` already exists; give it the extra degree of freedom of the heel.
4. **Pelvis hinge.** Split `_lean` into pelvis tilt (about 60%, rotating the Hips bone about the body's x axis) and spine (about 40%). Recompute `_hips_for` from the hip joint instead of the Spine bone.
5. **Optional, the shipped approach.** Blend toward an **authored crouch pose library**, retargeted onto the Godot humanoid profile from a CC0 source (see References), keyed on crouch depth. Keep IK only for foot plant and the hips height. This is what VRIK's "Animated" mode and Meta's generative legs effectively do.

### Headless test that would catch it (new `harness/unit/avatar_legs_unit.gd`, built from `probes/probe.gd`)

For every bundled avatar, driven at eye drops 0 / 0.25 / 0.45 / 0.68 m:

- **L1** standing: knee flex < 3°, both toes within ±1 cm of the floor, foot spacing within ±1.5 cm of the rest ankle spacing.
- **L2** every depth: toe/ankle y ≥ floor − 0.5 cm, so nothing sinks.
- **L3** every depth: ankle dorsiflexion ≤ 42°, inversion/eversion within ±12°, shin axial within ±15°. These come from the relative-rotation decomposition in `probe.gd:_rel_rotvec` (good for angles under 90°).
- **L4** every depth: horizontal angle between the knee direction and the toe direction ≤ 10°, i.e. the knee over the toes.
- **L5** monotonic: knee flex increases with depth, and the IK never flips. Catches jennifer's 0.68 m collapse.

(Note: `_rel_rotvec` mis-reports the knee's varus/valgus at flex > 120°, because it is a rotation-vector decomposition. For the knee, use hip–knee–ankle geometry, as `knee_flex` does.)

---

## Issue 2: Hands. The thumb goes where no thumb can, and finger poses are off

### What the probe measured (right hand, curl 0 → 1 = open → full grip + trigger)

| avatar | thumb tip → index MCP, open → fist (cm) | thumb tip → middle-intermediate at fist (cm) | thumb tip toward little finger, open → fist (cm) | palm normal vs assumed −Y |
|---|---|---|---|---|
| anna | 4.7 → **6.3** | 6.0 | −7.8 → −5.4 | **10°** |
| base_male | 7.3 → 9.4 | 8.5 | −9.5 → −7.3 | 7° |
| bruno | 8.5 → 11.8 | 10.6 | −12.3 → −8.4 | 3–4° |
| kyle | 11.3 → **14.1** | 12.9 | −11.2 → −7.4 | **9–12°** |
| sendagaya_shino | 7.3 → 9.3 | 8.3 | −9.4 → −7.0 | 7.5° |
| others (100Avatars family) | +3 to +4 cm away | 6.5–9.9 | barely crosses | ≈0° |

In a real fist or controller grip, the thumb tip ends on the index/middle **middle phalanx**, about 1–2 cm away, or on the face buttons. Ours ends 6–13 cm away and *further* from the index than when the hand was open. The Bee has no finger bones, so this doesn't apply to it.

### Likely causes in our code

- **`scripts/avatar_body.gd:629`**: `axis_g = dir.cross(Vector3.DOWN)` is used for **every** finger, *including the thumb*.
  - For the four fingers that is the knuckle line, which is correct when the palm faces exactly −Y.
  - The thumb, however, sits rotated about 80–90° about its own long axis (the thumbnail faces sideways). Its MCP and IP joints flex about an axis roughly *along the palm normal*, so flexing sweeps the tip across the palm toward the little finger.
  - Bending it about the fingers' axis gives **palmar abduction** instead: the tip drops away from the palm plane and swings away from the index. Those are the "positions that should not exist."
- **`:618/633`**: thumb max angles are Metacarpal 15°, Proximal 42° and Distal 39° (`× 0.6`). The thumb's IP joint flexes 70–90° (Physiotutors 85–90; AAOS-style 80), and the CMC needs **opposition**, a rotation, not just flexion.
- **`:629` again**: the "palms face −Y" assumption is off by 7–12° on anna, base_male, kyle and shino. Their fingers therefore curl partly sideways, which is the "finger poses need work on most avatars" part.
- **`:637-650`**: one curl scalar per group. Middle, ring and little fingers use identical curves, and there are no spread or abduction changes as the fist closes (real fingers converge).

### How others solve it

- **Godot XR Tools hands**: authored **hand poses** (Default, Grip 1–5, Grip Shaft, Hold, …) blended by an `AnimationTree` on the `grip`/`trigger` values (`parameters/Grip/blend_amount`). No procedural axes at all. The code is MIT; images are CC0 per the README, but the hand-mesh licence was not confirmed.
- **OpenXR controller-driven hand skeletons**: Godot's `XRHandTracker` has a `HAND_TRACKING_SOURCE_CONTROLLER` source, where joint poses are *inferred from controller inputs*. `XRHandModifier3D` can drive a skeleton's bones from it, in rotation-only mode, which keeps bone lengths. On Quest via the vendors plugin (already in `addons/godotopenxrvendors`), the runtime then supplies thumb-on-button and finger poses that Meta authored and tested. **Needs a headset check** that the data source extension is exposed on our runtime/build.
- **Thumb joint ranges** for clamps and pose authoring:
  - CMC flexion 45–50°, MCP 50–55°, IP 85–90° (Physiotutors).
  - Biomechanics maxima IP 60/−20°, MCP 60/−10°, CMC 20/−25° (core.ac.uk thesis).
  - Opposition is judged functionally: the tip reaches the little finger.

### Fix plan, in order

1. **Pose library instead of axes.** Author 5 hand poses once, on the Godot humanoid profile: open/relaxed, full grip, trigger-curl (index), thumb-on-face-button, thumb-up. Store per-bone **local rotations relative to rest**, retargeted per avatar through the humanoid profile (`RetargetModifier3D` / BoneMap). Blend them from `curl` and `thumb` the way XR Tools does. The XR Tools poses are a starting reference; check the mesh licence before copying any data.
2. **Correct the thumb axis immediately.** Even before (1), build each hand's real palm frame from the geometry, as `probe.gd:_hand_geom` does: palm normal = side · (fingers × (index − little)). Then:
   - fingers curl about the measured knuckle line;
   - the thumb's MCP and IP flex about (thumb direction × measured palm-*plane* direction toward the little finger);
   - the CMC gets a combined flexion + opposition rotation that brings the tip in front of the index/middle middle phalanx at grip = 1;
   - thumb IP max goes to about 80°.
3. Small spread: index and little fingers converge about 5–10° as the grip closes.
4. Later, `XRHandModifier3D` with the controller data source as an optional "runtime hand poses" mode, with (1) as the fallback.

### Headless test that would catch it (`avatar_hands_unit.gd`, from `probes/probe.gd:_hand_geom`)

For every avatar with finger bones:

- **H1** at curl = 1, the thumb tip is ≤ 2.5 cm from the index or middle intermediate bone, and its distance to the index MCP is *smaller* than at curl = 0.
- **H2** at every curl, the thumb tip stays on the palmar side (`palmar ≥ 0`) and never more than 1.5 cm through the palm plane.
- **H3** every finger's curl moves its tip within 10° of straight toward the palm normal (no sideways curl).
- **H4** per-joint angles stay inside the ROM table above (thumb IP ≤ 90°, MCP ≤ 60°, finger MCP ≤ 90°, PIP ≤ 110°, DIP ≤ 90°).

---

## Issue 3: Shoulders twist strangely on some avatars, and wrists too

### What the probe measured (right arm; poses front / down at the side / overhead / across the body / at the face × hand roll −90°/0°/+90°)

| avatar | max humeral twist vs rest | max wrist bend | max radial/ulnar deviation | max wrist twist |
|---|---|---|---|---|
| anna, base_male, kyle, olivia, shino | 88–93° | 63–77° | 61–72° | 35–43° |
| **bruno, kate, oldmoustache, seed_san, jennifer** | **148–159°** | 68–85° | 63–84° | 40–50° |
| jimmy | 107° | 75° | 62° | 35° |
| **bee_warrior** | 72° but **elbow-hinge error 73°** | 75° | 77° | 55° |

For comparison, real ranges are: humeral rotation about ±90° from neutral (ER about 90°, IR about 70°); wrist flexion about 74°, extension about 66°, ulnar deviation about 28°, radial deviation about 19° (goniometer study of 120 adults, via Physiotutors / PMC). **The wrist's sideways bend is 3× what anatomy allows**, and the upper-arm roll on the second group is about 60° past it.

### Likely causes in our code

- **`scripts/arm_ik.gd:142`**: `fu = _frame(E - S, n) * d.mu`. The upper arm's **roll is fully determined by the elbow-hinge plane** `n`. The swivel cost (`_cost`, `:162+`) penalises wrist bend, forearm twist, elbow-up, elbow-in, distance from the pole and frame-to-frame change, but **never the humeral rotation itself**. A swivel that keeps the wrist straight can therefore buy it with 150° of upper-arm roll, which shows as a twisted deltoid and shoulder.
  - The avatars that hit 150° are the ones whose `Shoulder`/`UpperArm` rest frames differ: bone +Y vs child 8–25° on the clavicle, per `probes/probe2.gd`. The rest-relative reference that "neutral" is measured against therefore differs per rig.
- **`arm_ik.gd:140`**: when the arm is nearly straight, `n = (E−S)×(W−E)` is tiny and the frame jumps to `pole_dir.cross(u)`. Near full reach, the upper arm and forearm roll can flip between frames.
- **`arm_ik.gd:166-168`**: `bend = fore.angle_to(hand_y)` is a single **isotropic** cone (`BEND_FREE` 20°), so 60° of ulnar deviation costs the same as 60° of flexion. Anatomy is an ellipse: about 74/66° flex/extend versus 19/28° radial/ulnar.
- **`arm_ik.gd:18/151`**: `FOREARM_SHARE = 0.55`. The other 45% of pronation/supination stays **at the wrist**, but anatomically the wrist cannot twist at all (the radius rolls over the ulna along the forearm). With no twist bones in VRMs, that 45% shows as a candy-wrapper twist at the wrist skin.
- **The Bee**: elbow-hinge error 73°. Its rig is an A-pose (arms 58° below horizontal) with no Hand bones (`boneY` probe: RightHand missing). The hinge `hinge_s = arm × forward` is computed from the drooped forearm, so the elbow bends about a tilted axis.

### How others solve it

- **Final IK VRIK**:
  - Shoulders: a *shoulder rotation mode* (YawPitch vs FromTo) and a *shoulder rotation weight* (how much the clavicle follows the hand), per the Resonite VRIK docs.
  - Elbows: a swivel offset plus a bend goal.
  - Twist: **TwistRelaxer** spreads twist over ForeArmTwist bones, relaxing each twist bone "using their initial rotations as the most relaxed pose" (Ready Player Me Final IK setup).
- **Godot 4.6**: `BoneTwistDisperser3D` "passes descendant bone rotation to ancestors", i.e. the stock twist-distribution modifier. The `ChainIK3D`/`IterateIK3D` family takes **per-joint `JointLimitation3D`** limits (Godot 4.6 IK article; IterateIK3D class reference). It also warns that ChainIK3D ignores twist.
- **Rigging practice**: decompose the wrist rotation into swing and twist, then distribute the twist over 2–3 forearm segments with falloff toward the elbow (for example 1.0 / 0.6 / 0.3; CMT swing-twist uses 0.5 / 1.0). That's how the candy-wrapper effect is avoided (Chad Vernon CMT docs; Polycount; SideFX forum).
- **Valve (Half-Life: Alyx)** chose hands only: "IK bodies are really jarring if they aren't perfect", and players "don't notice them missing" (AMA via VentureBeat/UploadVR). That's useful as a fallback mode, e.g. a hands-only option for avatars that fail the tests below.

### Fix plan, in order

1. **Anisotropic wrist cost and limit.** Decompose the hand-vs-forearm swing into flexion (about the palm's lateral axis) and deviation (about the palm normal), using the palm frame measured in Issue 2. Add cost outside an ellipse of flex 70 / ext 60 / radial 20 / ulnar 30. The swivel search then picks an elbow that turns deviation into flexion, which is what a real elbow does.
2. **Humeral-rotation term in the swivel cost.** Measure the upper arm's twist against a **swing-only reference**: rest frame swung onto the current upper-arm direction, which is the anatomical neutral and per-rig consistent. Add a cost outside ER 80° / IR 70°. This removes the 150° outliers.
3. **Twist distribution.**
   - Give 100% of pronation/supination to the forearm; the wrist keeps 0% twist.
   - To stop the elbow-end skin twisting, insert **two procedural twist bones** at load time: split the LowerArm vertex weights by position along the bone into LowerArm (elbow 0%), Twist1 (50%) and Twist2 (100%). Godot can `add_bone` and rewrite the skin weights in the import step.
   - Or try `BoneTwistDisperser3D` first, if it can work without dedicated twist bones.
4. **Straight-arm stability.** When `|n|` is small, keep last frame's hinge normal and blend toward the new one, instead of using the pole cross.
5. **Calibrate A-pose rigs (the Bee).** Compute `hinge_s` from a virtual T-pose: swing the rest arm up to horizontal first. Or reject rigs without a Hand bone from the anatomical solver and use the old chain.

### Headless test that would catch it (extend `harness/xr/xr_ik.py` / `joint_report`, or a new `avatar_arms_unit.gd` from `probes/probe.gd`)

For every avatar, 5 positions × 3 rolls (plus the existing 120-pose sweep):

- **A1** humeral twist versus the swing-only neutral stays within [−75°, +85°].
- **A2** wrist deviation within [−32°, +22°], flexion/extension within [−65°, +75°].
- **A3** `wrist_twist` (hand vs forearm) ≤ 10°, so all pronation is in the forearm.
- **A4** `elbow_hinge_err` ≤ 10° on every avatar, including the Bee.
- **A5** sweep the hand target out to 99% reach: the frame-to-frame upper-arm roll change stays ≤ 5°, so it doesn't flip.

---

## Issue 4: The Bee. Looking down, you see the inside of its own chest

### What the probe measured (`probes/probe2.gd`, `probe5.gd`)

- At level gaze, the Bee's eye point is **11.5 cm ahead of the headset**, i.e. the camera is behind its eyes. The front of its torso (Hips…UpperChest/Neck/Shoulder-skinned vertices, my-eyes layer 8) is **18.5 cm ahead of the camera**, the torso top is +0.2 to +1.5 cm *above* the camera, and the nearest torso vertex is 8 cm away. The camera is inside the thorax.
- Every other avatar converges to eye − camera = (0, y, 0): horizontal error 0, and the torso front is between −27 and +15 cm.
- The Bee's skeleton: `Hips` rest is at **z = +0.304 m** (forward, in skeleton space) and the Head at z = +0.21, against about 0 for every VRM. Calling `_eyes_on_headset` a second time brings the error to 0.0. So **the per-frame slide cap is the only thing in the way.**

### Likely cause in our code

- **`scripts/avatar_body.gd:937`** (`drive`) resets the Hips each frame to `_hips_rest.x/z + sit`. For the Bee (a Meshy-built GLB) the rest is 30 cm ahead of the model origin.
- **`:767`** `_eyes_on_headset` then limits the horizontal correction to 0.2 m (`limit_length(0.05 if waist else 0.2)`). 30.4 − 20 ≈ 11 cm stays uncorrected, so the camera ends up 11.5 cm behind the Bee's eyes, inside the head, collar and thorax. Looking down then shows the inside of the chest.
- The same offset shifts the hips 30 cm ahead of the feet's reference point, which is why the Bee's legs rake (Issue 1: shin 87°).
- **Secondary**: the Bee's fur collar tops out at about camera height even when correctly placed. Its torso front (+18.5 cm) is the deepest of any avatar, so even after the fix, looking straight down at 70° may graze the collar's back faces.

### How others solve it

- **VRChat**: the view position is the avatar's eye point, placed in the SDK. The body is moved so that point sits on the headset ("Eye Height is the height above 0 of an avatar's viewpoint", VRChat avatar-scaling docs). VRChat replied to a per-eye feature request that creators "can't offset the viewpoint freely" because it breaks their animation. The engine owns the alignment, not a clamp.
- **godot-vrm / VRM first-person**: first-person *mesh annotations* (`thirdPersonOnly` / `firstPersonOnly` / `auto`). `auto` builds a "(Headless)" copy without head-weighted vertices. Our `_route_layers` (`:413+`) uses those. The Bee is a GLB with no annotation, so its collar and neck vertices stay visible.

### Fix plan

1. **Normalise the rest Hips x/z at `setup()`.** Store `_hips_rest_xz` and offset `model.position` by `−_hips_rest_xz·_scale`, rotated by the 180° turn, so every avatar's hips start under its own origin. Or subtract it inside `drive()` when composing the Hips pose position. Keep the 0.2 m cap: it is a sanity limit on *motion*, not on rig offsets.
2. Also derive `_eye_fwd` and the foot targets from that normalised frame, which fixes the Bee's legs in Issue 1.
3. **Hide the Bee's collar from its own eyes.** In `_route_layers`, for unannotated GLBs, build a "(Headless)" copy that drops vertices whose dominant bone is Neck or Head. This is godot-vrm's `auto` rule extended one bone down for this rig. Plus a near-clip/neck "shrink" if needed.

### Headless test that would catch it (extend `avatar_body_unit.gd` E-series, from `probes/probe2.gd`)

For every avatar at head pitch 0 / −45 / −70°:

- **B1** horizontal |eye − camera| < 1 cm.
- **B2** no layer-8 (my-eyes) torso vertex within 10 cm of the camera.
- **B3** the camera is not behind the front of the torso while below the torso top: `!(torso_front_ahead > 3 cm && torso_top > cam.y)`.

Today the Bee fails B1 (11.5 cm), B2 (8 cm) and B3.

---

## Issue 5: Avatar height. Eye height ignores the avatar's size (head motion must stay 1:1)

### What the code does

- **`scripts/avatar_body.gd:121`**: `_scale = (user_height · 0.935) / _rest_eye`. Every avatar is resized so its eyes sit at the *player's* eye height, so the avatar's own size never reaches the camera.
- Native eye heights measured (`rest_eye`, model metres): jennifer **0.94**, anna 1.39, jimmy 1.40, seed_san 1.41, olivia 1.43, oldmoustache 1.43, shino 1.47, kate 1.48, kyle 1.49, bee 1.57, base_male 1.60, bruno **1.84**.
- The camera height itself is the XR Tools `PlayerBody` origin + `player_height_offset`. That offset is already used for the floor calibration (`rig_xr.gd:61, 170, 1279-1282`) and by "seated mode" (`calibrate_player_height`, `player_body.gd:504-507`).

### How others solve it

- **VRChat**: you see from the avatar's View Position. Scale = prefab eye height / target eye height, applied to the avatar root. Community reports say the *world/tracking* gets scaled (arm-length-based), so motion is scaled too. Known complaints: "longer-legged avatars' hands can't reach the floor, shorter ones sink through it", and "the floor is far below my real feet" (VRChat docs and feedback; SlimeVR config guide).
- **BONELAB**: measures the player (height, wingspan, inseam…), and the avatar's *proportions* drive stats (speed, strength, recoil) (UploadVR, Road to VR). I could not confirm from a primary source how BONELAB sets the camera for a taller avatar, so treat it as unknown.
- **HurricaneVR HexaBody** (BONEWORKS-style): the player can't be scaled; it uses an "Eye Height" field.
- **Research**: scaling tracking changes reach and performance, case-dependent (Effects of tracking scale on user performance in VR games, nweon summary).

### Fix plan (honours "head motion stays 1:1")

1. **Avatar size is the avatar's own.** Add a setting `avatar_size_mode`:
   - `native`: `_scale = 1`, or a per-avatar "intended height" from its card.
   - `match_me`: today's behaviour.
   - an optional size slider (clamped, for example 0.5–1.3× native, like VRChat's 0.2–5 m dial but sane for physics).
2. **Raise or lower the view by an offset, not a scale.** `player_height_offset = floor_offset + (avatar_eye_h − player_eye_h)`. A 1 cm head movement stays a 1 cm view movement, and only the origin of the view changes.
3. **Hands and the floor.** The same offset moves the controllers, so on a tall avatar you can't reach the floor and on a short one your hands go through it. Three options for Zeke to choose from:
   - (a) accept it, as VRChat does;
   - (b) apply the offset **only to the camera, not the hands**. That needs a hand-space correction, and the arms' IK absorbs it (we already stretch up to 15%);
   - (c) **fade the offset out as you crouch** (offset × eye_h/standing_eye_h), so floor contact is preserved. Vertical motion is then not exactly 1:1 near the floor, but horizontal motion is.
   - **I'd recommend (c) for a tall avatar and (a) for a short one.**
4. **Keep everything else consistent:**
   - the collision capsule height (physique);
   - the network look: send the avatar size instead of only `height_m`, and have `Net._clean_look` and `RemoteAvatar` use it;
   - walk speed and step height (avatar-proportion stats, like BONELAB);
   - the "Measure me" calibration (`rig_xr.gd:1284`), which must keep measuring the *player*, not the avatar.

### Headless test that would catch it (extend `scale_unit.gd` / `avatar_parity_unit.gd`)

- **S1** with `native` mode and a 1.74 m player: camera eye height = the avatar's native eye ±1 cm (bruno about 1.84, jennifer about 0.94).
- **S2** moving the headset +10 cm up or forward moves the camera exactly +10 cm (ratio 1.000 ± 0.005, so no world scaling).
- **S3** the avatar's eye stays on the headset at every height (reuses B1).
- **S4** in (c) mode, a controller placed at the real floor reaches the virtual floor at full crouch on the tallest and shortest avatars.
- **S5** remote parity: the other client sees the same size (`avatar_parity_unit` P-series).

---

## Top fixes, in order

1. **Feet from each avatar's own rest ankle** (position and spacing) instead of `0.11·scale` / `0.08·scale` (`avatar_body.gd:939-940, 351, 375`). Removes bent standing knees on 5 avatars, sinking toes, and jennifer's A-frame. Small change, test L1/L2.
2. **Normalise the Bee's Hips rest x/z at setup** (`avatar_body.gd:937` and `:767`). Fixes the camera inside the chest and the Bee's raked legs. Small change, test B1–B3.
3. **Toes follow the knees plus ankle limits** (`FootFlat` `:399-406`, `KNEE_OUT`/`SHIN_MAX` `:157/168`): foot yaw = knee opening, dorsiflexion ≤ 40°, then heel lift. Fixes the "ankle bends wrong" in every crouch. Tests L3/L4/L5.
4. **Thumb on its own axis, with a measured palm frame** (`avatar_body.gd:618-650`); next, a retargeted hand-pose library blended by grip/trigger/thumb (XR Tools style). Tests H1–H4.
5. **Arm IK: anisotropic wrist ellipse, a humeral-rotation cost, 100% of the twist in the forearm (+ procedural twist bones or `BoneTwistDisperser3D`), stable hinge near full reach** (`arm_ik.gd:140-151, 162-186`). Tests A1–A5.
6. **Avatar height by offset** (`avatar_body.gd:121` scale mode + `player_height_offset`). Needs Zeke's call on floor reach (option c recommended). Tests S1–S5.
7. **Pelvis hinge in the crouch** (split `_lean` between Hips and spine), then optionally an authored or mocap crouch blend.

---

## References

### Shipped systems and engine docs
- Final IK VRIK setup (Ready Player Me docs, alpha guide; Twist Relaxer, Animated locomotion): https://docs.readyplayer.me/ready-player-me/integration-guides/unity/setup-for-xr-alpha/setup-final-ik and https://docs.readyplayer.me/ready-player-me/integration-guides/unity/setup-for-xr-beta/setup-final-ik
- VRIK component parameters (shoulder rotation mode/weight): https://wiki.resonite.com/Component:VRIK
- VRChat feedback, VRIK spine overbend: https://feedback.vrchat.com/vrchat-ik-20/p/in-lock-both-mode-spine-scrunching-pathologically-overbends-the-neck
- Meta Movement SDK, Generative Legs + IOBT: https://developers.meta.com/horizon/blog/inside-out-body-tracking-and-generative-legs ; hip pinning / advanced samples: https://developers.meta.com/horizon/documentation/unity/movement-advanced-samples
- Godot 4.6 IK (TwoBoneIK3D, ChainIK3D, IterateIK3D, BoneTwistDisperser3D): https://godotengine.org/article/inverse-kinematics-returns-to-godot-4-6/ ; IterateIK3D (joint limitations): https://docs.godotengine.org/en/4.6/classes/class_iterateik3d.html
- Godot XRHandModifier3D / XRHandTracker (controller data source): https://docs.godotengine.org/en/latest/classes/class_xrhandmodifier3d.html (I only retrieved translated pages; the English page wasn't fetched)
- Godot XR Tools hand poses: https://godotvr.github.io/godot-xr-tools/docs/hand_poses ; repo licence (code MIT, images CC0, mesh licence unconfirmed): https://github.com/GodotVR/godot-xr-tools
- VRChat avatar scaling: https://creators.vrchat.com/avatars/avatar-scaling ; Udon player scaling: https://creators.vrchat.com/worlds/udon/players/player-avatar-scaling ; per-eye view position thread: https://feedback.vrchat.com/avatar-30/p/per-eye-view-position ; floor/IPD scale thread: https://feedback.vrchat.com/feature-requests/p/base-ipd-scale-adjustment ; SlimeVR VRChat config: https://docs.slimevr.dev/tools/vrchat-config.html
- BONELAB avatar system: https://uploadvr.com/bonelab-avatar-system/ ; Road to VR interview: https://roadtovr.com/?p=171242
- HurricaneVR HexaBody: https://cloudwalker2020.github.io/HurricaneVR-Docs/manual/hexabody/hexabody.html
- Half-Life: Alyx, no arms: https://gameinformer.com/interview/2020/03/23/valve-talks-half-life-alyx-and-why-arms-dont-work-in-vr ; AMA recap: https://venturebeat.com/ai/half-life-alyx-ama-everything-we-learned
- Swing-twist decomposition (CMT): https://chadmv.github.io/cmt/html/rig/swingtwist.html ; Cascadeur twist bones: https://cascadeur.com/help/rig/advanced_rigging/advanced_rigging_techniques/twist_bones

### Biomechanics
- Kim et al. 2015, *Lower Extremity Strength and the ROM in Relation to Squat Depth*, J Hum Kinet 45:59–69, **CC BY 3.0**. Cites Hemmerich 2006: squat dorsiflexion 38.5 ± 5.9°, hip flexion 95.4 ± 26.6°. https://pmc.ncbi.nlm.nih.gov/articles/PMC4415844
- Hemmerich et al. 2006, *Hip, knee and ankle kinematics of high ROM activities of daily living*, J Orthop Res (knee about 157° in a heels-up squat). Abstract only seen.
- Macrum et al. 2012, restricted dorsiflexion → knee valgus in a squat: https://pubmed.ncbi.nlm.nih.gov/22100617/
- Schoenfeld 2010, *Squatting kinematics and kinetics*, J Strength Cond Res 24(12) (already cited in the code): https://pubmed.ncbi.nlm.nih.gov/20182386/
- Wrist and hand ROM (Physiotutors): https://book.physiotutors.com/?p=232 ; wrist ROM study table: https://pmc.ncbi.nlm.nih.gov/articles/PMC3016043 ; thumb joint ranges thesis: https://core.ac.uk/download/147998492.pdf

### Mocap and pose sources (licences as stated by the source; verify each file before use)
- **Quaternius Universal Animation Library 1 & 2**: **CC0**, universal humanoid rig, Godot-ready (sitting, crawling; crouch not confirmed in the list). https://quaternius.com/packs/universalanimationlibrary.html · https://quaternius.itch.io/universal-animation-library
- **CMU Graphics Lab Motion Capture Database**: "free for all uses", citation requested. Not formally CC0. BVH conversion by B. Hahne, "no additional restrictions". Mirrors: https://github.com/meshula/cmu-mocap ; bone rolls are messy and need retarget cleanup.
- **Cologne (TH Köln) Motion Capture Database**: **CC BY 4.0**, BVH. https://mocap.web.th-koeln.de/about.php
- A Hugging Face CMU repack labelled CC BY 4.0 (attribution): https://huggingface.co/datasets/ckinpdx/CMUMDB
- Photo references: Wikimedia Commons `Category:Squat (exercise)`. Check the licence per file (most are CC BY-SA); no specific file was verified here.
- *Not usable as-is*: anything CC BY-NC (for example Bandai Namco's research motion set) if the game is ever distributed beyond personal use.

### Gaps and honesty notes
- **YouTube transcripts**: `yt-dlp` hit HTTP 429 (rate limit) on the first subtitle fetch. No talk transcripts were used. The only candidate found was "Inverse Kinematics – Godot 4.6 Tutorial" (Lukky, MbaPDWfbNLo). Retry later into `/home/iris/untrusted/yt_ik/`.
- I could not find a primary source for BONELAB's camera-vs-avatar-height behaviour, or for a measured "toe-out angle" in a squat. The 1:1 knee/foot yaw coupling and the 30° cap are **my estimates**, to be tuned against Zeke's own crouch.
- The probe's knee varus/valgus column breaks down above 120° of flexion (rotation-vector decomposition), so the conclusions above use hip–knee–ankle geometry instead.
