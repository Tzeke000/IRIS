# Full-body physics avatar (BONELAB-style) for VR Iris / LUMEN — research + design (2026-11-04)

**Summary (5 lines)**
1. BONELAB's body is three layers: tracked input → a "remap" solver → a **physics rig** of rigid bodies (pelvis, chest, head, arms, hands, a knee, and a **rolling ball as the feet**), with hands pulled toward the controllers by a **force-capped spring**. The reaction force goes back into the body, and that is why climbing and heavy objects feel real.
2. **Nobody has published a finished open-source version of this for Godot 4.** The XR Tools team says they "haven't cracked" Boneworks. The best things to learn from are godot-xr-kit (MIT, Godot 4.2, physics hands + a physical body), BIMOS (MIT, Unity, Boneworks-inspired) and BONELAB Fusion's PD controller (MIT). HexaBody is paid and closed, so it's out.
3. Our game already has half the pieces: hands that are blocked by the world, a force-based hold, a "solid to yourself" body, climbing + mantle, webs, and a strength stat. The pieces that are not physical are the **locomotion body** (XR Tools' kinematic capsule) and the **hands** (kinematic `move_and_slide`).
4. Multiplayer stays the way it is: **each player simulates their own body** (BONELAB Fusion works this way too). The server only sees and checks the poses, which grow from 3 transforms to about 10, roughly 6 KB/s per player. The real netcode cost is **player-vs-player shoving**, because two owners can't both simulate the same contact.
5. Smallest first step: **dynamic physics hands behind a setting** (Stable-PD drive, strength cap, ghost hand), with everything else left alone. The full rolling-ball body is stage 3 and needs Zeke's decisions below first.

Background already on disk (not repeated here): `research/bs_mods_rumble_bonelab_2026-10-26.md` §5 (rig layout, Fusion Stable-PD numbers, HurricaneVR, XR Tools collision hand), `research/bonelab_avatars_2026-10-22.md` §4 (Marrow SDK body measurements + mass/stats), `research/bonelab_bs_physics_review_2026-11-02.md` (items 4 and 9, hold-law risk H2). Those paths are in the game's research folder `/data/projects/vr_iris/research/`.

---

## 1. How BONELAB / Boneworks bodies work

### 1.1 The rig (from BONELAB Fusion source, Marrow SDK, interviews; SLZ has never published its tuning)
- **ControllerRig**: the headset and controllers as tracked.
- **RemapRig**: fits the real person onto the avatar's proportions (the "body remapping system") and adds walk/jog, jump and crouch intent.
- **PhysicsRig**: rigid bodies for **pelvis, chest, head, upper arm / forearm / hand ×2, a knee, and the feet as one locomotion ball**, plus a grounder. Legs can be switched between physical and kinematic, and the whole rig can go limp (ragdoll).
- **Avatar**: the visible mesh, posed by IK to follow the physics rig. It never collides itself.

### 1.2 Locomotion: ball + knee + pelvis
- The feet are a **high-friction sphere that is rolled by the stick** (torque or angular velocity on the ball). Friction against the floor moves the body. Slopes, stairs, moving platforms and being knocked about all come for free from the solver.
- The **knee** is a body between the ball and the pelvis. Its joint height is driven, so crouching and jumping mean lowering or raising and then driving the knee and pelvis. HexaBody (a documented Boneworks clone) adds a **fender**: a sphere slightly larger than the ball and raised a little, so the ball can't climb walls.
- **Air control** is a capped force on the knee (Fusion's `F = (clamp(v_err)·5 + v_target)·m_total`).
- SLZ said walking speed is capped around **4 m/s** by the physics step versus the collider size. A small ball rolled faster starts to skip.
- The **pelvis and chest** are driven toward the point under the head (spring + damper). The head body follows the headset. When the head is blocked (pushed into a wall), the body stays outside and the head body stops at the wall. The camera itself is always the real headset.

### 1.3 Hands: why BONELAB hands feel heavy
- Every hand is a **rigid body driven toward the controller by a PD controller with a force cap** ("hands can only accelerate so fast"). It is not teleported.
- Because the drive is a real force, Newton's third law gives a **reaction on the body**:
  - **climbing**: hand anchored to the wall, so the drive pulls the body up
  - **pushing off**
  - **heavy objects**: a 30 kg crate drags your hand down and pulls you off balance
- **Grabbing** is a joint between the hand and the object (BONELAB uses typed grips: cylinder, box, sphere and target grips with break forces). HurricaneVR does the same with a joint spring and damper of about spring/10, and max force = strength.
- **Best open control law found:** Fusion's `PDController` (Tan/Liu/Turk "Stable PD"), which stays stable at high gains. The numbers are in the 10-26 note §5. Our current hold law is **deadbeat** (it cancels all of the velocity error in one tick). That is the probable cause of flake **H2** (period-2 tumble), see 11-02 review §B1.

### 1.4 Comfortable vs nauseating
What reviewers report (Mixed, UploadVR, Android Central): BONELAB's physics body is **hard on comfort**. The unpredictability, being thrown, minecarts and a "terribly shaky camera" make people sick. Comfort options are minimal: smooth locomotion only, no vignette, no teleport. A 2019 community thread blames the *physically simulated head* for most of the sickness. That is anecdotal, not a developer statement.

General VR comfort practice says the same thing (I found no SLZ source for these specific rules; they come from general VR design):
- **The camera is the headset, always.** Physics may never rotate the view and should never translate it on its own. Body jitter must not reach the eyes.
- **Smooth out the ball's vertical bob.** The rig root (XROrigin) follows the pelvis with a filter. Small steps and bumps get eased, not passed through tick by tick.
- **Head in a wall → fade to black** rather than shoving the view back. We already do this (`_fade_in_walls`).
- **Big involuntary motion is the worst case**: knockback, explosions, being flung by another player. Cap the knockback anyone else can give you, and offer a "comfort body" (see §6 decisions).
- Keep our comfort pack (vignette, snap turn, teleport) working with the new body. BONELAB not having these is a known complaint, so we shouldn't copy that.

---

## 2. Open-source search: what exists (and dead ends)

| What | Engine / licence | Useful for | Verdict |
|---|---|---|---|
| **godot-xr-tools** (in the project, v4.5.2) | Godot 4, MIT | PlayerBody (kinematic CharacterBody3D), CollisionHand (kinematic `move_and_slide`), `hand_physics_bone.gd` (top-level bone bodies) | What we use now. The team says it hasn't matched Boneworks. Climb mode collapses the player to a sphere. |
| **godot-xr-kit** (patrykkalinowski) | Godot 4.2, **MIT** | Physics hands blocked by the world, heavier objects harder to move, two hands easier, **ghost hand** when far from the controller, reset when stuck; **physical player body** that collides and can push off surfaces | **Best Godot reference to read and adapt.** Its `new_playground` branch crashes on standalone (irrelevant: we're PCVR). |
| **Godot-Active-Ragdolls** (R3X-G1L6AME5H) | Godot 3 / Bullet era, MIT | RigidBody + 6DOF joints with motors chasing an animated skeleton | The approach only (Godot 3 API). Useful for stage 5 and for the sparring dummy. |
| **hellrot-vr** (iamrequest) | game-jam project, licence not checked | RigidBody locomotion, wall running, grapple | Read only if we want rigidbody wall-run ideas. |
| **BIMOS** (KadenZombie8) | Unity, **MIT** | "Physics-based VR character controller inspired by Boneworks": full-body IK, procedural legs, hand poses, ragdoll | A reference for structure. It lists its own limits (body colliders centred on the head, "too far forward"; physics tied to 144 Hz). That's a warning for us: centre the body on the *neck*, as `self_body.gd` already does (7 cm behind the eyes). |
| **BONELAB Fusion** (Lakatrazz) | Unity mod, MIT | `PDController.cs` numbers; how remote players get the **same physics body** (73 floats + 8 ellipses of avatar stats) | Already mined (10-26 §5). |
| **Marrow SDK** (SLZ) | Unity, **no licence** | Avatar measurement model | Read, don't copy (10-22 §4). |
| **HexaBody VR** | Unity Asset Store, **$59.99, closed** | The standard Boneworks clone | **No** (zero-spend rule, and the source is closed). The free itch demo is IL2CPP-protected. |
| **GorillaLocomotion** (Another Axiom) | Unity, MIT | Hand-push locomotion | Not a full body. See the movement-game doc. |

**Dead ends (documented so nobody repeats them):**
- No GDC or other SLZ talk on the body's knee/pelvis solver turned up. The Boneworks Wikipedia, UploadVR and GamesBeat pieces describe it only at a high level.
- No complete Godot 4 Boneworks-style body exists. Forum threads (Godot forum "Rigid bodies as hands", "Improving hand physics in VR") report the same problems we hit: a RigidBody under a controller fights its parent (fix: `top_level`), and fast hands with infinite strength break the solver.

**Engine facts that matter (Godot 4.7 built-in Jolt):** `Generic6DOFJoint3D` has angular/linear springs and motors with **per-axis force limits** in Jolt (other backends ignore them). The motor limit only applies when the motor is enabled, so check the spring-only path. Keep neighbouring bodies within about 10:1 mass ratio (`ragdoll.gd` already follows this).

---

## 3. What the code does today (read 11-04) and what would change

Paths relative to `/data/projects/vr_iris/project/`.

| Part | Today | In a physics body |
|---|---|---|
| **Locomotion body** | `rig_xr.gd` instances XR Tools `player_body.tscn`: a **kinematic CharacterBody3D capsule** (layer PLAYER_BODY L20, `move_and_slide`). Its engine platform logic is switched off; XR Tools' patched ground velocity is used instead. | Replaced by a **ball + knee + pelvis + chest** RigidBody3D chain. The XROrigin follows the pelvis (filtered). |
| **Movement providers** | About 8 write `PlayerBody.velocity` / `ground_control_velocity`: direct walk, turn, jump, XR Tools climb, `movement_mantle.gd` (order 14), `movement_armswing.gd` (order 11), teleport, flight (`flight.gd`), web sling (`web_sling.gd` sets velocity through a one-sided distance constraint on the chest). | **Keep them.** Write a facade with the same fields (`velocity`, `on_ground`, `ground_control_velocity`, `request_jump`) that turns "wanted velocity" into ball torque + knee force. Each provider is then re-verified one by one. The web becomes a real force or joint on the chest body, which is closer to Zeke's "realistic" web spec. |
| **Own body** | `self_body.gd`: an AnimatableBody3D torso + 2 leg capsules under the head, which only MY hands touch (L11). | It becomes the physics rig's chest/legs. The legs start collidable with the **world** too (stage 4). |
| **Hands** | XR Tools `CollisionHand` = kinematic force body moved by `move_and_slide`, with `max_push_force` patched to `Strength.max_force`. Palm 3 × 8.5 × 9 cm on the palm pose (10-30 fix). Grabs = XR Tools FunctionPickup (`pickup_surface.gd`) + `prop.gd::_hold_step` deadbeat force hold. | **Dynamic RigidBody3D hands** (about 0.4 kg each) with a Stable-PD drive toward `PalmTarget`, capped by `Strength`. Holds become a joint or PD between the hand and the prop, which fixes H2 and makes pushes mass-aware (a kinematic hand is never mass-aware: 11-02 §B1). |
| **Arms** | `arm_ik.gd` visual IK plus OWN_ARMS capsules (kinematic). | Physics forearm/upper arm come later (stage 4). Until then the IK arms stay visual. |
| **Strength / mass** | `strength.gd` (from avatar physique) caps push force. | The same stat caps the hand drives and the knee force, like BONELAB's strengthUpper/strengthLower. |
| **Desktop player** | `player_flat.gd`: CharacterBody3D, own WASD/jump code. | Gets the **same rig**, with hand targets from the mouse/keys (stage 3b). Otherwise VR and desktop play different physics games, which breaks the minigames' fairness. |
| **Remote players** | `remote_avatar.gd`: an AnimatableBody3D body + kinematic hand bodies (REMOTE_HAND L12), interpolated from 30 Hz poses. | The same, with about 10 kinematic proxies instead of body + 2 hands. Proxies are cheap. |
| **Bots / Iris** | `rig_iris.gd` / bot clients have their own movement. | Iris's floating eye body doesn't need legs. Bots can keep the kinematic path. |

**Things that will break and need re-testing:** every movement provider; seats (`seats.gd`); carrying Iris; standing on props (the 10-26 beach-ball launch bug comes back in a new form, because a ball standing on a spinning ball *will* be flung, so physics, not a patch, has to handle it); teleport hops (`teleport_go`); flight; the whole `vr.sh regress` locomotion suite. That's why the plan below is staged behind a setting.

---

## 4. Multiplayer / netcode cost

**What stays the same:** `net.gd`'s model is "every client is authoritative over its OWN rig; poses at 30 Hz; the server relays and checks." This is also BONELAB Fusion's model. Making player bodies **server-authoritative Jolt is not viable for VR**: the body under your own eyes would lag one round trip behind your head and hands, which is the textbook nausea case. The server stays authoritative for **game state** (props at rest, ownership claims, powers, minigame rounds, eliminations), as now.

**Bandwidth:** about 10 body transforms (position 12 B + compressed quaternion ~8 B) ≈ 200 B per pose × 30 Hz ≈ **6 KB/s up per player**, and N−1 times that down. At 12 players that's about 66 KB/s per client down. The 11-03 audit says home uplink is mostly voice relay, so this is small next to voice. *(Estimate; measure with the stress plan's Phase B bots.)*

**Server CPU:** remote bodies are kinematic proxies, similar to today's body + hand bodies, so cost grows linearly and stays small. The audit already names "server CPU on 2 cores: server-owned props + city movers" as the likely first bottleneck. Player proxies don't add to that much.

**The hard part: shoving other players.**
- Each owner simulates their own body and sees everyone else as **kinematic (infinite mass)**. If I walk into you, I stop dead, as if you were a wall, and you feel nothing until your own client sees my proxy overlap yours.
- Design (my proposal; I found no published Fusion detail on how they handle it): when my body or hand hits your proxy, my client sends a **contact impulse** to you through the server. The server checks distance, rate and a **cap**. Your client applies it to your own body. Result: shoves are mutual, delayed by about one RTT, and capped so nobody can launch you (comfort + griefing).
- This is exactly what spleef "knockback" and tag need, so it's worth building once and well.

**Anti-cheat:** stays light (friends-only, invite world). The server already checks reach (`near_enough`) and rates. Add a speed sanity check on the pelvis.

---

## 5. Staged plan (smallest first)

Effort: S ≈ an evening, M ≈ 1–3 days, L ≈ a week+. Each stage ships behind a Settings ▸ Comfort toggle, defaults OFF until Zeke tries it in the headset.

| Stage | What | Effort | Proves |
|---|---|---|---|
| **1. Physics hands** (smallest first step) | Dynamic RigidBody3D hand per side, Stable-PD drive to `PalmTarget` (Fusion numbers to start), force cap = `Strength.max_force`. Ghost hand (godot-xr-kit idea) when >0.15 m from the controller, snap back >1 m. Holds = a 6DOF joint hand↔prop (or PD); keep the deadbeat hold for the OFF path. | M | Mass-aware pushes; H2 tumble gone (`probes/heldpair_squeeze.gd`); two-hand heavy lifts; feel. |
| **2. Reaction force into the body** | Hand drive reaction `−F` (capped) applied to the existing PlayerBody velocity. Grab any rough surface to climb; heavy things pull you. Mantle stays as "finish the climb". | S–M | Climbing feels earned without replacing locomotion yet (11-02 review item 4). |
| **3a. Physics locomotion body (VR)** | Ball + fender + knee + pelvis + chest; the facade for movement providers; filtered origin; crouch/jump through the knee; web as a chest force. | L | Stairs, slopes, being knocked; all providers pass regress. |
| **3b. Desktop on the same rig** | `player_flat.gd` drives the same body; hands from mouse/keys. | M | Crossplay physics parity (needed for spleef/tag fairness). |
| **4. Physical limbs + player contact** | Forearm/upper-arm and leg bodies that collide with the world; the contact-impulse RPC for player-vs-player shoves with a cap. | L | You can shove a friend and they feel it. |
| **5. Ragdoll states** | Big hit / long fall: muscle strength drops (Active-Ragdolls approach), the camera **stays on the headset**, the body goes limp and you get up after a moment. | M | Spleef falls and knockouts look right. |

---

## 6. Risks
- **Motion sickness.** BONELAB's own reviews are the warning. Mitigation: the camera is never physics-driven, the vertical bob is filtered, shoves are capped, and a "comfort body" option keeps the current kinematic locomotion.
- **Regressions everywhere.** Locomotion is under every system (webs, flight, seats, teleports, carrying Iris). The toggle and stages keep the current path alive until the new one passes regress.
- **Solver stability:** hand (0.4 kg) ↔ body (70 kg) is far past 10:1. Don't joint the hands straight to the chest; connect them by forces (as BONELAB does), or add arm bodies between them to step the ratio.
- **Tick rates:** the client runs at 72 Hz (Quest via Air Link), the server at 90. Stable PD is dt-implicit, which is the reason to use it, but test the gains at both rates.
- **4 m/s ball limit** vs our desktop sprint 4.5 m/s and web flings: fast motion (web, flight) should switch the legs to kinematic, as `PhysicalLegs()/KinematicLegs()` does in BONELAB.
- **Avatars:** VRM proportions have to drive the rig size (the BONELAB ellipse model in 10-22 §4). A tiny or huge avatar can't be allowed to be unplayable (blend mass toward a reference, clamp height).
- **XR Tools drift:** replacing PlayerBody moves us away from upstream XR Tools. That's acceptable, but a large fork to maintain.

## 7. What Zeke must decide
1. **Opt-in or for everyone?** Is the physics body a comfort setting (some friends keep the current smooth body), or the only way to play?
2. **Is stage 1 (physics hands only) the go-ahead**, before any locomotion change?
3. **Shoving:** should players physically push each other **everywhere**, or only inside minigame arenas (spleef/tag)? Everywhere is more fun, but also a griefing and comfort risk in a social world.
4. **Ragdoll on big hits/falls:** yes or no, and should it ever take control of the view (recommend never)?
5. **Desktop players on the same physics body?** (Recommend yes; it's required for fair minigames.)
6. **Iris's body:** her floating eye + cable arms, should it get the same physics treatment, or stay as it is?

## Sources
- godot-xr-kit (MIT): https://github.com/patrykkalinowski/godot-xr-kit
- Godot XR Tools: https://github.com/GodotVR/godot-xr-tools (VERSIONS.md: climb collapses the player to a sphere)
- Godot-Active-Ragdolls (MIT): https://github.com/R3X-G1L6AME5H/Godot-Active-Ragdolls
- hellrot-vr: https://github.com/iamrequest/hellrot-vr
- BIMOS (MIT): https://github.com/KadenZombie8/BIMOS · demo https://kadenzombie8.itch.io/bimos-demo
- HexaBody (commercial): https://assetstore.unity.com/packages/tools/physics/hexabody-vr-player-controller-185521
- BONELAB Fusion (MIT): https://github.com/Lakatrazz/BONELAB-Fusion · Marrow SDK: https://github.com/StressLevelZero/MarrowSDK
- Godot forum: https://forum.godotengine.org/t/rigid-bodies-as-hands/67646 · https://forum.godotengine.org/t/improving-hand-physics-in-vr/94661
- Godot Jolt 6DOF limits: https://github.com/godot-jolt/godot-jolt/issues/110
- Boneworks body: https://www.uploadvr.com/boneworks-body-vr-game-mechanic/ · https://gamesbeat.com/how-boneworks-turns-your-body-into-its-key-mechanic/
- BONELAB comfort reviews: https://mixed-news.com/en/bonelab-review-vr-shooter-hit-or-failed-experiment/ · https://uploadvr.com/bonelab-review/ · https://www.androidcentral.com/gaming/virtual-reality/bonelab-review-the-new-standard-for-vr-physics-based-games
