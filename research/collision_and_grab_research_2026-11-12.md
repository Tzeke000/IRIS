# Collision that matches the model, and grabbing: research for LUMEN (2026-11-12)

**Zeke's ask (11-12):** "Collision should match the 3D model for EVERYTHING (guns, swords, buildings — not just bottle
bottoms). Research what VR games do for matching collision to the 3D model, and everything related: where you grab, how
you grab, and other stuff."

This is research only. No game code was changed and no tests were run. Every claim about another game or tool cites a
source. Where I could not find a primary source, the text says so; it does not guess.

---

## 0. Where LUMEN is today (read from the code, 11-12)

| What | Today | Source in the repo |
|---|---|---|
| Dynamic prop collider | Offline convex decomposition `<glb>.hulls.json` (V-HACD through Godot, or CoACD through `tools/coacd_models.py`). **Fallback:** ONE `create_convex_shape(clean=true, simplify=true)` hull per mesh. | `scripts/prop.gd` `_build_model` (~l.1127–1195) |
| Coverage | **118 prop .glb files, only 39 have a `.hulls.json`.** The other **79 use the fallback single simplified hull.** That includes **8 armory weapons: shield, shotgun, rifle, laser_rifle, railgun, pistol, laser_pistol, sword.** It also includes sonic_cannon, every city car/flycar/subway model, and the energy cells/docks. Each of these .glb files has exactly one mesh node and no `COL_` proxy, so each one collides as ONE convex lump. A pistol's lump fills the trigger guard and the gap between grip and barrel; a sword's lump is a wedge spanning blade and crossguard. | `find props -name '*.glb'` vs `*.hulls.json` |
| Hand-made proxies | `COL_*` meshes in a model become the hulls; a decomposition, if one exists, supersedes them. | prop.gd |
| Shape fixes already in place | 32-gon prism instead of Jolt cylinders (sustained side-push sinking); `MIN_THICK` 12 mm along the piece's own PCA axes for blades/arrow only; margin 0.002; `_small_margin`; compound-rest reports for flat dishes. | prop.gd l.1196–1240, 1385+ |
| Mass | density × (AABB volume × 0.55 per mesh), or `mass_kg`. **COM = Jolt's automatic COM from the collision pieces** unless `com_local` is set. The pint-glass case: V-HACD envelope COM 0.49 h vs CoACD walls 0.42 h. | prop.gd; known_flakes.txt S5b note |
| Fit metric | `tools/eval_hulls.py` + harness `shapes.py` S5b: coverage (exposed points with a piece within 5 mm) and "fat" (air 8 mm out along the normal lies inside a piece). Known limits: lantern cage 0.333, laser gun 0.270, pint glass (V-HACD) 0.127; the band is 0.10. | eval_hulls.py, shapes.py, known_flakes.txt |
| Grabbing | xr-tools pickables + `grips` dictionaries (`at`, `curl`, optional `axis`/`len`/`edge`/`reach`). A hand within `GRIP_REACH` (10 cm) slides onto the grip. A grip with an axis puts the handle in the fist, oriented from the controller's grip pose. Otherwise "held where you grab it", with finger curl from the object's size. Two-handed: bow, SONIC-1, rifles (`_fore`). | prop.gd l.447–490; armory.gd l.290–325; harness grips.py G1–G4 |
| Sockets | `socket.gd`: magnetic receptacle with plug points (`at`, `dir`), snap within SNAP_R. | socket.gd |
| Static world | city2: boxes/cylinders, plus some `ConcavePolygonShape3D` (backface). Street cars: static concave from a 3 cm voxel-remeshed, 2.4k-tri watertight copy plus box cores (car_fit_unit.gd). building_kit: boxes/cylinders/convex. Tests K1 (roof height) and K2 (wall edge ±1.4 m). | city2.gd, city_cars.gd, building_kit.gd, harness city2_colliders.gd |

**The wine bottle bug is a class bug, not a one-off.** `simplify=true` hands Godot's hull simplifier a budget, and the
simplifier spends it on the silhouette rather than the base, so a round base can collapse to a few points. The same fallback
is shaping 79 models right now, every gun and the sword among them.

---

## 1. How shipped VR physics games match collision to the visual

**The consistent pattern: no shipped game I found uses the render mesh directly as a moving collider. Moving things get
simplified convex pieces or primitives, chosen per object class. Exact triangle meshes are for static world geometry
only. "Match the model" means a fit that is tight where hands and contacts happen, not triangle-exact.**

| Game / engine | What they do | Source |
|---|---|---|
| **Half-Life: Alyx (Source 2)** | Physics props get a `PhysicsHullFile` node in ModelDoc. That is one convex hull around the enabled meshes, or "Hull Per Mesh", which "can give a simpler hull that still closely fits the render mesh". Complex meshes may need a separately built simplified physics mesh. Valve used "invisible arms … for physics detection only". A late-joining developer wrote that grip poses were **hand-authored per interactable, "several hundred" of them**; that is one unofficial account. I found no first-party talk on Alyx's collider tolerances. | [Valve wiki: HLA Physics Prop](https://developer.valvesoftware.com/wiki/Half-Life:_Alyx_Workshop_Tools/Modeling/Physics_Prop) (403 to my fetcher; quoted from the search snippet) · [Road to VR interview, Robin Walker](https://roadtovr.com/valve-half-life-alyx-game-design-interview-robin-walker/) · [janeng.com account](https://www.janeng.com/half-life-alyx/) |
| **s&box (Source 2 lineage)** | "Physics Hull From Render" is "fine for quick tests", but the result "usually isn't the best". **For jointed objects, use capsules, spheres and boxes instead of auto-generated phys models.** | [s&box wiki: PhysicsHullFromRender](https://wiki.facepunch.com/sbox/PhysicsHullFromRender) |
| **Blade & Sorcery (Unity, ThunderRoad SDK)** | Official modding guidance: mesh colliders are "quite expensive", so use primitives (box/capsule/sphere). **Blades: "consider using capsule colliders" — vanilla blades use capsules.** Also: **"very precise colliders can make items pass through each other"**, so primitives can be the better choice. Colliders sit in **ColliderGroups**, which **Damagers** (Slash/Pierce/Blunt) reference. Slash penetration depth runs from the blade's centre to its edge. **Handles are separate from colliders** (an axis with a length; side = which hand). Each handle has HandleLeft/Right poses; you add more grip angles by rotating copies (up to 32). **HolderPoint** sets the holster snap point and orientation. | [BasSDK: Items Best Practices](https://kospy.github.io/BasSDK/Components/Guides/Items/ItemsBestPractises.html) · [Items & Weapons](https://kaidevvy.github.io/BasSDK/Modding/SDK/Unity%20SDK/The-SDK-Items-and-Weapons.html) · [Damager](https://kaidevvy.github.io/BasSDK/Components/ThunderRoad/Damager.html) · [ColliderGroup](https://kaidevvy.github.io/BasSDK/Components/ThunderRoad/ColliderGroup.html) · [Handle](https://kaidevvy.github.io/BasSDK/Components/ThunderRoad/Handle.html) · [Creating Items](https://kospy.github.io/BasSDK/Components/Guides/Items/CreatingItems.html) |
| **Boneworks / Bonelab (Unity)** | "Everything that moves … is moved with physical forces and is simulated." Weight is real: VR needs to "actually make the object heavy". **Grips are dynamic: you can "rotate or slide [your] hand on a grip without releasing"** (a friction model). **Authored hand poses blend roll/pitch/yaw of the object relative to the hand**, so the hold is not rigid. Two-hand steadying. I found **no public doc** of the Marrow SDK grip classes or collider authoring; searches for the grip type names returned nothing. | [UploadVR: Boneworks feels like the first next-gen VR game](https://uploadvr.com/boneworks-feels-like-the-first-next-generation-vr-game/) · [Road to VR: Boneworks reveal](https://roadtovr.com/boneworks-reveal-stress-level-zero-physics-based-interactions-embodiment/) |
| **H3VR** | Melee is configured per object through collider lists (`HighDamageColliders`, stab/lodge colliders) in `FVRPhysicalObject.MeleeParams`. I found no doc of its grab-collider or hull practice. | [H3VR modding API: MeleeParams](https://h3vr-modding.github.io/docs/api/FistVR.FVRPhysicalObject.MeleeParams.html) |
| **Gorilla Tag, Hard Bullet, Saints & Sinners, Into the Radius, Gun Club, The Lab** | **I found no primary source** on their collider pipelines. A fan map-building guide for VR maps says: boxes/spheres preferred, mesh colliders only when necessary, and climbable surfaces as their own collider category. That source is not authoritative. | [fan map guide](https://sites.google.com/view/gorillakillerscorp/cyogf/map-building) |
| **Unity (engine rule)** | Concave mesh colliders: static or kinematic only, and two concave meshes never collide. "Use Mesh colliders for static scene geometry, and **compound colliders for moving GameObjects**." For accuracy in concave regions, use a compound of convex pieces. | [Unity Manual: Mesh colliders](https://docs.unity.com/en-us/engine/current/manual/physics-section/physics-overview/collision-section/collider-shapes/mesh-colliders/introduction) |
| **Unreal (engine rule)** | Artist-authored `UCX_` (convex) / `UBX_` / `USP_` / `UCP_` meshes, or the **Auto Convex** tool (V-HACD): Hull Count (example 32), Max Hull Verts (example 16), Hull Precision (voxels, example 50k–460k). "Larger values … closer to the visible geometry, but more expensive." | [Epic: Auto Convex Collision](https://dev.epicgames.com/documentation/unreal-engine/add-a-collision-hull-to-a-static-mesh-using-the-auto-convex-collision-tool-in-unreal-engine) · [Epic scripting guide](https://docs.unrealengine.com/4.26/ProductionPipelines/ScriptingAndAutomation/HowTos/SettingUpCollisionProperties) · [UE naming summary](https://weisl.github.io/collider_ue_guidline) |
| **Godot (engine rule)** | "Favour primitive shapes for dynamic objects." Concave works only in StaticBodies. Single convex (Quickhull) is "ideal for small objects"; multiple convex (V-HACD) is "more accurate for concave objects at the cost of performance". | [Godot docs: Collision shapes (3D)](https://docs.godotengine.org/en/stable/tutorials/physics/collision_shapes_3d.html) |
| **Jolt (our engine)** | Convex radius: the shape is "shrunken by the convex radius and then inflated again", which rounds edges. **`cDefaultConvexRadius = 0.05f`**. Godot-Jolt computes margin = **Collision Margin Fraction × the shape's smallest AABB axis**, capped by `Shape3D.margin`. MeshShape can be dynamic only under conditions: a dynamic mesh must not collide with other meshes or heightfields, "cannot calculate mass and inertia", and is costly. **Mass/inertia default to shape × density; shapes recentre on their COM.** CCD `LinearCast` is for fast small objects, but **"fast-rotating long objects can still tunnel"** because the cast uses the start-of-step orientation. **Enhanced internal edge removal** only works when edges are shared within ONE shape (put the pieces in one StaticCompoundShape). | [Jolt docs](https://jrouwe.github.io/JoltPhysics/) · [PhysicsSettings.h](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/PhysicsSettings.h) · [Godot: Using Jolt Physics](https://docs.godotengine.org/en/stable/tutorials/physics/using_jolt_physics.html) |

### Per object class: what the industry pattern looks like

- **Small props (cups, bottles, food):** a single hull if the shape is convex. If it is concave (a mug handle, a bowl's dish), use a convex decomposition. Many Unity/Unreal titles use primitives (a capsule for a bottle).
- **Weapons:** **authored primitives or a few hand-placed convex pieces**, not raw auto-decomposition. Blade & Sorcery uses capsules on blades and keeps handles and damage zones as separate, named collider groups. The reason given is gameplay: precise colliders snag and pass through each other.
- **Furniture:** convex decomposition or authored boxes (legs + seat + back).
- **Vehicles:** compound convex for the moving body (Unreal UCX / Auto Convex). Static parked cars can use a static mesh.
- **Buildings/levels:** static triangle mesh colliders are allowed and normal, usually a **simplified** collision mesh or authored boxes rather than the render mesh (Unity/Godot/Unreal docs above). Small decorative details are left out on purpose for performance and feel (Godot docs).

---

## 2. Tools and pipelines

| Tool | Facts | Source |
|---|---|---|
| **CoACD** (Wei et al., SIGGRAPH 2022, MIT) | Cuts the mesh with 3D planes plus a tree search. Its **collision-aware concavity** measures the distance between a shape and its hull "from both the boundary and the interior". It preserves "drawer handles, kettle spouts, inner rings of scissors, toaster slots … with fewer convex components". `threshold` default 0.05 (we use 0.03); `preprocess_resolution` 20–100 default 50; `-am box` approximates with boxes. | [arXiv 2205.02961](https://arxiv.org/abs/2205.02961) · [GitHub SarahWeiii/CoACD](https://github.com/SarahWeiii/CoACD) |
| **V-HACD 4** | Header-only. `m_concavity` is replaced by `m_minimumVolumePercentErrorAllowed`. **`m_shrinkWrap`** projects hull vertices back onto the source mesh. Fill modes: FLOOD / SURFACE_ONLY / RAYCAST; **use RAYCAST for open meshes**, because flood fill makes them hollow. One listing says it is end-of-life, with new development moved to CoACD. **Godot's built-in `create_multiple_convex_collisions` uses V-HACD** (our 32-hull cap). | [V-HACD repo mirror](https://github.com/Evangel63/v-hacd) · [TestVHACD man page](https://mankier.com/1/TestVHACD) · [MathWorks vhacdOptions](https://www.mathworks.com/help/nav/ref/vhacdoptions.html) |
| **Blender Collider Tools** (add-on) | Authoring path: auto-convex (V-HACD) plus primitives inside Blender, with engine naming conventions (UCX_ etc.). Matches our `COL_` proxy route. | [Collider Tools: Auto Convex](https://weisl.github.io/collider-tools_auto_convex/) |
| **Unreal Auto Convex** | V-HACD with Hull Count / Max Hull Verts / Precision. Epic's docs give no fit metric; you judge by eye in the editor. | above |

**Validation: what makes an automatic pipeline trustworthy.** No engine I found ships a fit metric; Unreal, Unity and
Godot all leave it to eyeballing. CoACD's paper scores fit with its own concavity measure (boundary and interior distance).
Our `eval_hulls.py` (exposed-surface coverage and fat along the normal) is already stricter than anything the engines
offer. What it lacks, and what the wine bottle shows:
1. **No contact-patch / base check.** A hull can score well on surface coverage and still have a 3-point base. Rest stability depends on the **support polygon** of the face it rests on.
2. **No per-model gate on all 118 models.** S5b checks a hand-picked set; 79 models were never decomposed at all.
3. **No mm tolerance per class.** "Fat" is a fraction; Zeke's ask is in visible millimetres.
4. **No check that mass/COM come from the visual volume.**

---

## 3. Grabbing

| System | Where you grab | How the hand takes shape | Source |
|---|---|---|---|
| **Blade & Sorcery** | **Handles**: axis plus length, so you can grab anywhere along it, with a side per hand. Telekinesis-only handles exist for big props. | Authored HandleLeft/Right poses, multiplied by rotated copies (up to 32 angles). | BasSDK links above |
| **Boneworks/Bonelab** | Authored grips that you can **slide and rotate along without releasing** (friction). | Authored poses blended by the object's roll/pitch/yaw relative to the hand. | UploadVR above |
| **Half-Life: Alyx** | Gravity-glove pull with select → lock-on → flick-pull. The catch window is "deliberately generous", and objects are steered with small impulses toward the hand. | Per-object authored poses ("several hundred", unofficial account). | [Road to VR: Alyx design details](https://www.roadtovr.com/these-details-make-half-life-alyx-unlike-any-other-vr-game-inside-xr-design/) · [DigiPen: Kerry Davis](https://www.digipen.edu/showcase/news/with-half-life-alyx-digipen-graduate-kerry-davis-helps-redefine-vr-games) · janeng.com |
| **SteamVR Skeleton Poser** (Valve) | Per-object poser component. | Authored pose ScriptableObjects plus a blending editor driven by analog/boolean actions. **It blends poses; it does not solve finger geometry.** | [Valve: Skeleton Poser](https://valvesoftware.github.io/steamvr_unity_plugin/articles/Skeleton-Poser.html) |
| **Hurricane VR** | Multiple posable grab points, chosen by **position and rotation**. Falls back to an "Offset" grab where no point fits. | Authored primary pose plus blendable poses and per-finger curl 0..1. **"Auto Pose" dynamic poser** ("Half-Life: Alyx style dynamic hand pose solver"). | [HVR: Hand Posing](https://cloudwalker2020.github.io/HurricaneVR-Docs/manual/hand_posing.html) · [HVR gitbook](https://cloudwalkingames.gitbook.io/hurricanevr/) |
| **Auto Hand** | Free grab anywhere. | **Procedural pose from a raycast hit on the target surface** (`hand.AutoPose(RaycastHit, Grabbable)`), plus optional custom poses. | [Auto Hand docs: Hand](https://earnest-robot.gitbook.io/auto-hand-docs/auto-hand/hand) · [FAQ](https://earnest-robot.gitbook.io/auto-hand-docs/auto-hand/faq) |
| **Meta Interaction SDK** | Pinch (small, a 0.05 m sphere between the fingertips) vs palm (large; strength starts at 0.9, releases below 0.85, hysteresis). **Three alignments: no alignment / object-to-hand / hand-to-object** (the last for fixed things like ladders). Grab surfaces (cylinder/sphere) define where a pose is valid. Procedural finger poses use iterations; more is more accurate. | [Meta: Grab specs](https://developers.meta.com/vr/design/grab_specs/) · [ISDK hand grab poses](https://developers.meta.com/horizon/documentation/unreal/unreal-isdk-hand-grab-poses/) |
| **godot-xr-tools** (ours) | GrabPointHandLeft/Right/Snap with `hand_pose`, `mode` (primary/secondary/general), `drive_position/angle/aim` for two-handed grabs, `snap_hand`, and `handle` (stops two grabs on one handle). | Pose resources per point. | [xr-tools: Grab Points](https://godotvr.github.io/godot-xr-tools/docs/grab_point) |

**Patterns that matter for "collision matches the model":**
- **Grab volumes are SEPARATE from physics colliders everywhere** (B&S Handles, HVR grab points, xr-tools grab points, ISDK grab surfaces). The collider decides what you hit and how it rests; the grab definition decides where the hand goes. Both should be authored or derived from the same model.
- **Free grab only looks right if the collider hugs the visual.** Auto Hand and the Alyx-style solvers place fingers by raycasting the collider. A fat collider leaves the fingers floating 1–3 cm off the visible surface. That is the same complaint as "touching a car registers a foot early" (city_cars.gd history).
- **Weapons always get authored handles with an axis** (slide along, roll free). LUMEN already has this (`axis`/`len`/`edge` grips), which matches B&S and Boneworks.
- **Hand clipping:** I found no primary source describing how these titles keep hands out of colliders beyond physics hands themselves (Boneworks) and invisible physics arms (Alyx). LUMEN's own physics-hand / ghost-hand approach is already in this family.
- **Climbing big things/buildings:** I found no primary source on Gorilla Tag's surfaces. The general pattern from engine docs and map guides: climbable static geometry uses simple static colliders, so the collider surface IS the climb surface. If it sits 30 cm off the wall, your hand grips air.

---

## 4. Related things worth adopting

1. **Mass / COM / inertia from the VISUAL volume, not the collider pieces.** Jolt computes mass and COM from shape × density and recentres shapes on their COM ([Jolt docs](https://jrouwe.github.io/JoltPhysics/)), so a fat or hollow-missing decomposition moves the COM. That is exactly the pint-glass case (0.49 h vs 0.42 h). `hulls.json` already stores `mesh_volume` (divergence theorem). The same pass can compute the mesh COM and inertia tensor. **Unresolved:** switching the pint glass to CUSTOM COM flipped water test I3 (diving bell) to a sink, and the cause is not understood (known_flakes S5b note). Do not roll custom COM out world-wide until that is explained.
2. **Thin blades and tunnelling.** Jolt LinearCast helps fast small objects, but **fast-rotating long objects can still tunnel** ([Jolt docs](https://jrouwe.github.io/JoltPhysics/)). That is a sword swing. B&S uses capsules on blades. LUMEN's MIN_THICK (PCA, blades only) plus pierce.gd's tip sweep is already the right shape of fix. Keep it, and add a capsule/thin-box authored blade option.
3. **Precise ≠ better for weapons.** B&S warns that very precise colliders make items pass through each other. Target "visibly matches at hand scale" (mm-level on the handle, the blade edge and the base), not every vertex.
4. **Convex radius / margin.** Jolt rounds every hull by its margin (default 0.05 m in raw Jolt; Godot-Jolt uses margin fraction × the smallest axis, capped by `margin`). The 0.002 margin LUMEN uses is right for small props. A test should assert no model ships with a margin > 5 % of its smallest dimension.
5. **Ghost contacts on static compounds.** Enhanced internal edge removal only works when edges are shared inside one shape ([Jolt docs](https://jrouwe.github.io/JoltPhysics/)). Building kits made of many separate box shapes will snag sliding props and feet at the seams. Merge each building's static boxes into one compound where possible.
6. **Sockets/holsters.** B&S HolderPoint (position plus arrow orientation; tip-down for blades) and xr-tools GrabPointSnap. LUMEN's socket.gd plug model already covers this; the gap is that weapons have no holster points of their own.
7. **Per-surface material.** B&S damagers and collider groups name zones (blade, handle, pommel). A LUMEN collider piece could carry a `zone` tag (blade/handle/guard/body) from the authoring step, so impacts, pierce and grab logic can read it.

---

## 5. Recommendation for LUMEN

### 5.1 The pipeline, per object class

| Class | Collider | Grab | Tolerance target (proposal; Zeke to confirm) |
|---|---|---|---|
| **Small round props** (bottles, glasses, cans, candles, vases) | **New "lathe" route:** detect a body of revolution (radial deviation of the visual mesh < 1 mm about its axis). Build a **stack of frustum prisms from the measured radius profile**, using the same 32-gon trick as `cyl_shape`. Base and rim are exact, the pieces are few, and the COM is analytic. Hollow vessels get a wall ring of pieces (CoACD did 56 for the pint). | free grab; curl from radius (exists) | silhouette ≤ 1 mm; **base support polygon ≥ 95 % of the visual base area** |
| **Other small props** | CoACD (threshold per model, stored in a sidecar list). Fallback **`create_convex_shape(true, false)`** — **never `simplify=true`**. | free grab + optional grips | proud-by p95 ≤ 3 mm |
| **Hand weapons** (guns, swords, axes, shield) | **Authored `COL_` proxies, B&S-style**: blade = thin box or capsule (MIN_THICK, CCD), guard, grip, pommel; gun = receiver box(es), barrel, grip slab, **trigger guard as 2–3 small hulls so the hole stays open**. Pieces carry a `zone` tag. Auto CoACD is acceptable only if it passes the same per-model gate. | **mandatory handle grip(s) with axis + len** (exists for most); a foregrip for two-handed guns (exists); a holster/HolderPoint | proud-by p95 ≤ 5 mm on handle/blade; trigger-guard hole open |
| **Furniture** (chairs, tables, shelves) | CoACD, plus the base check on every rest pose (feet must reach the floor) and the 11-12 "no overlapping spawn, no knife-edge rest" guard. | grips on backs/edges where it matters | proud-by p95 ≤ 10 mm; legs present as separate pieces |
| **Dynamic vehicles** (cars you can push/ride, flycars) | CoACD coarse (≤ 32 pieces) plus box cores, like the static cars' `.col.json` remesh but convex. | handles/doors as grips | proud-by p95 ≤ 30 mm |
| **Static parked cars** | As now: watertight 3 cm voxel remesh → concave static (already measured with car_fit_unit). | — | ≤ 30 mm (exists) |
| **Buildings/levels** | Static: authored boxes per module where the module is boxy (as now). A **simplified concave mesh** (voxel remesh + decimate, the cars' method) for anything ornamental or touchable. Merge each building's static shapes into **one compound** (internal-edge removal). | climbable = the static collider itself | touch/climb surfaces ≤ 20 mm; roofs/walls as K1/K2 |

**One rule over all of it:** every model the game loads as a physics object must have an **explicit collider source**:
`hulls.json` (CoACD/V-HACD), `COL_` proxies, a lathe/analytic shape, or a primitive the code names. The silent "one
simplified hull" fallback becomes a **test failure**, not a behaviour.

### 5.2 Guard tests that should FAIL on today's code

All of these can run offline or headless. Most extend `eval_hulls.py` / `shapes.py` and need no headset.

| ID | Guard | Fails today because |
|---|---|---|
| **C1 collider source** | Every `.glb` used by any prop/vis_model has hulls.json, COL_ proxies, or a declared analytic/primitive collider. Lint: no `create_convex_shape(…, true)` anywhere in `scripts/`. | 79/118 models fall back; `create_convex_shape(true, true)` at prop.gd:1160 AND floor_lever.gd:32, fountain.gd:42, training.gd:147, robot_arm.gd:44+70, temple_decor.gd:27 (7 sites; static bases still lose shape) |
| **C2 surface match, every model** | eval_hulls for ALL models (not a hand-picked set): exposed coverage ≥ 0.95 and **proud-by p95 ≤ class tolerance in mm** (table above). | the sword/pistol single hulls fill the guard/trigger cavities; lantern/laser gun are already known over |
| **C3 base contact patch** | For each stable resting face of the visual mesh (its convex hull's faces whose COM projection lands inside them), the collider's support polygon on that plane covers ≥ 95 % of the visual's, and its inradius is within 2 mm. | wine bottle base = 3 points |
| **C4 rest pose matches visual** | Dropped from 5 cm onto a flat floor, a prop settles within 2° and 2 mm of the pose its visual mesh predicts (headless Jolt; the propsettle.py style). | knife-edge bases rock or tip |
| **C5 mass/COM from the visual** | Collider-derived COM vs visual-mesh COM within 3 % of the object's height; mass within ±25 % of density × visual volume. | COM comes from pieces (pint glass 0.49 h vs 0.42 h); volume = AABB × 0.55 |
| **C6 handle coverage** | Every weapon: ≥ 1 axis grip whose `at`…`len` segment lies **inside the visual handle** (within 1 cm of the mesh's medial line). A closed-fist hand capsule at that grip does not penetrate the collider by > 5 mm. Two-handed weapons have a foregrip. | untested today: grips are placed by AABB fractions (`ext.y * 0.42`) |
| **C7 thin & fast** | Any piece thinner than 12 mm on a model with a damager/fast flag is in THIN_GUARD or has a CCD + tip-sweep test. | not enumerated today |
| **C8 static world match** | Extend K1/K2: for every building module and touchable static mesh, rays from 0.5 m out hit the collider within 20 mm of the render surface (render-vs-collider depth). | K2 allows ±1.4 m |
| **C9 budget** | Per prop ≤ 64 hulls (weapons ≤ 24), per scene a total cap; the decomposition time is recorded. | prevents the fix from turning into a perf regression |

C1–C3 and C5 are the cheapest and catch the wine-bottle **class** (Zeke 11-12: "every root cause gets a guard").

### 5.3 Migration order

1. **Kill `simplify=true`** in the fallback and at the 6 other sites (floor_lever, fountain, training, robot_arm ×2, temple_decor) (→ `create_convex_shape(true, false)`). One line; it fixes the knife-edge base class at once. Measure the hull vertex counts (Jolt is fine with ~100s of points per hull).
2. **Land C1 + C3 + C2 as baselines** (expected red), with the failing list as the work queue.
3. **Weapons:** author `COL_` proxies for the 8 undecomposed armory weapons and sonic_cannon (Blender, B&S-style pieces plus zone tags). Run C2/C6/C7.
4. **Lathe route** for bottles/glasses/cans/candles. It may also retire the pint-glass COM issue, since the COM becomes analytic; the I3 diving-bell flip must be understood first.
5. **Mass/COM/inertia from the visual mesh** (C5), behind an A/B env like `IRIS_MIN_THICK_MODE`, because of the I3 surprise.
6. **Furniture** + the 11-12 spawn-overlap guard.
7. **Grab:** surface-snapped free grab (the hand moves to the collider surface along the palm normal, fingers raycast-curled as in Auto Hand/HVR Auto Pose). Only worth doing after C2 passes, because it exposes every fat collider.
8. **Dynamic vehicles**, then the **static world** (C8, merged compounds).

### 5.4 Open questions for Zeke

1. **Tolerances:** are the proposed mm targets right (bottles ≤ 1 mm silhouette, weapons ≤ 5 mm, furniture ≤ 10 mm, vehicles/buildings ≤ 20–30 mm)? Or does "exact" mean tighter for some class?
2. **Weapons:** exact shape, or B&S-style "feels right" primitives? Blade & Sorcery deliberately uses capsules because precise colliders snag and pass through each other.
3. **Free grab:** should the hand **snap onto the visible surface** wherever you grab (fingers wrapping the real shape), or stay where your controller is (today)?
4. **Climbing:** should every building surface be climbable (then C8 must be tight everywhere), or only marked surfaces?
5. **Authoring:** OK for me to author `COL_` proxies in Blender for the weapons, or should everything be automatic (CoACD) and only checked?
6. **Perf budget** for the extra pieces: the C9 caps are guesses until a vantage capture is measured.

### Notes on my inbox
The inbox (`state/helpers/colresearch_1112.inbox`) was empty when checked at start and between steps; nothing to act on.
