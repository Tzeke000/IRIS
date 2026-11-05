# Movement game (tag in an obstacle course) + physics gravity gun — research + design (2026-11-04)

**Summary (5 lines)**
1. **Tag:** the game already has most of what tag needs: real hand contact between players (remote hand bodies, touch grades), climbing + mantle, arm-swing running, webs, server-decided teleports, and a build system that saves. A tag round is a server-side "who is it" plus a tag check.
2. **Course:** start with what exists (the City's Skyline ring course, or the temple), then a **course built with the build system** that friends can edit together. A Gorilla-Tag-style "push off with your hands" mode is possible (MIT Unity reference), but desktop players can't do it.
3. **Gravity gun:** Half-Life 2's physcannon (pull + hold at a distance, punt, a max mass). Here it's a held prop like the laser gun. The hold is a **force-capped PD spring** on the target, so heavy things sag and swing instead of floating like a cursor.
4. Gravity-gun lag is solved the same way grabs are solved: the gun's user **claims ownership** of the target prop and simulates it locally. Using it on *players* goes through the shove-impulse path from the full-body doc (capped, the victim's client applies it).
5. Smallest first steps: **tag** = a server "it" token + a hand-touch tag check in the temple; **gravity gun** = an offline held prop that pulls and holds one prop with the existing hold law.

---

## 1. Search: what exists
- **Gorilla Tag's locomotion is open source:** Another Axiom's `GorillaLocomotion` (Unity, **MIT**: `Player.cs`, `Surface.cs`). There's a VRChat Udon port that adapts arm length to the avatar. **No Godot port found.** "Godot XR Autohands" (MIT) does gesture locomotion, which is a different thing.
- **Gravity gun:** **no Godot 4 implementation found.** References:
  - Valve Developer Wiki: primary fire = **punt**, secondary = **pull and grab**, primary while holding = throw; `physcannon_maxmass 250`.
  - Valve's original `weapon_physcannon.cpp` in the public Source SDK 2013 (read for ideas; Valve's SDK licence, don't copy).
  - A Godot forum thread on HL2/Amnesia-style object handling (Godot 4.6.1): a PD-like controller run inside the held body's `_integrate_forces`, with a Jolt 6DOF-spring variant, and the warning that you have to **detect and prevent the extreme cases** (huge forces).
  - Not useful: a JS demo on Matter.js, a Unity asset, a Minecraft mod.
- Our own notes: `research/bs_mods_rumble_bonelab_2026-10-26.md` §6 already plans **telekinesis and stasis** (B&S ForceStasis). The gravity gun is the same physics in a hand-held tool.

## 2. Movement game: tag in an obstacle course

### 2.1 Modes (pick one to start)
| Mode | Rules | Why |
|---|---|---|
| **Classic tag** | One player is "it"; a touch passes it on; whoever is it when time runs out loses. | Simplest to build first. |
| **Infection** (Gorilla Tag's main mode) | Tagged players join the "it" team; the last free player wins. | Scales well from 3 to 12 players. |
| **Freeze tag** | Tagged players freeze in place; a free player's touch unfreezes them. | Co-op moments. |

### 2.2 How it reuses what exists
- **The tag itself is physical:** in VR, a tag = **my hand body touching your body or hand proxy** (the same contacts `rig_xr.gd` already grades for high-fives). Desktop: the E-grab key within 1.0 m in front.
- The **server decides**: it checks its own latest poses (hand within ~0.4 m of the target body, plus latency slack). The cooldown is 3 s before you can be tagged back.
- **Moving:** the arm-swing run, climbing holds + any-ledge mantle, jump, and webs if the mode allows them. Powers are granted or removed per round through the existing powers grant system.
- **Rounds:** start on a pad, teleport to start spots (`teleport_go`), announce on the PA, show "it" as a glow on the avatar. Iris's mood glow shows this already works on bodies.
- **Course options:**
  1. **Existing spaces now:** the temple courtyard (columns, dais, benches) or the City's Skyline ring course with the avenue and sky bridges.
  2. **A built course:** Build-system pieces (planks, beams, boxes, FIX in place) plus climbing holds, saved by the server in `builds.json`. Friends can build and change it together.
  3. **A hand-locomotion mode** (Gorilla Tag): move only by pushing off surfaces with your hands. That means porting GorillaLocomotion's ideas (MIT) onto our physics hands (full-body doc stage 1–2 helps). It's **VR-only**, so desktop players would need a separate role (for example "it" with a speed handicap) or be left out.

### 2.3 Fairness and comfort
VR and desktop speeds are already matched (VR walk 3 m/s, desktop run 4.5 m/s). Climbing is the big gap: desktop needs a "climb" key that works on the same holds at the same speed. Fast chases + webs are a lot for VR comfort: keep vignette and snap-turn on, and make webs a mode, not the default.

### 2.4 Cost
The server holds one "it" set and does a distance check per tag claim. No new per-tick traffic, because poses already flow.

## 3. Physics gravity gun

### 3.1 Behaviour (HL2 adapted to "physics only")
- **Pull** (hold the trigger, pointing at a prop within ~8 m): a force on the prop toward the muzzle, **capped by the gun's max force**. Heavy things come slowly or not at all. HL2 has a 250 kg ceiling. Here the cap is a **force**, not a mass gate, so a car just doesn't move.
- **Hold:** the prop floats ~1 m in front of the muzzle on a **Stable-PD spring** (the same law the full-body doc proposes for hands). It sags under its own weight, swings, bumps into things, and slips out if yanked too hard (break force). Not a kinematic cursor.
- **Punt** (the other button): an impulse along the aim, capped. The **recoil goes into the holder's hand** through the hold force, so a weak player's arm kicks back (strength-capped hold, already how `prop.gd` holds work).
- **Throw while holding:** release with the gun's velocity + a punt.

### 3.2 How it reuses what exists
- **The gun:** a prop subclass like `laser_gun.gd` (rail plug, grips, desktop left-click). The beam visuals come from the laser and web strands.
- **No-lag hold:** the gun's user **claims the target prop** exactly like a grab (`props.gd` ownership). The server's `request_own` needs a "gravity gun" range exception (reach ~8 m instead of hand reach). Then the user simulates the prop locally and streams it as usual.
- **Force math:** `force_fields.gd` (fans, magnets) already applies forces to props each tick. The pull is a cone-shaped force field on the end of a gun.
- **Players as targets:** off by default. If on: the server relays a **capped impulse** to the target player's own client (the shove path from the full-body doc). Never a hold on a player.
- **With other games:** in spleef it could pull tiles out of the floor (same physics as the web yank). In tag it could pull you toward a wall like a grapple.

### 3.3 Risks
- **Extreme forces:** cap force, cap the held prop's speed, and drop the hold when the error grows past a limit (the forum thread's warning; the 10-26 Spider-Man car incident is the same lesson).
- **Griefing:** pulling props out of other people's hands. The current ownership rule (`request_own` refuses while someone holds the prop) already blocks that. Keep it.
- **Tunnelling:** fast punted props: `continuous_cd` is already on for props.

## 4. Staged plan
| Step | What | Effort |
|---|---|---|
| Tag 1 (smallest) | Server "it" token + VR hand-touch / desktop E tag check + PA + glow, in the temple. Bots play it in `vr.sh mp`. | S–M |
| Tag 2 | Infection + freeze modes, round pads, timer, powers-by-mode. | M |
| Tag 3 | A built obstacle course (Build system) + desktop climb parity. | M |
| Tag 4 (optional) | Gorilla-style hand-locomotion mode (after physics hands). | M–L |
| Gun 1 (smallest) | Offline gravity-gun prop: pull + PD hold + punt on one prop. | S–M |
| Gun 2 | Networked: claim-at-range on the server, capped forces, recoil into the hand. | M |
| Gun 3 | Optional player punt (capped impulse path), spleef/tag interplay. | S |

## 5. Questions for Zeke
1. **Which tag first:** classic, infection (Gorilla Tag), or freeze?
2. **Course:** play in what exists (temple / City ring course), or build a dedicated course together with the build tools?
3. **Webs and other powers in tag:** allowed, a special mode, or never?
4. **A hand-push (Gorilla Tag) locomotion mode** is VR-only. Is that OK, or must every mode work for desktop players too?
5. **Gravity gun on players:** never, admins only, or allowed (capped)?
6. **Gravity gun strength:** a hard ceiling like HL2 (nothing over X kg), or purely force-capped (anything, but slowly)?
7. **Where does the gun live:** a pickup at the shooting range, a power in the menu, or a minigame-only tool?

## Sources
- GorillaLocomotion (MIT): https://github.com/Another-Axiom/GorillaLocomotion · VRChat port: https://booth.pm/en/items/3941902
- Valve wiki: https://developer.valvesoftware.com/wiki/Weapon_physcannon · https://developer.valvesoftware.com/wiki/Gravity_Gun
- Godot forum, HL2-style manipulation: https://forum.godotengine.org/t/hl2-amnesia-style-physics-object-manipulation/133886
