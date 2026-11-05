# SPLEEF minigame for VR Iris / LUMEN — research + design (2026-11-04)

**Summary (5 lines)**
1. Spleef, as in classic Minecraft: everyone stands on a breakable floor and breaks it out from under the others. Fall through and you're out; **last one standing wins**. Classic rules: no placing blocks, no hitting players (knockback is a variant).
2. The floor should be a **grid of tiles whose state the server owns** (whole / cracked / gone). *How* a tile breaks is real physics: the impulse of a shovel strike, a hand slam, a thrown snowball, **fire melting ice** (built on `fire.gd`), or a **web yank**. The falling debris is real rigid fragments simulated on each client (`shards.gd` already does this).
3. VR vs desktop stays fair if breaking is **melee range only**, every tool has the **same per-hit cooldown and damage cap** for both, powers are off inside the arena by default, and both rigs keep today's matched speeds.
4. Server cost is tiny: tiles are static colliders that get removed, a break is a small event, a round reset is one bitmask. Debris never goes over the network. 2–8 players per round fits a 16 × 16 m arena.
5. Smallest first step: an **offline prototype tile floor** where a hand slam or thrown prop breaks tiles through the existing Impacts system, plus a fall line, tested with `vr.sh test` bots. After that, a server-run round (lobby pad, countdown, elimination, reset).

---

## 1. Search: what exists
- **No Godot or open-source spleef implementation turned up.** The only "spleef" code results were a paid Fortnite UEFN Verse generator, which shows the usual feature set: tiles break when hit or stepped on, falling through eliminates you, last player standing wins.
- **Minecraft rules** (Minecraft Wiki and server wikis): started on a Classic multiplayer server in 2009. The floor is snow or wool; players use a **shovel** (or shears for wool) because it breaks blocks fast. Breaking snow with a shovel drops **snowballs**, which players throw. Rules vary by server: no placing, usually no hitting. Variants include **splegg** (an egg cannon breaks blocks at range), **bow spleef**, **TNT run** (blocks fall where you walk), multi-layer arenas, and no-jump.
- **Runtime fracture plugins for Godot 4** (for completeness): VoronoiShatter (MIT, Godot Asset Library), ShatterKit (paid, itch), Destruct3D (C#), Destructibles-CSharp. None mention Jolt. **Not needed:** our own `shards.gd` already makes real convex-hull fragments that split the volume and mass. True runtime fracture of a big slab would also be impossible to keep identical across peers.

## 2. Rules (proposed defaults; every one is Zeke's call)
- 2–8 players. Everyone spawns on the floor at evenly spaced points, facing the centre.
- **No placing, no hitting** (classic). Knockback is a toggle (needs the player-shove work from the full-body doc).
- You're out when your body drops **2 m below the floor** (or you stand over a hole for >0.5 s, which catches a floating cheat). Eliminated players are teleported to the spectator ring.
- Last one standing wins. If two fall within 0.3 s of each other, it's a draw.
- **Sudden death** after 90 s: the outer ring of tiles starts crumbling inward, one ring every 10 s, so rounds can't stall.

## 3. The floor: how breaking works with real physics

### 3.1 Options considered
| Option | Physics truth | Sync | Cost | Verdict |
|---|---|---|---|---|
| A. **Tile grid, server-owned state, physical damage + local physical debris** | Damage comes from real impulses; debris is real rigid bodies | One bit per tile, tiny events | Low | **Recommended** |
| B. Runtime Voronoi fracture of one slab | Highest | Non-deterministic across peers; you'd have to stream every fragment | High | No |
| C. Every tile a frozen RigidBody that unfreezes when broken | High | Falling tiles that hit players would need ownership + streaming | 256–1600 bodies, plus streaming | Only as the cosmetic half of A |

### 3.2 Recommended design (A)
- **Grid:** 16 × 16 tiles at 1 m (256 tiles), 0.25 m thick. Hexagons are prettier (Fall Guys' "Hex-a-gone" look) but harder to aim; start square.
- **Colliders:** test both, because I didn't measure which wins: (i) one StaticBody3D per tile, where breaking disables its layer/shape; (ii) chunks of 4 × 4 tiles per body. Note: in Godot, changing the shapes of a single big body rebuilds its compound shape, so one body with 256 shapes might hitch on every break. Measure before choosing.
- **Visuals:** one MultiMesh for the whole floor (one draw call). A cracked tile swaps to a cracked instance colour/texture.
- **Tile health = material physics.** Each tile has a material from `data/materials.json` (snow, ice, glass, wood…) and an HP that drops on hits, judged like `breaks.gd`: `v_eq = Δv·√((m_self+m_other)/m_other)·hardness^(5/6)` against a limit. A tile becomes *cracked* past 70 % (`CRACK_AT`, same as glass) and *gone* at 100 %. So a good shovel stab breaks it, a tap only cracks it, and a thrown mug might crack it.
- **When a tile goes:** the server broadcasts it. Every client hides it and spawns a seeded burst of real fragments locally with `shards.gd` (same pattern as breakables: same burst everywhere, no traffic, a budget of 40 pieces). The fragments fall into the pit and fade.

### 3.3 What breaks the floor (the tools; all physics)
| Tool | How it's physical | Reuses | Notes |
|---|---|---|---|
| **Shovel** (the classic) | A held prop with a blade "damager". Tile damage ∝ strike impulse at the blade edge. Stab down / scoop. | `prop.gd` grips, blade/penetration code (`pierce.gd`, `blade_sfx.gd`), `impacts.gd` | Default tool. Desktop: click = a fixed-impulse swing (see §4). |
| **Hands** (slam/punch) | Hand impact impulse, capped by `Strength` | the collision hand + Impacts | Weaker than a shovel. Good for "everyone starts empty-handed". |
| **Snowballs** (from broken snow tiles, as in Minecraft) | Real thrown rigid bodies. Impact Δv cracks tiles at range. | props, `throw_estimator.gd` | Gives a ranged option that still needs a real throw. |
| **Fire on an ice floor** | Heat flux from `fire.gd` melts ice tiles over a few seconds (HP drains with heat). Water puddles under them. | `fire.gd`, `fire_fx.gd`, `puddles.gd` | The first **elemental** spleef, in line with "fire powers first". Needs Zeke's powers doc first. |
| **Web yank** | A web strand stuck to a tile; pulling past the tile's anchor strength tears it out and flings it. | `web_sling.gd` (strands already have tension + breaking strength) | Spider-Man spleef. Pull force is real, so it's fair. |

### 3.4 Who judges a hit (latency)
- The striking player's client **owns its held shovel** (ownership claim on grab, as now), so it sees the contact first.
- It sends `tile_hit(tile, impulse, time)`. The server checks: is that player within reach of the tile (per its own pose of them), is the per-player rate under the cap, and is the impulse plausible. Then it applies the damage and broadcasts.
- The breaker sees a crack **immediately** (local prediction, visual only). The tile only disappears when the server says so, about one round trip later. That keeps one truth for whose feet are on what.

## 4. VR vs desktop fairness
- **Reach:** breaking only within ~1.5 m for everyone (melee). Ranged breaking only through physical projectiles (snowballs), which desktop players throw with a fixed release speed equal to a typical VR throw.
- **Damage and rate:** one tile per player per **0.25 s** at most, and each hit's damage is **capped**. A frantic VR flail and a desktop click-spam therefore both top out at the same rate.
- **Desktop shovel:** a click plays a strike with a fixed impulse equal to the median VR strike (calibrate it with Iris's bot and Zeke's headset).
- **Movement:** keep today's matched speeds (VR walk 3 m/s, desktop run 4.5 m/s, same jump). **Powers off in the arena** unless the round's mode turns one on (the server already grants powers; it can switch them off by zone).
- **VR comfort:** a short fall (pit 2–3 m onto a soft foam/slime floor), then a fade to the spectator ring. Never a long free-fall in VR.

## 5. Round flow (server-authoritative)
1. **Lobby pad** at the arena: step on to join (2–8). A host can start, or it auto-starts 10 s after the 2nd player joins.
2. **Countdown 3-2-1** spoken through the PA; the floor resets (one bitmask message).
3. **Grace 3 s:** no breaking.
4. **Play.** Eliminations are announced; spectators watch from the ring.
5. **Sudden death** at 90 s (ring crumble).
6. **Winner** announced, confetti, score saved in `server.cfg`-style stats. The next round can start from the pad.

The server already has the pieces: `request_action` / `do_action` with allow-lists and rates, server-decided teleports (`teleport_go`), the PA system, the build-save mechanism.

## 6. How many players and what it costs
- **Players:** 2–8 per round (16 × 16 m ≈ 32 m² each at 8). The world allows 16 (`MAX_PEERS`). A 24 × 24 m floor would fit 12+.
- **Server CPU:** 256 static colliders plus the players' existing proxies. Breaking a tile = removing a collider. Debris never exists on the server. Negligible next to server-owned props and city movers.
- **Bandwidth:** a break event is a few bytes; a reset is 32 bytes for 256 tiles. Debris is local and seeded, so zero.
- **Clients:** at most 40 live fragments (the existing budget) plus one MultiMesh. Cheap.

## 7. Where it lives
- **Recommended:** a **sky arena** reached from the Teleport HUB (a new "Spleef" pad next to Lumen / New York / …). It floats high above the temple, so the pit can be as deep as we like with nothing to dig, and spectators stand on a ring around it.
- **Alternative:** a Lumen rooftop (Glasshouse Heights has 8 rooftop pads), with the city view and real height. That feels great, but there are more things to fall onto.
- **Not** the temple courtyard: the props, stage and fountain would all have to avoid it.

## 8. Staged plan
| Stage | What | Effort |
|---|---|---|
| **1. Offline prototype** (smallest first step) | Tile grid + MultiMesh, tile HP from materials, hand slams and thrown props break tiles via Impacts, `shards.gd` debris, a fall line. Bot test in `vr.sh test`; the collider-layout A/B (§3.2). | S–M |
| **2. Networked round** | Server tile state + `tile_hit` validation, lobby pad, countdown, elimination, spectator ring, reset, PA announcements. Bots (and Iris) play rounds in `vr.sh mp`. | M |
| **3. Shovel + snowballs + desktop parity** | Shovel prop with blade damager, snowballs from snow tiles, desktop strike/throw calibration, rate/damage caps. | M |
| **4. Element and power variants** | Ice floor + fire (after Zeke's powers doc); web-yank spleef; powers-by-mode. | M |
| **5. Variants** | 2–3 stacked layers (classic multi-layer spleef), TNT-run (tiles crack where you stand), splegg-style ranged gun, knockback on (needs the shove work in the full-body doc). | M each |

## 9. Questions for Zeke
1. **What breaks the floor at launch?** Shovel only (classic), hands too, snowballs too? Fire/web come later as variants?
2. **Hitting / knockback allowed,** or classic "no touching"?
3. **Floor material:** snow (classic), ice (fits fire powers), glass (dramatic shatter), or a choice per round?
4. **Where:** a sky arena off the Teleport HUB, a Lumen rooftop, or somewhere else?
5. **Players per round** (2–8 OK?) and does the arena grow when more join?
6. **Powers in the arena:** off by default, with special modes that turn one on?
7. **Does Iris play?** Her bot can join as a player, but her body is a floating eye. Does she get legs and gravity in the arena, or is she the referee/announcer?
8. **Scores:** a persistent win count / leaderboard, or just per session?
9. **Fall feel:** a short drop + fade to spectating (comfortable), or a real fall into a pit you can see (more fun, less comfortable)?

## Sources
- Minecraft Wiki mirror, Spleef: https://wiki.ronlab.site/content/minecraftwiki_en_all_maxi_2025-11/Spleef · Snow block: https://minecraft.fandom.com/wiki/Snow_Block
- Spleef variants plugin (classic/splegg/TNT run/bow): https://modrinth.com/mod/arenaspleef
- UEFN spleef generator (feature set only): https://uefncentral.com/es/verse-generator/spleef
- Godot fracture plugins: https://indiedevwilliam.itch.io/shatterkit · VoronoiShatter https://godotengine.org/asset-library/asset/edit/16582 · Destruct3D https://gh.r-mj.com/OffByTwoDev/Destruct3D
