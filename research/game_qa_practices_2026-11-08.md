# How game studios test (esp. VR + multiplayer), and what LUMEN should change — 2026-11-08

Zeke asked on 11-08 how game developers test, and how that compares with what we do. This is read-only research by helper
`qaresearch_1108`. Companions: `research/regression_prevention_2026-11-07.md` covers merges breaking each other, Google
TAP, flaky tests, merge queues and the Sea of Thieves numbers. `research/ops_guards_2026-11-08.md` covers the machinery
around the tests. To avoid repeating those, this note covers studio practice, VR, multiplayer, Godot, and **how to cut
the 4 h regress and get more out of Zeke's headset time**.

**Source quality:** where I only had a search excerpt or a secondary write-up, the claim is marked *(excerpt)*. Two
fetches timed out (Riot's tech blog, node/33 and node/117), so the Riot figures are *(excerpt)*.

---

## 0. Our setup in numbers (measured today, not remembered)

- **Suites:** 92 offline, 87 multiplayer, 55 XR scripts and 74 GDScript unit files in the harness; a regress counts
  about 198 suites.
- **Full regress wall time:** the 11-05 host run took 141 suites in 7,683 s (2.1 h). The 11-06 lane-2 run took 187
  suites in 13,907 s (**3.9 h**, about 74 s per suite on average). The regress runs **serially in one lane**.
- **Spare capacity:** there are 4 lanes on the box (36 cores, 58 GB RAM, one V100).
- **Gap: there is no per-suite duration anywhere.** `run/history.tsv` records date, suite, pass, fail and the failing
  checks, but not seconds. Every speed-up below that splits or reorders suites needs that number first.

---

## 1. How studios actually test — what each one publishes

| Studio / game | What they do | Source |
|---|---|---|
| **Rare — Sea of Thieves** | About 10,000 tests, about 90 % unit ("run without creating or ticking world"). Also actor tests, integration tests (which can run over the network), behaviour-tree tests, and **screenshot tests compared against "last known good"**. A **core set runs before every check-in**, and longer tests run periodically on TeamCity. A change "has to run against the current build and pass all the tests". | [GDC 2019 Masella](https://gdcvault.com/play/1026366/Automated-Testing-of-Gameplay-Features), [slide mirror](https://sambuz.com/doc/automated-testing-for-multiplayer-game-ai-automated-ppt-presentation-901544) *(excerpt)*, [modl.ai summary](https://modl.ai/?p=3381) *(secondary)*, [GDC 2018 Soltani, Adopting Continuous Delivery](https://gdcvault.com/play/1025028/Adopting-Continuous) |
| Rare — the payoff | Weekly builds with a bug count "always under a hundred", so Microsoft put the game on **Adaptive Certification**, a trust-based fast path, instead of full certification on every update. The trust was earned by the automation. | [Stevivor interview](https://stevivor.com/features/interviews/sea-of-thieves-rare-can-bypass-microsofts-certification-system-at-release), [MSPoweruser](https://mspoweruser.com/microsoft-confident-sea-thieves-allowing-rare-bypass-traditional-certification-process-upon-release) *(excerpt)* |
| **Riot — League** | A test-automation team builds tools so other teams can write tests. About **100,000 test cases a day**. Gameplay-*feel* changes were "pretty much out of the question" for automation. | [Riot tech blog: Automated Testing for LoL](https://technology.riotgames.com/node/33), [Bug Blog](https://technology.riotgames.com/node/94) *(excerpt)* |
| **Riot — VALORANT** | Testing each service alone wasn't enough, so they built a **load harness**: a simulated player plus a scenario plus a player pool, skipping rendering and game logic. It ran **2 million simulated players** against a test shard. A separate game-server load test covered the simulation itself. | [Scalability and Load Testing for VALORANT](https://technology.riotgames.com/node/117) *(excerpt)* |
| **Ubisoft Reflections — The Division** | "Client bots" mimic human input and report issues. Uses: automated mission playthroughs with reports, **follow bots** so one tester can test multiplayer missions solo, and **street wandering to gather performance data**. Server-side bots also run natively on the server. | [80.lv GDC 2019 preview](https://80.lv/articles/gdc-using-ai-controlled-players-to-test-the-division/), [GDC news](https://gdconf.com/news/see-how-ubisoft-tested-division-automated-players-gdc-2019) *(excerpt)* |
| Ubisoft La Forge | Commit Assistant flags likely-buggy commits before they land. It was trained on about 10 years of Ubisoft code. | [Slashdot/Wired](https://games.slashdot.org/story/18/03/05/202226/), [80.lv](https://80.lv/articles/how-ubisoft-uses-ai-machine-learning-to-build-games/) *(excerpt)* |
| **EA SEED — Battlefield V** | 601 features would take about 0.5 M manual test hours (about 300 work-years). Scripted bots break whenever the feature changes, so SEED researches RL and curiosity-driven agents to **raise state coverage** and find exploits. It is published as research, and I found no production numbers. | [SEED ML research](https://www.ea.com/seed/news/seed-ml-research-aaa-game-testing), [CoG 2021 curiosity](https://www.ea.com/seed/news/cog2021-curiosity-driven-rl-agents), [arXiv 2103.13798](https://arxiv.org/pdf/2103.13798.pdf) |
| **Activision — Call of Duty** | CI that tracks **performance and memory** and handles intermittent failures, running automated tests across **hundreds of PCs**. Later tooling ("Compass") validates code, maps and release builds, and runs perf across modes, maps and hardware. | [GDC 2018 van Valburg](https://gdcvault.com/play/1025324/Automated-Testing-and-Profiling-for), [Game Developer summary](https://gamedeveloper.com/production/video-how-i-call-of-duty-i-handles-automated-testing-and-profiling), [Activision research](https://research.activision.com/publications/archives/automated-testing-in-call-of-duty) *(excerpt)* |
| **Bungie — Destiny 2** | QA splits the game into **modular "building blocks"** so the test scope of a frequent release is predictable. SDETs "improve automated tests that run prior to check-in". | [GDC 2023 Rios, Testing with Building Blocks](https://www.gdcvault.com/play/1029336/Testing-with-Building-Blocks-A), [job posting](https://hitmarker.net/jobs/bungie-destiny-qa-engineer-sdet-728553) *(excerpt)* |
| **Blizzard — Overwatch** | "Playtesting first": the data pipeline, build system and server stack were shaped around daily team playtests. | [GDC 2018 Hamilton](https://gdcvault.com/play/1025012/Tools-Tutorial-Day-Playtesting-Overwatch) *(excerpt)* |
| **Valve** | Playtesting means **fun, "not bug testing"**. "Game designs are hypotheses, playtests are experiments." Methods: direct observation ("what people do, not what they say"), think-aloud (warned that it distorts play), then a survey, an individual Q&A and a group Q&A, plus stats and design experiments. Half-Life: Alyx was built and playtested **one room at a time**, and Portal on a weekly Friday-test / Monday-review loop. | [GDC 2009 Ambinder, Valve's Approach to Playtesting](https://cdn.fastly.steamstatic.com/apps/valve/2009/GDC2009_ValvesApproachToPlaytesting.pdf) (read in full), [Road to VR, Walker on Alyx](https://roadtovr.com/valve-half-life-alyx-game-design-interview-robin-walker/), [GMTK](https://gmtk.substack.com/p/valves-secret-weapon) *(excerpt)* |

**The common shape, by studio size:**
- **AAA:** a fast pre-check-in core set, then periodic or nightly long runs, then perf and memory tracking per build on a
  device farm, then bots for coverage, soak and load, then a big manual QA team (often outsourced, e.g. Keywords), then
  external playtests, then platform certification.
- **Indie / small VR:** mostly manual QA plus playtests, a CI build, and maybe a smoke test. Owlchemy, Lucky VR and the
  Keywords testers below are the published VR examples, and their automation is far thinner than ours.
- Vendor guides converge on the same tiering: a **~60 s smoke on every commit** (launch, load each critical scene,
  basic input, no crash for N s; "a failed smoke test should stop the line"), and **nightly full regression + perf +
  soak** ([Bugnet: smoke tests](https://bugnet.io/blog/how-to-build-automated-smoke-tests-for-game-builds),
  [Ixie](https://www.ixiegaming.com/blog/automated-game-testing-that-delivers-bots-toolchains-and-ci-cd/)). These are
  vendor posts, so treat the durations as examples, not standards.

## 2. VR specifically: what can and can't be automated

**Can be automated, and the tools exist:**
- **Simulated headsets on a real OpenXR runtime.** Monado's conformance was certified *on a simulated device*, and its
  `XR_MND_headless` extension gives an XrSession with no compositor
  ([Collabora](https://collabora.com/news-and-blog/news-and-events/monado-2100-officially-conformant-openxr-implementation.html),
  [monado.freedesktop.org](https://monado.freedesktop.org/), [MR !93](https://gitlab.freedesktop.org/monado/monado/-/merge_requests/93)).
  This is what `vr.sh xr` does.
- **Record a real session once, replay it as a test.** Meta XR Simulator **session capture** records the inputs given to
  an OpenXR app into a VRS file and replays them in CI with no headset ("no-code automated tests": "you record real inputs
  once, then replay them"). The replay output is compared with the baseline, and the docs warn to give the CI job a
  timeout. ([XR Simulator automated testing](https://developers.meta.com/horizon/documentation/native/xrsim-native-automated-testing/),
  [session capture](https://developers.meta.com/vr/documentation/native/xrsim-session-capture/).) Meta's older
  **AutoDriver** did the same on device, recording headset, controller and hand tracking, but it is VrApi-only and has
  **no OpenXR support** ([AutoDriver](https://developers.meta.com/horizon/documentation/unity/ts-autodriver/)). I found
  **no generic open OpenXR record/replay layer**. For a Godot game the recorder belongs in our own rig.
- **Agent-driven testing.** Meta "XR Operator" (experimental, SDK v205) lets MCP agents drive an app in the simulator,
  take screenshots and report pass/fail ([Road to VR](https://www.roadtovr.com/meta-ai-vr-dev-tool-test-quest-games/)
  *(excerpt)*). That is the same idea as Iris driving bots over the control socket.
- **Unity's XR Device / Interaction Simulator** drives XR input from keyboard or scripts
  ([Unity docs](https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@3.4/manual/xr-interaction-simulator-overview.html)).
  It is an iteration aid, and Unity has no official guide to using it in automated tests *(excerpt)*.
- **Perf floors as hard gates.** Meta's store check VRC.Quest.Performance.1 sets a **60 fps floor** at ≥ 72 Hz, with
  loading screens and fades exempt ([Meta VRC](https://developers.meta.com/horizon/resources/vrc-quest-performance-1),
  [common VRC failures](https://developers.meta.com/horizon/resources/publish-common-vrc-failures/) *(excerpt)*).
  Certification is a checklist run on real hardware, and Meta's reviewers use the same published test plan.

**Can't be automated (every source agrees):** comfort and sickness, feel, scale, readability, first-time understanding.
- Lucky VR (PokerStars VR) found simulating input and head tracking "extremely challenging" and recommended skipping it
  ([Meta blog, 2019](https://developers.meta.com/horizon/blog/developer-perspective-test-automation-for-multiplayer-vr/)).
  We already did that hard part.
- Owlchemy says many things "can only be verified in the headset". They also used a mannequin head ("Bob") on a tripod
  for captures ([Game Informer, 2023](https://www.gameinformer.com/2023/12/26/vomit-and-ginger-candy-inside-the-vr-qa-process)).
- **How VR QA works in practice** (same Game Informer piece): about 50 % of an 8 h shift in the headset at Keywords
  Gurgaon, with a **10-minute break every 20 minutes** to file bugs. A full regression pass follows every update. In
  Cosmonious High that meant physically doing all 18 mixing recipes, "murder on the arms". Testers keep anti-nausea
  supplies on hand.
- **Fresh testers.** Owlchemy tests with "users who have never tried VR before", because "once people know the game,
  their testing becomes essentially useless" for the first-time experience
  ([Meta, Developer Perspectives: Job Simulator](https://developers.meta.com/horizon/blog/developer-perspectives-job-simulator)).
- **Measuring comfort.** The SSQ is the standard sickness questionnaire. Use it **before and after** a session and read
  the *difference*, because absolute scores mislead ([Bimberg et al. 2020](https://www.uni-weimar.de/de/hci/the-ssq-in-vr-research)).
  It misses eye strain and ergonomics ([Hirzle et al., CHI 2021](https://perceptualui.org/publications/hirzle21_chi/)).
  Exercise **every comfort setting** you ship ([QAwerk](https://qawerk.com/blog/vr-game-testing/), vendor).

## 3. Multiplayer specifically

- **Bots plus churn, unattended overnight.** Lucky VR ran 8 bots per room: a host bot creates the room, clients join,
  the host leaves (forcing a host migration), and the loop repeats. That gave **~16 h overnight, a few hundred
  migrations**. A room counts as broken if its game state stops changing for a set time or a bot stops answering RPCs.
  Timestamps in the logs are matched to a recorded spectator camera. The result: "severe, game-breaking issues …
  every host migration to 0 in a matter of a few weeks"
  ([Meta blog](https://developers.meta.com/horizon/blog/developer-perspective-test-automation-for-multiplayer-vr/)).
- **Network conditions.** Use OS-level tools (netem, clumsy) or in-engine wrappers. Unity's Boss Room tested **100–150 ms
  desktop, 200–300 ms mobile, 5–10 % loss**. Delay alone without loss and jitter is "unrealistic". Also try
  500–1000 ms to check that the game fails gracefully
  ([Unity Netcode docs](https://mp-docs.dl.it.unity3d.com/netcode/2.3.2/tutorials/testing/testing_with_artificial_conditions/),
  [Bugnet](https://bugnet.io/blog/how-to-test-your-game-on-slow-network-connections)). Godot options: netfox
  `NetworkSimulator` (disabled when `CI` is set) and `LaggyMultiplayerPeer`
  ([netfox](https://foxssake.github.io/netfox/latest/netfox.extras/guides/network-simulator),
  [laggy peer](https://awesome.ecosyste.ms/projects/github.com%2Fricmzn%2Flaggy-multiplayer-peer)).
- **Load and scale.** Riot used simulated players that skip rendering (above). Unreal's Gauntlet launches "four clients
  and one server" style sessions; it needs **unique ports per parallel test**, and the forums report thin docs for
  N-client soak runs ([Epic Gauntlet](https://dev.epicgames.com/documentation/unreal-engine/gauntlet-automation-framework-overview-in-unreal-engine),
  [forum](https://forums.unrealengine.com/t/gauntlet-tests-for-a-dedicated-server-with-multiple-clients/2558869)).
  Sea of Thieves also ran **public "scale test" weekends** before launch
  ([Windows Central](https://windowscentral.com/rare-launching-sea-thieves-scale-test-weekend)).

## 4. Godot-specific tooling

- Godot ships no GDScript test framework. **GUT** (`godot --headless -s addons/gut/gut_cmdln.gd -gdir=… -gjunit_xml_file=… -gexit`)
  and **gdUnit4** (`GdUnitCmdTool.gd --run-tests`, with a GitHub Action) are the community standards. Both emit JUnit
  XML. Run `--headless`, and **import the project first** (`--headless --import`)
  ([gdUnit4](https://github.com/MikeSchulze/gdUnit4), [HelpMeTest CI guide](https://helpmetest.com/blog/godot-ci-cd-testing/) *(vendor)*).
  The engine itself uses doctest (`--test`) ([Godot docs](https://docs.godotengine.org/de/4.5/engine_details/architecture/unit_testing.html)).
- For multiplayer, Godot offers "Customize Run Instances" in the editor and server+client in one scene tree. **I found no
  published Godot multi-instance CI framework and no Godot studio testing postmortem.** Our bot-over-socket, lanes and
  Monado setup is beyond anything public for Godot.
- **Our `unit/*.gd` runner is home-made.** Moving to GUT or gdUnit4 would buy JUnit XML, per-test timing and parametrised
  tests. It is not urgent; the per-suite timing is the part that matters (§5).

## 5. Where we stand: ahead / behind / different

**Ahead of most published practice, AAA included in places:**
- A **real OpenXR runtime with a simulated headset in automation** (Monado null compositor, scripted Touch / Touch Plus /
  Index). Lucky VR called this "extremely challenging" and skipped it, and Meta only recently shipped a recorder.
- **Bots over a control socket in real MP sessions** (server + me + Tester), with a netem lag proxy (`mp/lagproxy.py`).
- **Real exported client smoke ×3 on the target OS** as a deploy gate (`clientsmoke`). This matches the industry "smoke
  stops the line" rule.
- **Perf captures at fixed vantage points** with a baseline (`xr cityperf`), the same idea as The Division's street
  wandering and CoD's per-build perf.
- **Flake quarantine with owner and expiry**, per-suite history, triage against a live baseline, a touchmap, a merge
  gate, and parallel lanes. These are big-company habits most indies lack.

**Behind / missing:**
1. **No per-suite timing**, so no sharding, no smart ordering, and no honest ETA per suite.
2. **The full regress is serial** (3.9 h in one lane while 3 lanes idle). Studios run long tiers in parallel across
   farms (CoD: "hundreds of PCs").
3. **No explicit tiers.** We have `land` (cc + branch suites + unit) and full regress, but no named ≤ 10-minute smoke
   tier and no mid tier. Sea of Thieves runs a core set per check-in; vendors suggest about 60 s per commit.
4. **No soak / churn test**: hours of join/leave/teleport with jitter, watching memory and frame-time drift.
   Lucky VR's overnight churn bots found what manual testing never could.
5. **Zeke's headset time is unstructured and leaves no replayable trace.** Studios turn real sessions into repeatable
   tests (Meta session capture), run playtests against stated hypotheses (Valve), and time-box them (Keywords 20/10).
6. **No "fresh eyes" tester.** Zeke knows the game, so first-time understanding is untested (Owlchemy).
7. **No screenshot diff vs last-known-good** (Sea of Thieves). Our stills (`cityshots`, `lumenshots`) are looked at by
   a person, if at all.

**Different, and fine:** we have no QA team and no certification. Our "QA team" is the bots plus Iris, and our
"certification" is the deploy gates. Our test mix is heavier on integration than Rare's 90 % unit split. That is the
nature of physics and VR features, but contract-style unit tests for shared modules (already proposed in the 11-07
note) would move cheap checks down the pyramid.

## 6. Top 5 changes, cost vs payoff

| # | Change | Cost | Payoff |
|---|---|---|---|
| **1** | **Record seconds per suite, then shard the full regress across the 4 lanes.** (a) `history_append` adds a duration column, with start and end stamps per suite in the regress log. (b) `regress --shards N` assigns suites to lanes by **longest-processing-time-first bin packing** on the median of recent durations, falling back to the median of known suites when there is no history ([rstest sharding](https://python-rstest.readthedocs.io/en/stable/guides/sharding/), [Tuist](https://tuist.dev/blog/2026/03/25/test-sharding)). Keep **GPU/XR suites on one shard** (one V100) and the memory-heavy suites serial (skill rule), and spread CPU suites over the rest. | ~½ day; lanes already exist | **Estimate, not measured:** 3.9 h → roughly 1.3–1.8 h, floored by the XR shard's total. The first timed regress gives the real floor. |
| **2** | **Named tiers.** **T0 smoke ≤ 10 min**: cc + unit + one fast suite per area (boot, grab, walk, MP join, XR boot). **T1 = land**: T0 + touchmap suites. **T2 = full sharded regress**: nightly and every 3rd land, as now. **Order every run by failure history** (recent reds and quarantined suites first, then newly touched) so a red shows in the first minutes, the TAP / predictive-selection idea already cited 11-07. Add `# tier: 0` tags in suite headers. | ~½ day | Fail fast; most lands wait ~10–20 min instead of the branch's whole list. The nightly T2 stays as the backstop. |
| **3** | **Structured headset sessions for Zeke** (checklist below), plus an in-game **"mark" button** that logs a timestamp, pose and state to a file. His spoken notes then line up with logs, the Lucky VR timestamp trick. | ~2 h (checklist + one button) | Every minute of his headset time goes to what bots *can't* judge: feel, comfort, scale, readability. Nothing is re-checked by hand that a suite already proves. |
| **4** | **Record his sessions as replayable tracks.** The XR rig writes head and controller poses plus button states per frame to a file (opt-in, Settings ▸ Debug), and `vr.sh xr replay FILE` feeds it to the Monado sim as a scripted track. **Assert outcomes, not frames**: the can ends held, he reached the ledge (Jolt isn't deterministic). This is Meta session capture done in our rig, since no OpenXR layer exists. | ~1–2 days | Each bug Zeke finds by hand becomes a permanent regression test from his *real* motion. Real motion differs from scripted motion, and our scripted tracks miss that. |
| **5** | **Nightly MP soak in a free lane:** 4–8 bots for 2–3 h with join/leave/teleport/host churn under lagproxy jitter (100–150 ms, 5 % loss, occasional 500 ms spikes). Gate on "state stopped changing" or "a bot stopped answering", RSS growth, server tick drift, and the error count. | ~1 day, reusing `mp/` bots + `lagproxy.py` | Catches leaks, desyncs and rare-race bugs that 2-minute suites can't. Lucky VR went from breakage "every host migration to 0 in a few weeks". |

**Also worth doing later:** perceptual screenshot diffs of `cityshots`/`lumenshots` against last-known-good (Sea of
Thieves); an occasional fresh-eyes playtester, a friend who has never played, for the first-time path (Owlchemy).

## 7. Zeke's headset session checklist (draft for change #3)

Keep each block to 20 minutes, then a break. That is the professional VR QA rhythm (Keywords Gurgaon), and Zeke isn't
paid to suffer.

1. **Before:** Iris states the session's **1–3 questions** (Valve: playtests are experiments), e.g. *"does the ledge
   catch feel fair? is the new street readable at night?"* Iris names what the bots already proved, so he skips it.
   Comfort baseline: "0–10, how do you feel?"
2. **During:** he plays normally. He presses **mark** (+ says a word) on anything odd; that's the whole job. No
   think-aloud required (Valve: it distorts play). Iris watches the telemetry, not him.
3. **Feel-only list** (only what automation can't judge, ~1 min each when relevant): motion comfort on every locomotion
   or comfort setting touched · scale (does the thing feel the right size?) · hand feel (grab, throw, weight) · text
   and sign readability at distance · audio direction · anything new on the wrist menu reachable without looking.
4. **After:** comfort again (0–10; the *difference* is the signal, per the SSQ guidance), then 3 questions: *what
   felt wrong · what felt best · anything you expected to work that didn't.* Iris turns each mark into a bug,
   a list item, or a recorded replay test (#4).

## 8. Sources not opened in full (marked *(excerpt)* above)

Riot node/33 and node/117 returned HTTP 504 twice, so the figures come from search excerpts. The GDC Vault pages are
members-only, so only titles and abstracts were read. Valve's 2009 PDF was read in full (text extracted).
