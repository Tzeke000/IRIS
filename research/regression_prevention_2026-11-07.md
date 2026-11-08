# Keeping new work from breaking what's already built — LUMEN, 2026-11-07

*Written by a helper for Iris (regress_guard_1107) after the integ4-1107 regress. Tools: harness branch `guard-1107`
(`vr.sh touchmap / conflicts / regdiff / baseline / testkeys`). Skill: `vr-multiplayer-merge-test-deploy` §6–§8.*

---

## For Zeke — the one-screen version

**What goes wrong.** Each feature is built and tested on its own branch, and it passes. Then we merge four or five of
them together, and tests for things nobody touched go red: co-op lifting, the wrist-menu buttons, running speed,
throwing. That happened again tonight (integ4) and on every integration this week.

**Why it happens.** There are four reasons, and only one of them is "something actually broke":

1. **Features share more than their files.** Two branches that never edit the same file still share the physics
   engine, the frame budget, the wrist menu, the buttons, the boot order, and settings like your height. Adding a new
   step-up check, a new menu row, or a thousand new static bodies changes the conditions every other feature runs
   under. This is the biggest group: **9 of the 30 cross-branch reds we could trace were "the new feature used up the
   headroom an old test relied on"** (frame time, boot time, draw calls).
2. **Some reds were already red before.** Five of tonight's ten were already failing or flaky on the live game. Nobody
   compared against the live game *per check* before calling them new.
3. **The test itself went stale.** Tonight's updater test (7 reds) failed because we installed the *real* signing keys,
   and a real release build correctly refuses the test's throwaway keys. The game was right; the test needed a test copy.
4. **The test machinery tripped.** Lanes, killers, cold caches, a job that never started, a log written somewhere
   nobody looked, a filter that threw away a whole run's output. Six of the 30.

**What we'll change.**

- **Before a branch is called done, it runs the suites it could break, not just its own.** A new tool reads the
  branch's diff and lists those suites: `vr.sh touchmap BRANCH`.
- **Before branches are merged together, a second tool lists where they overlap:** the same function, button,
  setting, or hot path (physics, frame order, network). Those spots get their suites run on the merged result:
  `vr.sh conflicts A B C`.
- **Every red gets compared check-by-check against the live game** before anyone calls it new: `vr.sh regdiff` /
  `vr.sh baseline`. Pre-existing reds go on a dated list with an owner and an expiry. They don't block the new work,
  and they don't get quietly forgotten.
- **Tests that depend on development-only settings run on a test copy.** Test keys are one example. The real build
  keeps its real settings: `vr.sh testkeys`, and the updater suite now does this by itself.
- **Smaller, more frequent merges.** Land each branch within hours, not in batches of five. Research is clear that
  integration risk grows with branch age and batch size.
- **A merge queue.** Branches land one at a time onto a tree that was just green. Each is tested *on top of* the
  previous one before it lands. Main stays deployable.

---

## 1. What actually broke — the taxonomy from our own history

Sources: helper reports `state/helpers/{integ_1105,regfix_1105,r4land_1106,triage_1106,triage2_1106,integ2_1106,
integ3_1106,integ3fix_1106,integ4_1107,integ4fix_1107}`, `scratch_integ3/triage_*.txt`, `scratch_integ4fix/triage.md`,
the integ4 regress log `/data/vrlanes/1/run/regress_1107_155418.log` (journal `iris-worker-integ4_regress-*`), the rerun
journal `iris-worker-integ4_rerun-*`, `known_flakes.txt`, and the merge commits on project `main` and harness `master`.
C = confirmed by a log or commit. H = hypothesis, still to be tested.

| when · where | red | culprit | cause | evidence | caught before merge by | cheapest guard |
|---|---|---|---|---|---|---|
| 11-05 main | joints J0, relics T3, round3b T1, stationsmp M0–M8, authgate A5, reconnect R1 | carhole-1105 | **perf/boot budget** | C: ~3,000 shapes added to the in-tree Cars body pushed boot to 4.7 s. Built off-tree in f629759 | timing each boot chunk on the branch | a boot-chunk budget check (cc/bootclean) |
| 11-05 integ | stationsmp M1/M2/M8, coop CO1–3 ("tester id 1") | street-1105 + stage-1105 | **two O(n²) boot steps, over budget only together** | C: 222f79f, 288027f | `conflicts` (both on the `boot` hot path) | the same budget check |
| 11-05 integ | stations MC4 (mic kicked off the table) | spleef-1105 (2,560 tile bodies) | **physics headroom / contact shift** | C: 8203a7c, a8a4184 | touchmap: `spleef*` → stations? No, so: **@regress after any physics-heavy branch** | flaky ×3 of stations after physics-heavy merges |
| 11-05 integ | xr/run R4 (later holes H2 hit the first fix) | spleef-1105 (+1.2 ms per physics tick) | **perf headroom** | C: 8b48762 → 2f76ce5 | touchmap row `spleef*.gd → xr/run` (now in the table) | `vr.sh xr r4perf` A/B on the branch |
| 11-05 integ | city_big CB13a | skyway × stage (`city_skylanes.gd`) | **shared file** | C: 190a5c1, the 855 m helipad became a 9th pad and reordered the tour | `conflicts`: same file, near hunks | run city_big after any `city_*` conflict |
| 11-05 integ | xr/swim X2–X4 | water-1105 | **a branch's new suite never run on its own head** | C: e3fa3c4, haptics aliased at 10–22 fps | branch DoD: run every suite the branch adds ×3 | the land gate |
| 11-05 main | bee B4 | avatarfix-1104 | **test outdated by an intended change** | C: fa8b741, the headless first-person split | touchmap: the token grep finds the layer constant | tests read the game's constants, not copies |
| 11-05 main | xr/mech M5, I1, S4 | stride-1105 | **tuning drift** (walk speed is now Froude-based, not 3 m/s) | C: harness 6d3b67a | `conflicts`/touchmap on the `decl` changed | tests read speeds from the probe |
| 11-05 main | xr/menu M9c, M10c | avatarscale + irisbody (each added a settings row) | **UI layout shared by two features** | C: regfix harness | `conflicts`: both edit `settings_menu.gd` | find UI elements by name, never by row position |
| 11-05 main | powers RS1 | holes-1105 rescue × the Lumen gate | **latent bug exposed by timing** | C: rescue spot 0.64 m from the gate trigger | — | a deterministic RS1b |
| 11-05 integ | mp/stagemix 0/5 | — | **harness/env**: stale `~/.local/bin/vr.sh` | C | — | the PATH vr.sh `exec`s the harness one (done 11-06) |
| 11-06 | cloth, relics, pierce, coop, spleefmp crashed in lanes | — | **harness/env**: bot killed at 150 s in a busy lane | H | — | bot timeout tied to the suite timeout |
| 11-06 engine | renderbudget RB2/RB4 (+80–100 draws) | batch-1 content | **render budget eaten** | C: 1c5b98a, ae2d35f, 83280a0 | `RB_ATTR` per-branch draw diff | margin warning < 5 % |
| 11-06 integ2/3 | renderbudget avenue 271–272 vs a 270 ceiling | lumenfx (+25) + batch 2 | **render budget eaten** | C: c06ae89 | the same | the same |
| 11-06 engine | xr/mech P1/P2 (punch grabs) | lower fps after batch 1 | **frame-time gate** (squeeze age read on the wall clock) | C: bc88807 | the `slow` probe (3 fps) | timing gates use game time |
| 11-06 engine | renderbudget "1 pass" | engine-1106 grew vantages 12 → 20 | **harness**: killed at 900 s, partial passes counted | C: 5f44021 | — | a killed suite prints `FAIL XT` (done) |
| 11-06 integ2 | netsafe crashed on its first runs | — | **fixture depends on dev config** (co-op server config untracked, exists only in the main harness) | C | — | track it, or seed it in up.sh |
| 11-06 integ2 | mp/hear crashed in a lane | — | **harness**: lane-private `/tmp` | C: a230888 | — | — |
| 11-06 integ3 | bee CRASH at exit | telesafe-1106 | **lifecycle order** (static registries) | C: bisect 0/6 → 6/6, 3e7699a | exit-under-load in unit | — |
| 11-06 integ3 | spleef SP0, SL1a–c | telesafe (the lift is now a 7 s ride) | **test outdated by an intended change** | C: 4d731ae | touchmap: `teleport.gd` → spleef suites | grep tests that use the changed mechanic |
| 11-06 integ3fix | xr/swim X4 | — | **harness**: cold shader cache per worktree | C: 0f1706e | — | shared shader cache (done) |
| 11-06 integ3fix | xr/quiver Q2/Q4 | fps 15 → 11–13 | **perf exposes an old bug** (the arrow pose never reached physics) | C: 5d18729; the feature was removed instead (7cc9017) | — | — |
| 11-07 integ4 | xr/stepup V2, xr/menuair A2 | batch3safety's own new tests | **a test bug shipped with the branch** | C: red on the branch head too; fixed in 4f80574, fd906a1 | branch DoD: own suites ×3 | the land gate |
| 11-07 integ4 | **mp/updatemp U1–U7** | **8c39413** (real keys committed *after* the branch suites ran) | **fixture depends on dev config** | **C**: server log "manifest does not verify … no updates served"; `update_core.gd` `test_switches_allowed()` is false once STATUS = "release" | touchmap: `scripts/updater/*` → mp/updatemp (in the table) | **`vr.sh testkeys`; updatemp now runs a test copy** (`updatemp-testkeys-1107` a9a9965) |
| 11-07 integ4 | mp/coop CO1 | ? b28aeec | ? | **H**: rose 0.081 m (needs ≥ 0.2), lift driven by the wall clock (1.0 s). integ4 failed 2/3, main passed 2/2. Candidates: the new step-up acting on the bots' desktop bodies next to a ~0.25 m box, or load | conflicts/touchmap would have listed it (`player_flat.gd`, `step_up.gd`) | `vr.sh flaky mp/coop 5` on integ4 vs b28aeec^1 |
| 11-07 integ4 | xr/wristfit touch(real/aim) | batch3safety: Spawn = one tap (db64535) | **UI layout / shared node** | **H**: Spawn adds a 4th row (45 → 50 presses). Both misses are row 3, ¼ from the bottom edge. aim = grip passes; only the real-Touch aim model fails | touchmap: `wrist_menu.gd → xr/wristfit` (in the table) | wrist UI edits trigger wristfit + menu |
| 11-07 integ4 | xr/run R4 (+R3) | borderline on live + new per-tick work | **perf headroom** | **H**: live 3.46/3.58 on 11-06. New per tick: `movement_stepup` test_move, the wrist late physics push (9da1481), batch3body IK | touchmap `step_up.gd → xr/run` | r4perf A/B; judge R4 in game time |
| 11-07 integ4 | mp/coop5 K3 (+K6), mp/coopon O2, xr/web W4d, xr/throw T1/T3, xr/body crash | already on main | **pre-existing** | C: main K3 failed tonight at 80/2 and 300/1; W4d fails in almost every regress since 11-04; body crashes at `xr_body.py:110` since 11-06 | `vr.sh regdiff` vs live runs | dated known_flakes entries with an owner |
| 11-07 integ4 | xr/ledge E3, xr/body | — | **flaky/load** (5 other jobs + an 18:06 runtime outage during the regress) | C: passed on rerun | — | — |

**Counts by primary cause** (30 reds):

| cause | count |
|---|---|
| perf/budget headroom eaten | 9 |
| harness/environment | 6 |
| test outdated by an intended change | 4 |
| pre-existing on live / flaky | 4 |
| UI layout or button meaning shared by two features | 2 |
| a branch's own new test never run (or red) on its head | 2 |
| shared file | 1 |
| lifecycle order | 1 |
| latent timing bug | 1 |
| unresolved (CO1) | 1 |

The **fixture depends on dev config** cause shows up twice: netsafe and updatemp.

**Hot files** (edited by more than one branch in the same integration):
- `scripts/net.gd`: integ-1105, integ2, integ4
- `scripts/rig_iris.gd`: integ-1105, integ2, integ4
- `scripts/rig_xr.gd`: integ-1105, integ4
- `scripts/city_big.gd` / `city_skylanes.gd`: integ-1105, integ2
- `scripts/xr_probe.gd`, `shot.gd`, `scripts/player_flat.gd`, `settings_manager.gd`/`settings_menu.gd`: integ-1105
- `scripts/props.gd`: integ2
- `world.gd` / `project.godot`: integ4
- the harness's `vr.sh` (help text and regress lists): conflicted in **every** integration

**The pattern.** Only one of the 30 was a classic "two branches edited the same file". The rest are **shared resources**
(frame time, boot time, draw calls, contact sets), **shared meaning** (a menu row, a button, a walk speed, a lift
duration), **stale assumptions** in tests, and **test machinery**. A tool that only looks for file overlap would have
caught almost none of them. The guard tools look at budgets, hot paths, declared names, quoted strings (probe commands,
buttons, prop ids), settings keys and functions as well.

---

## 2. What others do — the research, with sources

**Integrate early and often; long-lived branches are the risk.**
- Fowler, *[Patterns for Managing Source Code Branches](https://martinfowler.com/articles/branching-patterns.html)* (2020):
  two changes can merge with no textual conflict and still "fail to build, or … build but fail at run-time". Frequent
  integration "increases the frequency of merges but reduces their complexity and risk". *[SemanticConflict](https://martinfowler.com/bliki/SemanticConflict.html)*
  (2011): the fixes are self-testing code plus merging more often. Branches that live under a day behave much like
  continuous integration.
- [trunkbaseddevelopment.com](https://trunkbaseddevelopment.com/) (Hammant): short-lived branches, feature flags, branch
  by abstraction. DORA ([trunk-based development capability](https://dora.dev/capabilities/trunk-based-development/), from
  *Accelerate*, 2018): three or fewer active branches, merged to trunk at least daily, no integration phases *(unverified)*.
- **The number that matters for us:** Brun, Holmes, Ernst & Notkin, *[Proactive Detection of Collaboration Conflicts](https://homes.cs.washington.edu/~mernst/pubs/vc-conflicts-fse2011.pdf)*
  (ESEC/FSE 2011). **33 % of the merges git called clean were actually build or test conflicts.** Textual conflicts
  lived 9.8 days on average. Their tool, Crystal, merges and tests *speculatively in the background* to warn early. That
  is the idea behind `vr.sh conflicts` and a merge queue. Palantír ([Sarma et al., TSE 2012](https://digitalcommons.unl.edu/csearticles/104))
  showed that awareness of parallel changes lets people resolve conflicts earlier *(unverified)*. Semantic-conflict
  detection with generated tests: da Silva et al., [ICSME 2020](https://www.cse.chalmers.se/~bergert/paper/2020-icsme-testconflicts.pdf);
  [SAM](https://arxiv.org/pdf/2310.02395) (2023) *(unverified)*.

**Keep main green with a merge queue (the "Not Rocket Science Rule").**
- Graydon Hoare's rule, behind bors for Rust (about 2013): *"automatically maintain a repository of code that always
  passes all the tests"*. Original post: [graydon2.dreamwidth.org/1597.html](https://graydon2.dreamwidth.org/1597.html)
  (couldn't fetch it; the wording is confirmed second-hand by [forum.bors.tech](https://forum.bors.tech/t/other-apps-that-implement-the-nrsrose/51)).
- [GitHub merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue):
  each PR is tested on a temporary branch = latest base + every PR queued ahead of it. A failing PR is removed and those
  behind it are rebuilt without it.
- Uber, *Keeping master green at scale* (Ananthanarayanan et al., EuroSys 2019; [summary](https://blog.acolyer.org/2019/04/18/keeping-master-green-at-scale/)):
  the iOS mainline was green only **52 %** of a sampled week. With 16 concurrent changes there is about a 40 % chance of
  a problem. SubmitQueue uses speculative builds and **build-target conflict analysis**: changes that can't affect each
  other's targets are tested independently. Shopify's merge queue ([engineering blog](https://shopify.engineering/successfully-merging-work-1000-developers),
  *unverified*): CI runs before merging to master.
- **For us:** `tools/land.sh` already merges one branch at a time with cc + the branch's suites. The missing piece is
  that the land gate runs only the branch's *own* suites. The queue needs the *touchmap* suites of every branch ahead of
  it, plus the conflicts pair-suites.

**Choose tests by the change, keep a full run as backstop.**
- Google TAP: Memon et al., *[Taming Google-Scale Continuous Testing](https://web.eecs.umich.edu/~xwangsd/courses/w23/readings/googletest.pdf)*
  (ICSE-SEIP 2017). Of 5.5 M affected tests only 63 K ever failed, so selecting by dependency plus history pays.
- Facebook, *[Predictive Test Selection](https://arxiv.org/abs/1810.05286)* (Machalica et al., 2019): half the test cost,
  still over 95 % of individual failures and over 99.9 % of faulty changes caught.
- [Microsoft Test Impact Analysis](https://learn.microsoft.com/en-us/vsts/build-release/test/test-impact-analysis): run the
  impacted tests + previously-failing + new tests, with a **periodic full run as backstop** *(unverified)*. Bazel
  [`rdeps`](https://bazel.build/query/guide) = reverse dependencies.
- **For us:** suites drive the game over a control socket, so there is no import graph to walk. The touchmap combines a
  **maintained glob → suites table** (tools/touchmap.tsv, which learns from every red) with a **token grep**: declared
  names, quoted probe commands, button names, settings keys. The nightly full regress is the backstop.

**Flaky tests: measure them, quarantine with an owner, never delete.**
- Google ([Micco 2016](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)): about **1.5 % of
  test runs** flaky, and about **84 % of pass→fail transitions** involved a flaky test *(figures from search excerpts)*.
- Facebook, *[Probabilistic flakiness](https://engineering.fb.com/2020/12/10/developer-tools/probabilistic-flakiness/)*
  (2020): ask how flaky a test is, not whether, with a per-test score *(unverified)*.
- Spotify, *[Test Flakiness](https://engineering.atspotify.com/2019/11/test-flakiness-methods-for-identifying-and-dealing-with-flaky-tests)*
  (Palmer, 2019): publishing a flakiness table cut flakiness **6 % → 4 % in two months**. The "Odeneye" grid of tests × CI
  runs separates the cases: scattered reds = a flaky test, a whole red column = infrastructure. Flakybot reruns before merge.
- Luo et al., *[An Empirical Analysis of Flaky Tests](https://mir.cs.illinois.edu/marinov/publications/LuoETAL14FlakyTestsAnalysis.pdf)*
  (FSE 2014): async wait 45 %, concurrency 20 %, test-order dependency 12 %. Most order-dependency fixes **clean shared
  state**. iDFlakies ([Lam et al. 2019](https://cs.gmu.edu/~winglam/publications/2019/LamETAL19iDFlakies.pdf), *unverified*):
  50.5 % of flaky tests were order-dependent. In Godot, **autoloads and static vars are that shared state**; the
  integ3 bee exit crash came from static registries.
- **For us:** `vr.sh history SUITE` already gives the Odeneye view per suite. `known_flakes.txt` needs an **owner and an
  expiry** per line (§4d).

**Compare against a baseline, then bisect.**
- Chromium CQ ([cq.md](https://chromium.googlesource.com/chromium/src/+/282cd8a/docs/infra/cq.md), *unverified*): a
  failing suite is retried *without the patch*. If it also fails at tip-of-tree it is pre-existing. A change is rejected
  only if a test fails with it **and passes without it**. That is exactly `vr.sh regdiff` / `vr.sh baseline` + `vr.sh triage`.
- [`git bisect run`](https://git-scm.com/docs/git-bisect): exit 0 = good, 1–127 = bad, 125 = skip.

**Feature flags + kill switches.**
- Hodgson, *[Feature Toggles](https://martinfowler.com/articles/feature-toggles.html)* (2016–17): release, experiment, ops
  (kill switch) and permission toggles. Test the production config, the toggles you'll turn on, the off fallback, and
  optionally all-on. **Treat toggles as inventory with an expiry date.** We already ship `IRIS_*` env A/B switches and
  `IRIS_COOP_GRAB`. A branch that changes shared behaviour behind a flag lets a red be cleared by flipping it back, not
  by reverting a merge.

**Contract tests for shared modules.** [Pact](https://docs.pact.io/), consumer-driven contracts: the consumer's tests
define what it relies on, and the provider is tested against exactly that. Internally that means a shared module (the
wrist menu, `hand_orient`, `prop.gd`'s hand joint, `Settings`) gets a small **contract unit test per consumer**, e.g.
"the wrist menu's row N is at height H" or "the hand joint reaches the target within X ms". A change that breaks the
contract fails a fast unit test, not a 6-minute XR suite.

**Games.**
- Rare, *[Automated Testing of Gameplay Features in Sea of Thieves](https://gdcvault.com/play/1026366/Automated-Testing-of-Gameplay-Features)*
  (Masella, GDC 2019; figures from a [slide mirror](https://www.sambuz.com/doc/automated-testing-for-multiplayer-game-ai-automated-ppt-presentation-901544)):
  about 10,000 tests, about 90 % unit. A core set runs before every check-in and longer ones periodically, including
  under high latency. Shipped 100+ times in two years.
- Riot ([Merrill 2016](https://technology.riotgames.com/node/33), *unverified*): about 100,000 test cases a day.
- **Jolt determinism** ([Architecture: deterministic simulation](https://raw.githubusercontent.com/jrouwe/JoltPhysics/master/Docs/Architecture.md)):
  deterministic **only if every API call that changes the simulation happens in exactly the same order**, on the same
  binary. So **a new static body anywhere, or a node added earlier in the tree, legitimately changes contact results in
  an unrelated test.** The [Godot Jolt notes](https://docs.godotengine.org/en/stable/tutorials/physics/using_jolt_physics.html)
  say not to rely on determinism. Physics checks need tolerances and repeated runs, and they shift when *unrelated*
  content changes. Nothing is broken when that happens; you should *expect* it after physics-heavy merges.
- **OpenXR input collisions** ([XrActionSetCreateInfo](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrActionSetCreateInfo.html),
  *unverified*): when two active action sets bind the same input, the higher priority wins, and **at equal priority all
  bindings fire**, which is the "two features on one button" case. We read raw buttons in code (`rig_xr.gd` `ax_button`
  / `by_button`), so the engine can't arbitrate. The updater's "LEFT X = Yes / LEFT Y = No (only while asking)" next to
  the existing X/Y uses is an example. `conflicts` flags any two branches touching logic that names the same button.
- **Dev vs release config** ([Godot feature tags](https://docs.godotengine.org/en/stable/tutorials/export/feature_tags.html),
  *unverified*): custom tags apply only to exported builds, so editor-run tests don't see release-only config, and the
  reverse is true too. Tonight's updatemp is the reverse case: a committed release constant turned the test hooks off.

**Change risk.** Just-in-time defect prediction ([Kamei et al., TSE 2013](https://sail.cs.queensu.ca/data/pdfs/TSE_ALarge-ScaleEmpiricalStudyOfJust-in-TimeQualityAssurance.pdf),
*unverified*): reviewing 20 % of the effort finds 35 % of defect-inducing changes. The `hot path` tags in `conflicts`
(physics, order, protocol, layers, boot) are a hand-made version of that.

---

## 3. The process for LUMEN

### (a) A feature branch is "done" when it has proven all of this

1. **Its own suites ×3 on its own head.** Each new or changed suite passes 3/3 (`vr.sh flaky SUITE 3`). integ4's stepup
   V2 and menuair A2 were red on the branch head itself, and water's swim suite had never run on its branch.
2. **The touchmap suites ×1** (`vr.sh touchmap BRANCH --base auto --cmd` → run the printed lines). Anything red is
   checked against live with `vr.sh baseline <suite>` + `vr.sh regdiff`, before the branch is called done.
3. **Budgets measured, not assumed**, if the branch adds bodies, draws, per-tick work or boot work. Run `vr.sh xr r4perf`
   (physics ms/tick), `RB_ATTR=1 vr.sh xr renderbudget` (draws), and the boot `[sky]`/build chunk times. Write the delta
   in the report. **A branch may spend at most half the remaining headroom of any budget.** More than that needs Iris's
   OK and a named owner.
4. **Shared behaviour changes go behind a switch** (`IRIS_*` env / a Settings key) whose off-state is the old behaviour,
   for one release. A red after merging can then be cleared by flipping the switch.
5. **Tests reference behaviour by name, not position or magic number.** Menu rows by label, speeds from the probe,
   layers from the game's constants. (4 of the 30 reds were tests that hard-coded what an intended change then moved.)
6. **Commit messages carry `suites:` lines** for every suite the branch knowingly affects. `land.sh` and touchmap read them.

### (b) How integration works

1. **Before merging anything:** `vr.sh conflicts B1 B2 … --base main`. Every HIGH gets one of:
   - a planned merge order (riskiest last, as in the skill §1),
   - an extra suite after both are in (the tool prints them),
   - a talk with the branch owner (two features meaning different things by one button or row).
2. **Merge one branch at a time** (`vr.sh land` or the integ branch). After each: import + cc + its own suites + **its
   touchmap suites** + the pair-suites `conflicts` printed for it and anything already merged.
3. **Full regress every ~3 merges and after the last.** Run it as a plain worker job, never inside a Claude helper
   (that cgroup got OOM-killed at 12 G on 11-07). If it auto-moves to a lane, the log is in `/data/vrlanes/N/run/` and
   `run/latest_lane` now points there.
4. **Prefer small, frequent lands.** Land a verified branch the same day, rather than five branches in one night. The
   smaller the batch, the fewer pairs to analyse (4 branches = 6 pairs, 8 = 28).
5. **Baselines are live and fresh.** The nightly regress on main/master (`tools/install_nightly.sh`) is the default base
   for regdiff. Also keep the last 2–3 live runs, since one base run is noisy: `vr.sh regdiff LOG BASE1 BASE2 BASE3`.

### (c) Red triage — every ✗ gets a verdict

1. `vr.sh regdiff THIS_LOG LIVE_LOG [LIVE_LOG2 …]` → NEW / PRE-EXISTING / FIXED / NO-BASE per check. Use the journal
   directly if needed: `journalctl --user -u 'iris-worker-X-*' -o cat > x.log`.
2. NO-BASE (the suite never ran on live) → `vr.sh baseline SUITE…` (or `--from LOG`) → regdiff again.
3. NEW → `vr.sh triage LOG --only suite` (rerun ×2 on the same tree → FLAKY; live ×2 → PRE-EXISTING; `--engine-ab` →
   code vs engine). Record the load (`uptime`) before and after. Tonight's regress shared the box with 5 jobs and an
   outage.
4. Still REAL → bisect over the merge commits (`git bisect run` with one check as the probe), then **reproduce the
   mechanism on demand**: the `slow` probe, a planted value, `IRIS_*` off.
5. Verdict table in the report (suite · runs · load · verdict · evidence · owner). GO/NO-GO is one sentence on top.

### (d) Quarantine rules (known_flakes.txt)

- An entry needs **a reason, a date, an owner and an expiry** (`# … owner=iris until=2026-11-21`). An expired entry
  stops being "(known)" and comes back as a red, so a quarantined check can't hide forever. W4d has been "known" since
  11-04 and fails in nearly every regress; that's a bug in the test or the game, not noise.
- A check is quarantined only after it is shown flaky (pass and fail on the same tree, ≥ 1 of 5 each), or pre-existing
  on live with an owner. **Never delete a check to get green.**
- Review the list weekly with `vr.sh history SUITE` (the Odeneye view). A known flake that fails > 50 % of runs gets
  promoted to a real bug.
- Built: `known_mark` / `triage.py` honour `until=` (guard-1107 d90ca95). Existing lines have no `until=` yet; add them at the next review.

### (e) Release-config hazards

The release turns several **dev/placeholder** values into **real** ones. Each one can silently turn a test's hooks off,
or turn a test environment into production:

| hazard | what bites | guard |
|---|---|---|
| **updater keys** (`update_keys.gd` STATUS placeholder/test → release) | test hooks (`--update-keys`, test PEMs, test answers) are honoured only in a non-release build. With the real keys committed, mp/updatemp U1–U7 go red | `vr.sh testkeys` / the updatemp test copy (`updatemp-testkeys-1107`). The *publish* gate (cli.py `gate`) still refuses test keys for friends, so a test copy can't ship |
| **brand / user-dir** (`VR Iris` → `LUMEN`) | tests and fixtures that hard-code `app_userdata/VR Iris` (xr/data, mp/xs, backup scripts) read an empty dir | `tools/brand_check.sh <worktree>`; harness paths read the brand constant, never a literal. `unit/brand_unit` |
| **versions** (`build_version.txt`, release tags) | the installer refuses a version that's already a tag; a test that pins a version string ages out | `vr.sh version check`; tests read the version |
| **AUTH_VERSION / protocol hash** | a server or client mismatch refuses joins; a test with an old fixed hash breaks | updatemp fakes an old client on purpose (`--test-auth-v`); everything else reads the live hash |
| **untracked config only in the main harness** (co-op server cfg, mp/xs invites, xr/data settings) | a fresh worktree or lane lacks it ("netsafe crashed") | track it or seed it in the launcher. Never depend on a file only the main checkout has |
| **exported vs editor build** | a release export strips textures, drops debug, ignores env hooks | `MP_SERVER_BUILD=<export>` for mp suites; `bootclean`; client smoke on the real GPU |

Rule: **a commit that flips a release value (keys, brand, version, protocol, a default setting) names the suites that
read it** (`suites:` line), and touchmap's table maps the file to them. `scripts/updater/* → mp/updatemp`,
`scripts/brand*.gd → unit/brand_unit, bootclean, mp/updatemp` are in the table now.

---

## 4. The tools (harness branch `guard-1107`)

| command | what | verified |
|---|---|---|
| `vr.sh touchmap [BRANCH] [--base auto] [--cmd\|--list]` | suites a branch could break: table + diff tokens + its own | 13 unit checks; retro-run on integ4's branches. batch3safety-1107 (base auto → lumen-rename-1107) lists **xr/ledge, xr/run, xr/wristfit, xr/body, xr/web** among its top 23 (tonight's reds from it). updater-1107 lists **mp/updatemp** 2nd and **mp/coop, coop4, coop5, coopon, cooppop** (net.gd). It does *not* list xr/throw, which is pre-existing anyway |
| `vr.sh conflicts A B … [--base auto]` | pairwise HIGH/MED/LOW overlap + "after merging both, run" | unit checks. On integ4's 3 branches (base lumen-rename): batch3safety × updater = HIGH `rig_xr.gd` near hunks + HIGH `settings:profile/height_m`, MED net.gd + boot. The two body branches: no overlap. **On tonight's unmerged city stack** (`--base auto`): lobby × holo2 = HIGH `holo_board.gd` (the conflict Iris predicted by hand) **and HIGH `stage_tower.gd`** (not predicted), pair suites unit/flycar_lights_unit, unit/stage_mixer_unit, xr/renderbudget; `shot.gd` near hunks in most pairs (an append list, "keep both") |
| `vr.sh land` (+ touchmap) | the land gate also runs each branch's touchmap suites (≤ 25/branch, `LAND_TOUCHMAP=0` = off) | `land --dry integ4-1107` read-only: 25 touchmap suites added |
| `vr.sh regdiff LOG BASE…` | per-check NEW / PRE-EXISTING / FIXED / NO-BASE | unit checks incl. the journal prefix and multi-base; on integ4 vs the 11-06 live-ish runs: updatemp = NO-BASE, coop5 K3/coopon O2/W4d = PRE-EXISTING |
| `vr.sh baseline [--ref] [--from LOG] [--against LOG] SUITE…` | the same suites on the base trees → a real log + run/latest (+ regdiff) | `--dry` plan only (no suites run tonight) |
| quarantine expiry | `known_flakes.txt` notes may carry `owner= until=YYYY-MM-DD`; past it a FAIL prints `(known-EXPIRED …)` and counts as NEW (vr.sh, triage.py) | unit check |
| `vr.sh testkeys [--project P]` | test-keys copy of a release-keyed project | unit check + real run on wt_integ4_project: 1.6 s, original untouched |
| `vr.sh rerun journal:UNIT` / `lane:N` | rerun from a worker journal / a lane log | the error path; the parse is covered by the regdiff checks |
| lane.sh `run/latest_lane` | a lane run leaves a pointer in the caller's harness | `bash -n` only (needs a lane run) |
| updatemp test copy | `mp/updatemp.py` on `updatemp-testkeys-1107` (off integ4) | `test_copy` self-test on wt_integ4_project. **The suite itself was not run** (no Godot tonight) |

Unit checks: `python3 -I tools/test_guard.py` (13, temp git repos, no Godot).

## 5. Tooling gotchas from tonight (now in the skill)

1. **`worker.py run --after T --detach` never started its job.** From reading `scripts/server/worker.py` (not
   reproduced): a dependant is admitted only when every parent is **verified** (store status verified/committed). A
   parent worker's own `ok` reaches only `output_pending`, so without `worker.py verify T --ref ACT` the dependant waits
   forever. `--wait-deps` defaults to 0 (= check once), and nothing re-polls it. Fix: verify the parent, or chain the
   two jobs in one script.
2. **A foreground `vr.sh regress` "wrote no run/latest".** It did write one, but it had auto-moved to lane 1 (lane 0
   busy), so the log and run/latest were in `/data/vrlanes/1/run/`, and the next lane-1 job overwrites that run/latest.
   Fix: lane.sh now leaves `run/latest_lane` in the caller's harness; `vr.sh rerun` picks it when it's newer;
   `rerun lane:N` / `rerun journal:UNIT` read the other places; headers name `project=` / `harness=`.
3. **A baseline job's grep filtered all its output away** (`scratch_integ4/main_base2.txt` holds only "##### mp coop").
   The queue's `step()` piped every suite through `grep -E '^(✓|✗|…)'` and kept nothing raw. Rule: **a job's only
   output is never filtered.** Tee the raw output to a file and grep a copy. `vr.sh baseline` writes every line.
4. Also seen: raw outputs of two concurrent runs in the same lane overwrote each other in `/data/vrlanes/1/run/raw`
   (coop/coopon). Read `run/raw/*_fail*.txt` from the harness that ran the suite, and don't compare raw files across
   concurrent runs in one lane.
