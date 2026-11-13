# Faster deploys + how to test each kind of thing in LUMEN — research + plan for update #4 (2026-11-13)

Helper `fastdeploy_1113`, **research and planning only**. I ran no Godot, regress, land, or deploy. Every claim about our
system is tagged with the file it came from, e.g. `[land.log]`. Paths: `H` = `/data/projects/vr_iris/harness`,
`ST` = `IRIS/state/vr_regress`. Builds on `research/game_qa_practices_2026-11-08.md` (studio practice; not repeated here)
and `state/helpers/testaudit_1112.full.md` (redundancy audit).

---

## 0. TL;DR

Our deploys are slow because **reds don't mean anything**, not because tests are slow. Over the last 8 regresses, 6 were RED with
9–18 "NEW" failures each `[ST/regress.log]`. The deploy-#3 gate regress (11-12 22:53, 106 min) had 10 NEW failures. Triage
(55 min) found **0 real**: 5 flaky and 5 *pre-existing* (already red in earlier regresses) `[state/helpers/triage3_1112.full.md]`.
So each deploy pays roughly **regress 106 min + triage 55 min ≈ 2 h 40 min** to learn "nothing blocks". On top of that, 7 of 17
lands on 11-11/12 went RED, burning **228 of 666 land-minutes** `[ST/land.log, computed]`. Three of those were
`xr/renderbudget` draw ceilings missed by 1–2 % on a single sample (112 > 110, 271 > 270) `[ST/land_1112_*.log]`.

**The five biggest wins (§3):**
1. "NEW" means new vs the **last N regresses**, not vs `known_flakes.txt` → triage ≈ 0 on most deploys (−40…55 min/deploy).
2. **Auto-rerun reds inside the regress** (same lane, ×2, before the regress ends) → flake verdicts arrive with the regress (−20…30 min).
3. Perf ceilings judged on a **median of 3 captures with a noise band** + INCONCLUSIVE for precondition misses → stops ~40 % of land reds (−60…120 min/night).
4. **Freshness + touchmap pre-flight on the branch** (rebase on main, run its touchmap suites *before* it enters the land queue) → no "26 behind main" reds, no land red on a suite the branch never ran (−30…60 min/night).
5. **Warm process reuse for offline + mp suites** (xr already has warmlane) → ~35 of ~196 serial minutes are boot `[testaudit_1112]` → about −10…15 min of sharded wall time per regress.

Each item ships with a test that fails on the old code (§3).

---

## 1. Where the hours went (11-11 12:00 → 11-13 00:00, measured)

| bucket | minutes | evidence |
|---|---|---|
| Lands, total | 666 (17 lands, 10 green / 7 red) | `[ST/land.log]` START→GREEN/RED stamps |
| …of which RED lands (all redone) | **228** | same |
| RED causes | renderbudget ceiling 1–2 % over ×3 (00:56, 05:12, 21:06); xr/body T3 slap dB under load ×1 (04:11, a `solo:` suite); furniture F5 real (construct → chairfix) ×1; flicker suites CRASHED ×1 (06:53); fullbreak F4 real (propguard ring) ×1 (20:21) | `[ST/land_1112_001041/041340/202250/033210/063706/201203.log, land_1111_232808.log]` |
| CONFLICT re-lands | 3 starts lost (wizscale `unit/desktop_unit.gd`, towerlane `tools/suite_tiers.txt`) | `[ST/land.log 11-12 00:07/00:09]` |
| Deploy-#3 gate regress | 106 | `[ST/regress.log 11-12 22:53]` |
| Triage of that regress | 55 (22:53 → 23:48) | `[state/helpers/triage3_1112.*` mtimes] |
| Stale branch | irisarms 26 commits behind main ⇒ red that wasn't its own; fixed by merging main | `[memory/live_list.md 11-12 20:0x]` |
| Land red on a suite the branch had never run | propguard + irisarms → fullbreak F4 (the branch's own suites didn't include fullbreak) | `[live_list 11-12 20:2x; land_1112_201203.log]` |
| Box contention | auto-regress deliberately STOPPED, helpers queued behind the quiet-box lease, "helpers starved" | `[live_list 11-12 20:11, 16:3x]`; lease design `[H/tools/quietbox.sh]` |

**Regress reality** `[ST/regress.log]`: 8 regresses since 11-09 → 92–123 min sharded (249 min serial on 11-09).
GREEN = 3, RED = 5, and the RED ones were never actually blocking (each was followed by a deploy or a triage "nothing blocks").
Serial sum of recorded medians = 13,523 s ≈ 225 min over 215 suites `[H/tools/suite_durations.tsv]`. By family:
mp 83 min (46), xr 69 (30), offline 56 (53), unit 12 (75). The single biggest suite is `xr/renderbudget`, median 775 s / max 1029 s.

**Why our regresses are always red, by mechanism:**
- **`newfails` compares to `known_flakes.txt` only.** A check that was red in the previous 3 regresses but never got a
  quarantine line counts as NEW every time: 5 of 10 on 11-12 (coop5 K4, shard_regress R12, share6b F2, remount R7, mech S4)
  `[triage3_1112]`. `vr.sh regdiff` already does per-check NEW vs PRE-EXISTING against base logs `[H/vr.sh:602]`, but the
  headline number and the RED/GREEN verdict don't use it.
- **Flakes are found after the fact, by a separate serial job** (`tools/triage.py`: reruns serial, lane 0 or one lane) `[H/tools/triage.py docstring]`.
- **Single-sample thresholds on noisy measures.** RB2 draw ceilings are "perf-1105 result + ~15 %" `[H/xr/xr_renderbudget.py:21]`.
  The same build measured 105 / 102 / 112 at lumen_spleef_floor `[live_list 11-12 21:07]`, so a ±5 % spread on one sample decides a land.
- **Precondition misses report as FAIL.** There's no INCONCLUSIVE anywhere; about 75 checks in about 40 suites fold a precondition into the assertion `[testaudit_1112]`.
- **Quarantine has no teeth.** 75 lines in `known_flakes.txt`, only 10 with `until=` `[H/known_flakes.txt, grep]`. Most quarantines never expire,
  so the list only grows.

---

## 2. Research — what the literature and studios say (sources per topic)

### 2.1 Keep main green: merge queue, batching, freshness
- **"Not Rocket Science Rule"** (Graydon Hoare / bors, Rust): automatically maintain a repository that never fails its tests by
  testing the *merge result* before it becomes main, not the branch alone. That's exactly our "green alone ≠ green together".
  https://graphite.com/blog/bors-google-tap-merge-queue · https://github.com/graydon/bors · https://blog.janestreet.com/making-never-break-the-build-scale/
- **Uber SubmitQueue** (EuroSys '19): a merge queue with a *speculation graph*. It predicts which pending changes will pass and builds
  likely outcomes in parallel. Follow-up (ICSE-SEIP '25) reports −53 % CI resources and −37 % P95 wait from smarter batching/speculation.
  https://www.uber.com/blog/research/keeping-master-green-at-scale · https://lethain.com/static/blog/2019/eurosys19uber.pdf ·
  https://conf.researchr.org/details/icse-2025/icse-2025-software-engineering-in-practice/4/CI-at-Scale-Lean-Green-and-Fast
- **GitHub merge queue**: batches (min/max group size + wait timeout) and a "build concurrency" cap. Vendors warn that a deep queue
  degrades badly when tests are flaky, because one flake ejects a whole batch.
  https://www.rwx.com/docs/github-merge-queues · https://docs.trunk.io/merge-queue/administration/advanced-settings
- **Google TAP / Memon et al., "Taming Google-Scale Continuous Testing"** (ICSE-SEIP '17): Google can't test each change alone, so it
  batches and then *culprit-finds* on red. Few tests ever fail, and failing tests sit "close" to the changed code (supports touchmap-style selection).
  https://research.google/pubs/taming-google-scale-continuous-testing/

**For us:** we already do the bors part (land = merge, then test on main) `[H/tools/land.sh]`. What's missing:
(a) **freshness before the queue**: test the branch *merged with current main* in its own worktree before it may queue, so a land
only finds *interaction* reds; (b) **batch + bisect**: land N ready branches as one batch, and on red, bisect (split in half, re-land)
instead of undoing all of them and re-landing one by one by hand (what happened at 20:21 → 20:22 `[ST/land.log]`).

### 2.2 Test selection (what runs per land)
- **Meta, Predictive Test Selection** (Machalica et al., ICSE-SEIP '19): a model trained on history picks tests per change and
  catches >95 % of failing tests / >99.9 % of faulty changes while running a fraction of them. https://arxiv.org/abs/1810.05286
- **Mozilla SETA**: tests that historically caught regressions run on every push, and the rest every 10th push or 90 min. Jobs per push dropped
  ~400 → ~240, with full runs kept on the main branches. https://kimmoir.blog/2015/06/16/test-job-reduction-by-the-numbers/ ·
  https://hacks.mozilla.org/2020/07/testing-firefox-more-efficiently-with-machine-learning/
- **Sea of Thieves (Rare)**: ~10 k tests, a **core set before every check-in, longer tests periodically**; "actor tests" treat engine
  objects as first-class fixtures. https://gdcvault.com/play/1026366/Automated-Testing-of-Gameplay-Features ·
  https://sambuz.com/doc/automated-testing-for-multiplayer-game-ai-automated-ppt-presentation-901544

**For us:** we have the tiers (smoke ≤ 10 min + touchmap ≤ 25/branch at land, full every 3rd land + nightly) `[H/tools/tiers.txt, land.sh]`.
That is already SETA-shaped. The gap is that touchmap is a static table + diff tokens `[H/tools/guard.py]`, and `run/history.tsv` (3,280 rows,
34 regresses `[H/run/history.tsv]`) isn't used to add "suites that went red after changes to these files before". It's cheap to add, and
it's the light version of predictive selection (§3 item 6).

### 2.3 Flaky tests: how big companies handle them
- **Google** (Micco 2016): ~1.5 % of all test runs are flaky; ~16 % of tests have some flakiness; **84 % of pass→fail transitions in
  CI were flakes**. Policy: rerun failures, and flag tests flaky above a threshold, out of the blocking set.
  https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html
- **Microsoft "Flakes"** (Lam et al., ICSE '20 lifecycle study; ISSTA '19 RootFinder): a service that detects flakes by *rerun data*,
  reports them, and **quarantines them with an owner bug**, used by 100+ teams. https://devblogs.microsoft.com/engineering-at-microsoft/improving-developer-productivity-via-flaky-test-management/ ·
  https://www.microsoft.com/en-us/research/publication/root-causing-flaky-tests-in-a-large-scale-industrial-setting/ ·
  https://learn.microsoft.com/azure/devops/pipelines/test/flaky-test-management
- **Chromium Findit / Flake Portal**: a failing test is **retried in its own process inside the same run**. Pass on retry = "hidden flake",
  recorded but not blocking. The Flake Analyzer reruns history to find where the pass rate changed (culprit).
  https://chromium.googlesource.com/infra/infra/+/9ffe1ba58c936d4edf5852da63c470325cf490e8/appengine/findit/docs/flake_portal.md ·
  https://sites.google.com/chromium.org/cat/findit/flaky-failures/overview
- **GitHub** cut flaky builds 18× mainly by auto-retrying and auto-attributing: https://github.blog/engineering/reducing-flaky-builds-by-18x/

**For us:** in-run retry (Chromium) + history-based "pre-existing" (Google/Flakes) + quarantine *with owner + expiry* (Microsoft).
We have the third half-built (`known_flakes.txt` `owner= until=` → "(known-EXPIRED)" `[H/vr.sh:90]`) but only 10/75 lines use it.

### 2.4 Determinism and tolerances (physics)
- **Jolt** is deterministic only on the *same binary + platform + same order of operations*. Cross-platform needs
  `JPH_CROSS_PLATFORM_DETERMINISM` (~8 % cost), and the Godot-Jolt binding does not promise it.
  https://jrouwe.github.io/JoltPhysics/md__build_2_r_e_a_d_m_e.html
- So physics checks must be **bands derived from physics** (e.g. v²/2μg slide distance ± k), measured over repeated runs, and a
  check's band should be wider than its measured run-to-run spread (vrgame-testing skill: "Jolt is not deterministic — use tolerances and
  repeated runs"). Our harness already does this for many checks (xr/mech ice `[H/vr.sh header]`).

### 2.5 Visual / rendering correctness
- **Unreal Screenshot Comparison**: per-channel + brightness tolerance, plus a separate `MaximumLocalError` / global error. The default
  tolerance is "Low" because AA adds per-pixel noise.
  https://dev.epicgames.com/documentation/unreal-engine/screenshot-comparison-tool-in-unreal-engine ·
  https://docs.unrealengine.com/4.27/en-US/API/Developer/FunctionalTesting/FComparisonToleranceAmount/index.html
- **Unity Graphics Test Framework**: `PerPixelCorrectnessThreshold` (ΔE per pixel) + `AverageCorrectnessThreshold` (image-wide).
  https://docs.unity3d.com/Packages/com.unity.testframework.graphics@7.3/api/UnityEngine.TestTools.Graphics.ImageComparisonSettings.html
- Lesson: golden images need **perceptual (ΔE) + local + global** thresholds and a re-baseline workflow, or they flap. For us, **metrics
  beat goldens** wherever a metric exists (draw counts, coverage %, temporal luminance variance for flicker). Goldens are only for "did the
  look change" review images that a human (Zeke) approves.
- **Photosensitivity (DK-116 flicker)**: WCAG 2.3.1 "three flashes or below threshold". A flash = an opposing luminance change of
  ≥ 20 cd/m² (darker < 160) per ITU-R BT.1702, which also treats saturated-red transitions as harmful; the Harding test is the broadcast scanner.
  https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html ·
  https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf ·
  https://backlothelp.netflix.com/hc/en-us/articles/360057622133
  → a measurable, standard-based guard: count opposing luminance swings per 1 s window in the rendered frames, **fail at > 3/s over
  ≥ 25 % of the screen** (WCAG area rule ≈ a 341×256 px region at 1024×768; scale to our render size).

### 2.6 Networking, multiplayer, server hangs
- Previous research covered netem / bots / Gauntlet (`game_qa_practices_2026-11-08.md` §3). The new item is **DK-120 (server hang 11-12
  23:22: main+worker spinning, UDP backlog)** `[live_list 11-12 23:27]`. The standard pattern is a **liveness watchdog that measures the main
  loop's progress** (tick counter advancing), *not* the process being alive. On a stall > N s: dump all thread stacks (we already do this for
  clients: `vr.sh clienthang` `[H/vr.sh header]`), then restart. Test shape: inject a deliberate main-loop stall via a test-only hook (refused when not
  in a test build) and assert the watchdog fires, dumps, and restarts within the window, and the clients rejoin (mp/reconnect R1 already measures rejoin).

---

## 3. Ranked implementation plan for update #4 ("faster deploys")

Each item gives: what · files · proof (it must fail on the old code) · minutes saved · risk. Order = saved minutes ÷ effort.

### #1 — NEW means "new vs the recent regresses", not vs known_flakes  (DK-82)
- **What:** `newfails` / regress verdict = checks that FAIL now **and** passed in each of the last K=3 finished regresses for that suite
  (from `run/history.tsv` + the logs it names), minus known_flakes. Print three numbers: `NEW n · PRE-EXISTING n · KNOWN n`. RED = NEW > 0.
  PRE-EXISTING checks red in ≥ 3 regresses in a row get **auto-filed** to known_flakes with `owner=iris until=+7d` and a "pre-existing" note
  (shown in the summary, never silent — Zeke's no-silent rule).
- **Files:** `H/vr.sh newfails()` (~l.887) → call `tools/guard.py regdiff` (already does per-check NEW/PRE-EXISTING `[H/vr.sh:602]`);
  `run/history.tsv` needs per-*check* rows (it has per-suite pass/fail counts today `[H/run/history.tsv head]`) — add `run/check_history.tsv`
  (`stamp suite check PASS|FAIL|INC`), written by `history_append`.
- **Proof:** a unit test feeding three synthetic regress logs: check X red in all three → 4th log reports X as PRE-EXISTING (old code: NEW).
  Then replay the real 11-12 22:53 log against 11-11 02:08 / 11-11 15:59 / 11-12 09:37 → expect NEW ≤ 5 (the 5 triage3 called flaky), not 10.
- **Saved:** triage shrinks to the truly new. 11-12: 10 → 5 suites to triage, ~25–30 min saved. Combined with #2, ~55 min/deploy.
- **Risk:** a real regression that lands while the check is already red hides. Mitigation: pre-existing lines expire in 7 days and the
  regress prints them; the per-check *message* changing (e.g. different vantage over the ceiling) counts as NEW.

### #2 — Retry reds inside the regress (in-run flake verdicts)
- **What:** at the end of each shard (same lane, same tree, still holding the lease), rerun every suite with a non-known FAIL **×2**.
  Report `FLAKY (1/3)` / `CONSISTENT (3/3)`. FLAKY checks don't count toward RED, are listed, and are written to check_history (so #1's
  "N in a row" catches a flake that's getting worse). This is the Chromium "retry in its own process" + Google "rerun policy" shape (§2.3).
- **Files:** `H/tools/shard_regress.sh` (after `run_shard`: a retry pass per shard from its `.out` FAIL lines); `tools/shard.py merge` prints the
  verdict column; `tools/triage.py` keeps only the LIVE / engine A/B steps (its TREE step becomes the in-run retry).
- **Proof:** fake suite `tools/test_flakeme.py` that fails on its first run per lane (marker file), passes after. Regress with it → FLAKY,
  verdict GREEN. Old code: NEW failure, RED. Plus a 3/3 failing fake → CONSISTENT, RED.
- **Saved:** triage's TREE step (the bulk of the 55 min) runs in parallel lanes, inside the regress: ~+5–10 min of regress wall time, −30–45 min triage.
- **Risk:** retries can mask load-induced reds. Retry in the same lane but record the load (`qb_load`) before/after, as triage.py already does.

### #3 — Noise-aware perf ceilings + INCONCLUSIVE preconditions  (DK-82)
- **What (a) perf:** `xr/renderbudget` RB2/RB3: capture each vantage **3× (median)**, and judge over-ceiling only when the median is over *and* by more
  than the vantage's measured spread (store min/median/max per vantage in `renderbudget_last.json`). Draw counts are a near-deterministic metric
  *per frame*; the spread comes from streaming/LOD timing, so also wait for "no pending loads" before capturing (a precondition → INCONCLUSIVE).
- **What (b) preconditions:** `testkit.py` gets `need(cond, why)` → `INCONCLUSIVE` + one retry (the design in `testaudit_1112` "K1-class fix");
  summary counts INC separately; **3 regresses in a row INCONCLUSIVE ⇒ FAIL** (so a broken precondition like grabbing can't hide). Convert the
  co-op/lift family first (kart K1, coop4, coop5, m, holdset) + xr carry/throw/grabco/bow2/web (~75 checks `[testaudit_1112]`).
- **Files:** `H/xr/xr_renderbudget.py` (~l.286–300), `H/tools/testkit.py`, `H/xr/xrlib*.py` `check()`, `H/vr.sh summary()`/`newfails`,
  `tools/shard.py merge` (INC column), `run/check_history.tsv` (from #1).
- **Proof:** (a) replay the three 11-12 RB2 reds' numbers (112/110, 271/270, 73/60) through the new judge with their spreads: 112 and 271 → pass
  (within spread), gallery_skywalk 73 > 60 → still FAIL (it was a real overage that irisarms2 fixed `[live_list 17:4x]`). (b) kart with a forced
  grab miss (env hook `KART_DROP_GRAB=1`) → INCONCLUSIVE (old: FAIL K1); with the lift law broken (mass×2) → FAIL.
- **Saved:** 3 of 7 land reds on 11-12 were RB2 noise (46 + 59 + 44 = 149 min of red lands + re-land time) `[ST/land.log]`. Also renderbudget ×3
  costs ~+5 min per run. Net ~60–120 min per busy night.
- **Risk:** a 3× capture adds ~4 min to a 13-min suite. Keep 1× for single runs (`vr.sh xr renderbudget`) and 3× only when gating (land/regress).

### #4 — Branch pre-flight: fresh vs main + its own touchmap, BEFORE the land queue
- **What:** `vr.sh ready BRANCH` (run by each helper at the end, in its own worktree and its own lane, *not* the quiet box):
  (1) merge current main into the branch (or refuse if more than M=10 commits behind with conflicts) — this is the "keep branches fresh" rule;
  (2) `godot --import` + cc; (3) its own suites + its **touchmap list** (the same list land would run); (4) writes
  `state/vr_regress/ready/BRANCH.json` = {main_sha_tested, suites, verdict}. `vr.sh land` refuses (or warns, `LAND_READY=warn`) a branch
  whose ready file is missing / red / tested against a main more than K commits old. Lands then mostly confirm and only catch *interaction* reds.
- **Files:** new `H/tools/ready.sh` (reuse land.sh's suite-selection code — factor `select_suites()` out of land.sh steps "suites the merged
  range asks for" + touchmap); `H/tools/land.sh` step 1 check; helper prompt template (IRIS `scripts/server/helper.sh` brief) gets "finish with `vr.sh ready`".
- **Proof:** `tools/test_land_tiers.sh` gets: branch 26 behind main with a ready file from an old main → land refuses (old: lands then goes red);
  branch whose touchmap includes fullbreak but ready ran without it → refuse.
- **Saved:** 11-12: the irisarms stale red and the propguard F4 red would have shown up in the helper's own time (not on the shared box):
  the 20:21 red + undo + re-land ≈ 9 + 44 min, plus earlier the irisarms2 helper cycle. ~30–60 min/night of *serialized* box time.
- **Risk:** helpers take longer (+10–20 min each), but in parallel lanes, not on the shared box. Touchmap lists must stay ≤ 25 (already capped `[land.sh]`).

### #5 — Warm process reuse for offline + mp suites  (DK-82 "boot reuse")
- **What:** like xr warmlane (`VR_WARM=1`: one game per series, reset between suites, `cold:` opt-out `[H/vr.sh:854, 1036–1041, 1246]`):
  `_shard` keeps **one offline bot** and **one up.sh MP stack (server + me + Tester)** alive across consecutive suites of the same kind, with a
  `reset` control command between suites (reload the world scene, clear props/powers/settings, reseed). Suites that change engine/project
  settings or boot their own stack stay `cold:`.
- **Files:** `H/vr.sh` `_shard` + `test`/`mp` paths; the bot's `rig_iris.gd` control socket `reset` (project); `tools/suite_tiers.txt` `cold:` list;
  `tools/test_warmlane.py` extended.
- **Proof:** (a) a **leak check**: run suite A then B warm vs B cold → identical check results (run the pair matrix over the smoke tier); (b) a
  deliberate leak (A sets gravity ×2 and doesn't reset) → the reset restores it (warm B passes) — and a guard that **diffs a world fingerprint**
  (prop count, physics settings, player state hash) after reset vs a fresh boot, failing the shard if they differ (Zeke's "class not instance" rule:
  no suite can ever leak into the next one).
- **Saved:** ~35 of ~196 serial min are boot `[testaudit_1112]`. Warm offline (53 suites × ~18 s saved) + warm mp (~40 up.sh-world suites × ~15 s)
  ≈ 26 serial min ≈ **8–12 min of a 5–6-shard regress**, and 1–3 min off every land.
- **Risk:** state leaking between suites = order-dependent results, the classic flake source. The fingerprint guard is the mitigation, and it's why this is #5, not #1.

### Smaller items (do if time)
6. **History-weighted touchmap** (light predictive selection, §2.2): `guard.py touchmap` adds suites that went red within 2 days after
   past changes to the same files (from check_history + git log). Proof: replay 11-12 propguard → fullbreak is selected. Saves red lands, not minutes per se.
7. **Batch-land with bisect:** `vr.sh land A B C` red ⇒ automatically `--undo`, then land A+B / C halves in turn (Google TAP culprit-finding).
   Saves the manual undo + re-land loop (20:21 → 20:22 → 21:06).
8. **Quarantine hygiene:** every `known_flakes.txt` line must have `owner=` + `until=` (testlint rule; land refuses new lines without them). Only 10/75 do today.
9. **Split renderbudget** (775 s, max 1029) into 3 vantage groups → LPT can spread it over both XR slots; it's the regress's critical path on shard 0.
10. **Cut the 2 zero-check suites** (settle, rolltest) and the testaudit top-8 (≈ 6 serial min) `[testaudit_1112]`.

**Expected effect per deploy (gate regress + triage):** today ~106 + 55 = 161 min. After #1+#2: ~110 + ~10 (LIVE/engine A/B only for true NEW)
≈ 120 min. After #5+#9: ~100–105 min. Per night of lands: −2…3 h of red-land churn after #3+#4.

**Order of build:** #1 (history + newfails; pure tooling, no game risk) → #2 → #3b (testkit INCONCLUSIVE) → #3a → #4 → #5. #1–#3 are harness-only
(no protocol change, no client deploy). #5 touches the bot's control socket (project-side, server-only).

---

## 4. How we test each kind of thing (shape · setup · pass/fail · cost · tier)

Tier: **L** = land (smoke/touchmap), **R** = full regress (every 3rd land + deploy gate), **N** = nightly only, **H** = human in headset.
"Fail-on-old" = the test must be shown red on the code before the fix (the commit message records it). Every check states its precondition with `need()`
(→ INCONCLUSIVE + 1 retry; 3 regresses INC in a row ⇒ FAIL).

| category | test shape | setup | pass/fail measure | cost | tier |
|---|---|---|---|---|---|
| **Physics: settle / tip / rest** | spawn → wait for sleep or ≤ T s → read pose | offline bot, prop at its real spawn; precondition: nothing else within 1 m | upright within ±X°, base on floor ±3 cm, still (v < 1 cm/s) — `furniture F5`, `propsettle` | 15–60 s | L (touchmap) / R |
| **Physics: world-wide audit** | ONE scan of every spawned object (class guard, Zeke 11-12) | offline bot, whole world loaded | no AABB overlap at spawn, no knife-edge (est. tip < 5° or flat-based), nothing floating > 2 cm, nothing under the floor | 1–3 min | R + L when props/scenes change |
| **Physics: collision ≈ shape** | per-prop: ray/point sample the collider vs the render mesh | headless, no bot needed; per class rule from `collision_and_grab_research_2026-11-12.md` (C1–C9) | max gap mesh↔collider ≤ class tolerance; convex count ≤ budget | seconds/prop, batch | R (all props), L (changed props) |
| **Physics: dynamics laws** | drive an impulse/drop, compare with the closed form (v²/2μg, e, Δv = J/m) | offline bot; repeated ×3 when the band is tight | measured within a band **wider than measured run spread** (Jolt not deterministic across order, §2.4) | 20–60 s | L (smoke: push, fullbreak) / R |
| **Grabbing / hands** | layered: unit (grab rules) → offline bot grab → xr sim with scripted controller tracks | Monado null compositor + scripted Touch/Index tracks (xr/hold, grabco, carry) | held = joint present + prop follows hand ≤ 2 cm p95; release velocity; two hands don't interpenetrate | unit s / xr 1–4 min | L (xr/hold smoke) / R |
| **Locomotion** | scripted input track → trajectory | xr sim (arm-swing, stick), desktop unit for the maths | speed/jump arc/ledge catch within physics-derived bands; no camera move without input (comfort) | 1–2 min | L (xr/run) / R |
| **Rendering perf** | fixed vantages, **median of 3** captures, same box, quiet (`solo:`) | xr sim on V100; precondition: loads settled, box quiet (else INCONCLUSIVE) | draws/prims/GPU ms ≤ ceiling *and* over by more than measured spread (§3 #3); A/B = same box, interleaved runs | 5–13 min | R (gate), L only when render/scene files change |
| **Visual: flicker / z-fight (DK-116)** | render a short sequence at the reported spot; temporal metric | xr or desktop render, fixed camera + the moving element (elevator, hologram) | (a) z-fight: per-pixel depth/colour toggling rate in a region ≤ ε; (b) **photosensitivity: ≤ 3 opposing ≥ 20 cd/m²-equivalent luminance swings per 1 s over ≥ 25 % area (WCAG 2.3.1 / BT.1702)**, saturated-red transitions counted | 1–2 min per spot | R + L when shaders/materials/lobby change |
| **Visual: text fit / signs** | metric, not golden: label rect vs panel rect | headless UI or render | no overflow/clip, min px height at the reading distance | seconds | L (unit) |
| **Visual: LOD / visibility** | per-section audit + vantage stills | `tools/lod_audit.gd`, undercull checks | every MeshInstance has a vis range or LOD; nothing culled while in view (screen coverage > 0 at the vantage); stills to Zeke only for "does it look right" | 1–5 min | R; stills = H review |
| **Look changes** | golden images, perceptual ΔE + local + global thresholds | fixed vantage, fixed time of day, AA off for capture | not a gate, a **review diff**: changed images go to Zeke | minutes | N + H |
| **Networking: protocol + auth** | unit hash + handshake | `unit/proto_hash_unit`, mp/handshake | hash bump ⇒ server+clients deployed together (deploy gate) | s / 1 min | L (smoke) |
| **Networking: sync under bad network** | N bots + netem (latency/loss/jitter) | local server + bots; netem on the lane's private netns | positions agree ≤ X cm after Y ms; ownership hand-off has no pops (p95) | 1–5 min | R (+ N with heavier profiles) |
| **Networking: reconnect** | kill/restart server mid-session | mp/reconnect | clients rejoin by themselves ≤ 30 s, state restored | ~1 min | R |
| **Server hang (DK-120)** | test-only stall hook in the server main loop | dedicated server export (build_server), 2 bots | watchdog sees the tick counter stop > N s → stack dump file → restart → bots rejoin ≤ 30 s; the hook is refused in a release build (a test proves that too) | ~2 min | R + deploy gate (server export) |
| **Audio / voice** | signal metrics, not ears | mp/micvad, hear, voice_fade unit | VAD on/off timing, attenuation dB vs distance curve, no clipping; graded per drawn frame ⇒ `solo:` | 1–3 min | R |
| **UI / menus** | drive via control socket / xr pointer track | xr/menu, desktop unit | each control reachable and does its action; settings persist (schema unit) | 1–2 min | L (unit) / R |
| **Content placement** | world audits (see physics audit) + reachability | headless navmesh/ray probes from spawn | nothing floating/overlapping; every interactive reachable from a standing player (height 0.4–2.0 m, within arm reach of a walkable spot) | 1–3 min | R + L when scenes change |
| **Client smoke** | the real exported Windows client ×3 | `vr.sh clientsmoke` (tower/win101) | first frame, online, no hang report | ~3 min | deploy gate |
| **Human-only** | Zeke's headset checklist | — | comfort, feel, scale, "looks right", sound feel, photosensitivity sign-off on flagged spots | — | H after deploy |

**Rules that apply to every row:**
1. **Precondition ≠ property:** `need()` for the setup (bot grabbed, loads settled, box quiet), `check()` for the claim.
2. **Bands from physics + measured spread**, never a single-sample threshold on a noisy measure.
3. **Fail-on-old proof:** a new check must be shown red on the pre-fix commit (testkit can record `--expect-fail-on SHA`).
4. **Measures that load skews are `solo:`** (frame time, dB per frame, p95 jitter) `[H/tools/suite_tiers.txt]`.
5. **Every root cause gets a class guard** (world-wide audit rows above), not a single-instance test.

---

## 5. Inbox
Checked at start (00:03) and before writing; no lines from this run.

## 6. What I'm not sure about
- Minutes saved are estimates from one night's logs + testaudit's single boot sample. #5's number has the widest error.
- I did not verify that `guard.py regdiff` handles sharded logs as base logs; #1 assumes it does (vr.sh says regdiff reads them unchanged `[H/vr.sh:72]`).
- The renderbudget run-to-run spread (105/102/112) is one vantage, one night `[live_list]`. #3a needs a 5× measurement per vantage first (`vr.sh flaky xr/renderbudget 5`, a solo run).
- Land durations are START→END and include lease waiting.
