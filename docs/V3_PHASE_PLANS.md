# V3 phase plans — Phase 1 close-out, Phase 2, the rest of Phases 3–4, Phase 5

Written 2026-11-03. Phase definitions come from the V3 spec §11. The current system is described in
[`ARCHITECTURE_V3.md`](ARCHITECTURE_V3.md); what Phases 3–4 built is in [`V3_PHASE3_4.md`](V3_PHASE3_4.md).
The Phase 1 plan lives in Iris's memory (`v3_phase0/phase1_plan.md`).

**Rules for every plan below:**
- Each step fits one helper session (`scripts/server/helper.sh`, locked down, own worktree).
- Iris reviews and merges every step.
- Nothing touches canonical memory without a fresh backup.
- Every live behaviour sits behind a flag whose off value is the rollback.
- "Zeke" means the step needs his hands, ears, a restart window or a decision. Everything else is Iris's to do.
- Proposed numeric targets are marked *(proposed)*. They're starting points to agree on, not measurements.

---

## Phase 1 close-out (prerequisite for anything "v3 boots")

Steps 1.1–1.6 are built (01e8000, 48c8b76, cd6a059, 1aec733, af11185). Exit-gate status on 11-03:

| Gate | Status |
|---|---|
| 1 store + claim tests | PASS |
| 2a kill -9 after a reply, no duplicate | PASS (b034453 + live restart evidence) |
| 2b DM while the host is down | offline PASS, **live untested** |
| 2c corrupt DB | offline PASS |
| 3 3-day parity soak | **PASS on the numbers**: `state/v3/parity_soak.json` has 4,375 samples over ~3.0 days, 1 brief mismatch, 0 persistent, 0 gaps. Iris still has to accept it |
| 4 latency ≤ v2 +10 % | PASS for v2 + dual-write only (4d5be86) |
| 5 rollback drill | sandbox PASS, **live untested** |
| 6 checklist rows | 5/6 (P1-H2 leases log-only) |

| # | Step | Gate it closes | Zeke? |
|---|---|---|---|
| 1.A | **Install the current start gate.** `~/iris_start.sh` is older than `scripts/zorin_launcher/iris_start.sh` and never calls `stack_launch.sh`. Diff it, install it, and confirm one v2 start writes `state/v3/launched.env` with `stack=v2`. | precondition for 5 | restart window (the gate runs at start) |
| 1.B | Accept the soak: record the soak numbers in the Phase 1 plan. Flip the Discord source of truth to `events.sqlite`, keeping the JSON writer one more week as rollback. | 3 | no |
| 1.C | Create `~/IRIS-v3` (worktree, branch `v3`) and `~/.config/iris/v3.env`. Run a staging boot (`~/IRIS_BOOT_TEST`, `IRIS_ROLE=staging`). | precondition | no (staging needs no window) |
| 1.D | Live rollback drill: `stack_switch.sh v3` → boot → `stack_switch.sh v2`. Then make a v3 start fail on purpose and watch the supervisor roll back. Memory counts must be unchanged (JSONL lines, Chroma count). | 5 | **yes**, two restarts |
| 1.E | DM-while-down: stop the host, Zeke sends one DM, start the host. It must arrive exactly once, with its age. | 2b | **yes** |
| 1.F | Flip file leases from log-only to deny, **and** make out-of-process helpers declare leases that `worker.py` checks at launch. The SDK hook never sees them. | 6 (P1-H2) | no |

**Risks:**
- 1.A changes the one path every boot takes. Keep the old gate as `~/iris_start.sh.prev` and test with `~/IRIS_BOOT_TEST` first.
- 1.D on a bad day leaves Zeke without me for a supervisor cycle. Only do it while he's there.

---

## Phase 2 — Responsiveness

**Spec:** an independent microphone/STT, echo handling, an interrupt controller, conversational priority,
and measured latency.

### Already done (don't redo)

| Item | Evidence |
|---|---|
| The ears are independent of the cognition process | `iris-ears` unit; host `voice_reader` |
| Interrupt tiers ack/duck/pause/barge/defer, live | a05b64e, 8d43625, aefba8f, 8d0c1f6; `IRIS_DUPLEX_TIERS=1` |
| Streaming ASR sidecar, live, whisper fallback | 30bfb2b, `stream_asr: on` |
| Continuation merge; an answer after a question is a turn | ce0356f, bd6cd7e |
| Source priority + inbox checkpoint (now/after/defer/delegate) | `_SOURCE_PRIORITY`, 37d0483 |
| Stage latencies, including tool-blocked | 83e62b5, 4d5be86, 0378445 |
| AEC3 built tower-side | b1e10e5, **default off** |
| Live words in the orb app | 0f676d9 |

Measured 11-02 (`latency_report.py`), medians:

| Stage | Median | Events |
|---|---|---|
| speech end → transcript | 1.4 s | 15 |
| speech end → first sound out | 8.7 s | 15 |
| first speak → first sound out | 1.2 s | 27 |

### Remaining steps

| # | Step | Exit check | Zeke? |
|---|---|---|---|
| 2.1 | **Latency baseline with streaming ASR.** Run `latency_report.py` over 3 days of real use and record p50/p90 per stage: speech_end→stt_done, stt_done→decision, decision→first_audio, tool_blocked. Split each into whisper-only and stream turns using `state/health/stream_asr_turns.jsonl`. This is the Phase 2 reference table. | table filed with n ≥ 50 voice turns per stage | no |
| 2.2 | **First live streaming sentence check.** Compare sidecar finals with whisper on the same turns, from the logged turns. Report WER where both exist and the fallback rate. | fallback rate and agreement reported | no |
| 2.3 | **"Never hears its own voice" acceptance (§12).** Search the transcript for user turns logged during my playback windows whose text matches my own last spoken line (fuzzy match). Count them. Add a regression test that feeds a recorded self-echo turn through the tier classifier. | 0 self-echo user turns over 7 days *(proposed)*; test green | no |
| 2.4 | **Interrupt acceptance run.** Twenty scripted trials on the headset: 5 "mm-hm", 5 coughs/noise, 5 "wait"/"stop", 5 full sentences. Record the tier chosen against the expected tier from the body log (`[barge-*]` lines, 69cca4a reasons). | ≥ 18/20 correct tier, 5/5 stops honoured *(proposed)* | **yes** (his voice, ~10 min) |
| 2.5 | **AEC on the tower.** Install livekit in the tower venv, set `IRIS_AEC_REF=1` (relaunch the orb app) and `IRIS_AEC=1` (mic_source respawns). Check `state/mic_source.log` for "ECHO-CANCELLED" and rising `erle_db`. Follow duplex design §8 step 3. | 30 s of my speech on speakers with Zeke silent → 0 barges | **yes** (tower, speakers, app relaunch) |
| 2.6 | **Loud-speaker double-talk.** First ask Zeke how he uses speakers. Then try a tuned AEC3 config, or AEC output for VAD + a linear-AEC transcript path with a voiceprint guard. Offline bench on the 10-31 AEC recordings first. | his words kept in ≥ 3/4 loud-room bench clips with no echo leak *(proposed)* | decision + live test |
| 2.7 | **Plumb `critical`.** The host sets a flag while I read out a number or a warning, so DEFER holds until the sentence ends. Unit test, plus one live check. | test green; no barge mid-number in 2.4's trials | no |
| 2.8 | **Echo residual from the tower** (`residual_fn`), so the over-residual test uses real numbers instead of the RMS floor. Requires 2.5. | classifier logs real residuals | tower deploy |
| 2.9 | **Look-up-phrase live tests** (awareness P1-T2/T4b/R5). One long foreground `sleep 600` Bash, then a routine DM, then "Iris, look up". Record whether `interrupt()` kills the Bash. If it does, change the interrupt path to checkpoint-and-continue for foreground tools. | result recorded; work not lost | **yes** |
| 2.10 | **Deliberate defer / delegate trials** (P1-T3b/T3c). During real work, once deliberately defer a DM and answer it later; once delegate via `helper.sh` and reply. Both must show in the event store. | two event traces filed | **yes** (he sends the DMs) |
| 2.11 | **Coordinator split (§3 C) — design only.** Write down what moves out of the host (queue, priority, checkpoint decisions) and what stays with me (meaning, commitments). Don't build until 2.1–2.4 pass, since it risks the working host. | design reviewed by Iris | decision |

### Phase 2 exit gates

1. 2.1's table exists, and the v3 build (after Phase 1 close-out) is within +10 % of it on every stage.
2. 2.3: 0 self-echo turns in 7 days *(proposed)* plus a regression test.
3. 2.4: ≥ 18/20 correct tiers and 5/5 stops *(proposed)*.
4. 2.5: AEC on, 0 self-barges on speakers in the 30 s test. Or Zeke decides headset-only and AEC stays off. That's a recorded decision, not a pass.
5. 2.9: the look-up path doesn't lose work.

### Phase 2 risks

- AEC can mask Zeke's voice on loud speakers. This was measured 10-31. Rollback: unset `IRIS_AEC`.
- The GPU is at ~31.8/32.8 GB with the stream sidecar. Any new model in this phase goes through `gpu_admit.py` first.
- Tier tuning against live speech can over-fit one evening. Use the logged reasons (69cca4a) across days.

---

## Phases 3–4 — remaining work

**Built** (17f7484, cd6a059, 6490642, b0ba697, 0377171, 1aec733, c62cb84, 44e612a, 8ecdb41, 0378445):
task dependencies, incident bundles, fault injection, worker lockdown, the delegated≠done ledger,
health snapshot + aging + whisper-load probes, and GPU load-time admission for the stream sidecar.

### A. Restart during a task, across a real reboot (§12 "Reboot during a task")

| # | Step | Exit check | Zeke? |
|---|---|---|---|
| 3.1 | **Drain the verify backlog.** 71 tasks sit in `output_pending` because nobody runs `worker.py verify`. Write `worker.py review`: list each with its report path and act id, so Iris can verify, fail or cancel in bulk. Without this, "reconcile after reboot" has nothing clean to reconcile against. | open-task count < 10 *(proposed)* | no |
| 3.2 | **Reboot drill script** `scripts/server/reboot_task_drill.py prepare`. Creates three tasks: a `none` side-effect worker (`sleep` + file write, `--key` set), a `reversible` one, and an `external` one that would post a marker message to a test channel (fake/dry-run target). Snapshots the store. | dry-run in staging passes | no |
| 3.3 | **The reboot.** Run `prepare`, reboot the VM (needs `--arm`), and let the unattended boot come back. | — | **yes** — reboot window |
| 3.4 | **`reboot_task_drill.py check`.** The `none` task resumed under the same key, with no duplicate row. The `external` task is in `needs_review`, not retried. The marker was sent at most once (count in the test channel). Integrity is ok, the parity soak continues, and a v2 DM arrives once. | all assertions PASS, recorded in an incident bundle | no |
| 3.5 | Enable `state/incidents/capture_on_red` once the first manual bundle has been reviewed for redaction. | `state/incidents/` exists; one reviewed bundle | no |
| 3.6 | GPU-hog drill (V3_PHASE3_4.md §3.1), with the mouth speaking. | mouth fluent; `admit` defers; VRAM recovered | **yes** — window |
| 3.7 | CPU delegation: install `user@.service.d/iris-delegate.conf` (sudo) and apply it at the 3.3 reboot. Afterwards `cpu` appears in the user cgroup's `subtree_control`. | P2-R4b flips to PASS | **yes** (same reboot) |
| 3.8 | GPU admission for the mouth and ears too (`gpu_admit.py` ExecStartPre, sized from measured peaks). These are core units, so on refusal they wait rather than fail. | units carry the gate; test that a refused admission logs and retries | restart of each unit |

### B. Dormant Ollama diagnostic helper (§8)

**Already there:**
- Ollama is installed, with models on `/data` and the limits drop-in installed (MAX_LOADED_MODELS=1, KEEP_ALIVE=5m).
- The deterministic CPU supervisor = the watchdog + `health_snapshot.py`.
- Incident bundles provide the evidence.
- The only small general model present is `qwen2.5:0.5b`.
- `iris-little-*` are Iris's own fine-tunes and **must not be used or modified** for this.

| # | Step | Exit check | Zeke? |
|---|---|---|---|
| 4.1 | **Corpus + retrieval, no LLM.** `scripts/server/diag_corpus.py` indexes incident bundles, `docs/`, `V3_PHASE3_4.md`, test names + last results, and selfcheck history into a local index. Embeddings come from the installed `nomic-embed-text` or plain BM25 (CPU). It has a query CLI. | top-5 retrieval returns the right doc for 10 hand-written fault questions | no |
| 4.2 | **Wake gate.** `diag_helper.py wake` checks `gpu_admit.py NEED` (model + KV cache), RAM, the little-brain park flag (`state/little_brain/pilot_deliberately_off.json`), quiet hours and an active speaking window. On a "no" it defers or runs CPU-only. It logs every decision. | unit tests for each refusal reason | no |
| 4.3 | **Model choice.** Pick one small coding/reasoning model that fits in ≤ ~4 GB VRAM *(proposed)*. Free download, zero spend. Pin its digest. | model card + licence reviewed | **decision** (which model; disk on `/data`) |
| 4.4 | **The helper.** It runs as a `worker.py --lockdown helper` unit with RuntimeMaxSec and a MemoryMax. Input: one incident bundle + retrieved context. Output: `state/incidents/<id>/diagnosis.md` with observed / inferred / suggested test. Read-only; it may run only sandbox tests (`fault_inject.py`, `pytest` in a scratch worktree). Over budget → it's killed and the evidence stays. | runs on 3 past incidents; budget-kill test passes | no |
| 4.5 | **Independent verification.** A suggested fix is never applied by the helper. Iris applies it; `health_snapshot` / the named test must flip, and that check — not the helper's claim — closes the incident. Wire it to the act ledger. | one incident closed end-to-end through the ledger | no |
| 4.6 | **Evaluation set.** A held-out set of ≥ 20 labelled past incidents *(proposed)*, with the right diagnosis written by Iris. Score the helper. No fine-tuning until this exists and the base model's score is known (§8). | score recorded | no |

### Phases 3–4 exit gates

- 3.4 PASS on a real reboot.
- 3.6 PASS.
- 3.7 PASS.
- 4.4 + 4.5 done on at least one real incident, with zero changes made by the helper itself.

### Phases 3–4 risks

- A reboot drill on a bad day costs Zeke ~10 min of me being away (the last measured unattended recovery was ~9.5 min, 10-05).
- The `external` drill task must target a test sink, never Zeke's DM.
- The GPU is nearly full. The helper must defer far more often than it runs, and that's correct.

---

## Phase 5 — Experiments (PersonaPlex, A/B, regression gates, cutover)

**Spec:** a PersonaPlex sandbox, A/B comparisons, and performance regression gates. Cut over only after
staged acceptance.

**Already done:** PersonaPlex was measured on the V100 on 10-30.
- It needs 19.5 GB VRAM.
- Each 80 ms frame takes 70.7 ms (88 %); a second run took 83 ms per step, slower than real time, cause unknown.
- Voice-clone similarity was 0.57 / 0.63.
- It **cannot load beside Fish** (free VRAM < 19.5 GB).
- The duplex design recommends A (Fish + duplex coordinator) as the default, with B as a swap-in "call mode" experiment only.

| # | Step | Exit check | Zeke? |
|---|---|---|---|
| 5.1 | **Regression gate suite (do this first; it guards every other phase).** `scripts/server/regress_gate.py` runs `latency_report.py` (stage p50/p90 against the 2.1 reference), `health_snapshot` (no red for 24 h), the memory counts (JSONL + Chroma unchanged except for new appends), the parity soak, and the full `pytest`. Output is PASS/FAIL against stored baselines in `state/v3/baselines.json`. | runs green on today's v2 | no |
| 5.2 | **v2 vs v3 benchmark (§12).** After Phase 1 close-out, run 5.1 for 3 days on v2, then 3 days on v3, with the same day mix as far as possible. | v3 within +10 % on each stage *(spec gate)*, no memory loss | **yes** (stack switch windows) |
| 5.3 | **PersonaPlex step-time regression.** Offline, in a scratch venv and **only with Fish stopped**: find why step time went 58.7 → 83 ms, strip the extra `other_mimi` pass, and re-measure. | step < 80 ms sustained over 5 min, or a written "infeasible" | **yes** — mouth down (~4 min Fish warm-up after) |
| 5.4 | **int8 fit test.** Check whether int8 weights fit beside Fish without breaking `gpu_admit`'s reserve. Estimated ~8 GB, unverified. Only an upstream-supported path; otherwise record "not supported". | measured peak VRAM, or a recorded no | same window as 5.3 |
| 5.5 | **End-to-end client** for "call mode": constrained persona text, Iris's cognition still decides content (§10). PersonaPlex only does turn-taking and speech, or it isn't used. | design note + a working offline loop | no |
| 5.6 | **Blind A/B (§10).** A = Fish + duplex tiers, B = PersonaPlex call mode. Zeke rates naturalness, voice consistency, interruption accuracy and recovery, without knowing which is which. ≥ 10 short sessions each *(proposed)*. | ratings filed | **yes** |
| 5.7 | **Staged cutover.** Make v3 the default only if 5.2 passes and every Phase 2 + 3–4 gate holds. Keep v2 bootable with `stack_switch.sh v2` for ≥ 2 weeks *(proposed)*. Any regress-gate FAIL rolls back. | 2 weeks green on v3 | **yes** — decision |

### Phase 5 exit gates

- 5.1 is in place and green.
- 5.2 meets the spec gate (no memory loss, no worse responsiveness, no unverified repairs).
- 5.6 is decided by Zeke.
- 5.7 has run 2 weeks green.

### Phase 5 risks

- Any PersonaPlex test takes my voice down. Never do it unannounced, and never during quiet hours.
- A/B ratings from one listener are small-sample. Treat them as his preference, which is what matters, not as a general result.
- PersonaPlex generates its own content. Letting it answer would replace Iris's reasoning, which §10 forbids. 5.5 must prove it can't.

---

## Summary: what needs Zeke

| When | What |
|---|---|
| One restart window | 1.A gate install (+ check) |
| Two restarts, while present | 1.D rollback drill |
| A few minutes | 1.E DM-while-down; 2.4 interrupt trials; 2.9/2.10 look-up and defer trials |
| Tower session | 2.5 AEC, 2.6 the speakers question |
| One planned reboot | 3.3 + 3.7 together |
| A GPU window | 3.6, and 5.3/5.4 with the mouth down |
| Decisions | 2.6 headset vs speakers, 2.11 coordinator split, 4.3 model, 5.6 A/B, 5.7 cutover |

Everything else (1.B, 1.C, 1.F, 2.1–2.3, 2.7, 3.1, 3.2, 3.4, 3.5, 3.8, 4.1, 4.2, 4.4–4.6, 5.1, 5.5) is Iris's.
