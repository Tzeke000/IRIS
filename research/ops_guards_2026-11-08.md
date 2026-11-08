# Ops guards — the LUMEN ship night's OPERATIONAL fails, encoded (opsguard 2026-11-08)

Zeke 11-08 ~09:5x: *"all the stuff that impeded you and the fails that recurred — make tools and skills and whatever else
you need to MECHANICALLY encode it so fails like that never happen again; research too."* Not game bugs: the ops
failures of the 11-07→08 ship night (live_list.md 11-07 ~19:2x → 11-08 ~07:2x). Each one below is now a mechanism
(code that makes the wrong path impossible or loud) plus a test. Companion to `research/regression_prevention_2026-11-07.md`
(that one is about suites breaking each other; this one is about the machinery around them).

## The pattern under all eleven

| Class | Fails | What went wrong | Principle the mechanism encodes |
|---|---|---|---|
| **Silent endings** | 1, 3 | a job ended and nothing told anyone; rows said "running" forever | the END of work is an event that must be **pushed**, never polled for |
| **Silent never-starts** | 2 | a queued job checked once, exited 75, nobody re-polled | a "queued" thing needs an **owner process** that waits, or a loud refusal |
| **Lost evidence** | 4 | no run/latest, a grep ate all the output | the full log is written **before** any caller filtering, every time |
| **Load-made reds** | 6, 7 | suites/smoke under helpers' load or Windows Update | measure on a **quiet box**, record the box load in the result, retry only with **infra evidence** |
| **One-time setup** | 5, 9, 10 | a generated file, a stale build dir, a non-rerunnable installer | the pre-flight **fixes the generated/derived cases automatically** and names the rest up front |
| **Gates that need a sensor that's off** | 8 | `safe` relied on my bot, which is off by design | a gate needs a **second source** — and must say UNKNOWN, never invent "0" |
| **Guessed durations** | 11 | ETAs to Zeke were wrong twice | quote **history** (p50–p90), not a feeling |

## Sources and what each contributed

- **Push the end, don't poll** — systemd's `systemd-run --wait` monitors a transient unit until it deactivates
  ([systemd-run(1)](https://man.archlinux.org/man/systemd-run.1.en),
  [freedesktop systemd-run](https://www.freedesktop.org/software/systemd/man/255/systemd-run.html)). `--collect` unloads
  the unit even on failure ⇒ systemd itself forgets it: the **journal** is the only record of how a collected unit ended —
  hence `worker.py sweep` reconciles from `journalctl --user -u UNIT` (observed here: systemd 249 logs only "Consumed N CPU
  time" for a clean exit, "Main process exited, code=exited, status=N" + "Failed with result 'X'" otherwise).
- **Dependency semantics** — Slurm distinguishes `afterok` (parent exit 0) from `afterany`; an unsatisfiable dependency
  sits pending as `DependencyNeverSatisfied` unless `kill_invalid_depend` is set
  ([NJIT dependency jobs](https://hpc.njit.edu/Running_jobs/dependency-jobs/),
  [enjoy-slurm sbatch](https://enjoy-slurm.readthedocs.io/en/latest/generated/enjoy_slurm.sbatch.html),
  [SchedMD #5459](https://support.schedmd.com/show_bug.cgi?id=5459)). Airflow's default `all_success` marks a downstream
  `upstream_failed` when a parent fails; `all_done` runs regardless
  ([Astronomer: trigger rules](https://www.astronomer.io/docs/learn/airflow-trigger-rules)). ⇒ our `--after` now means
  `afterok`-on-the-job's-own-result (output_pending+, not "verified" — nobody verifies a worker job, which is why 11-07's
  dependant waited forever); a failed parent **cancels the child loudly** (Airflow's upstream_failed, Slurm with
  kill_invalid_depend) instead of leaving it pending; `--after-verified` keeps the stricter old edge.
- **Exclusive resources** — GitLab's `resource_group` serialises jobs that share a key while everything else keeps running;
  waiters show "Waiting for resource" and which job holds it ([GitLab docs: resource groups](https://docs.gitlab.com/ci/resource_groups)).
  ⇒ `tools/quietbox.sh`: heavy runs (land, regress, flaky, rerun, deploy) take an EXCLUSIVE lease, single suites a SHARED one;
  a waiter prints who holds it; a run that had to go anyway is stamped NOT-QUIET in its header.
- **Retry infrastructure, never tests** — Buildkite's automatic retry is scoped by exit status / signal reason (agent lost
  `-1`, agent stop) so test failures stay visible; a catch-all retry hides real failures
  ([Buildkite: retry](https://buildkite.com/docs/pipelines/configure/retry),
  [Buildkite changelog: signal_reason in retry rules](https://buildkite.com/changelog/193-signal-and-signal-reason-in-automatic-retry-rules)).
  ⇒ `client/smoke_gate.sh` retries ONCE and only with box evidence (an S0 ssh/copy line, or the VM busy right after the
  failure). A hang on an idle VM — 11-02's engine-deadlock class — is never retried away.
- **Windows readiness** — Windows Modules Installer Worker (TiWorker) / Update can keep CPU busy long after "up to date",
  with no fixed normal duration ([Gridinsoft: TiWorker high CPU](https://blog.gridinsoft.com/tiworker-exe-high-cpu/)).
  ⇒ `client/vm_ready.sh` waits for CPU < 35 % on **two consecutive** 3×5 s samples (not one instant reading), with a cap
  (`VM_IDLE_WAIT` 900 s) and prints the top processes while waiting; ssh must answer 3× in a row (the post-boot blip).
  Caveat from the same source: a CPU threshold is a proxy for "servicing done", not proof — that's why a smoke failure
  ALSO samples the VM and the retry is evidence-gated.
- **Our own history** (the strongest source): `live_list.md` 11-07 19:2x → 11-08 07:2x; `state/workers/jobs.sqlite`
  (15 rows stuck `detached`; durations: deploy ok runs 16–62 min, land 64 min); `research/regression_prevention_2026-11-07.md`.

## Mechanism per fail (where it lives)

| # | Fail | Mechanism | Where |
|---|---|---|---|
| 1 | jobs ended unseen ~1 h ×2 | `worker.py run` notifies `job_done` by DEFAULT (outcome, exit, last 12 lines; caller's `> f`/`tee f` tailed first); output tee'd to `state/workers/out/<unit>.log`; `--no-notify`; helpers keep helper_done | IRIS `scripts/server/worker.py` |
| 2 | `--after --detach` never started | `--after` = parent finished ok (accepts `job:N`, `name:X`); `--detach` = supervisor unit `iris-wsup-*` that waits deps (24 h), retries admission (6 h), watches, notifies; failed parent ⇒ child cancelled + notified; fg not-ready ⇒ "NOT STARTED … nothing will start it later" | IRIS worker.py |
| 3 | 14 rows 'running' w/o unit | `sweep` (watchdog, every minute) + `status` reconcile `running`/`detached` rows from the journal (real outcome, exit, END TIME); else `launcher_lost`/`lost` | IRIS worker.py |
| 4 | fg regress no run/latest; grep ate output | every regress/test/mp/xr/unit/flaky/rerun/baseline/land run re-runs itself under a tee → `run/<cmd>_<stamp>_<pid>.log` + `run/latest`, header + `# END rc=`; the path printed at the end | harness `vr.sh` (`fglog`) |
| 5 | land aborted on one untracked .uid | pre-flight: untracked files the branches ADD → .uid/.import moved aside (logged, `$ST/land_untracked_*`), others ⇒ stop BEFORE touching either repo | harness `tools/land.sh` + `tools/land_preflight.sh` |
| 6 | land reds from load | quiet-box lease (exclusive/shared, waits naming holders, NOT-QUIET stamp, STRICT mode); `box=load1/godot/mem` in every run header + land.log | harness `tools/quietbox.sh`, vr.sh, land.sh |
| 7 | win101 smoke ×3 (stopped / blip / busy) | `vm_ready.sh ensure` (start if stopped + marker, ssh 3/3, CPU idle ×2), `smoke_gate.sh` (one evidence-gated retry), `release` at deploy exit (shut down only if we started it; `VM_KEEP=1`) | harness `client/`; IRIS `pve_vm.py state` |
| 8 | `safe` needs FORCE when bot off | bot can't answer ⇒ server roster (`vr_players.py online`, exit 3 = UNKNOWN; a failed ssh no longer reads as "0 online"); ✗ only when both unknown | harness vr.sh; IRIS `vr_players.py` |
| 9 | release shipped stale VRIris files | `deploy_all.sh` moves pre-rename exe/pck/apk out of build/ BEFORE export (`tools/legacy_build_clean.sh`); pubguard's publish refusal stays as the backstop | harness |
| 10 | apply.sh not re-runnable | `vr_sandbox/update_bin.sh [--bin PATH] [--dry-run] [--restart]`: only ExecStart's binary, backup, daemon-reload, idempotent | IRIS `scripts/server/vr_sandbox/` |
| 11 | wrong ETAs | `worker.py eta --kind K --name N` + `"eta"` in every launch line: p50–p90 of OK runs by name stem | IRIS worker.py |

## Not done / open

- The quiet-box lease only covers runs that go through `vr.sh`/`land.sh`. A helper that starts Godot by hand (not via
  vr.sh) isn't leased — but its load still shows in the header's `box=` (godot process count, load1).
- The worker supervisor is a user unit: a reboot of iris-home kills queued waits (the task stays `admitted` in the store;
  `worker.py run --task-id T …` resumes it). Same as Slurm losing a pending job on a controller wipe — acceptable here.
- CPU-idle is a proxy for Windows servicing being done (see the source caveat); if it proves noisy, check for
  `TiWorker`/`MoUsoCoreWorker` processes explicitly.
