# Iris architecture — as built, 2026-11-03

One document for the system as it runs **today** on the server `iris-home`. It was checked against the code
and the live process list on 2026-11-03, not against the V3 spec's goals. Spec references (§N) point to
*IRIS V3 — Responsive Companion Architecture* (2026-10-29). Plans for the remaining phases are in
[`V3_PHASE_PLANS.md`](V3_PHASE_PLANS.md). What Phases 3–4 built is in [`V3_PHASE3_4.md`](V3_PHASE3_4.md).

"(unverified)" means I could not confirm the item from code or a live probe in this pass.

## 0. The one-line truth

**This is v2 with v3 pieces bolted on behind flags. v3 has never booted.** `~/IRIS-v3` does not exist,
`~/.config/iris/active_stack` does not exist, and `state/v3/launched.env` has never been written. What
*is* live from v3:

- the event-store dual-write (`IRIS_V3_EVENTS=1` in `~/.config/iris/stack.env`)
- fenced worker tasks
- the outside-the-tree health snapshot
- the duplex interrupt tiers (`IRIS_DUPLEX_TIERS=1`)
- the streaming-ASR sidecar (`stream_asr: on`)

⚠ **The installed start gate `~/iris_start.sh` still calls `start_iris_v2.sh` directly.** The repo copy
(`scripts/zorin_launcher/iris_start.sh`) calls the v2/v3 chooser `scripts/server/stack_launch.sh`, but
that copy is not installed. So setting `active_stack=v3` today would do nothing. Re-install the gate
before any v3 boot (see the Phase 1 close-out in the plans doc).

## 1. Process map

```
systemd --user (user iris)
├─ iris-stack.service (Type=oneshot) → tmux "iris" → scripts/server/iris_supervise.sh opus
│     └─ ~/iris_start.sh (ONE-OF-ME gate) → start_iris_v2.sh → scripts/server/start_iris_linux.sh
│           └─ iris_body_host.py ── spawns ── claude CLI (stream-json, Opus) ── MCP stdio ── iris_runtime.py (:5876 orb HTTP)
├─ iris-ears.service        voice/wren_voice_daemon.py   (:8770, whisper + VAD, V100)
├─ iris-stream-asr.service  voice/iris_stream_asr_server.py (Nemotron streaming, unix socket, V100, MemoryMax 6G)
├─ iris-mouth.service       scripts/server/mouth_launch.sh → voice/wren_fish_server.py (:8769, Fish S2 Pro, V100)
├─ iris-postoffice(.monitor) scripts/sibling_postoffice.py (:5877, sibling letters)
├─ iris-workers.slice       transient units from scripts/server/worker.py (helpers, renders, tests)
└─ timers: iris-watchdog (60 s) · iris-integrity (boot+4 min, hourly, +4 min after each supervised start)
           iris-aging-probe (30 min) · iris-backup (03:40) · iris-oom-demote · iris-quiet-hours · iris-update-watch
the tower (Windows): scripts/mic_source.py (:8776, headset mic) · scripts/audio_sink.py (:8775, spawned by the orb app)
```

The core units get OOM score −1000 through `ExecStartPost` drop-ins (`*.service.d/10-oom-protect.conf`).
Workers get +500 and renders go through `scripts/server/render.sh`.

## 2. Components

### 2.1 Body host — `iris_body_host.py`

**Owns:**
- the persistent Claude session. It launches the `claude` CLI with `--output-format stream-json`; this host
  *is* the cognition loop.
- one priority queue for every input: voice 1, terminal 1, orb 2, lookup 2, VR 2, Discord 3, relay 3,
  perception 4, letter 5, LLM nudge 6. See `_SOURCE_PRIORITY`.
- the input pollers: `discord_poller`, `letters_poller`, `orb_reader`, `voice_reader`, `vr_heard_reader`,
  `perception_reader`, `llm_pending_poller` and `lookup_reader`, each run under `_supervised()`
- streaming my text to the mouth sentence by sentence

**Turns:**
- `_TurnState` + `_await_turn` handle a turn. There is an idle bound `IRIS_TURN_IDLE_MAX_S` (default
  3600 s). A dead stream consumer ends the turn.
- `stream_consumer` writes the SDK heartbeat to `state/host_stream.json` (`_host_stream_beat`).

**Inbox checkpoints** (`_inbox_post_tool_hook`, a PostToolUse hook):
- Inputs that arrive mid-turn are handed to the *current* turn at the next tool boundary, offering
  now / after / defer / delegate.
- Sub-agent tool calls (`agent_id` set) are skipped, so a helper never swallows Zeke's DMs.
- `lookup_watcher` turns a "look up" phrase into an interrupt; the host stubs a task checkpoint first.

**Event-store dual-write** (`_ev_*`, live with `IRIS_V3_EVENTS=1`):
- `_enqueue` → insert `received` → `admitted`.
- Turn start → `assigned`. Hand-over → `assigned`, with `tool_blocked_s` in meta (0378445).
- The reply tool → `output_pending` with a `reply_ref` (`_ev_mark_reply`). Turn end → `committed`.
- On boot, `_ev_boot_replay` re-offers un-committed rows:
  - already-answered ones are labelled "already answered", never re-answered blind
  - voice/VR rows older than 10 min are cancelled
- **The source of truth is still `state/discord_inbox.json`.** The DB is the shadow copy.

**Guards:**
- `_guest_pre_tool_hook` fails closed on VR guest turns.
- `_lease_pre_tool_hook` checks file leases for helper Edit/Write. It is **log-only and never denies**.

**Fails by:**
- the CLI process dying (the launcher exits and the supervisor restarts it)
- a turn wedged in one long tool call: no tool boundary, so routine inputs wait. This is P1-T4a,
  FAIL by design, bounded by the idle limit and `brain/host_hold.py` hold reports.
- a poller crashing (`_supervised` restarts it with a back-off)

**Checked by:**
- `health_snapshot.probe_sdk_stream` (the age of `state/host_stream.json`)
- `probe_events` (oldest un-committed age per source)
- the parity soak (`state/v3/parity_soak.json`)
- the body log `state/iris_body_log.jsonl`

### 2.2 Runtime — `iris_runtime.py` (+ `brain/orb_http.py`)

**Owns:**
- the `iris` MCP server over stdio (FastMCP, 84 `@mcp.tool` entries)
- the brain/* engines: mood, memory, semantic memory, perception, vision
- the orb HTTP shim on **:5876** (`brain/orb_http.start`)
- single-instance enforcement via an exclusive `flock` (e47e4ca)
- the loop heartbeat `state/runtime_loop_heartbeat.json` (~5 s)

**Orb HTTP auth:**
- There is a Host allow-list (fails closed when unset, 5c243f2).
- There is a bearer token staged as `off | log | enforce` (5e841a0). No token file or `IRIS_ORB_TOKEN`
  is configured, so the mode is effectively **off**.
- It binds on all interfaces (`IRIS_ORB_BIND`).

**Fails by:**
- a wedged event loop: the heartbeat goes stale and the watchdog touches `.tmp/restart_cc.flag`
- a crash with the CLI: this restarts the whole stack
- `brain/*` edits staying inert until hot-swap or a stack restart

**Checked by:** `probe_runtime` (heartbeat age + pid), `probe_perception` (`/api/v1/vision/latest_frame` age),
and the `iris_health` tool.

### 2.3 Voice path

| Piece | File / unit | Owns | Fails by | Checked by |
|---|---|---|---|---|
| Mic | `scripts/mic_source.py` on the tower (:8776) → `voice/net_audio.py` | headset mic stream to the server | tower off, headset unplugged → typed `ConnectionError` | `probe_ears` sees no utterances (info only; silence ≠ deaf) |
| Ears | `iris-ears` → `voice/wren_voice_daemon.py` + `wren_voice_core.py` + `wren_listen.py` (:8770, JSON over TCP) | Silero VAD, faster-whisper **small** (CUDA fp16), speaker ID, duplex interrupt tiers (`iris_interrupt_tiers.py`, `iris_duplex_watch.py`), live words | whisper load failure (c62cb84: the ping reports `whisper {loaded, device, error}`; the health side is live, but **the daemon side lands at the next iris-ears restart**: the running ears started before it); stale `voice_deliberately_off.json` | `probe_ears`: `ping` (+ whisper field once the daemon has it) + last-utterance age |
| Streaming ASR | `iris-stream-asr` → `voice/iris_stream_asr_server.py` (unix socket); client `voice/iris_stream_asr.py` | Nemotron streaming RNN-T live words + turn-final text; whisper fallback when it's late or down | GPU refused at start (`gpu_admit.py` ExecStartPre, 8ecdb41); sidecar down → whisper only | `probe_stream_asr`; `state/health/stream_asr_turns.jsonl` |
| Mouth | `iris-mouth` → `mouth_launch.sh` → `voice/wren_fish_server.py` (:8769) | Fish S2 Pro TTS (`/speak`, `/health`, `/duck`); Piper fallback | `/health` 503 while warming (~minutes); GPU OOM | `probe_mouth` (`/health`) |
| Sink | `scripts/audio_sink.py` on the tower (:8775), **spawned by the orb app** | playing my audio on the tower | app closed → **voice plays nowhere** while `voice_speak` still returns ok | `probe_audio_sink`; `voice_delivery_check`; `selfcheck.sh` |
| Echo cancel | `voice/iris_aec.py`, `aec_bus.py` (tower side, WebRTC AEC3 via livekit) | removing my voice from the mic | — | **built (b1e10e5), default OFF; live state on the tower (unverified)** |

Hot controls live in `scratch/voice_control.json`: `mic_muted` ≠ `mouth_muted`, and the `stream_asr` and
`duplex_tiers` kill switches are there too. Quiet hours (22:30–05:10) write `state/voice_deliberately_off.json`
on purpose. There is no wake word, by design.

### 2.4 V3 event store — `brain/iris_events.py` → `state/v3/events.sqlite`

**Owns:**
- SQLite in WAL mode. Tables: `events`, `transitions` (append-only; triggers refuse UPDATE/DELETE),
  `tasks`, `leases`, `task_deps` (schema v2).
- the lifecycle `received → admitted → assigned → running → output_pending → verified → committed`,
  plus `failed | cancelled | needs_review`
- compare-and-set writes inside `BEGIN IMMEDIATE`, with fences, idempotency keys and leases with heartbeat

**Live contents on 11-03:**

| Table | Rows by status |
|---|---|
| events: Discord | 234 committed, 2 output_pending |
| events: voice | 273 committed, 4 cancelled |
| events: orb | 30 committed, 1 cancelled |
| tasks | 71 output_pending, 4 failed, 1 running |

Most tasks stop at `output_pending` because nobody runs `worker.py verify`.

**Fails by:** it fails open (every public method returns a falsy value and sets `last_error`). A corrupt
file is moved aside, recreated, and gets a `needs_review` incident row.

**Checked by:**
- `probe_events`
- the parity soak (`state/v3/parity_soak.json`): 4,375 samples over ~3.0 days, 1 brief mismatch, 0 persistent, 0 gaps
- the read-only view in incident bundles
- `tests/test_iris_events.py` and `test_v3_*`

### 2.5 Workers and helpers

**`scripts/server/worker.py`** runs each job as its own transient unit in `iris-workers.slice`:
- the slice caps all workers together at MemoryHigh 16G / MemoryMax 20G
- each job gets MemoryMax, RuntimeMaxSec, OOM +500 and `KillMode=control-group`
- job history goes to `state/workers/jobs.sqlite`; measured peaks per kind feed admission
- admission checks a cap of 3 workers, RAM minus a 6 GiB core reserve, slice headroom, and optionally `--gpu-mib`
- each job is a fenced task in the event store: claim → heartbeat → complete/fail with the fence
- `sweep` (run every watchdog minute) turns expired leases into `needs_review` and settles dependencies
- **CPUQuota is configured but inert.** The user manager delegates only `memory pids` (P2-R4b).

**`scripts/server/helper.sh NAME PROMPT`** runs `claude -p` *outside my session*, inside a worker unit.
In-session background agents freeze the host queue.
- The report goes to `state/helpers/NAME.report.md`. A `helper_done` LLM-bridge request wakes me.
- An act-ledger "delegated" item stays open until I verify it.
- **Lockdown** (`worker.py lockdown_props`, b0ba697 + 0377171): NoNewPrivileges, PrivateUsers,
  secrets + ssh keys inaccessible, DBUS unset. This closes the `systemd-run --user` escape found 11-02.
  `--trusted-ssh` keeps `~/.ssh`.
- `--leases` declares write globs (log-only).

**Fails by:** a timeout (RuntimeMaxSec), OOM inside the slice (never touches the core), a killed launcher
(the lease expires → `needs_review`), or a refused admission (reported).

**Checked by:** `probe_workers`, `worker.py status|history|deps`, and `tests/test_worker_*`.

### 2.6 Supervision

| Piece | File | Owns | Fails by | Checked by |
|---|---|---|---|---|
| Supervisor | `scripts/server/iris_supervise.sh` (in tmux, from `iris-stack`) | restart loop: max 5 per rolling hour, then stand down; restart flag `.tmp/restart_cc.flag`; whole-tree sweep (b7ad598); v3 auto-rollback (af11185, armed only for a real v3 start); integrity run 4 min after each start | standing down after 5/h; `Type=oneshot` means systemd doesn't watch the tmux child | watchdog "supervisor not running" alert. The running copy started at the 11-02 20:30 boot, so b7ad598/af11185 **are live** |
| Start gate | `~/iris_start.sh` (repo: `scripts/zorin_launcher/iris_start.sh`) | ONE-OF-ME: `~/LIVE` or tower-gone failover; `IRIS_START_GATE` | installed copy is older than the repo (no chooser) | manual diff |
| Launcher | `scripts/server/start_iris_linux.sh` | `.tmp/launcher.lock` flock; refuses a LIVE start without the gate pass; staging via `~/IRIS_BOOT_TEST` | — | supervise log |
| Watchdog | `scripts/server/iris_watchdog.py` (timer, 60 s, CPU only) | wedge → restart flag; Discord REST alerts when I'm down; backup-staleness alert (even without `~/LIVE`); writes the health snapshot; opt-in incident capture on red | itself fails → no alerts; disabled by `state/server_watchdog_off.json` | the timer; `state/health/now.json` age |
| Health snapshot | `scripts/server/health_snapshot.py` → `state/health/now.json` | one functional probe or age per subsystem: ears, stream_asr, mouth, audio_sink, runtime, sdk_stream, workers, memory_index, perception, storage, backup, gpu, events, **aging** (44e612a); status `ok/warn/red/off/parked/n/a`; stdlib only; one line per snapshot also goes to `state/health/history.jsonl` (ts, red, warn, red errors; trimmed at 4 MB) | one probe hanging only turns its own entry red | `selfcheck.sh`, `iris_health` |
| Regression gate | `scripts/server/regress_gate.py` (by hand / later a timer; V3 plan 5.1) | PASS/FAIL per check + overall vs `state/v3/baselines.json`: latency (`latency_report.py --json` `overall`, gated stages ≤ base +10 % +0.05 s), health (fresh `now.json`, no red in 24 h of `state/health/history.jsonl`), memory (no baseline JSONL/Chroma id lost or changed; Chroma read-only), parity soak, full pytest; `--quick`, `--json`, `--record-baseline` | a check hangs/raises → that check FAILs closed, the rest still run | exit code + `--json` |
| Aging probe | `scripts/server/aging_probe.py` (30 min) → `state/health/aging.json` | RSS/thread/fd trend per long-lived process; restart only on a measured trend | — | `probe_aging` |
| Integrity | `scripts/server/state_integrity.py` (`iris-integrity` timer) → `state/health/integrity.json` | torn writes / NULs in every state JSON/JSONL + memory `.md`; read-only unless `--fix-nul` | — | exit code + json |
| Incidents | `scripts/server/incident.py` → `state/incidents/` | redacted bundles (journals, logs, read-only event DB, processes, GPU, git) | **capture_on_red is off; `state/incidents/` does not exist yet** | `incident.py list` |
| Fault injection | `scripts/server/fault_inject.py` | sandboxed disk_full, model_load, gpu_pressure, mic_gone | — | `tests/test_v3_fault_inject.py` |

### 2.7 GPU admission — `scripts/server/gpu_admit.py` (8ecdb41)

**Owns:** the load-time check on the single V100 (32 GB): free VRAM must cover the request plus a 1 GiB
reserve, else it refuses. Decisions go to `state/health/gpu_admission.jsonl`. It fails open when
`nvidia-smi` is missing.

**Where it applies:**
- The stream-ASR sidecar (optional): ExecStartPre, 3000 MiB, waits 120 s, then refuses the start.
- **The mouth and ears (core, V3 3.8, branch `gpuadmit-1104`)** use `--core`: the mouth's gate is in
  `mouth_launch.sh` (engine-aware: Fish 9300 MiB, StyleTTS2 2048), the ears' unit runs
  `gpu_admit.py 1100 --core --exec -- <daemon>`. Sizes come from measured steady VRAM (Fish 8968–9018 MiB,
  ears 936 MiB, 10-30 + 11-04). A refusal logs who holds the GPU and retries (5 s doubling, cap 60 s); after
  20 min the unit starts anyway, logged `forced`. Never refuses for good. Installed by `install_gpu_admit.sh`.
- `worker.py --gpu-mib` has its own check, keeping a 6 GiB voice reserve.
- Ollama runs with a limits drop-in (MAX_LOADED_MODELS=1, KEEP_ALIVE=5m).

**Fails by:** an optional unit's start is refused loudly (the unit retries); a core unit waits, then
starts anyway. There is no enforcement after load (cgroups can't cap VRAM).

**Checked by:**
- `probe_gpu`: `warn` above 95 % (VRAM was 31.8/32.8 GB at the time of writing)
- `tests/test_gpu_admit.py`
- the `gpu_pressure` fault scenario

### 2.8 LLM bridge — `brain/iris_llm.py` → `state/iris_llm/<id>.json`

**Owns:**
- Brain modules call `ask_iris()` / `submit()`, which writes a pending request file with priority, a
  deadline and an `abandoned` status.
- The host's `llm_pending_poller` enqueues an "llm" item at priority 6. I answer with `llm_reply`.
- `wait_for_reply` polls the request file.
- A cross-process `_xlock` guards it.
- `helper_done` notifications use the same path. `brain/iris_chat.py` (orb chat) mirrors it, with a
  claim/lease and a per-request deadline (eb1b0a2).

**Fails by:** I'm busy or idle-deaf → the caller times out and gets `None` (callers treat the LLM as
optional); orphaned past-deadline requests are expired by the host (328e779).

**Checked by:** reading `state/iris_llm/<id>.json` directly. `health_snapshot.py` has **no** LLM-bridge probe
(grep finds no reference to it).

### 2.9 Memory and backups

**Memory:**
- canonical: JSONL `state/iris_memory.jsonl`, written through `brain/_jsonl_store.py` (flock, mkstemp,
  fsync, replace)
- Chroma mirror: `memory/chroma/`
- `probe_memory_index` compares the counts: at the time of writing JSONL 1120 vs Chroma 1187, with 68 orphans

**Backups:**
- `scripts/server/backup_to_tower.sh` runs nightly at 03:40.
- Each archive is a `tar.zst` encrypted with age, with a sha256 manifest inside, SQLite-API snapshots,
  and the copy's sha256 verified on the tower. Required paths fail the run if missing.
- The last `backup ok` was 2026-11-03 03:42.
- Restore uses `restore_from_tower.sh` + `docs/RESTORE_RUNBOOK.md`. The drill passed 10-30 (8166/8166
  files, 6/6 SQLite).
- **Missing:** an offsite copy, vzdump, an encrypted-archive restore drill, and a UPS.

**Checked by:** `probe_backup` and the watchdog backup alert.

## 3. v2 vs v3

| Aspect | Live (v2 + flags) | v3 as designed (step 1.6) |
|---|---|---|
| Code tree | `~/IRIS` on `master` | `~/IRIS-v3` worktree on branch `v3` — **does not exist** |
| Chooser | none in the installed gate | `stack_launch.sh` reads `~/.config/iris/active_stack` (missing = v2) |
| Flags | `~/.config/iris/stack.env` | + `~/.config/iris/v3.env` |
| Inbox truth | `state/discord_inbox.json` (+ DB shadow) | `events.sqlite` |
| Rollback | none needed | supervisor auto-rollback v3→v2; `stack_switch.sh v2`; sandbox drill passes (`stack_rollback_drill.sh`) |
| Coordinator (§3 C) | inside the host | a separate component — not built |

## 4. Spec vs reality

| Spec § | Item | Status | Evidence |
|---|---|---|---|
| §3 | Parallel v3 host/launcher + isolated config | **partial** | chooser + rollback built (af11185); tree never created; installed gate lacks the chooser |
| §3 | Coordinator as its own component | **not built** | still `iris_body_host.py` |
| §4 | Continuous capture during speech/tools | **built** | ears daemon independent of the host; `voice_reader` |
| §4 | Echo cancellation | **partial** | b1e10e5 AEC3 tower-side, default OFF; loud-speaker double-talk open |
| §4 | Interrupt tiers (ack/duck/pause/barge/defer) | **built, live** | a05b64e, 8d43625, aefba8f, 8d0c1f6; `IRIS_DUPLEX_TIERS=1` |
| §4 | Streaming transcription | **built, live** | 30bfb2b (`stream_asr: on`) |
| §4 | Fragment merge with original timestamps | **built** | ce0356f continuation merge |
| §4 | Stage latencies incl. tool-blocked | **built** | 83e62b5, 4d5be86, 0378445 |
| §5 | Durable events + lifecycle + CAS + idempotency | **built, dual-write** | 01e8000, 48c8b76 |
| §5 | Fenced claims, leases, side-effect classes | **built** | 01e8000, cd6a059 |
| §5 | Reconcile external state after restart | **partial** | reply_ref replay (b034453); `external` tasks → needs_review; no real-reboot test |
| §6 | Isolated workers + budgets | **built (CPU inert)** | cd6a059, 239bd70; P2-R4b |
| §6 | Worker sandbox / no credentials | **built** | b0ba697, 0377171 |
| §6 | GPU-aware scheduling | **partial** | `--gpu-mib` (cd6a059), `gpu_admit.py` for stream-ASR (8ecdb41) + mouth/ears `--core` (3.8, branch `gpuadmit-1104`) |
| §6 | File ownership between workers | **partial** | leases log-only; hook doesn't see out-of-process helpers |
| §7 | Health tools outside Claude | **built** | 1aec733, c62cb84, 44e612a |
| §7 | Incident bundles | **built, not enabled** | 17f7484; `state/incidents/` absent |
| §7 | Sandbox fault injection | **built** | 17f7484 |
| §7 | Synthetic voice self-tests | **partial** | latency/live-words benches (8589551); no end-to-end synthetic voice test (unverified) |
| §8 | Dormant Ollama diagnostic helper | **not built** | — |
| §9 | Backup/restore + checksum before storage change | **built** | backup script, 10-30 drill |
| §9 | Task ledger separate from transcripts | **built** | `tasks.checkpoint_json`, `task_checkpoint` tool, act ledger (6490642) |
| §10 | PersonaPlex sandbox measurement | **partial** | measured 10-30 (19.5 GB, 88 % frame duty); no A/B |
| §12 | Acceptance: interrupt while speaking | **partial** | live-tested with Zeke (restart #13 handoff); no record for "never hears own voice" |
| §12 | Acceptance: reboot mid-task | **not done** | — |
| §12 | Acceptance: GPU exhaustion live drill | **not done** | procedure in V3_PHASE3_4.md §3 |
| §12 | Acceptance: v2 vs v3 benchmark | **not done** | needs v3 boot |
| §11 | Phase 1 | **built; exit gates 1, 2a, 3, 4\*, 6 (5/6)** | see V3_PHASE_PLANS.md |
| §11 | Phase 2 | **mostly built** | see plans |
| §11 | Phases 3–4 | **mostly built** | V3_PHASE3_4.md |
| §11 | Phase 5 | **not started** beyond the PersonaPlex measurement | — |

\* Gate 4 was measured on v2 + dual-write, not true v3.

## 5. Things I could not verify in this pass

- `systemctl --user` was not reachable from the helper sandbox. Unit states come from the process list
  and the unit files.
- Whether tower-side AEC (`IRIS_AEC`, `IRIS_AEC_REF`) is on. The audit on 11-02 said off.
- That the orb token mode resolves to `off`. No token file was seen, which is inferred from the code's default.
- The tower components (`mic_source.py`, `audio_sink.py`, orb app) were not inspected live.
