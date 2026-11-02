# V3 Phases 3 + 4 — task dependencies, incident bundles, fault injection

Built 2026-11-02 on top of the Phase 1 pieces (event store `brain/iris_events.py`, `scripts/server/worker.py`,
the watchdog's `state/health/now.json`). Everything here is opt-in: with no `after`, no flag file and nobody
running the harness, behaviour is unchanged.

## 1. Task dependencies (Phase 3)

- A task declares `after: [task_ids]`: `iris_events.create_task(..., after=[...])`,
  `worker.py run --after T1,T2 ...`, or `worker.py task new --goal G --after T1`.
- **The gate is in the store** (`claim()`), so every claimer respects it: a task can't be claimed until every
  parent is **verified or committed**. A worker's own `ok` only reaches `output_pending` ("a result exists").
  Someone has to check it first: `worker.py verify T --ref <act id>`.
- A parent that is `failed`, `cancelled` or missing, or a cycle, makes the dependant **`failed`** with
  the note `not started: dependency X is failed (…)`. This runs in `settle_dependencies()`, which
  `worker.py sweep` calls every watchdog minute. Requeueing a parent and then the child re-evaluates the gate.
- Cycles: creation can't make one, because parents must already exist. `add_dependencies` refuses any edge
  that would close one, and a hand-made cycle is reported `blocked` and settled to `failed`.
- **Restart-safe:** the wait lives only in `state/v3/events.sqlite`. Kill `worker.py run --wait-deps` at any
  point, re-run with `--task-id T` or the same `--key`, and it resumes. No duplicate task gets created.
- Exit codes: 75 = still waiting, 65 = a dependency failed.
- `worker.py deps T` prints `ready | waiting | blocked` and the reason.
- Schema v2 adds the `task_deps` table. A v1 DB gains it the first time new code opens it, and old code
  keeps working: it only warns that the schema is newer.

## 2. Incident bundles (Phase 4)

`scripts/server/incident.py capture REASON [--subsystem S] [--force]` · `list` · `show ID [--file F]`

Each bundle is written to `state/incidents/<ts>_<slug>/`, with mode 0700 for the directory and 0600 for the files. It contains:

- `manifest.json`
- the health `now.json` and `integrity.json`
- journals for `iris-stack`, `iris-mouth`, `iris-ears` and `iris-watchdog`, plus the subsystem's own unit
- the supervise and watchdog log tails
- a **read-only** view of the event store (`mode=ro`). It never goes through `EventStore`, whose
  corruption recovery would move the file.
- processes ranked by RSS
- GPU, disk and memory
- git HEAD and the dirty files

Limits and masking:

- **Caps:** 256 KiB per file (the newest part is kept), 4 MiB per bundle (later files are skipped and listed),
  and the newest 100 bundles.
- **Masking:**
  - the privacy-guard list and its generic patterns
  - token shapes
  - every *value* in `~/.iris_private.env` and `config/private.local.json`. Those files are read in memory and never copied.
- **Disk full / read-only `state/`:** the half-written bundle is removed and the bundle goes to
  `/dev/shm/iris-incidents` instead (RAM, so it's lost on reboot). The manifest records this as `primary_failed`.
- **`--subsystem` rate limit:** one bundle per subsystem per hour. `--force` overrides it.

**Watchdog (opt-in):** when a subsystem in `now.json` is `red` (not `warn`, not during boot grace, and only with
`~/LIVE`), the watchdog starts `incident.py capture` in its own transient unit (`systemd-run --user`,
RuntimeMaxSec=120), at most once per subsystem per hour. **Off** unless
`state/incidents/capture_on_red` exists (`touch` to enable, `rm` to disable) or `IRIS_INCIDENT_CAPTURE=1`.

## 3. Fault injection (Phase 4)

`scripts/server/fault_inject.py run disk_full|model_load|gpu_pressure|mic_gone|all [--keep]`

Every scenario runs in a fresh `/tmp/iris-fi-*` staging root:

- `IRIS_ROLE=staging`, with CUDA hidden and Hugging Face offline
- no live ports; voice-status writes are redirected into the staging root
- the child process refuses to run anywhere else

Each scenario asserts three things: the system **degrades**, nothing **crashes**, and an **incident is recorded** through the real
`capture_red_incidents → incident.capture` path.

| Scenario | Fault | Asserted |
|---|---|---|
| disk_full | 1 MiB tmpfs over `<staging>/state` inside `unshare -rm` (no sudo; only that namespace sees it), filled to ENOSPC | the event store fails open (insert → None, `database or disk is full`) and recovers when space returns (integrity ok); storage probe red; `write_health` contains it; the bundle lands in the fallback dir, no half-written `.tmp_` dir |
| model_load | whisper model path doesn't exist; smart-turn model missing | `_warm_whisper` logs FAILED and returns; `_rich_health` → `smartturn` degraded; incident |
| gpu_pressure | fake `nvidia-smi` reports 31.3/32.5 GiB | `worker.py admit --gpu-mib 4096` defers ("GPU free … < 4096"); gpu probe `warn`; the watchdog does **not** auto-bundle a warn, so the harness captures one explicitly |
| mic_gone | `IRIS_MIC_SOURCE` → a dead port; a source that dies mid-stream | typed `ConnectionError` fast (no hang); the daemon's real `_handle_connection` answers `{"ok": false}` and still pongs on the same connection; ears probe red against a dead staging port; incident |

### Gaps the harness found (not fixed here: they're decisions)

- **A failed whisper load doesn't show in health.** The ears daemon still answers `ping`, and `probe_ears` only
  pings, so health stays `ok` while the ears can't transcribe. Possible fix: have `probe_ears` call `status` and turn red on a failed
  load. That needs a design decision because it changes the live verdict.
- **GPU exhaustion is only ever `warn`**, so the watchdog never bundles it. That's by design in `health_snapshot`.

### Procedures that need the REAL stack (deliberately not automated)

1. **Real GPU hog beside a speaking mouth** (spec §12, "Exhaust GPU headroom"). It shares the live V100 with
   the mouth and ears, so run it only in a window Zeke OKs, with Iris watching:
   - `nvidia-smi --query-gpu=memory.free --format=csv`, then pick N so free − N stays ≥ 8 GiB.
   - Start a capped hog in the worker slice:
     `worker.py run --kind fi_gpu --name hog --gpu-mib N --timeout 120 -- python -c "import torch,time; x=torch.empty(N<<20,dtype=torch.uint8,device='cuda'); time.sleep(90)"`.
   - While it runs, ask Iris to speak, and check that `worker.py admit --gpu-mib 4096` defers.
   - Expect the mouth to stay fluent, the hog to die at RuntimeMaxSec, and the VRAM to come back.
   - If the mouth stutters, `worker.py stop hog`.
   - Capture with `incident.py capture "gpu hog drill" --subsystem gpu --force`.
2. **Real headset unplug / tower mic source down.** Unplug the headset, or stop `mic_source.py` on the tower:
   - Expect ears listen calls to return errors, not hang.
   - `probe_ears` stays `ok` (the daemon still pongs). The same gap as above.
   - Plug it back in: the next listen works without a restart.
3. **Full ears daemon on a staging port** (`WREN_VOICE_DAEMON_PORT=18770`, `IRIS_EARS_ON_BOOT=0`). Not
   automated, because the daemon's status and live-words files are written relative to the repo, and from
   `~/IRIS` that's the live orb state. Do it on VM 104 or a scratch clone only.
