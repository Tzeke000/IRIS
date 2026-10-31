# VR bot sandbox (audit H2), prepared 2026-10-30, NOT applied

**Order of work:**
1. In the game repo, `git apply game_code_file.patch`, then rebuild and deploy the bot (`vr.sh parse` first). The patch adds `--code-file=` and `IRIS_MOOD_FILE`.
2. Run `./apply.sh --dry-run`, then `./apply.sh`. The body is offline for about 10–30 s. If it fails to join within 90 s, apply.sh rolls back on its own.
3. Run `./verify.sh`. Add `--wake-test` to push one labelled fake gesture into heard.jsonl. That line wakes me.
4. Commit the reader changes that `apply.sh` applied to the IRIS working tree (`iris_readers.patch`).
5. Undo with `./rollback.sh [--purge]`.

**Files:**

| File | Role |
|---|---|
| `iris-vr-presence.service` | The system unit, and the single source of truth for the sandbox. |
| `egress_rules.sh` | Per-uid iptables rules. |
| `test_sandbox.sh` | Boots the bot in a throwaway sandboxed copy (no `--join`, ctl port 17790). |
| `test_probe.sh` + `probe.sh` | Filesystem and network checks run from inside the same sandbox. |
| `lib.sh` | Turns the unit's sandbox block into `-p` arguments. |

**Paths:**

| Path | Contents |
|---|---|
| `/data/irisvr/hear` | Opus spool. The bot writes it; `vr_hear` reads and deletes it. |
| `/data/irisvr/gestures` | Gesture wakes. `vr_hear` relays them into heard.jsonl. |
| `/data/irisvr/voice` | My TTS wavs, reached through the host link `/tmp/iris_vr_voice`. |
| `/data/irisvr/in` | Mood mirror, kept current by user unit `iris-vr-mood-mirror.path`. |
| `/var/lib/irisvr` | The bot's HOME and Godot `user://`. |
| `/etc/irisvr/` | Invite credential and egress env. Root only, never mounted into the bot. |
