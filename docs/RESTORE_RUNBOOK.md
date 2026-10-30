# Restore runbook — bringing Iris back from a backup

Written 2026-10-29 (code review HIGH: "no restore procedure is written down or tested").
Scripts: `scripts/server/backup_to_tower.sh` (nightly, `iris-backup.timer` at 03:40) and
`scripts/server/restore_from_tower.sh`. No addresses live in this file; the tower is reached through
the `tower` ssh alias (`~/.ssh/config`, git-ignored), the rest comes from `config/private.local.json`.

## What a backup holds

One `iris-home_YYYYmmdd_HHMM.tar.zst` per night on the tower (`D:\IrisBackups`), plus a
`.sha256` sidecar. Paths inside are relative to `$HOME`:

| Always (a missing one FAILS the backup) | When present | Only in ENCRYPTED backups |
|---|---|---|
| `IRIS/state` (minus songs, GGUFs, stock Vector backup) | `IRIS/profiles`, `IRIS/faces` | `~/.ssh` |
| `IRIS/memory` (Chroma + JSONL) | `~/.config/iris`, `~/.config/systemd/user` | `~/.claude/channels` (Discord token) |
| Claude memory dir `.claude/projects/-home-iris-IRIS/memory` | `.iris_private.env`, `.iris_sibling_secret`, `LIVE` | `~/.claude/settings.json` |
| `ava_core/IDENTITY.local.md`, `USER.local.md` | cold-wake msg, app `.env.local` | wire-pod `apiConfig.json`, `certs`, `session-certs` |
| `config/private.local.json` | little-brain v16 adapter + Modelfile, wire-pod `jdocs` | |

Inside every archive made since 2026-10-29: `IRIS_BACKUP_MANIFEST.sha256` (every file's sha256),
SQLite files as consistent snapshots (sqlite backup API), Chroma copied until two passes agree.
Retention: newest of each of 7 days, 4 ISO weeks, 6 months.

**Not in the backup** (re-create or re-download): the git repo itself (GitHub), the Python venvs,
Ollama models and the q8 GGUFs, `IRIS/state/tzeke_songs`, the vault `~/ClaudeCodeMemory` (its own git
remote — `vault.sh`), Blender/apps under `~/apps`, the whole-VM image.

**Encryption** is OFF until an age recipient public key is put in
`~/.config/iris/backup_age_recipients.txt` (one `age1...` line). Then archives end in `.tar.zst.age`
and restoring needs the matching private key (`--identity FILE`). **Where that private key lives is
Zeke's decision** — not on this VM and not on the tower, or the encryption protects nothing.

## A. Restore test (do it every quarter, on a scratch VM or a scratch dir)

1. List what's there: `scripts/server/restore_from_tower.sh --list`
2. Restore the newest into an empty scratch dir — this touches nothing else:
   `scripts/server/restore_from_tower.sh --latest --into /data/restore_test`
   It checks the sidecar sha256, decrypts (if `.age`), `zstd -t`, extracts, `sha256sum -c` the
   manifest, and runs `PRAGMA integrity_check` on every SQLite file. Exit 0 = all passed.
3. Semantic memory opens and answers (CPU is fine):
   ```
   CUDA_VISIBLE_DEVICES= ~/venvs/iris-v100/bin/python -c "
   import importlib.util; s=importlib.util.spec_from_file_location('sm','$HOME/IRIS/brain/iris_semantic_memory.py')
   m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.configure('/data/restore_test/IRIS')
   print(m.count(), len(m.search('Zeke', k=3) or []))"
   ```
   Expect a count in the same range as the live store and 3 hits.
4. On a scratch VM, also run steps B.4–B.6 there (staging dry run boots the runtime with no cognition).
5. Delete the scratch copy (it holds secrets): `chmod -R u+w /data/restore_test; rm -rf /data/restore_test`.
6. Write the date + result in `state/backup.log` by hand (`restore test ok/failed: ...`).

Tested 2026-10-29 against a fresh `--dry-run --keep-local` archive: 8127 files verified, 5/5 SQLite ok,
Chroma 1175 entries, search returned hits; a flipped byte failed the sidecar check, and without the
sidecar `zstd -t` caught it.

## B. Real recovery (the server's disk died)

1. **New VM**: Ubuntu 22.04 on the Proxmox host, V100 passed through, user `iris`. Pins and driver:
   `setup/server_pinned_stack_bootstrap.sh` (do not freestyle CUDA/torch versions — Volta).
2. **Repo + venv**: clone the repo to `~/IRIS`; build `~/venvs/iris-v100` per
   `setup/server_pinned_stack_bootstrap.sh` / `setup/server_perception_bootstrap.sh`
   (`requirements.txt`).
3. **Reach the tower**: put an ssh key the tower accepts in `~/.ssh` and a `tower` alias in
   `~/.ssh/config` (if the backup was encrypted, `~/.ssh` is inside it — restore with step 4 first
   from a copy carried over by hand, then fix the alias).
4. **Restore**: with nothing of mine running yet,
   `scripts/server/restore_from_tower.sh --latest --live --yes [--identity KEYFILE]`
   — it extracts to a staging dir, verifies everything, and only then copies over `$HOME`.
   For an older night: `--archive iris-home_YYYYmmdd_HHMM.tar.zst`.
5. **Services**: `scripts/server/install_supervision.sh --dry-run`, then for real (installs + enables
   stack, mouth, ears, quiet hours, worker slices, watchdog + backup timers). The Vector / wire-pod /
   post-office units come back from `~/.config/systemd/user` in the backup —
   `systemctl --user daemon-reload` and `systemctl --user enable` them; the root OOM helper and the
   ollama drop-in need sudo (`scripts/server/systemd/ollama.service.d/`).
6. **Prove it before going live**: `IRIS_ROLE=staging scripts/server/start_iris_linux.sh` (runs
   `staging_dryrun.py`, no cognition), then `memory_search` from a session, then create `~/LIVE` and
   start `iris-stack`. Re-mint the self-cron (CORE memory says how).
7. **After**: check `systemctl --user list-timers` shows `iris-backup.timer` and `iris-watchdog.timer`,
   and that the next night's `state/backup.log` line says `backup ok`.

## C. Alerts that point here

The watchdog DMs Zeke when `state/backup.log` has no `backup ok` for 36 h or `iris-backup.timer` is
not enabled/active — even when `~/LIVE` is absent. A run that FAILS (required path missing, sha256
mismatch on the tower, archive under half the previous size → nothing pruned) logs `backup FAILED: ...`
and the unit ends `failed`, which the watchdog also reports.
