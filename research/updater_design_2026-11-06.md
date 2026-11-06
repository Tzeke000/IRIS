# LUMEN auto-updater: research + design (2026-11-06)

Research and design only. Nothing in the game project or the harness changed. The prototype lives in
`/data/projects/vr_iris/updater_proto/` (its own git repo; 23 unit tests, all passing).

**What Zeke asked for (11-05/06):** Steam-style updates. At launch the game checks its version against the temple
server. If it's behind, it asks "Update now? Yes / No", downloads only what changed, checks that the files are signed by
us, and restarts itself. Settings and invites survive the update, and they also survive the rename to LUMEN.

---

## 1. The recommendation in 15 lines

1. **Delta = content-addressed chunks (the Steam/casync model), not per-version patch files.** The publisher cuts every
   file into chunks named by their SHA-256. Each release adds only the chunks no earlier release had. A client downloads
   the chunks it doesn't already hold, from **any** older version, so there's no patch chain.
2. **The chunker understands PCKs:** chunk boundaries follow the resources inside `LUMEN.pck`. Small resources are grouped
   into chunks and a group ends on a path hash. One re-export then changes about 7 chunks instead of about 100.
   Exe/dll files use fixed 1 MiB chunks. Chunks are zlib-compressed for now (zstd later).
3. **One signed manifest per channel:** `stable/manifest.lumen` = a magic line + the base64 signature + canonical JSON
   (product, channel, version, **seq**, issued, **expires**, launch, files[path, size, sha256, chunks[id, size, csize]]).
4. **Signature = ECDSA P-256 / SHA-256.** Godot 4.7.2's `Crypto.verify` checks it with no plugin, and so does Windows CNG.
   Ed25519 (Sparkle/minisign's choice) **fails to load in Godot** (`Error parsing key -15488`, probed on the real binary).
5. **The client checks the signature on the bytes before parsing the JSON.** It then applies policy: right product and
   channel, `seq` ≥ the installed one (rollback protection), not expired (freeze protection), and no unsafe paths. The
   friend's own files are never touched: invite.txt, join.txt, tunnel.txt, MyAvatars/, tools/wg_*.
6. **Every chunk is checked against its signed id, and every assembled file against its signed sha256, before any swap.**
7. **Keys:** the signing key lives **only on iris-home** (`~/.config/lumen_sign/`, 0600, passphrase-encrypted) and is
   **never on CT 110**. Every client embeds **two** public keys: the release key plus an offline recovery key kept by
   Zeke, which allows rotation.
8. **Where updates are served: CT 110, over the existing invite-gated DTLS game connection.** It uses a small, frozen
   update protocol (raw `send_bytes` on its own ENet channel) that the server accepts from **any** game version with a
   valid invite. That means no new port, no new service, and no change to nftables. The store sits read-only at
   `/srv/vrtemple/updates`.
9. **In-game flow:** connect → the server says `latest seq/version` → "Update now? Yes / No". On Yes the game stages
   everything into `<install>/_update/` while still running (download, verify, assemble), then starts the swap stub and
   quits.
10. **Swap stub = `tools/lumen_launch.exe`** (about 50 KB of C, cross-built with the llvm-mingw we already have, no
    console). It waits for the game to exit, writes a journal, renames each changed file into `_update/backup/` and the
    new one into place (same volume, so each rename is atomic), writes `installed.json`, and starts the new game.
11. **The shortcuts point at the stub,** so it runs before every boot. If the journal says `swapping` (power cut) → roll
    back. If it says `pending_boot` and the new game hasn't confirmed by the 3rd launch → roll back. The new game
    confirms once it reaches the menu, and that deletes the backup.
12. **Resume** is per chunk: a verified chunk is kept, so killing the update anywhere costs at most one 1 MiB chunk.
    Damaged local files get re-downloaded (local chunks are re-hashed, never assumed good).
13. **Settings:** `use_custom_user_dir=true`, `custom_user_dir_name="LUMEN"` ⇒ `%APPDATA%\LUMEN\`. The first autoload
    **copies** (never moves) settings/social/course/avatars_private/... out of `app_userdata\VR Iris` once, guarded by a
    marker. `settings.cfg` gets `[meta] schema=N` with one migration step per change, and unknown keys are kept.
14. **The rename itself ships as ONE last full Setup** (`LUMEN-Setup.exe`). It finds the old install, carries over
    invite/join/tunnel/MyAvatars, removes the old files, and installs the first updater-enabled build with its
    `installed.json`. Every update after that goes through the updater.
15. **Measured on real builds:** fr_build → fr2_build (two exports 2 h apart) is **0.81 MB** to download instead of a
    600 MB Setup, rebuilt byte-for-byte, in 4.9 s of staging and 2.1 s of swapping.

---

## 2. What we have today (learned from the harness)

- Export = `VRIris.exe` (90–109 MB) + **one 476 MB `VRIris.pck`** + 2 dlls (`libgodotopenxrvendors`, `libtwovoip`).
  The Quest `VRIris.apk` sits in the same `build/` dir. The PCK is format **v4** (4.7.2): header, file data, directory
  at the end, flags = relative file base, not encrypted.
- Setup = NSIS 3.08 built headless (`harness/installer/iris_setup.nsi`), per-user, no admin, installs to
  `%LOCALAPPDATA%\Programs\VR Iris`. invite.txt, join.txt and tunnel.txt sit next to the exe (read by net.gd).
  Running Setup again is already a working "repair/update" (keeps the key, the tunnel and MyAvatars) and refuses while
  the game runs (exit 4).
- user:// is still the default `%APPDATA%\Godot\app_userdata\VR Iris\` (settings.cfg, social.cfg, course.cfg,
  avatars_private/, avatar_cache/ ...). The server's user:// on CT 110 (`/srv/vrtemple/home/.local/share/godot/
  app_userdata/VR Iris/`) holds **the DTLS private key, server.cfg and invites**, so the rename must migrate the
  server too.
- Version = `YY.M.D.N` stamped into the pck (`installer/version.py`, `pck_version.gd`). The handshake (net.gd) rejects
  `AUTH_VERSION` mismatches with "version mismatch … update the game to join". That is the hook the updater replaces.
- Already proven on win101 (bugpass-1104): **you can RENAME a running exe**, but you can't open it for writing. The pck
  is held open by the game, so swaps must happen after the game exits.
- Transfer precedent: `avatar_share.gd` already moves 48 MB blobs over the DTLS connection (16 KB chunks, window 16,
  its own channel, sha256-checked).

---

## 3. Research: how others do it

### Godot's own patching (4.4+; 4.7 has delta encoding)
- Godot (4.4+) can export a **patch PCK** holding only the changed files (`--export-patch <preset> <path> --patches a.pck,...`).
  The 4.7 we run can also **delta-encode** them (I did not pin the exact version that added it) (`patch_delta_encoding`, zstd level 19, a min-reduction threshold,
  include/exclude filters). In the 4.7.2 source (`core/io/file_access_pack.h`) the per-file flags are
  `PACK_FILE_REMOVAL` and `PACK_FILE_DELTA`, and `FileAccessPatched` applies the delta when the file is read.
- Patches are loaded with `ProjectSettings.load_resource_pack(path, replace_files=true)`. The docs say to do it **as
  early as possible (an autoload's `_init()`)**.
- Docs caveats: each delta patch **adds load time for that resource, cumulatively**. Delta patches need the **exact**
  base packs in the same order ("re-exporting previous versions may differ slightly, which can make patching fail").
  Turning off compression helps patch size.
- **Why we don't use it:** a patch pck can't change the exe or the dlls (an engine patch, like the N9 fix, changes the
  exe). Patches pile up into a chain with growing load cost. Autoload scripts and `project.godot` are already loaded
  before any patch can be, and those are exactly the files we change most. And it needs the exact old pck kept for
  every release. It's a mod system, not an updater.
- Encryption (`script_encryption_key`, `PACK_DIR_ENCRYPTED`) is AES with a key compiled into the exe. That is
  obscurity, **not authentication**, so it can't stand in for a signature.

### Binary delta formats
- **bsdiff:** good patches, slow, a lot of memory on 400 MB inputs. **bidiff** (Rust) handles >2 GiB.
- **zstd `--patch-from`:** uses the old file as a dictionary. zstd's own benchmarks say it's far faster than bsdiff at
  similar size. **Measured here** on fr→fr2 with `-19 --long=30`: **0.61 MB patch, 3 min 27 s to create, 2.3 s to apply.**
- **Courgette** (Chrome) disassembles executables for tiny exe patches. Its complexity isn't worth it when the exe
  changes only on engine updates.
- **Content-defined chunking (casync/desync, and Steam):** chunks named by checksum in a plain chunk store, plus an
  index that maps a file to its chunks. Similar versions share most chunks, and any version can reassemble any other.
  **SteamPipe** splits files into about 1 MB chunks, compresses them, and only the new or changed chunks are downloaded.
  The weakness is its **fixed** boundaries: reshuffled assets inside a pack file defeat it. So pack layout matters.
- **Measured here (fr→fr2):**

  | method | download | per-release work | works from any old version | resumable |
  |---|---|---|---|---|
  | whole Setup | ~600 MB | none | yes | no |
  | fixed 1 MiB chunks (Steam-like) | **93.6 MB** | ~30 s | yes | per chunk |
  | **PCK-aware chunks (chosen)** | **0.81 MB** (7 chunks) | ~30 s | **yes** | **per chunk** |
  | zstd `--patch-from` | 0.61 MB | 3.5 min **per old version** | no (one patch per pair) | no |

  On a bigger change (Nov 4 → Nov 6: two days of assets + the patched engine) the chunked pck delta was 80.9 MB
  compressed. The same comparison with fixed chunks was 380 MB.

### Signing, and the threat model (TUF)
- **TUF (The Update Framework)** lists the attacks an updater must survive: arbitrary install, **rollback** (serving an
  older signed release), **freeze** (serving a stale manifest forever), mix-and-match, endless data, and key compromise.
  Its tools are versioned metadata, **expiry**, threshold signatures and separate roles (root/targets/snapshot/timestamp).
  Community write-ups stress that rollback/freeze defence **only works if the client keeps trusted state** (the last
  seq it saw).
- **We take from TUF:** a signed monotonic `seq`, an `expires` date, a manifest that names every file + hash (no mix-and-
  match), bounded sizes everywhere (no endless data or decompression bombs), and a recovery key for rotation. Full TUF
  (four roles, thresholds) is overkill for one publisher and a friends-only game. The design leaves room to grow into it.
- **Sparkle / WinSparkle:** an EdDSA (Ed25519) signature on each update, with the public key embedded in the app. DSA is
  deprecated. Same shape as ours; we only change the curve because of Godot.
- **minisign / signify / libsodium:** Ed25519 again. It would be a fine choice if the verifier were native code, but
  ours runs in GDScript.
- **Velopack (Squirrel's successor):** installs to `%LocalAppData%\<id>\current\` plus an `Update.exe` beside it that
  `apply`s / `patch`es / `start`s. Delta packages are generated at pack time against the previous release, with a
  **fallback to the full package** if a delta fails. We copy the "separate small updater exe beside the app" part and
  the full fallback (our fallback = the full Setup).
- **Google Omaha:** a separate updater service plus a server-side "what's latest" protocol. Too heavy (a service, admin).
- **Steam:** depots, manifests, about 1 MB content-addressed chunks, a client that only fetches missing chunks. That's
  the model we follow, plus our pck awareness.

### Replacing a running program on Windows
- A running exe can't be opened for writing (sharing violation) but **can be renamed**. That was measured on win101,
  and Microsoft/StackOverflow sources say the same, with a "usually" attached. The game's open pck can be neither.
- `MoveFileEx(..., MOVEFILE_DELAY_UNTIL_REBOOT)` needs admin for most paths, only queues a registry entry (the return
  value says nothing about the actual replace), and needs a reboot. It isn't suitable for a per-user, no-admin game.
- The standard pattern, used by Velopack/Squirrel and most game launchers: **a separate small updater waits for the
  app to exit, swaps the files, and relaunches.** That is what we do. The stub updates itself with the rename trick
  (rename `lumen_launch.exe` → `.old`, put the new one in place, delete `.old` next time).

---

## 4. The design in detail

### 4.1 Publisher (iris-home)
- `vr.sh release` (new, later) = export → stamp version → `lumen_publish` into a store dir on iris-home → **sign** →
  rsync **chunks first, manifest last** to CT 110 `/srv/vrtemple/updates/` (root-owned, mounted read-only into
  `vr-temple.service`). A client therefore never sees a manifest whose chunks aren't there yet.
- `seq` = a counter that only goes up, kept beside `versions_used.txt` (the same rule as version tags: never reused).
  `version` stays the human `YY.M.D.N`.
- `expires` = issued + 365 days. Re-signing the current manifest yearly (or on every release) keeps it alive. A friend
  whose client is fed an old manifest for a year sees "can't verify updates".
- Exclude `*.apk` (the Quest build shares the export dir; the bench caught it). The Quest gets its own channel later
  (`quest/manifest.lumen`), the same mechanism minus the exe swap.
- Store GC: a chunk no manifest from the last N releases references can be deleted (desync's `prune` idea).

### 4.2 Keys
| key | where | used for |
|---|---|---|
| release P-256 private | iris-home `~/.config/lumen_sign/release.pem` 0600, passphrase-encrypted; passphrase in `~/.iris_private.env` | every release |
| recovery P-256 private | **offline**: made once, kept by Zeke (USB/paper), never on any server | signing a manifest that rotates in a new release key |
| both public keys | compiled into the game (`res://update/keys.pem`) | verification |

- **CT 110 (the internet-facing box) holds no signing key.** Breaking into the temple server lets an attacker refuse or
  slow updates, but not forge one. That's the property that matters most.
- A compromised iris-home *can* sign. That is the honest residual risk. Recovery: sign a rotation release with the
  offline key that drops the old release key.
- Helpers never touch the key (helpers are locked out of secrets). Iris signs from her own session.

### 4.3 Transport: the updater lane on the game connection
- **Why not HTTPS?** It would mean a new listening service and port on CT 110, a new nftables rule, a second port
  forward for friends without WireGuard, and certificate handling. Since content trust comes from the signature, the
  transport only has to provide **access control** (invited people only) and **privacy**. The existing DTLS + invite
  connection already does both.
- **The frozen lane:** a client whose `AUTH_VERSION` doesn't match sends a hello with `kind="updater"`. The server checks
  the **invite** (same rules) **before** the version check. If the invite is good, it admits the peer as *updater-only*:
  no spawn, no world state, only the update ops. The lane uses `SceneMultiplayer.send_bytes` on a dedicated channel with
  a tiny fixed binary format, so it doesn't depend on node paths or RPC tables that change between versions:
  `01 GET_MANIFEST`, `02 GET_CHUNK <32-byte id>`, replies `81/82 <id> <part> <of> <bytes ≤16 KB>`, a window of 16
  (avatar_share's numbers). **This lane's format must never change.** A bump means a new op number, never an edited one.
- Rate limits: one updater session per invite, a bounded bandwidth share (play traffic first), and chunk ids served
  only if they appear in the current manifest (no arbitrary file reads; the store is content-addressed and read-only).
- Fallback: if the lane fails, the game says "Couldn't update. Ask Zeke for the new Setup". The full Setup still works
  as a repair.

### 4.4 Client flow (GDScript `Updater` autoload + the stub)
1. **Boot:** the stub runs `on_launch` (roll back on an unfinished swap, count unconfirmed boots), then starts
   `LUMEN.exe`. The game's first autoload runs **Migrate** (4.6), then Updater reads `installed.json` (verifies it).
2. **Check:** after connecting, the server sends `latest {seq, version, manifest_bytes}` (or the client asks on the lane
   when rejected for its version). If `seq` is newer: **"Update now? Yes / No"**. "No" lets you keep playing if the
   versions are compatible; if not, the game says it can't join until it updates.
3. **Stage** (in-game, behind a progress bar): fetch `manifest.lumen` → `Crypto.verify` → policy → plan (re-hash local
   chunks) → fetch the missing chunks → decompress (bounded by the signed size) → sha256 = id → write
   `_update/chunks/<id>` → assemble `_update/files/<path>` → sha256 = signed → write `_update/ready.lumen` +
   `swap_plan.txt` (path, sha256, one per line, so the C stub needs no JSON parser).
4. **Swap:** the game starts `tools/lumen_launch.exe --apply <pid>` (hidden, no console) and quits. The stub waits for
   the PID, re-hashes the staged files against `swap_plan.txt` (CNG BCrypt SHA-256), and journals `swapping`. It renames
   live → `backup/` and staged → live, writes `installed.json`, journals `pending_boot`, and starts the game.
5. **Confirm:** once the new game reaches the menu it deletes `backup/`, `files/`, `chunks/` and the journal.
6. **Rollback:** journal `swapping` on boot ⇒ put every backup back, delete files the update added, restore
   `installed.json`. More than 2 unconfirmed boots ⇒ the same, plus `rolled_back.txt` so the game can say "the update
   didn't start, so the previous version was restored".

### 4.5 Failure modes
| failure | result |
|---|---|
| tampered manifest (any byte) / wrong key | refused before parsing; nothing staged |
| old signed manifest replayed (rollback) | refused: `seq` < installed |
| stale manifest served forever (freeze) | refused after `expires`; the game says it can't verify updates |
| tampered or truncated chunk | refused at its hash; nothing swapped |
| a "signed" manifest with `../`, `C:\`, `invite.txt`, `MyAvatars/` ... | refused by policy (defence in depth) |
| wifi drops mid-download | the next try fetches only the missing chunks |
| power cut mid-swap | the stub rolls back on the next launch |
| the new version crashes on boot | rolled back after 2 unconfirmed boots |
| a damaged local file (bit rot) | its chunks fail the re-hash and are re-downloaded (the update also heals) |
| staged file edited after staging | the stub's re-hash refuses the swap |
| disk full | staging fails before any swap; free space checked first (≈ the changed files' size) |
| antivirus locks a file during the swap | the rename fails → journal still `swapping` → rollback (to test on win101) |

### 4.6 Settings + the LUMEN rename
- **Client user dir:** set `application/config/use_custom_user_dir=true`, `custom_user_dir_name="LUMEN"`
  ⇒ `%APPDATA%\LUMEN\` (and `~/.local/share/LUMEN` on the Linux server). It no longer depends on the project name.
- **Migrate (first autoload, before Settings loads):** if `user://.migrated_from_vr_iris` is missing and
  `%APPDATA%\Godot\app_userdata\VR Iris` exists, it **copies** `settings.cfg, social.cfg, course.cfg, avatars_private/,
  invites/, builds.json, gesture_templates.json` (caches are skipped; they rebuild), never overwrites, then writes the
  marker. The old dir stays (the uninstaller already asks before deleting player data). Reference:
  `updater_proto/lumen_update/migrate.py` + tests.
- **The server's carry list is different and critical:** `temple_server.key`, `server.cfg` (the invites!), `invites/`,
  `builds.json`, `avatar_cache/`, gesture templates. Do that move by hand in `deploy_temple_server.sh` (stop → copy →
  chown vrtemple → start), not by in-game magic. Keep the DTLS cert's CN `iris-temple`: renaming it means a new cert,
  and every old client would fail to connect *before* it could update.
- **Schema steps:** `settings.cfg [meta] schema=N`, one step per change (rename a key, convert a value). A *newer*
  schema is left alone (the rollback case), and unknown keys are kept.
- **Install dir + exe rename:** a one-off **LUMEN-Setup.exe**. It reads HKCU's old `VR Iris` uninstall key, copies
  invite.txt/join.txt/tunnel.txt/MyAvatars from the old dir, deletes the old game files + shortcuts + uninstall key,
  and installs to `%LOCALAPPDATA%\Programs\LUMEN` with `installed.json` (the signed manifest of exactly that build) and
  the stub. The WireGuard tunnel service `iris` stays as is (renaming it would mean re-enrolling every friend's key).
  This Setup also becomes the "full fallback".

---

## 5. Test plan (harness, for when it gets built)

**Unit (done in the prototype, 23 tests):** chunk coverage + pck parse; one edit + one insert ⇒ ≤6 new chunks; a real
build pair < 2 MiB; signature round trip; tampered body, signature or key refused; key rotation; rollback, expiry and
channel refused; signed-but-evil paths refused; full v1→v2 (removed file gone, invite + MyAvatars intact); tampered
chunk and tampered manifest refused with nothing touched; a replayed old manifest refused; an interrupted download
resumes fetching only the missing chunks; power cut after each of the first 7 renames rolls back cleanly; a new version
that never boots is rolled back after 2 launches; a staged file tampered before the swap is refused; bit-rot healed;
apk excluded; user-dir copy-once; schema steps; **GDScript `Crypto.verify` + DEFLATE decode in the real 4.7.2-n9 binary
accept our envelope and refuse a tampered one or the wrong key.**

**Integration (win101, `vr.sh update test`, to write):**
- U1 install build N with LUMEN-Setup `/S`; publish N+1 to a test store on a **test** temple (own XDG + invite,
  never the live CT 110 store); launch → "Update now?" → Yes (driven via the bot/control socket) → the game restarts on
  N+1 (`[version]` in the log) → the bytes on disk equal build N+1.
- U2 a tampered manifest on the test server ⇒ the game says "update refused", stays on N, still joins if compatible.
- U3 kill the network halfway (netem/firewall on the test server) → relaunch → resumes; count the chunks re-fetched.
- U4 `taskkill /F` the stub mid-swap → relaunch → rolled back to N, the game runs.
- U5 publish an N+1 that exits immediately → 3 launches → back on N, `rolled_back.txt` shown.
- U6 settings survive: set name/height/volume/blocked list on N → update → same values; plus the VR Iris → LUMEN
  migration from a real old `app_userdata\VR Iris` copy.
- U7 a friend without admin: a standard user (the I8-style UAC case). The update must need **no** UAC prompt.
- U8 the game running twice / Setup running during an update ⇒ refused cleanly (the exit-4 logic).
- U9 no console flash and no focus steal (the tower rule): the stub is built `-mwindows`.

---

## 6. Open questions for Zeke

1. **"No" means what?** If the versions are still compatible, may a friend keep playing on the old version, or is
   updating required to join? (Recommendation: required when `AUTH_VERSION` changed, optional otherwise.)
2. **The offline recovery key:** willing to keep one on a USB stick or paper, never on a computer that's online? Without
   it, a compromised iris-home can only be fixed by sending everyone a new Setup.
3. **Auto-update without asking** for small updates (< 5 MB), or always ask? (You said "ask"; asking is the default.)
4. **The rename Setup:** OK that the switch to LUMEN is one more full Setup for every friend (~600 MB, the last one)?
5. **Old `VR Iris` user data:** leave it after migrating (safe; recommended), or delete it after LUMEN's first good boot?
6. **Quest/apk:** updates for the Quest build too (sideload flow, different), or Windows only for now?

---

## Sources
- Godot docs, *Exporting packs, patches, and mods* (4.7): https://docs.godotengine.org/en/4.7/tutorials/export/exporting_pcks.html
- Godot docs, *File paths in Godot projects* (user:// + custom user dir): https://docs.godotengine.org/en/stable/tutorials/io/data_paths.html
- Godot 4.7.2-stable source, `core/io/file_access_pack.{h,cpp}` (PCK v4 layout; `PACK_FILE_DELTA`/`REMOVAL`), `modules/mbedtls/crypto_mbedtls.cpp`, `main/main.cpp` (`--export-patch`, `--patches`)
- The Update Framework specification: https://theupdateframework.io/specification · overview: https://en.wikipedia.org/wiki/The_Update_Framework · client-state caveat: https://dev.to/ioneyrezende/tuf-rollback-and-freeze-defenses-depend-on-trusted-client-state-2bg6
- Steamworks, SteamPipe (about 1 MB chunks, only changed chunks downloaded): https://partner.steamgames.com/doc/sdk/uploading · https://partner.steamgames.com/documentation/steampipe
- casync / desync (content-addressed chunk store + index): https://github.com/folbricht/desync · https://sources.debian.org/src/casync/2%2B20201210-2/README.md
- zstd as a patching engine: https://github.com/facebook/zstd/wiki/Zstandard-as-a-patching-engine · https://github.com/facebook/zstd/pull/1959
- bidiff (bsdiff-family, >2 GiB): https://github.com/divvun/bidiff
- Sparkle EdDSA: https://sparkle-project.org/documentation/eddsa-migration · WinSparkle: https://github.com/vslavik/winsparkle
- Velopack: https://docs.velopack.io/integrating/overview · https://docs.velopack.io/packaging/operating-systems/windows · https://docs.velopack.io/reference/cli/content/update-windows
- MoveFileEx / delay-until-reboot: https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-movefileexa
- Our own measurements: `updater_proto/bench_real.py`, `updater_proto/godot_probe/probe.gd`, zstd `--patch-from` run (§3)
