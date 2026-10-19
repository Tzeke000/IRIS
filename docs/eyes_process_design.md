# Eyes out of the runtime — design (2026-10-19)

**Status:** DESIGN + OFFLINE PROTOTYPE. Nothing live has changed: the runtime still owns
the camera, no unit is installed, and no runtime or `brain/*` code was edited.
Prototype: `scripts/eyes/`. Transport tests: `tests/test_eyes_transport.py`.
Zeke OK'd the fix. The cut-over (§8) is a separate session that needs him to say so.

## 1. Why

`iris_runtime.py` (the MCP server) runs about 600 threads. Its RSS creeps because of
glibc arena fragmentation: mallinfo2 `in_use` stays flat at about 1.45 GB, while
free-held memory grows from 1.2 to 2.5 GB over a day. With the eyes parked at night
the growth stops, and RSS falls by about 0.5 GB. py-spy shows the busy threads are
`iris-video-capture`, `iris-face-worker` and `iris-body-worker`.

The goal is to run the eyes in their own process, so that:

- their heap can't bloat the runtime;
- they can be capped, measured and restarted alone;
- a vision crash or hang takes down only the eyes, not me.

**The prototype confirms the vision pipeline is the source, and narrows it.** Fed
only a video file, with no runtime around it, the eyes process shows the same
signature: in_use flat, free-held +61 MB/min. In an A/B, that growth **comes with
the two MediaPipe workers** (expression + eye-tracker); face+body alone is flat. With
`MALLOC_ARENA_MAX=2` all four workers are flat (§9.3). So:

- **Step 0 (cheap, already queued):** measure the runtime after the restart that
  lands `MALLOC_ARENA_MAX=4`. If the runtime goes flat, the leak half of this
  project is solved without the split.
- **The split is still worth doing,** for the other half of the goal: the eyes can
  be restarted, capped and watchdogged alone, and a hung camera or a model crash no
  longer touches me. The split by itself *contains* growth; it doesn't cure it.

## 2. Today's wiring (map, file:line as of `34125a5`)

### 2.1 Producers: everything runs inside `iris_runtime.py`

| Producer | Where | Notes |
|---|---|---|
| Capture loop `_iris_video_capture_loop` | `iris_runtime.py:4228` | Thread `iris-video-capture` is started at `:4711`, and only if InsightFace booted (`:4708`). Opens `cv2.VideoCapture(0)` with MJPG 1280x720@60 (`:4254`) and deadline-paces at 30 fps. Publishes `g["_raw_frame_slot"]=(pub_ts, frame)` at `:4562` and `frame_store.push_frame(annotated)` at `:4633`. Measures fps into `_capture_fps_ema/_measured` (`:4573`) and `_cam_props` (`:4287`). |
| Enrollment hook (inside the capture loop) | `iris_runtime.py:4581-4622` | When `g["_enroll_request"]` is set, saves RAW frames to `faces/<pid>/enroll_*.jpg`, calls `update_known_faces()`, and writes `g["_enroll_result"]`. |
| Perception workers `_perception_worker` | `iris_runtime.py:4491`, started at `:4527` | Threads `iris-{face,hands,expr,attn}-worker`, 30 Hz over `_raw_frame_slot`. |
| Face step | `iris_runtime.py:4333` | `InsightFaceEngine.analyze_frame` (`brain/insight_face_engine.py:330`). Writes `_face_results`, `_recognized_{person_id,confidence,age,gender}`, `_face_{age,gender}` and `_person_present_since_ts`. Fires `face_appeared/face_lost/face_changed` (`:4372/4380/4387`). Then `face_tracking.tick_from_capture` (`:4398`) writes `_face_tracking_state/_last` and can fire `new_person_detected` (`brain/face_tracking.py:249`). |
| Hands step | `iris_runtime.py:4415` → `brain/iris_hands.py:127` | Writes `_hand_results` (`iris_hands.py:183`) and fires `hand_wave/gesture_*`. Gated on `_hands_enabled` and faces. |
| Expression step | `iris_runtime.py:4428` | `_expression_detector` (MediaPipe). Writes `_current_expression` and fires `expression_changed`. |
| Attention (eye-tracker) step | `iris_runtime.py:4457` | `_eye_tracker.analyze` (MediaPipe facemesh). Writes `_attention_state`, `_looking_at_screen`, `_gaze_region` and fires `attention_changed`. |
| Body worker `iris-body-worker` | `brain/person_track.py:351/376`, started from `brain/iris_bootstrap.py:345` | YOLOX-s ONNX (`brain/yolox_person.py:149`) plus ByteTrack. Reads `frame_store.get_buffered_frame(1.0)` (a different clock from the raw slot, `person_track.py:326-332`). State lives in the module-level `_state`, not in `g`. |
| Human pose loop | `brain/body_pose.py:597`, started from `iris_bootstrap.py:328` | YOLO11n-pose. Reads frame_store, `_face_results`, `person_track.track_boxes()` and the head bearing (`_attention_state_obj`). Writes `g["_human_pose_live"]` (`:643`). |
| face_liveness (SHADOW, flag OFF) | `brain/face_liveness.py:539`, `maybe_autostart` `:684` | Reads `_raw_frame_slot` (`:410`), `_face_results` and the `_insight_face` gallery. Writes `state/face_liveness_shadow.jsonl`. |
| unknown_capture | `brain/unknown_capture.py:147` | Watches `_face_results` and drives the `_enroll_request` hook into `faces/_drafts/`. |
| Engines in `g` | `insight_face_engine.py:573`, `expression_detector.py:212`, `eye_tracker.py:420` | `_insight_face`, `_expression_detector`, `_eye_tracker`. `eye_tracker` also has a module singleton, `get_eye_tracker()`. |

### 2.2 Consumers

**Frames**, through `brain/frame_store.py` (`push_frame` at `:149`,
`get_buffered_frame` at `:158`, `peek_buffer_age_sec` at `:172`, plus direct reads of
`_buffer_frame/_buffer_ts`):

- **orb_http** (runs in the runtime, :5876):
  - `/api/v1/vision/latest_frame` (`brain/orb_http.py:950`) and `/camera/live_frame` (`:908`) annotate and JPEG-encode the buffered frame (`_display_jpeg` `:70`);
  - `/camera/mjpeg` (`:917`) polls `_fs._buffer_ts` directly.
- **Servo and attention:**
  - `attention_smooth` (`tools/system/attention_smooth_tool.py:668,734,766,843,1383`; it runs a TrackerVit fast path on frames at `:659`);
  - `attention_engage` sentry (`~:231`);
  - `attention_multi` (`:94,261`);
  - `room_map` (`:196,569`);
  - `brain/visual_attention.py` (7 sites);
  - `ptz_predict` (`brain/ptz_predict.py:377`);
  - `head_odometry` (`brain/head_odometry.py:106`);
  - `object_lock` (`brain/object_lock.py:145`).
- **scene_memory keyframes** (`brain/scene_memory.py:235,502`; thread from `iris_bootstrap.py:320`). Writes `state/scene_memory/keyframes/<day>/*.jpg` and `keyframes.jsonl`.
- **Tools:**
  - `study_face` (`:343`, plus the enroll hook and `g["_insight_face"]` embedding at `:127`);
  - `face_pipeline_probe` (runs `analyze_frame` in-process; `repair` rebuilds `_SINGLETON`);
  - `human_pose` (`:110`);
  - `gpu_park` (`_frame_age_s`);
  - `eyes_reload` (`_buffer_ts`);
  - `eyes_debug`, `body_verify`, `orb_frame_patch`.
- **Health:** `iris_health_probes.probe_camera` (`brain/iris_health_probes.py:113-174`) uses `peek_buffer_age_sec` and compares `_buffer_frame` content to detect a frozen frame. `brain/health.py:41`.
- **Raw slot:** the perception workers, `vision_fps_probe` (`tools/system/vision_fps_probe_tool.py:24`) and `face_liveness.raw_frame`.

**Results**, read from `g`:

- `_face_results` has about 30 read sites: orb snapshot and annotate, servo, room_map, attention_follow/engage/search, study_face, human_pose, face_liveness, scene_memory, unknown_capture, inner_monologue, heartbeat emotion, camera_annotator, debug routes.
- `_recognized_person_id/_confidence` are read by iris_attention_sources (`:486`), runtime_presence (`:433`), ambient_snapshot (`:73`), orb `/snapshot` (`:388`), question_engine and sleep_mode.
- `_face_age/_gender` → orb current_person.
- `_person_present_since_ts` → orb time_at_machine.
- `_current_expression`, `_attention_state`, `_looking_at_screen` and `_gaze_region` are read by orb, inner_monologue, heartbeat, ambient_snapshot, wake_detector, voice_loop, scene_memory and camera_annotator.
- `_hand_results` → camera_annotator, the servo gesture input (`attention_smooth:600`) and pointing_ledger.
- `_human_pose_live` → scene_memory, human_pose, room_map, camera_annotator.
- person_track `track_boxes()/target_offset()/status()` are read by body_pose, scene_memory, human_pose and attention_smooth (`:743,903`). attention_search calls `step()` inline.
- `_capture_fps_measured`, `_cam_props` and `_perc_*_last_ms` → vision_fps_probe and app_status_board.

**Engine objects in `g`** are used directly by the runtime in many places:

- `_insight_face`: face_liveness, runtime_presence, person_onboarding, camera.py, orb_http (`:1412,1926`), operator_server, study_face, face_db, face_pipeline_probe, vision_fps_probe, enroll_face (`iris_runtime.py:2687`), iris_health (`:3159`).
- `_eye_tracker` / `get_eye_tracker()`: reply_engine (`:491`), camera.py, operator_server, eye_tracking tool, and orb `/camera/calibrate_gaze` (`:1680`). **calibrate opens `cv2.VideoCapture(0)` itself** (`brain/eye_tracker.py:167`).

**Perception wakes.** The face worker fires on `g["_signal_bus"]`. The host
(`iris_body_host.py` `perception_reader` `:1393`) polls `GET :5876/api/v1/signals`
every 1.5 s, wakes on `face_appeared/face_lost/unknown_capture` (`:199`), and reads
`/api/v1/snapshot.current_person`. In-runtime fallback: `iris_attention_sources._camera_loop`
(`:474`) polls `_recognized_person_id` and stands down while the host is polling.

**Control and park:**

- `body_pause_flag` (`.tmp/body_pause.flag`, `brain/iris_paths.py:128`) is the ONLY thing the capture loop obeys (`iris_runtime.py:4247`). It releases the device.
- `gpu_park` (`tools/system/gpu_park_tool.py:180`) stops sentry, follow, servo, pose and body loops, rests hands, runs `_eyes_rest` (`runtime_repair_tool.py:144`, which nulls the engines), and writes the pause flag.
- `eyes_quiet_hours` (`tools/system/eyes_quiet_hours_tool.py:77/113/125`) calls `gpu_park` on the quiet-hours schedule. **It only starts if `__main__` has `_iris_video_capture_loop`** (`:128`).
- `eyes_reload` writes and clears the pause flag. Its deep mode is Windows `pnputil` and is dead on Linux.
- `servo_deliberately_off.json` is read only by servo autostart.

**PTZ.** The device has no "PTZ process". Control-plane access is separate from the
capture fd already today:

- `attention_smooth._V4l2AbsJog` (`:474-596`) holds its **own** `os.open('/dev/video0', O_RDWR)` and sends `VIDIOC_S_CTRL` pan/tilt.
- `visual_attention.V4L2PtzActuator` (`:214-325`) shells out to `v4l2-ctl --set-ctrl/--get-ctrl`.
- Fallback: HID jog (`_PixyJog`).

**Second camera openers (hazards with any single owner):**

- `frame_store.read_live_frame_with_meta` (`:110`), reached by `vector_overhead` → `vector_vision`, when the buffer is older than 1.5 s;
- `eye_tracker.calibrate` (`:167`);
- the boot device probes `_probe_camera_open` (`iris_runtime.py:464`) and `_scan_video_device_indices`.

**Count:** 7 producer threads plus the engine bootstraps, about 60 distinct consumer
sites in about 35 files (≈15 frame readers, ≈30 result readers, ≈12 engine-object
users), and 3 control paths (pause flag, gpu_park/quiet hours, enroll handshake).

## 3. Decision summary

```
 iris-eyes.service (NEW process, ~/venvs/iris-v100)        iris_runtime.py (unchanged role)
 ┌──────────────────────────────────────────────┐          ┌───────────────────────────────────┐
 │ V4L2 STREAMING fd (sole owner, flock)        │  /dev/shm│ frame_store  ── ring reader ──────┤ orb latest_frame/mjpeg,
 │ capture 30fps ─▶ FrameRing writer ───────────┼─────────▶│  (get_buffered_frame shim)        │ scene_memory, servo TrackerVit,
 │ face / hands / expr / attn / body / pose /   │          │                                   │ room_map, study_face crops …
 │ liveness workers (unchanged brain/ modules)  │  unix    │ eyes_link shim thread ────────────┤ writes the SAME g[...] keys,
 │ enroll hook, face_tracking                   │  socket  │  (subscription mirror + RPC)      │ re-fires events on signal_bus,
 │ BusServer: snapshot push + events + RPC ─────┼─────────▶│  EyesFaceProxy in g[_insight_face]│ proxies engine calls
 │ ◀── head bearing (set_head) ─────────────────┼──────────┤ servo (attention_smooth) + PTZ    │ PTZ control fd stays HERE
 └──────────────────────────────────────────────┘          └───────────────────────────────────┘
```

- **Moves out:** the camera's streaming fd, capture, every frame-rate detector and
  tracker (InsightFace, hands, expression, eye-tracker, YOLOX+ByteTrack, pose,
  liveness shadow), face_tracking, and the enrollment frame saver.
- **Stays:** the servo and all PTZ control (attention_smooth/follow/engage/search,
  room_map, ptz_predict, visual_attention), scene_memory, the orb HTTP server,
  health, wakes, and every tool.
- **Frames:** a `/dev/shm` seqlock ring of raw BGR frames (4 × 2.76 MB). Readers get
  a read-only mapping and copy into their own buffer (~1 ms).
- **Results:** a Unix socket at `$XDG_RUNTIME_DIR/iris/eyes.sock` with ears-style
  newline JSON (`{"cmd","args"}` → `{"ok","result"}`), plus a push subscription
  (snapshot on every new result, rate-capped, 1 Hz heartbeat, and events).
- **Compatibility:** a runtime-side `eyes_link` shim keeps the existing `g[...]`
  contract, so ~90% of the consumers need no edit.

## 4. Design

### 4.1 Process boundary and who owns the device

**The eyes own the camera's STREAMING fd. The runtime keeps PTZ control.**

UVC/V4L2 lets any number of processes open the node. Only the streaming queue
(`VIDIOC_REQBUFS/STREAMON`) is exclusive. Control ioctls (`VIDIOC_S_CTRL` pan/tilt,
`v4l2-ctl --set-ctrl`) work from any fd. That is already how today's single process
works: cv2's capture fd and `_V4l2AbsJog`'s control fd are two different fds on
`/dev/video0`, and the kernel doesn't care whether they sit in one process or two.
So the split **does not change the device semantics the PTZ relies on.**

Why the servo does not move (phase 1):

- It is control-plane. Its memory cost is small and it isn't one of the busy threads.
- `attention_smooth_tool.py` is ~1650 lines entangled with `g` (gesture input,
  `_attention_state_obj`, rest stash, guide holds, sentry, study_face retargets).
  Moving it means moving the whole attention stack, which is a second project.
- Every tool that "looks" (room_map `look`, `attend`, guide head, study_face,
  ptz_predict) keeps working with no RPC hop.

What the servo now pays: person_track's `target_offset()` crosses the socket. The
eyes publish `body.offsets[label]`, the Kalman-derived dx/dy/vx/vy for every bound
label, in each body result. The servo reads the mirrored copy. Measured transport is
in §9. **Phase 2 option**, if the hop ever hurts pursuit: move `_servo_loop` plus the
`_V4l2AbsJog` fd into the eyes and leave a thin command API (`look`, `attend`, `hold`,
`release`) in the runtime.

The head bearing flows **into** the eyes: body_pose (`is_static_box`, `_LAST_HEAD`)
and the pose loop need the current pan/tilt. The servo already knows it, so the shim
sends `set_head {pan_deg, tilt_deg, ts}` on change (at most 10 Hz).

**Single-owner guard.** The streaming owner takes `flock($XDG_RUNTIME_DIR/iris/camera.lock)`
before it opens the device, and holds it while streaming. The eyes refuse to start
capture without it. The runtime's legacy capture loop must take the same lock once
the code lands (§8 step 1), so a wrong flag can never produce two streamers.

The second-opener hazards in §2.2 get fixed in the same change:

- `read_live_frame_with_meta` → ring-only;
- `eye_tracker.calibrate` → an eyes RPC;
- the boot device probe skips when the flag is on.

### 4.2 Frame transport — shared-memory ring (chosen) vs JPEG over a socket

| | shm ring (raw BGR) | JPEG over socket |
|---|---|---|
| Cost per consumer read | ~1 ms memcpy of 2.76 MB into a **reused** buffer (§9) | decode 8–15 ms CPU plus a fresh allocation per frame |
| Producer cost | ~0.85 ms copy per frame | JPEG encode ~5–10 ms per frame at 720p, every frame, whether anyone reads it or not |
| Pixel fidelity | exact (InsightFace crops for study_face and the TrackerVit need raw) | lossy |
| Allocation churn in the runtime | none when the reader reuses its buffer | new bytes plus a new ndarray per read, **the thing we are fighting** |
| Robustness | needs a seqlock and inode-change handling (done, tested) | trivially robust |

Chosen: **the ring.** It lives in `scripts/eyes/shm_ring.py`:

- **Format:** a raw mmap of `/dev/shm/iris_eyes_frames`, not `multiprocessing.shared_memory`, whose resource tracker on Python 3.11 unlinks a segment when an *attaching* process exits.
- **Slots:** 4. Each slot has a seqlock (`lock` odd while writing) and carries `seq`, `capture_ts`, `pub_ts` and the shape.
- **Header:** `latest_seq`, `writer_pid`, a `heartbeat_ts` that ticks even while parked, and `state` (starting / streaming / parked / stopped).
- **Readers:**
  - map PROT_READ;
  - re-map automatically when the writer restarts (new inode);
  - refuse frames older than `max_age_sec` or not newer than `newer_than`. The `newer_than` check is the cross-process form of person_track's stale-frame guard, and it fixes the two-clock problem at `person_track.py:326`: one `seq`, one `capture_ts`, for everyone.
- Correctness relies on x86-64 TSO store/load ordering. The server is x86-64, and this is noted in the module.

The orb is the one consumer that wants JPEG. It keeps encoding in the runtime, as it
does today (once per new frame, cached). If the MJPEG tab turns out to be a churn
source, phase 2 has the eyes serve MJPEG on their own loopback port and the orb
proxies it.

### 4.3 Results transport — Unix socket, ears protocol, plus push

`scripts/eyes/eyes_bus.py`:

- **RPC:** one JSON line in, one out, the same envelope as the ears daemon (`voice/wren_voice_daemon.py:166-225`).
  - Prototype commands: `ping`, `status`, `snapshot`, `park`, `unpark`.
  - Production adds: `enroll_start/status/cancel`, `reload_faces`, `analyze_image {path|ring_seq}`, `embed {path|ring_seq, bbox}`, `set_head`, `calibrate_gaze`, `set {hands_enabled, …}`, `tune`.
- **Subscribe:** the server pushes a `snapshot` whenever any producer publishes a new result, rate-capped at the subscriber's hz and with a 1 Hz heartbeat when idle. It also pushes `event` lines as they happen. A slow subscriber's queue drops the **oldest** entries; the eyes never block.
- **Snapshot shape** (p50 about 6 KB, max about 11 KB with landmarks):
  - `eyes_state`, `pid`, `ts`;
  - `frame{seq, capture_ts, fps}`;
  - one block per producer: `faces`, `recognized`, `body{boxes, tracks, bindings, offsets}`, `expression`, `attention`, plus `hands`, `pose` and `liveness` in production.
  - Each block carries `{frame_seq, frame_ts, done_ts, ms, value}`.
- **Staleness rules** use the **client's** clock against the result's *source frame* `capture_ts`:
  - fresh ≤ 0.35 s, aging ≤ 1.2 s (the same bands as frame_store), stale beyond that;
  - link dead = no line for 2 s;
  - ring writer dead = heartbeat older than 2 s.
- **Rule: blindness is never absence.** `presence_view()` returns `face_count=None` (unknown), not 0, whenever the eyes are down, parked or stale. This is the cross-process form of CORE's "a fresh age is not a live camera".

Why not HTTP or ZeroMQ: the ears already use JSON lines on a socket, there are no new
dependencies, and a Unix socket can't be reached from the LAN or Tailscale. The file
mode is 0600.

### 4.4 The runtime side: the `eyes_link` shim (one module, one thread)

This is what keeps the consumer edits small. Under the flag, `_eager_init_engines`
skips the InsightFace, expression and eye_tracker bootstraps and the capture thread,
and `iris_bootstrap` skips the body, pose and liveness loops. Instead it starts
`brain/eyes_link.py`, which does three things.

**1. Mirror.** A `Subscriber` on the socket writes the **same `g` keys** from each
snapshot:

- `_face_results`, `_recognized_*`, `_face_age/_gender`;
- `_current_expression`, `_attention_state`, `_looking_at_screen`, `_gaze_region`;
- `_hand_results`, `_human_pose_live`;
- `_capture_fps_measured`, `_cam_props`, `_perc_*_last_ms`.

`_person_present_since_ts` is kept in the shim. When the link or the results go
stale, it writes `_face_results = None` (already the meaning "not seeing" in deep
rest, `runtime_repair_tool.py:172`) and the `_recognized_*` keys go to `unknown`.

**2. Events.** It re-fires eyes events on the real `g["_signal_bus"]`, so
`/api/v1/signals`, the host's wakes and the attention arbiter don't change.
Face transitions are recomputed **in the shim** with a third state, UNKNOWN:

- known → unknown → same known fires nothing;
- an eyes restart never produces a `face_lost` + `face_appeared` wake pair;
- the shim fires `eyes_down` / `eyes_up` (ledger only, not wakes).

**3. Proxies:**

- **`frame_store` ring mode.**
  - `get_buffered_frame` reads the ring into one cached buffer per `seq`: N consumers in the same tick share one copy. Readers that keep a frame past the next tick get `.copy()`.
  - `push_frame` becomes a no-op.
  - `peek_buffer_age_sec` reads the ring header.
  - `_buffer_ts` and `_buffer_frame` become module `__getattr__` properties, for the orb MJPEG, iris_health and eyes_reload readers that touch them directly.
  - `_raw_frame_slot` becomes a property that returns `(capture_ts, frame)` from the same cache, which serves vision_fps_probe and face_liveness.
- **`EyesFaceProxy`** goes in `g["_insight_face"]`. It is duck-typed:
  - `available` is true only when the link is alive AND the eyes report the engine is up;
  - `provider()`, `known_count()`, `update_known_faces()`, `add_face()`, `analyze_frame(frame)` and `embed(...)` become RPCs;
  - `analyze_frame` of a ring frame sends its `seq`; an arbitrary image is written to `/dev/shm` and its path is sent.
  
  This covers face_pipeline_probe, study_face, enroll_face, face_db, person_onboarding, orb debug and health.
- **`EyesExprProxy` / `EyesEyeTrackerProxy`:** `available` plus `calibrate()` as an RPC. `get_eye_tracker()` callers get the proxy.
- **The enroll handshake.** The shim watches `g["_enroll_request"]`, sends `enroll_start {pid, count, interval_s, require_face}`, polls `enroll_status`, writes `saved_paths/remaining` back into the same dict, and sets `g["_enroll_result"]` exactly as the loop does now. study_face, unknown_capture and enroll_face don't change.

### 4.5 Lifecycle

**The unit** is `iris-eyes.service`, a systemd `--user` unit with the same shape as
`iris-ears.service`. A draft is at `scripts/eyes/iris-eyes.service.draft`; it is not
installed.

```ini
[Unit]
Description=Iris eyes (camera capture + vision workers, V100)
After=iris-mouth.service
[Service]
Type=notify                      # READY=1 after models are warm + camera streaming
NotifyAccess=main
WorkingDirectory=%h/IRIS
Environment=MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2
ExecCondition=%h/venvs/iris-v100/bin/python %h/IRIS/scripts/eyes/flag_on.py   # flag file says eyes-process mode
ExecStart=%h/venvs/iris-v100/bin/python %h/IRIS/scripts/eyes/iris_eyes.py --camera
WatchdogSec=20                   # WATCHDOG=1 only while frames advance OR parked on purpose
Restart=always
RestartSec=5
MemoryHigh=3G                    # reclaim pressure before…
MemoryMax=4G                     # …the eyes (not me) get OOM-killed and restarted
[Install]
WantedBy=default.target
```

- **Watchdog.** A hung `VIDIOC_DQBUF` (a classic UVC failure) stops the pings; systemd kills and restarts the unit. That gives the eyes a real hang recovery for the first time.
- **OOM.** The eyes are deliberately **not** in the `iris-oom-protect` list (−1000). They are the restartable part. Under memory pressure, the kernel should take the eyes before the core.
- **Pause flag** (`body_pause_flag`, used by voice_body_pause and "let me use the cam"): the eyes watch the same file. They release the streaming fd, set the ring state to parked and keep heartbeating. This is the same contract as today.
- **Quiet hours and gpu_park.** "Park" becomes `systemctl --user stop iris-eyes`, which frees the camera, about 1.3 GB of VRAM and the whole heap. "Unpark" is `start` (warm-up was 10 s in the prototype).
  - `eyes_quiet_hours` and `gpu_park` gain an eyes-process branch.
  - The `__main__._iris_video_capture_loop` check at `eyes_quiet_hours_tool.py:128` must accept "the eyes unit exists".
  - **Side benefit: a fresh eyes heap every morning.** This is the in-process night-park effect, made total.
- **eyes_reload** becomes `systemctl --user restart iris-eyes`, the "restart my eyes alone" Zeke wanted. Deep mode, the Windows `pnputil` path, is retired on Linux.
- **The runtime when the eyes are down:**
  - `_face_results=None` and `presence_view` reads unknown;
  - `latest_frame` returns 204, as it does today for a stale frame;
  - `iris_health` reports `camera: eyes_down` with a reason taken from the ring header (absent, stopped, dead heartbeat, or parked);
  - the servo refuses to start on a stale frame, as it does today (`attention_smooth:1638`);
  - nothing in the runtime blocks on the socket: RPCs time out in 1 s, and the mirror is a background thread.
- **Startup order:** none required. The ring reader and the subscriber attach lazily and reconnect, so either side can start first.

### 4.6 GPU

| | VRAM | Notes |
|---|---|---|
| Measured in the prototype, face + body | 1.26–1.27 GB | InsightFace buffalo_l (640 + 320 apps) plus YOLOX-s, ORT CUDA EP |
| MediaPipe expression + attention | +0 | CPU |
| Production adds pose (YOLO11n-pose, torch) | about +0.3–0.5 GB (estimate, not measured) | |

Today the same models sit in the runtime's 3.2 GB CUDA context. After the split, the
runtime's own CUDA usage should drop to whatever non-vision engines it still loads,
which needs measuring at cut-over. **Two CUDA contexts cost about 300–400 MB of
overhead each.** That is acceptable on the 32 GB V100.

## 5. Effect on every consumer

| Consumer | Change needed | Risk |
|---|---|---|
| orb latest_frame / live_frame / mjpeg (`orb_http.py:70-99,908-950`) | none (frame_store ring mode; `_buffer_ts` becomes a property) | low |
| orb /snapshot, /debug (`:380-669,1914-1974`) | none (mirrored keys) | low |
| orb calibrate_gaze (`:1680`) | none (proxy calibrate → eyes RPC) | med: second-opener removed |
| attention_smooth servo (`:600-1650`) | none for keys and frames. **person_track calls switch to `eyes_link.body_offset(label)` / `track_boxes()`** (thin `brain/person_track` facade under the flag) | **HIGH: pursuit latency and the inline-step fallback** |
| attention_search (`:105-116`, calls `step()` inline) | facade: `step()` becomes a no-op and `target_offset` reads the mirror | med |
| attention_follow / engage sentry / multi / room_map / ptz_predict / visual_attention / head_odometry / object_lock | none (frames via ring, faces via mirror) | low |
| scene_memory (`scene_memory.py:235-300,502`) | none; `body_pose.live()` and `track_boxes()` come from the mirror | low |
| study_face (`study_face_tool.py:127,276-512`) | none (enroll handshake plus EyesFaceProxy embed) | **med: multi-step handshake** |
| enroll_face (`iris_runtime.py:2652-2727`) / unknown_capture | none (handshake) | med |
| face_pipeline_probe | `repair=true` becomes an eyes restart RPC; the rest goes through the proxy | low |
| face_db, person_onboarding, camera.py, operator_server | none (proxy) | low |
| face_liveness (shadow, OFF) | moves into the eyes with its tool's flag; the tool reads `liveness` from the mirror | low (it is off) |
| human_pose tool / body_pose loop | the loop moves to the eyes; the tool's `read` → RPC; `live()` → mirror | med |
| vision_fps_probe | rewrite to read the ring seq/ts and eyes `status` (it checks thread names) | low |
| gpu_park | eyes branch: stop/start the unit, verify via the ring header instead of thread names | **med: thread-name checks** |
| eyes_quiet_hours | eyes branch (see §4.5); fix the `__main__` check | med |
| eyes_reload | → `systemctl --user restart iris-eyes` | low |
| runtime_repair `_eyes_rest`/`_hands_rest` | → park RPC / `set hands_enabled` | low |
| iris_health camera probe (`iris_health_probes.py:113-174`) | add an "eyes" block: unit active, ring state, heartbeat age, fps, frame-content diff by seq | low |
| boot device probes (`iris_runtime.py:435-473`) | skip under the flag (the runtime must never open the device) | low |
| perception wakes (host `perception_reader`, signals route) | none (events re-fired on the real bus; shim de-dupes across eyes restarts) | **med: wake storms if de-dupe is wrong** |
| iris_attention_sources camera fallback | none (reads mirrored `_recognized_*`) | low |
| ambient_snapshot / inner_monologue / heartbeat / wake_detector / voice_loop / question_engine / sleep_mode / app_status_board | none | low |
| vector_overhead → `read_live_frame_with_meta` | ring-only under the flag (no device open) | low |
| legacy Ava paths (background_ticks, heartbeat.py, startup.py) | none (not on the Iris path) | — |

## 6. Risks

1. **Pursuit regression.** The servo now sees body and face results one socket hop
   later. Prototype: capture → client p50 is about 44 ms, of which transport is
   about 9–12 ms p50 and 17–28 ms p95 (§9.1).
   - Mitigations: the subscription is result-driven, and the shim subscribes at 60 Hz.
   - The Kalman velocity is already in `offsets`.
   - Phase 2 moves the servo itself if Zeke sees lag.
2. **Wake storms or missed wakes** if the shim's transition logic diverges from the
   worker's. Mitigation: port the face-transition code verbatim, add the UNKNOWN
   state, and unit-test the restart de-dupe offline before cut-over.
3. **Two streamers** (flag mismatch). Mitigation: camera flock plus the
   `ExecCondition` flag check, and the runtime takes the same lock.
4. **Growth moves, not cured.** The eyes reproduce the creep (§9). The eyes being
   restartable and capped is the containment. The cure (§10) is still owed.
5. **Duck-typed proxies miss a method.** A runtime caller that reaches into
   `_insight_face._app` or `._known` would break. Mitigation: grep all
   attribute uses at implementation time, and give the proxy a `__getattr__` that
   logs and raises clearly.
6. **systemd watchdog false kills** during model warm-up or a long enroll.
   Mitigation: `READY=1` only after warm-up, `EXTEND_TIMEOUT_USEC` during enroll, and
   pings tied to "frames advance OR parked on purpose".
7. **/dev/shm leftovers** after a SIGKILL. These are harmless: the writer `O_TRUNC`s
   on start, and readers key on the inode. Total ~11 MB.
8. **The x86 memory-ordering assumption** in the seqlock. It is documented, and the
   multi-process torn-frame test covers it on this host.
9. **The ring can be read by any process of user `iris`.** That is the same trust
   boundary as today's `state/` files; the socket and ring modes are 0600.

## 7. What the prototype does NOT do (production work list)

- **Not built yet:**
  - camera capture (`--camera`), which needs the flock, MJPG 720p@60 request and pause-flag yield ported from `iris_runtime.py:4254-4300`;
  - the enrollment hook;
  - face_tracking (it writes live `state/`);
  - the pose loop, the liveness shadow and the hands step;
  - `set_head`;
  - sd_notify;
  - the `eyes_link` shim and the proxies;
  - the flag file and `flag_on.py`.
- **Runtime edits** behind the flag: the `_eager_init_engines` and `iris_bootstrap` skips, the frame_store ring mode, the person_track facade, and the gpu_park / quiet-hours / eyes_reload / health branches.

## 8. Migration: feature flag, cut-over, rollback

The flag is `state/eyes_process.json` → `{"enabled": true|false, "since": …, "why": …}`
(absent means false). The runtime reads it at boot. The eyes unit reads it in
`ExecCondition`, so both sides key off **one file**. Dual-run is impossible because
there is only one camera, so:

1. **Land the code with the flag OFF.** The runtime behaves byte-for-byte as today
   apart from taking the camera flock. Unit installed but not enabled. Offline soak:
   the eyes on a video file (as in this prototype), and the shim pointed at it from a
   **test harness**, not the live runtime, with the transition and de-dupe tests green.
2. **Cut-over window**, with Zeke present (needs his face for verification) and after
   the pre-restart contract (handoff, cold-wake audit):
   - set `enabled: true`;
   - `systemctl --user enable --now iris-eyes`. The ExecCondition passes, but capture waits on the flock, still held by the live runtime;
   - restart the stack. The runtime boots without capture, the eyes take the flock and stream, and the shim attaches.
3. **Verify checklist:**
   - **Process level:**
     - `iris_health.camera` is fresh and its source is eyes;
     - runtime thread count drops by about 400 (ORT pools, §9), and VRAM moves to the eyes pid;
     - `systemctl --user restart iris-eyes` costs about 15 s of blindness with **no** face_lost/face_appeared wake pair;
     - a pause-flag yield and resume works;
     - a quiet-hours stop and start works.
   - **Things to look at:**
     - orb camera tab live;
     - `latest_frame` 200;
     - `/snapshot.current_person` = zeke when he sits down, with a `face_appeared` wake reaching the host;
     - servo pursuit judged by Zeke's eye (the real test);
     - `room_map look`;
     - `study_face` on him: gallery written, enroll_result set.
   - Then watch runtime RSS and mallinfo over 24 h. It should stay flat, while the eyes' RSS carries the creep.
4. **Rollback** (any failed check, or Zeke says so):
   - `systemctl --user disable --now iris-eyes` (releases the flock and the device);
   - set `enabled: false`;
   - restart the stack. The runtime boots the in-process capture exactly as today.
   
   Nothing else changes state, so rollback is about one minute plus a stack restart.
   Keep the flag-OFF path for at least two weeks before deleting the in-process loop.

## 9. Prototype measurements (2026-10-19, offline)

Setup:

- **Input:** `/data/iris_eyes_test/keyframes_30s.mp4`, 900 frames at 1280x720/30 fps, built from 60 scene_memory keyframes (40 with faces), slow-panned, looped.
- **Detectors:** the runtime's own InsightFace (CUDA EP, 5 known people) and YOLOX-s + ByteTrack, plus MediaPipe expression and eye-tracker in the full run.
- **Client:** the 30 Hz `eyes_client.py`.

### 9.1 Transport (the cost of the boundary)

| Metric | Value |
|---|---|
| Ring write, eyes side (2.76 MB copy into shm) | p50 0.6–0.9 ms, p95 ~1.1–1.8 ms |
| Ring read into a reused buffer, client side | p50 0.93 ms, p95 1.27 ms, max 15 ms (one GC/scheduling outlier in 5,742 reads) |
| Frame age when the client reads it | p50 17 ms, p95 33 ms (i.e. ≤ one frame) |
| Torn-read retries | **0** in about 9,000 live reads. The seqlock is exercised by the 2-slot multi-process stress test instead. |
| RPC `ping` round trip | p50 1.3 ms, p95 2.2–2.8 ms |
| Snapshot size | p50 ~6 KB, max ~11 KB (106-point landmarks dominate) |
| Face result: eyes done → client, 30 Hz cap, face+body | p50 11.5 ms, p95 28 ms |
| Face result: eyes done → client, 60 Hz cap, all 4 producers | p50 9.0 ms, p95 17 ms |
| Face result: camera frame → client (end to end) | p50 43–45 ms, p95 65 ms |
| Body (YOLOX+ByteTrack) frame → result in the eyes | p50 18–27 ms |
| Subscriber queue drops | 0 |

Transport latency is dominated by the rate cap and the GIL. Every snapshot is built
and JSON-encoded in Python while the workers run. That is acceptable for the orb,
wakes and presence. For the servo it adds roughly a third of a frame. Two cheaper
options if it matters: per-producer delta messages (encode only the changed block),
or phase 2 (the servo into the eyes).

### 9.2 Throughput and GPU

| | |
|---|---|
| Capture | 29.7–29.9 fps sustained over 5 min (paced at 30) |
| InsightFace `analyze_frame` | p50 14–17 ms, p95 31–37 ms. The p95 is the every-3rd-frame identity pass. |
| YOLOX + ByteTrack `step` | p50 13–17 ms, p95 ~19 ms |
| MediaPipe expression / attention | p50 ~24 ms each (CPU) |
| VRAM, eyes process | **1.25–1.27 GB** (face + body; MediaPipe adds 0) |
| Model warm-up (process start → serving) | **10 s** (InsightFace 9.7 s, YOLOX 0.2 s). This is the blind window of an eyes restart. |
| Threads | 410 (face+body), 490 (+ MediaPipe: 72 `mediapipe/*` threads). **~400 are unnamed ORT intra-op pool threads.** Measured: each default ORT CUDA session adds ~36 threads (one per core on this 36-core box); `intra_op_num_threads=1` adds 0. |

### 9.3 RSS and glibc heap: the growth reproduces, and the A/B narrows it

| Run (all looped on the same video) | Length | RSS start→end | mallinfo in_use | **free-held slope** | RSS slope |
|---|---|---|---|---|---|
| face + body + expr + attn, default arenas | 5 min | 2014 → 2323 MB | flat ~1.18 GB | **+61 MB/min** | +68 MB/min |
| face + body only, default arenas | 3 min | 1884 → 1894 MB | flat ~1.13 GB | **−0.8 MB/min** | +4 MB/min (warm-up) |
| face + body + expr + attn, `MALLOC_ARENA_MAX=2` | 3 min | 1850 → 1870 MB | flat ~1.18 GB | **+3 MB/min** | +8 MB/min (warm-up) |

What this shows (one sample per arm, minutes not a day, so treat it as strong
evidence, not proof):

1. **The vision pipeline alone reproduces the runtime's signature**: in_use flat,
   free-held climbing. No MCP, orb or servo is involved.
2. **The growth comes with the two MediaPipe workers (expression + eye-tracker),
   not with InsightFace or YOLOX.** py-spy's "busiest threads" were the frame-rate
   threads; the *fragmenting* allocations come from the MediaPipe pair. I did not
   isolate which of the two.
3. **Capping arenas removes it** (free-held about 55–70 MB instead of 227 → 525 MB)
   at identical fps and latency. The runtime's `MALLOC_ARENA_MAX=4` (commit
   `34125a5`, lands next restart) is therefore likely to help a lot by itself.
   4 is not 2, though, so it has to be measured.

## 10. Follow-ups this points at (not this round)

- **Thread pools.** Each ONNX Runtime session builds its own intra-op pool with one
  thread per core. With 36 cores and about 11 sessions (InsightFace 640 and 320 apps
  × 5 models, plus YOLOX), that is about 400 native threads in the eyes alone, which
  accounts for most of the runtime's ~600. Each thread can take its own glibc arena.
  - Set `SessionOptions.intra_op_num_threads=1–2` for CUDA EP sessions (CPU work is tiny there), in `insight_face_engine` and `yolox_person`.
  - Free the 320-det "small" FaceAnalysis after the gallery loads.
  - This is the likely real cure. It is a `brain/` edit, so it is out of scope here.
- Set `MALLOC_ARENA_MAX=2` in the eyes unit (in the draft). For the runtime, verify
  that `=4` gives the same flat line (§9.3).
- Isolate expression vs eye-tracker as the fragmenter (one more 3-minute A/B). Both
  use MediaPipe's own ~70-thread pool, so capping that pool may be the targeted fix.
- Phase 2: the servo into the eyes, and the eyes serving MJPEG.
