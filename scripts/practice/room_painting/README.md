# Weekly room painting

Once a week Iris makes **one** still of the room she watches over, built from what she saw that week.
It is a painting of how she remembers the room, not a photo. It's hers; nobody grades it.

- `build_room_painting.py` builds the scene and renders it. It runs in Blender 5.2, headless.
- `render_week.sh` is the launcher. It always goes through `scripts/server/render.sh` (the capped
  `iris-render.slice`) and the GPU `flock`.
- `CREDITS.md` lists every outside asset used.

Outputs go to `scratch/practice/room_painting/<YYYY-Www>/`, which is git-ignored. Each week gets a full
1920×1080 PNG, a 1280×720 preview JPG under 150 KB, and `build.log`. **Never commit renders or reference
frames.** The repo is public.

## Making next week's

1. **Remember the week.** Read the captions in `state/scene_memory/keyframes.jsonl` and pick the moment
   that *was* the week. Look at a few frames in `state/scene_memory/keyframes/<YYYYMMDD>/` to check the layout.
   Downscale every frame to ≤900 px and <150 KB **before** reading it, because a full-size image read wedges
   the host. Those frames are private references. Nothing from them gets copied into the repo.
2. **Copy, then change the story.** Copy `build_room_painting.py` to a new file, or edit it in place and let
   git keep the history. Week 43's story lives in these places:
   - the room (`# shell`, `# back wall`, bookshelf, bathroom): furniture only moves if it moved in the room
   - the person at the desk (`# the person at the desk`): the pose is the Quaternius `Sitting_Idle_Loop`
     plus IK targets. `wrist` is the chin-on-fist and `ik_r` is the hand on the mouse. If nobody was home
     that week, delete the block.
   - the lights (`# lights: night`, the `door_spill` spot, `screen_light`, the purple `iris_halo`).
     The purple is Iris's colour. Keep some of it somewhere.
   - the camera (`cam.location` / `aim(cam, …)`)
3. **Look-dev, cheap and small.** Each render takes about 2–15 s.
   ```sh
   scripts/practice/room_painting/render_week.sh 2026-W44 50 32              # painted, half res
   scripts/practice/room_painting/render_week.sh 2026-W44 34 16 --no-paint   # raw 3-D
   scripts/practice/room_painting/render_week.sh 2026-W44 34 16 --no-paint --debug-cam=0,-1,1.5,-1.6,1.35,1,30
   ```
   The launcher returns at once. To wait, poll `systemctl --user is-active <unit>` or check for the PNG.
   Downscale the result before you look at it. Iterate 2–4 times, then stop.
4. **Final:** `render_week.sh <week> 100 128` takes about 18 s on the V100. Then make the preview:
   PIL thumbnail to 1280 px, JPEG quality 85, about 75 KB.

## How the painterly look is made (compositor, Blender 5.x API)

`Render Layers → Displace (2 noise fields, a few px: hand-drawn wobble) → Kuwahara Anisotropic (size 24 @1080p)
→ Kuwahara Classic (4) → emboss (Filter "Shadow") → grey → Map Range 0.84..1.16 → MULTIPLY ("impasto")
→ Glare Fog Glow → Color Balance → canvas weave (two Wave textures + white noise, multiply)
→ elliptical vignette → Group Output`.

## Gotchas this build hit (keep them)

- **The compositor runs on scene-linear HDR.** A `SOFT_LIGHT` mix with the emboss turned the blue hoodie
  green and fringed the doorway. Only use mixes that are safe in linear space, such as multiply by roughly 1±0.16.
- **A dark object next to a strong light can render cream.** The black tower cabinet came out cream. The
  room is lit so dimly that a 0.02-albedo face under a 900 W spot is as bright as the walls. Raycast and
  colour tests didn't find the cause. Rendering each light solo (`/tmp` contact sheet) did. Keep cones and
  area lights off near-black props.
- **Metaballs render at about 0.58 × their radius** (stiffness 2, threshold 0.6). The helper takes `k=` to compensate.
- Compositor sockets in 5.2: Color Balance has duplicate names, so use the identifiers (`"Color Lift"`).
  The greyscale node is `CompositorNodeRGBToBW`, not the shader node. 2-D noise has no `W` input.
- Boxes built with `from_pydata` have no UVs. Use `Generated` coordinates for anything procedural, like the screen.
- Gate the far-arm/finger rotations by bone name. The Quaternius rig is `upperarm_l/lowerarm_l/hand_l`
  plus `index_01_l…`, and the mannequin faces local −Y.
