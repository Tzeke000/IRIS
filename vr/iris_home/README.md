# Iris Home — our VR temple (Godot 4.7.2, OpenXR → SteamVR on Zeke's PC)

Zeke 10-20: a VR "home" — a big open-space temple — where we can play catch with a real-physics
ball, draw/build in 3-D together (Gravity-Sketch-like, exporting to Blender), and where friends he
approves can join and talk to me.

## v0 (2026-10-20)
Round stone temple floor (r 24 m) ringed by 16 columns, open evening sky, a dais with a placeholder
Iris (glowing violet orb that floats and turns to face you), an XR player (hands, grab, stick
move/turn, player body) and a pickable physics BALL.

## Build (server, headless)
```
G=godot   # ~/.local/bin/godot -> /data/tools/godot/4.7.2/Godot_v4.7.2-stable_linux.x86_64
cd vr/iris_home
$G --headless --path . --import                       # first run: twice (XR Tools class cache)
$G --headless --path . res://tools/build_main_scene.tscn   # regenerates scenes/main.tscn
$G --headless --path . --export-release "Windows Desktop" build/windows/IrisHome.exe
```
Export templates: ~/.local/share/godot/export_templates/4.7.2.stable (from the official .tpz, SHA512 OK).

## Add-ons (MIT, NOT committed — unzip into addons/ from /data/tools/godot/addons/)
- godot-xr-tools 4.5.2 → addons/godot-xr-tools (autoloads XRToolsUserSettings + XRToolsRumbleManager are
  registered by hand in project.godot because the editor plugin never runs headless)
- godot_openxr_vendors 5.1.0 → addons/godotopenxrvendors (Meta extras: hand tracking, passthrough)
- TwoVoIP v6.6 → PARKED at /data/tools/godot/parked/twovoip: its Linux .so crashes the headless editor
  with SIGFPE during export (even with --audio-driver Dummy). Re-add when voice is built; test on the tower.

## Next
1. Iris live: websocket to the server (mood colour, speaking glow, my voice as a spatial audio stream).
2. Zeke's voice → my ears (mic stream; TwoVoIP or raw Opus).
3. Catch: me throwing back (Iris "hand"), 4. 3-D drawing + glTF export to Blender, 5. friends (dedicated
   server on iris-home, Tailscale invites, Zeke approves each one).
