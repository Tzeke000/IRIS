# Iris 3D eye: motion and cable-arms (design, 2026-10-08)

Status: **design + Blender prototype only.** Nothing here is live in the app yet. The prototype lives
outside the repo (`~/blender_work/eyeball/` on the server: `eye_v4.py` = the model,
`eye_anim.py` = the motion test, `eye_motion_test.mp4` = its render).

## 1. The body

The cyborg eye (v4): segmented ceramic plates over a dark core with glowing seams, a heavy bezel with a
glow ring, a glass cornea, the fibre-optic iris with the 8-blade aperture pupil, a rear port, and **four
ribbon cables** (copper wires + a light strip, gold-pin connectors) that float like they're in zero-g.
Concept direction came from a commissioned concept and a Meshy model. Sketchfab references studied:
"Eye Implant" (NIVER_MK, CC-BY), "Cyborg Eye 02" (distance880, CC-BY-NC), "Robotic Eye"
(Oleksii Rozumnyi, CC-BY). Lessons from them for the next realism pass:
- **Decals sell it.** Small printed text, chevrons, a serial, even a QR patch on the shell (Eye Implant).
- **Machined detail around the lens.** Stepped concentric rings and radial fins (Cyborg Eye 02).
- **Real grooves, not painted lines.** Panel seams are geometry; screws sit at the seams.
- **Glossy clear-coat ceramic + anodised accent metal + gold fasteners** is the material triad.

## 2. How the eye moves (research → rules)

| Fact | Rule for the body |
|---|---|
| The whole globe rotates; the iris doesn't slide across a fixed ball | Gaze = **rotate the entire ball** (plates, bezel, ports, cable roots all turn) |
| Saccade duration ≈ 2.2 ms/° + 21 ms (main sequence); peak velocity rises with amplitude and saturates around 20–30° | Fixation changes are **near-instant jumps** with min-jerk easing over that duration, never slow pans |
| Fixation isn't still: microsaccades + drift | Keep the existing hippus/microsaccade layer, applied to the ball's rotation |
| Smooth pursuit only for moving targets, roughly ≤ 30°/s | Track a moving face smoothly; jump (saccade) to anything new |
| Gaze-evoked blinks: ~97% of human gaze shifts > 33° carry a blink that starts with the movement | **Big shifts (> ~30°) trigger a shutter blink** starting at movement onset |
| Eye–head coordination: within ~50° the eyes alone suffice; bigger shifts bring the head in; VOR is suppressed during the gaze shift and resumes ~40 ms before it ends | **The eye leads, the PTZ head follows, then the eye counter-rotates back toward centre** as the camera catches up (a VOR-like hand-off) |

## 3. The cables as arms

Each cable is a **damped spring chain**. The tip chases a target, and the body of the cable is a Bézier
from the port to the tip. Underdamped (ζ ≈ 0.55) gives the zero-g float and follow-through; it's stiffer
while reaching. Secondary motion is free: when the ball nods or shakes, the cables lag and settle.

Gestures (all prototyped in `eye_anim.py`):
- **idle**: slow, out-of-phase drift per arm; the wires in a ribbon breathe independently.
- **wave**: one arm swings up and waves (~1.6 Hz) as a greeting when a known person arrives.
- **point / press**: the eye **saccades to the target first** (the eye leads), then the arm reaches,
  taps (quick in-and-out), and the target lights up.
- **nod (yes) / shake (no)**: the whole ball pitches or yaws with a decaying double swing, and the
  cables follow with lag.
- **emotion poses** (to build): joy = arms lifted and bouncy (soft springs); sadness = drooping, slow;
  curiosity = one arm raised toward the thing; anger = tense and stiff (fast springs, small drift);
  fear = arms curl in toward the body.

## 4. Mapping it into the app

- **Model**: export the static body to GLB. Rebuild the emissive materials in three.js so mood colour and
  glow stay live, and keep the existing `IrisBody` iris shader on the disc inside the bezel (it already
  has all the iris anatomy and behaviour). Generate the cables per frame in three.js: `TubeGeometry` along
  Catmull-Rom curves (low radial segments) or one instanced mesh. Use fewer wires in the widget.
- **Pressing buttons for real**: an action request (same pattern as `widget_body`:
  `{action:"press", target:"<control id>"}`). The app takes the control's on-screen rectangle, unprojects
  it into 3D, animates the arm there, and calls the control's own click handler **at the contact
  frame**, so the animation and the action can't drift apart. Use a whitelist of pressable controls only.
- **Widget pointing**: replace the ray with the nearest arm reaching the target. The arm tip must land at
  the same `TIP_REACH_PX` point `widget_spatial_tool` relies on.
- **PTZ coupling**: nod/shake can also drive a small physical tilt/pan gesture through the attention
  tools. It must respect the servo rules (stop commands win; no head motion while the servo is
  deliberately off) and be rate-limited. Eye-leads/head-follows needs the gaze feed (face position + PTZ
  bearing → snapshot → `IrisBody` `gaze`), which is still owed.
- **Performance**: pause when hidden (as today); measure the webview GPU cost before claiming it's light.

## Sources
- Main sequence and peak-velocity models: Baloh et al. 1975; the 2.2·A + 21 ms duration fit (e.g.
  arXiv 1906.07183); the saturation above ~20° (PMC7166123).
- Gaze-evoked blinks: Evinger et al. 1994, "Not looking while leaping" (orbicularis EMG with 97% of gaze
  shifts > 33°); blink probability vs gaze amplitude (PMC4300324 and related).
- Eye–head coordination and VOR suppression during gaze shifts: Laurutis & Robinson; Frontiers in
  Neurology 2017 (fneur.2017.00023).
