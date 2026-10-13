"""Blast Off v4 — the camera-feed WARNING HUD as a separate compositor pass over the finished beauty frames.

    blender -b --factory-startup --python-exit-code 1 --python post_hud.py -- \
        --plan OUT_DIR/hud_plan.json --src OUT_DIR/frames --out OUT_DIR/hud [--start S --end E] [--pct 100]

Why a separate pass (research #1 §5, tested 0.42 s/frame at 1080p): Standard view (no second AgX tonemap on the
text), and a wording change costs minutes, not a 3-D re-render. The build writes hud_plan.json: for each Blender
frame on a camera-feed shot -> [feed label, mode, fisheye, breakup, beats, kick]. Zeke's brief: WARNING-ONLY
(no telemetry), B612 font, the approved wording "⚠ BOOSTER 3 FAULT — EXPLOSION IMMINENT".
  mode 1  before the blast: the red warning box blinking on the half-beat + a red edge pulse on the kicks
  mode 2  SIGNAL LOST (black + static)
  mode 3  after the blast, mid-tumble: "⚠ BOOSTER 3 LOST — ATTITUDE ERROR" (NOT yet approved by Zeke — edit
          WARN3 below; it only costs this pass)
  fisheye: a dark circular lens edge; breakup > 0: digital breakup (RGB split, tear band, block pixelation)
Frames are read from --src/f_####.png and written to --out/f_####.png (temp name, then rename). Resumable.
"""
import argparse
import json
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=1)
ap.add_argument("--end", type=int, default=10 ** 6)
ap.add_argument("--pct", type=int, default=100)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args(argv)

WARN1 = "⚠ BOOSTER 3 FAULT — EXPLOSION IMMINENT"       # approved by Zeke 10-13
WARN3 = "⚠ BOOSTER 3 LOST — ATTITUDE ERROR"            # placeholder wording (not approved yet)
LOST = "SIGNAL LOST"

plan = json.load(open(a.plan))
frames = {int(k): v for k, v in plan["frames"].items()}
font_path = plan.get("font") or "/data/iris_assets/fonts/b612/B612-Bold.ttf"
FONT = bpy.data.fonts.load(font_path)
mono = os.path.join(os.path.dirname(font_path), "B612Mono-Bold.ttf")
FONT_M = bpy.data.fonts.load(mono) if os.path.exists(mono) else FONT

sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1080, a.pct
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.render.use_compositing = True
sc.render.use_sequencer = False
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
W, H = 1920 * a.pct / 100.0, 1080 * a.pct / 100.0
S = a.pct / 100.0

cg = bpy.data.node_groups.new("HUD", "CompositorNodeTree")
cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
N_, L_ = cg.nodes.new, cg.links.new


def val(v=0.0):
    n = N_("ShaderNodeValue")
    n.outputs[0].default_value = v
    return n


def math_(op, x, y):
    n = N_("ShaderNodeMath")
    n.operation = op
    for k, s in enumerate((x, y)):
        if isinstance(s, (int, float)):
            n.inputs[k].default_value = s
        else:
            L_(s, n.inputs[k])
    return n.outputs[0]


def mix(fac, a_, b_, blend="MIX"):
    n = N_("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    for k, s in ((0, fac), (6, a_), (7, b_)):
        if isinstance(s, (int, float)):
            n.inputs[k].default_value = s
        elif isinstance(s, tuple):
            n.inputs[k].default_value = s
        else:
            L_(s, n.inputs[k])
    return n.outputs[2]


def text(string, size, color, x, y, font=FONT, halign="Left"):
    """coloured text with a transparent background, its anchor translated to (x, y) px from the centre"""
    t = N_("CompositorNodeStringToImage")
    t.inputs["String"].default_value = string
    t.inputs["Font"].default_value = font
    t.inputs["Size"].default_value = size * S
    try:
        t.inputs["Horizontal Alignment"].default_value = halign
    except TypeError:
        pass
    bw = N_("CompositorNodeRGBToBW")
    L_(t.outputs["Image"], bw.inputs["Image"])
    col = mix(1.0, t.outputs["Image"], color + (1.0,), "MULTIPLY")
    sa = N_("CompositorNodeSetAlpha")
    L_(col, sa.inputs["Image"])
    L_(bw.outputs["Val"], sa.inputs["Alpha"])
    tr = N_("CompositorNodeTranslate")
    L_(sa.outputs["Image"], tr.inputs["Image"])
    tr.inputs["X"].default_value = x * S
    tr.inputs["Y"].default_value = y * S
    return t, tr.outputs["Image"]


def over(bg, fg, fac):
    n = N_("CompositorNodeAlphaOver")
    L_(bg, n.inputs["Background"])
    L_(fg, n.inputs["Foreground"])
    if isinstance(fac, (int, float)):
        n.inputs["Factor"].default_value = fac
    else:
        L_(fac, n.inputs["Factor"])
    return n.outputs["Image"]


img_node = N_("CompositorNodeImage")
src_img = None
base = img_node.outputs["Image"]
# --- digital breakup: RGB split + a horizontal tear band + block pixelation, scaled by 'Brk'
brk = val(0.0)
seed = val(0.0)
sep = N_("CompositorNodeSeparateColor")
L_(base, sep.inputs["Image"])
chans = []
for k, ch in enumerate(("Red", "Green", "Blue")):
    cc = N_("CompositorNodeCombineColor")
    for j, nm in enumerate(("Red", "Green", "Blue")):
        if nm == ch:
            L_(sep.outputs[nm], cc.inputs[nm])
        else:
            cc.inputs[nm].default_value = 0.0
    cc.inputs["Alpha"].default_value = 1.0
    tr = N_("CompositorNodeTranslate")
    L_(cc.outputs["Image"], tr.inputs["Image"])
    L_(math_("MULTIPLY", brk.outputs[0], (k - 1) * 14.0 * S), tr.inputs["X"])
    chans.append(tr.outputs["Image"])
split = mix(1.0, mix(1.0, chans[0], chans[1], "ADD"), chans[2], "ADD")
band = N_("CompositorNodeBoxMask")
band.inputs["Position"].default_value = (0.5, 0.45)
band.inputs["Size"].default_value = (1.2, 0.09)
tear = N_("CompositorNodeTranslate")
L_(split, tear.inputs["Image"])
L_(math_("MULTIPLY", brk.outputs[0], 90.0 * S), tear.inputs["X"])
torn = mix(band.outputs["Mask"], split, tear.outputs["Image"])
pix = N_("CompositorNodePixelate")
pix.inputs["Size"].default_value = max(int(18 * S), 2)
L_(torn, pix.inputs["Color"])
pband = N_("CompositorNodeBoxMask")
pband.inputs["Position"].default_value = (0.35, 0.7)
pband.inputs["Size"].default_value = (0.5, 0.18)
blocky = mix(math_("MULTIPLY", pband.outputs["Mask"], math_("GREATER_THAN", brk.outputs[0], 0.5)), torn, pix.outputs["Color"])
img = mix(math_("MINIMUM", math_("MULTIPLY", brk.outputs[0], 10.0), 1.0), base, blocky)
# --- feed grade: a touch of desaturation + contrast, like a downlinked video feed
fish = val(0.0)
lensedge = N_("CompositorNodeEllipseMask")
lensedge.inputs["Size"].default_value = (1.0, 1.0)
lb = N_("CompositorNodeBlur")
lb.inputs["Size"].default_value = (90 * S, 90 * S)
L_(lensedge.outputs["Mask"], lb.inputs["Image"])
img = mix(math_("MULTIPLY", fish.outputs[0], math_("SUBTRACT", 1.0, lb.outputs["Image"])), img, (0.0, 0.0, 0.0, 1.0))
# --- top-left feed label + blinking REC (no telemetry: Zeke wants the warning only)
lab_node, lab = text("TRACKING CAM 07", 36, (0.92, 0.95, 0.92), -900, 470, FONT_M)
rec_on = val(1.0)
rec_node, rec = text("● REC", 36, (1.0, 0.12, 0.1), 640, 470, FONT_M)
img = over(img, lab, 0.9)
img = over(img, rec, rec_on.outputs[0])
# --- the warning: a red box + white text, blinking on the half-beat; a red edge pulse on the kicks
wmode = val(0.0)                                         # 1 = warning box visible (modes 1 and 3)
blink = val(1.0)
kick = val(0.0)
box = N_("CompositorNodeBoxMask")
box.inputs["Position"].default_value = (0.5, 0.18)
box.inputs["Size"].default_value = (0.66, 0.085)
redbox = mix(box.outputs["Mask"], img, (0.75, 0.02, 0.02, 1.0))
img = mix(math_("MULTIPLY", math_("MULTIPLY", wmode.outputs[0], blink.outputs[0]), 0.86), img, redbox)
warn_node, warn = text(WARN1, 46, (1.0, 1.0, 1.0), 0, -350, FONT, "Center")
img = over(img, warn, math_("MULTIPLY", wmode.outputs[0], math_("ADD", 0.35, math_("MULTIPLY", blink.outputs[0], 0.65))))
edge = N_("CompositorNodeEllipseMask")
edge.inputs["Size"].default_value = (1.05, 1.05 * 16 / 9)
eb = N_("CompositorNodeBlur")
eb.inputs["Size"].default_value = (160 * S, 160 * S)
L_(edge.outputs["Mask"], eb.inputs["Image"])
epulse = math_("MULTIPLY", math_("MULTIPLY", math_("SUBTRACT", 1.0, eb.outputs["Image"]), wmode.outputs[0]),
               math_("ADD", 0.15, math_("MULTIPLY", kick.outputs[0], 0.6)))
img = mix(epulse, img, (1.0, 0.0, 0.0, 1.0), "ADD")
# --- SIGNAL LOST: black + static + white text
lost = val(0.0)
static = N_("ShaderNodeTexWhiteNoise")
static.noise_dimensions = "4D"
sv = N_("CompositorNodeCombineColor")
for nm in ("Red", "Green", "Blue"):
    L_(math_("MULTIPLY", static.outputs["Value"], 0.22), sv.inputs[nm])
sv.inputs["Alpha"].default_value = 1.0
img = mix(lost.outputs[0], img, sv.outputs["Image"])
lost_node, lost_t = text(LOST, 64, (1.0, 1.0, 1.0), 0, 0, FONT, "Center")
img = over(img, lost_t, lost.outputs[0])
go = N_("NodeGroupOutput")
L_(img, go.inputs["Image"])
sc.compositing_node_group = cg

os.makedirs(a.out, exist_ok=True)
done = skipped = 0
t_all = time.time()
for f in sorted(frames):
    if f < a.start or f > a.end:
        continue
    srcp = os.path.join(a.src, f"f_{f:04d}.png")
    dst = os.path.join(a.out, f"f_{f:04d}.png")
    if not os.path.exists(srcp):
        continue
    if not a.overwrite and os.path.exists(dst) and os.path.getsize(dst) > 1024:
        skipped += 1
        continue
    label, mode, fisheye, brk_v, beats, kick_v = frames[f]
    if src_img is None:
        src_img = bpy.data.images.load(srcp)
        img_node.image = src_img
    else:
        src_img.filepath = srcp
        src_img.reload()
    lab_node.inputs["String"].default_value = label
    warn_node.inputs["String"].default_value = WARN1 if mode == 1 else WARN3
    wmode.outputs[0].default_value = 1.0 if mode in (1, 3) else 0.0
    lost.outputs[0].default_value = 1.0 if mode == 2 else 0.0
    blink.outputs[0].default_value = 1.0 if (beats * 2.0) % 1.0 < 0.5 else 0.0
    kick.outputs[0].default_value = float(kick_v)
    rec_on.outputs[0].default_value = 1.0 if (f // 15) % 2 == 0 else 0.0
    fish.outputs[0].default_value = float(fisheye)
    brk.outputs[0].default_value = float(brk_v)
    static.inputs["W"].default_value = f * 1.37
    tmp = os.path.join(a.out, f".tmp_f_{f:04d}.png")
    sc.frame_set(f)
    sc.render.filepath = tmp
    t = time.time()
    bpy.ops.render.render(write_still=True)
    os.replace(tmp, dst)
    done += 1
    print(f"POST {f} {time.time() - t:.2f}", flush=True)
print(f"[post_hud] {done} HUD frames written, {skipped} existing skipped -> {a.out} ({time.time() - t_all:.0f} s)",
      flush=True)
