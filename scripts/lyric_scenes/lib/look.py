"""Look-dev setup (Blender 5.2): colour management, EEVEE render defaults, HDRI world, and the 5.x compositor post
chain. One copy of the boilerplate police / rooftop / rocket4 / boss v1 each re-typed (AgX + look + exposure, EEVEE
ray tracing / fast GI / shadows / motion blur START / PNG 8-bit, the RLayers → mist haze → Fog Glow [→ Streaks] →
Lens Distortion → vignette chain, the HDRI + Mapping world with an optional camera-ray background). The numbers stay
per scene; the plumbing lives here.

    from lib import look as L
    L.color(SC, look="Medium High Contrast", exposure=-0.5)        # "AgX - " prefix added if missing (5.2 needs it)
    L.eevee_defaults(SC, samples=32, res=(1920, 1080), fps=30)      # strict: a renamed property raises, never skipped
    L.eevee_volumes(SC, start=0.5, end=4000, tile="2", samples=96)
    W, wn = L.hdri_world("/data/iris_assets/<scene>/<hdri>/x_2k.hdr", rot_deg=L.hdri_rotation(az_now, 118), camera=1.4)
    P = L.post_chain(SC, mist=dict(start=30, depth=1100, color=(0.022, 0.024, 0.032), k=0.55),
                     glare=[dict(type="Fog Glow", threshold=1.2, strength=0.4),
                            dict(type="Streaks", threshold=8.0, strength=0.3, streaks=2)],
                     lens=(-0.008, 0.012), vignette=0.72)            # → {"tree", "nb", "rl", "out", "image", ...}
    bake(P["tree"], 'nodes["Lens"].inputs["Distortion"].default_value', dist)   # chain nodes carry stable names

5.x COMPOSITOR (blender-photoreal-lookdev §8): scene.compositing_node_group + a CompositorNodeTree with a Group Output
— there is no scene.node_tree and no Composite node. Glare/Lens options are INPUT SOCKETS (Type, Quality, Threshold…).
The RLayers "Mist" output only exists once view_layer.use_pass_mist is on (mist_haze() turns it on first).
GRAIN: lib/render/scene.sh already adds grain at mux (GRAIN=5, ffmpeg noise) — film_grain() is for when the grain must
vary per shot (rocket4's onboard cams); don't use both.
"""
import math

import numpy as np

from .bpyx import NB

AGX_LOOKS = ("AgX - Punchy", "AgX - Greyscale", "AgX - Very High Contrast", "AgX - High Contrast",
             "AgX - Medium High Contrast", "AgX - Base Contrast", "AgX - Medium Low Contrast", "AgX - Low Contrast",
             "AgX - Very Low Contrast")          # Blender 5.2.1, sRGB display (tests/bpy_test_look.py re-checks them)


def _bpy():
    import bpy
    return bpy


# ------------------------------------------------------------------------------------------------ pure helpers

def resolve_look(view, look, valid):
    """The look name Blender accepts: exact, or '<view> - <look>' (a bare 'Punchy' raises TypeError in 5.2)."""
    if look in (None, "", "None"):
        return "None"
    if look in valid:
        return look
    pref = f"{view} - {look}"
    if pref in valid:
        return pref
    raise ValueError(f"look {look!r} not available for view {view!r}; valid: {list(valid)}")


def _valid_from_error(msg):
    """the enum names listed in Blender's 'enum "X" not found in (...)' TypeError"""
    if "(" not in msg:
        return ()
    inner = msg[msg.rindex("(") + 1:msg.rindex(")")]
    return tuple(x.strip().strip("'\"") for x in inner.split(",") if x.strip())


def equirect_dir(u, v):
    """Equirect image coords (u, v in 0..1, v from the BOTTOM row as in Image.pixels) → world direction the default
    Environment Texture shows there: azimuth = π − 2πu (Cycles' direction_to_equirectangular; image centre = +X),
    elevation = (v − 0.5)·π. MEASURED (tests/bpy_test_look.py renders toward it). NB boss v1's moon code used
    (−cos a, sin a) with a = (u − 0.5)·2π — that is 180° off in azimuth."""
    az = math.pi - 2 * math.pi * u
    el = (v - 0.5) * math.pi
    return np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])


def hdri_rotation(azimuth_now, azimuth_want):
    """Mapping Z rotation (radians) that moves a light seen at world azimuth `azimuth_now` (unrotated HDRI) to
    `azimuth_want` (both radians, from +X, CCW)."""
    return (azimuth_now - azimuth_want + math.pi) % (2 * math.pi) - math.pi


def brightest(pix, w, h):
    """(azimuth, elevation, (ix, iy)) of the brightest pixel of an equirect image given as a flat RGBA float array
    (Image.pixels order, bottom row first)."""
    lum = np.asarray(pix, np.float32).reshape(h, w, 4)[:, :, :3].sum(2)
    iy, ix = np.unravel_index(int(np.argmax(lum)), lum.shape)
    d = equirect_dir((ix + 0.5) / w, (iy + 0.5) / h)
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2]))), (int(ix), int(iy))


# ------------------------------------------------------------------------------------------------ colour + render

def _strict_set(obj, values, where):
    bad = []
    for k, v in values.items():
        if v is None:
            continue
        try:
            setattr(obj, k, v)
        except (AttributeError, TypeError, ValueError) as e:
            bad.append(f"{where}.{k}={v!r}: {e}")
    if bad:
        raise AttributeError("render settings not applied (renamed in this Blender?):\n  " + "\n  ".join(bad))


def color(sc, view="AgX", look="Medium High Contrast", exposure=0.0, gamma=1.0):
    """View transform + look + exposure. Returns the look name actually set."""
    vs = sc.view_settings
    vs.view_transform = view
    try:
        vs.look = look if look not in (None, "") else "None"
        name = vs.look
    except TypeError as e:
        name = resolve_look(view, look, _valid_from_error(str(e)))
        vs.look = name
    vs.exposure, vs.gamma = exposure, gamma
    return name


def eevee_defaults(sc, samples=32, res=(1920, 1080), pct=100, fps=30, mb=True, shutter=0.5, raytrace=True,
                   rt_scale="2", fast_gi=True, shadow_rays=1, shadow_steps=6, png8=True, view="AgX",
                   look="Medium High Contrast", exposure=0.0):
    """The EEVEE block every scene re-typed (police L2059, rocket4 L5206, boss v1 L81, rooftop L60). Motion blur
    position START — required by lib/camera's cut-safe holds. Strict: a renamed property raises instead of the old
    try/except-pass dance that silently skipped it. Pass look=None to leave colour management alone."""
    r = sc.render
    r.engine = "BLENDER_EEVEE"
    _strict_set(r, dict(resolution_x=res[0], resolution_y=res[1], resolution_percentage=pct, fps=fps, fps_base=1.0,
                        use_motion_blur=mb, motion_blur_shutter=shutter, motion_blur_position="START"), "render")
    if png8:
        _strict_set(r.image_settings, dict(file_format="PNG", color_mode="RGB", color_depth="8"), "image_settings")
    ee = sc.eevee
    _strict_set(ee, dict(taa_render_samples=samples, use_raytracing=raytrace, ray_tracing_method="SCREEN",
                         use_fast_gi=fast_gi, fast_gi_method="GLOBAL_ILLUMINATION", use_shadows=True,
                         shadow_ray_count=shadow_rays, shadow_step_count=shadow_steps), "eevee")
    if raytrace:
        _strict_set(ee.ray_tracing_options, dict(resolution_scale=rt_scale), "eevee.ray_tracing_options")
    if look is not None:
        color(sc, view, look, exposure)
    return sc


def eevee_volumes(sc, start=0.5, end=400.0, tile="8", samples=64, distribution=0.8, shadows=True, shadow_samples=12):
    """EEVEE volumetrics range/quality (rocket4 + boss). volumetric_start/end are NOT animatable — per-shot ranges go
    through lib/render/frames.py's per-frame scene props (vol_start/vol_end)."""
    _strict_set(sc.eevee, dict(use_volume_custom_range=True, volumetric_start=start, volumetric_end=end,
                               volumetric_tile_size=tile, volumetric_samples=samples,
                               volumetric_sample_distribution=distribution, use_volumetric_shadows=shadows,
                               volumetric_shadow_samples=shadow_samples), "eevee")
    return sc


# ------------------------------------------------------------------------------------------------ world

def hdri_world(path, rot_deg=0.0, strength=1.0, camera=None, name="HDRI", scene=None):
    """HDRI world: Environment Texture (absolute path, logged for credits via lib/importer.load_img) through a Mapping
    (Z rotation in DEGREES — find the key by rotating, skill §5). camera = what CAMERA rays see instead (Light Path
    'Is Camera Ray', boss v1 / rooftop): None → the same HDRI; a float → the HDRI × that strength; an RGB tuple → a flat
    colour (HDRI only lights and reflects). Returns (world, nodes dict: env, mapping, bg[, bg_cam, mix])."""
    from . import importer
    bpy = _bpy()
    sc = scene or bpy.context.scene
    im = importer.load_img(path, color=True)          # .hdr/.exr load linear
    w = bpy.data.worlds.new(name)
    sc.world = w
    nt = w.node_tree
    nt.nodes.clear()
    nb = NB(nt)
    out = nb.n("ShaderNodeOutputWorld")
    tc, mp = nb.n("ShaderNodeTexCoord"), nb.n("ShaderNodeMapping")
    env = nb.n("ShaderNodeTexEnvironment")
    env.image = im
    nb.link(tc.outputs["Generated"], mp.inputs["Vector"])
    nb.link(mp.outputs["Vector"], env.inputs["Vector"])
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot_deg))
    bg = nb.n("ShaderNodeBackground")
    nb.link(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nodes = dict(env=env, mapping=mp, bg=bg, out=out)
    if camera is None:
        nb.link(bg.outputs[0], out.inputs["Surface"])
        return w, nodes
    bgc = nb.n("ShaderNodeBackground")
    if isinstance(camera, (int, float)):
        nb.link(env.outputs["Color"], bgc.inputs["Color"])
        bgc.inputs["Strength"].default_value = float(camera)
    else:
        nb.set(bgc.inputs["Color"], tuple(camera))
        bgc.inputs["Strength"].default_value = 1.0
    lp, mx = nb.n("ShaderNodeLightPath"), nb.n("ShaderNodeMixShader")
    nb.link(lp.outputs["Is Camera Ray"], mx.inputs["Fac"])
    nb.link(bg.outputs[0], mx.inputs[1])
    nb.link(bgc.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs["Surface"])
    nodes.update(bg_cam=bgc, mix=mx)
    return w, nodes


def hdri_key(im):
    """(azimuth, elevation, (ix, iy)) of the brightest pixel (sun / moon) of a loaded equirect image, unrotated world
    frame. Then rot = hdri_rotation(az, az_want) → hdri_world(rot_deg=degrees(rot)) puts it at az_want (boss v1's idea,
    with the azimuth convention corrected)."""
    w, h = im.size
    px = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(px)
    return brightest(px, w, h)


# ------------------------------------------------------------------------------------------------ compositor (5.x)

def comp_tree(sc, name="Post", device="GPU"):
    """New compositor tree wired as the scene's → (tree, NB, render-layers node, group-output node)."""
    bpy = _bpy()
    cg = bpy.data.node_groups.new(name, "CompositorNodeTree")
    cg.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    nb = NB(cg)
    rl, go = nb.n("CompositorNodeRLayers"), nb.n("NodeGroupOutput")
    sc.compositing_node_group = cg
    sc.render.use_compositing = True
    if device:
        sc.render.compositor_device = device
    return cg, nb, rl, go


def _named(nd, name):
    if name:
        nd.name = nd.label = name
    return nd


def mist_haze(nb, image, rl, sc, color=(0.022, 0.024, 0.032), k=0.55, cap=0.85, start=30.0, depth=1000.0,
              falloff="QUADRATIC", name="Haze"):
    """Atmospheric haze between camera and subject: mix(image, color, min(mist × k, cap)). Turns the mist pass on and
    sets the world mist range. k may be a socket (a keyed Value node → per-shot haze, rocket4)."""
    sc.view_layers[0].use_pass_mist = True
    if sc.world is None:
        raise ValueError("mist_haze needs scene.world (mist range lives on the world)")
    ms = sc.world.mist_settings
    ms.start, ms.depth, ms.falloff = start, depth, falloff
    fac = nb.m("MINIMUM", nb.m("MULTIPLY", rl.outputs["Mist"], k), cap)
    nd = _named(nb.n("ShaderNodeMix", data_type="RGBA"), name)
    nb.set(nd.inputs[0], fac)
    nb.set(nd.inputs[6], image)
    nb.set(nd.inputs[7], tuple(color))
    return nd.outputs[2]


def glare(nb, image, type="Fog Glow", threshold=1.2, strength=0.4, size=None, quality="Medium", streaks=None,
          angle=None, fade=None, name=None):
    """Compositor Glare (5.x: options are inputs). Keep threshold high so only true highlights glare.
    type ∈ Bloom, Fog Glow, Streaks, Ghosts, Sun Beams, Kernel, Simple Star. angle in degrees."""
    g = _named(nb.n("CompositorNodeGlare"), name)
    for k, v in (("Type", type), ("Quality", quality), ("Threshold", threshold), ("Strength", strength),
                 ("Size", size), ("Streaks", streaks), ("Streaks Angle", None if angle is None else math.radians(angle)),
                 ("Fade", fade)):
        if v is not None:
            g.inputs[k].default_value = v
    nb.set(g.inputs["Image"], image)
    return g.outputs["Image"]


def lens_distortion(nb, image, distortion=-0.006, dispersion=0.012, fit=True, name="Lens"):
    """Lens Distortion: small negative distortion + 0.005–0.015 dispersion (chromatic fringe)."""
    ld = _named(nb.n("CompositorNodeLensdist"), name)
    ld.inputs["Distortion"].default_value = distortion
    ld.inputs["Dispersion"].default_value = dispersion
    ld.inputs["Fit"].default_value = fit
    nb.set(ld.inputs["Image"], image)
    return ld.outputs["Image"]


def vignette(nb, image, amount=0.72, size=1.25, blur=300, name="Vignette"):
    """Soft vignette: edges multiplied down to `amount` (rooftop 0.72, rocket4 0.70)."""
    em = nb.n("CompositorNodeEllipseMask")
    em.inputs["Size"].default_value = (size, size)
    bl = nb.n("CompositorNodeBlur")
    bl.inputs["Size"].default_value = (blur, blur)
    nb.link(em.outputs["Mask"], bl.inputs["Image"])
    nd = _named(nb.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY"), name)
    nd.inputs[0].default_value = 1.0
    nb.set(nd.inputs[6], image)
    nb.link(nb.mr(bl.outputs["Image"], 0.0, 1.0, amount, 1.0), nd.inputs[7])
    return nd.outputs[2]


def film_grain(nb, image, amount=0.05, name="Grain"):
    """Multiplicative per-frame grain (rocket4): 4-D white noise whose W follows the frame (two LINEAR keys with linear
    extrapolation — valid at every frame, no driver). amount = peak-to-peak gain (0.05 → ±2.5 %); a socket works too.
    NOT together with scene.sh's mux grain (GRAIN=5)."""
    from .bpyx import bake
    wn = _named(nb.n("ShaderNodeTexWhiteNoise", noise_dimensions="4D"), name)
    g = nb.m("ADD", nb.m("MULTIPLY", nb.m("SUBTRACT", wn.outputs["Value"], 0.5), amount), 1.0)
    cc = nb.n("CompositorNodeCombineColor")
    for i in range(3):
        nb.link(g, cc.inputs[i])
    nd = nb.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    nd.inputs[0].default_value = 1.0
    nb.set(nd.inputs[6], image)
    nb.link(cc.outputs[0], nd.inputs[7])
    fc = bake(nb.nt, f'nodes["{wn.name}"].inputs["W"].default_value', [1.0, 2.0], frames=[1, 2], thin=False)
    fc.extrapolation = "LINEAR"
    return nd.outputs[2]


def post_chain(sc, mist=None, glare=(), lens=(-0.006, 0.012), vignette=None, grain=None, name="Post", device="GPU"):
    """RLayers → [mist haze] → glare… → [lens distortion] → [vignette] → [grain] → output, as one call.
    mist = mist_haze kwargs; glare = list of glare kwargs (applied in order; named Glare0, Glare1…);
    lens = (distortion, dispersion) or None; vignette = amount or None; grain = amount or None.
    Returns {"tree", "nb", "rl", "out", "image"} (image = last socket, already linked to the output)."""
    cg, nb, rl, go = comp_tree(sc, name, device)
    img = rl.outputs["Image"]
    if mist:
        img = mist_haze(nb, img, rl, sc, **mist)
    for i, gk in enumerate(glare or ()):
        gk = dict(gk)
        gk.setdefault("name", f"Glare{i}")
        img = globals()["glare"](nb, img, **gk)
    if lens:
        img = lens_distortion(nb, img, *lens)
    if vignette:
        img = globals()["vignette"](nb, img, vignette)
    if grain:
        img = film_grain(nb, img, grain)
    nb.link(img, go.inputs["Image"])
    return dict(tree=cg, nb=nb, rl=rl, out=go, image=img)

