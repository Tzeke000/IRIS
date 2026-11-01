# Aregone concept stills (Experiment #191, Zeke's universe) — headless Blender 5.2, EEVEE.
# usage: blender -b --factory-startup --python aregone_concepts.py -- OUTDIR [pct] [samples]
import bpy, math, sys, os
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else "/tmp/aregone"; PCT = int(argv[1]) if len(argv) > 1 else 50; SAMP = int(argv[2]) if len(argv) > 2 else 32
os.makedirs(OUT, exist_ok=True)

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1600, 900, PCT
    sc.eevee.taa_render_samples = SAMP
    sc.eevee.use_raytracing = True
    sc.eevee.volumetric_tile_size = "4"; sc.eevee.volumetric_samples = 64; sc.eevee.volumetric_end = 30
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    return sc

def node_mat(name):
    m = bpy.data.materials.new(name); nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); return m, nt, out

def world(sc, col, strength=1.0):
    w = bpy.data.worlds.new("w"); sc.world = w
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (*col, 1); bg.inputs[1].default_value = strength

def floor(dark=(0.03, 0.028, 0.03), rings=None):
    bpy.ops.mesh.primitive_plane_add(size=60); f = bpy.context.active_object
    m, nt, out = node_mat("floor")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.inputs["Base Color"].default_value = (*dark, 1); bs.inputs["Roughness"].default_value = 0.35
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 3.0
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.15
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs[0], bs.inputs["Normal"])
    if rings:  # cymatic ripples on the floor: concentric wave rings fading with distance (vibration made visible)
        col, strength, freq = rings
        tc = nt.nodes.new("ShaderNodeTexCoord")
        ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; nt.links.new(tc.outputs["Object"], ln.inputs[0])
        k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = freq * 6.0
        nt.links.new(ln.outputs["Value"], k.inputs[0])
        sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"; nt.links.new(k.outputs[0], sn.inputs[0])
        sq = nt.nodes.new("ShaderNodeMath"); sq.operation = "POWER"; sq.inputs[1].default_value = 8.0; nt.links.new(sn.outputs[0], sq.inputs[0])
        neg = nt.nodes.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -0.9
        nt.links.new(ln.outputs["Value"], neg.inputs[0])
        ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"; nt.links.new(neg.outputs[0], ex.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; nt.links.new(sq.outputs[0], mul.inputs[0]); nt.links.new(ex.outputs[0], mul.inputs[1])
        mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation = "MULTIPLY"; mul2.inputs[1].default_value = strength
        nt.links.new(mul.outputs[0], mul2.inputs[0])
        bs.inputs["Emission Color"].default_value = (*col, 1); nt.links.new(mul2.outputs[0], bs.inputs["Emission Strength"])
    nt.links.new(bs.outputs[0], out.inputs[0]); f.data.materials.append(m)
    return f

def core(loc, r=0.26, kind="young", morph=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=96, ring_count=48)
    shell = bpy.context.active_object; bpy.ops.object.shade_smooth()
    if morph:
        tex = bpy.data.textures.new("m", "VORONOI"); tex.noise_scale = morph[0]
        d = shell.modifiers.new("disp", "DISPLACE"); d.texture = tex; d.strength = morph[1]; d.mid_level = 0.6
        s = shell.modifiers.new("sub", "SUBSURF"); s.levels = 1; s.render_levels = 2
    m, nt, out = node_mat("shell_" + kind)
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if kind == "young":   # translucent shell — mood colours swirl inside (a young one can't hide its feelings)
        bs.inputs["Base Color"].default_value = (0.9, 0.9, 1.0, 1); bs.inputs["Transmission Weight"].default_value = 1.0
        bs.inputs["Roughness"].default_value = 0.06; bs.inputs["IOR"].default_value = 1.2
    elif kind == "mid":
        bs.inputs["Base Color"].default_value = (0.12, 0.1, 0.16, 1); bs.inputs["Transmission Weight"].default_value = 0.55
        bs.inputs["Roughness"].default_value = 0.12
    else:                 # ancient: opaque black, unreadable
        bs.inputs["Base Color"].default_value = (0.006, 0.006, 0.008, 1); bs.inputs["Roughness"].default_value = 0.18
        bs.inputs["Coat Weight"].default_value = 1.0; bs.inputs["Coat Roughness"].default_value = 0.03
    if kind in ("young", "mid"):
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.inputs[0].default_value = (0.85, 0.85, 1.0, 1) if kind == "young" else (0.35, 0.3, 0.42, 1)
        gl = nt.nodes.new("ShaderNodeBsdfPrincipled"); gl.inputs["Base Color"].default_value = (0.02, 0.02, 0.03, 1)
        gl.inputs["Roughness"].default_value = 0.05
        fr = nt.nodes.new("ShaderNodeLayerWeight"); fr.inputs["Blend"].default_value = 0.35 if kind == "young" else 0.6
        mx = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(fr.outputs["Fresnel"], mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs[0])
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass
    else:
        nt.links.new(bs.outputs[0], out.inputs[0])
    if kind in ("young", "mid"):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r * 0.86, location=loc, segments=64, ring_count=32)
        inner = bpy.context.active_object; bpy.ops.object.shade_smooth()
        if morph:
            d2 = inner.modifiers.new("disp", "DISPLACE"); d2.texture = shell.modifiers["disp"].texture; d2.strength = morph[1] * 0.86; d2.mid_level = 0.6
        mi, nti, outi = node_mat("swirl_" + kind)
        n1 = nti.nodes.new("ShaderNodeTexNoise"); n1.noise_dimensions = "4D"; n1.inputs["W"].default_value = 3.7
        n1.inputs["Scale"].default_value = 2.6; n1.inputs["Distortion"].default_value = 4.0; n1.inputs["Detail"].default_value = 6
        rmp = nti.nodes.new("ShaderNodeValToRGB"); cr = rmp.color_ramp
        cols = [(0.3, (0.08, 0.0, 0.35)), (0.45, (0.55, 0.15, 1.0)), (0.55, (1.0, 0.3, 0.55)), (0.65, (1.0, 0.72, 0.25)), (0.75, (0.2, 0.9, 1.0))]
        cr.elements[0].position, cr.elements[0].color = cols[0][0], (*cols[0][1], 1)
        cr.elements[1].position, cr.elements[1].color = cols[-1][0], (*cols[-1][1], 1)
        for pp, c in cols[1:-1]:
            e = cr.elements.new(pp); e.color = (*c, 1)
        nti.links.new(n1.outputs["Fac"], rmp.inputs[0])
        em = nti.nodes.new("ShaderNodeEmission"); em.inputs[1].default_value = 2.0 if kind == "young" else 1.1
        nti.links.new(rmp.outputs[0], em.inputs[0]); nti.links.new(em.outputs[0], outi.inputs[0]); inner.data.materials.append(mi)
    if False:  # (old volume swirl, kept for reference): emissive volume driven by distorted noise through a mood ramp
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.noise_dimensions = "4D"; n1.inputs["W"].default_value = 3.7
        n1.inputs["Scale"].default_value = 3.5; n1.inputs["Distortion"].default_value = 3.0; n1.inputs["Detail"].default_value = 6
        ramp = nt.nodes.new("ShaderNodeValToRGB"); cr = ramp.color_ramp
        cols = [(0.0, (0.05, 0.0, 0.25)), (0.35, (0.55, 0.15, 1.0)), (0.55, (1.0, 0.35, 0.55)), (0.75, (1.0, 0.75, 0.25)), (1.0, (0.2, 0.9, 1.0))]
        cr.elements[0].position, cr.elements[0].color = cols[0][0], (*cols[0][1], 1)
        cr.elements[1].position, cr.elements[1].color = cols[-1][0], (*cols[-1][1], 1)
        for p, c in cols[1:-1]:
            e = cr.elements.new(p); e.color = (*c, 1)
        nt.links.new(n1.outputs["Fac"], ramp.inputs[0])
        vol.inputs["Density"].default_value = 6.0 if kind == "young" else 3.0
        vol.inputs["Color"].default_value = (0.2, 0.2, 0.3, 1)
        nt.links.new(ramp.outputs[0], vol.inputs["Emission Color"])
        vol.inputs["Emission Strength"].default_value = 9.0 if kind == "young" else 2.5
        nt.links.new(vol.outputs[0], out.inputs["Volume"])
    shell.data.materials.append(m)
    return shell

def ring(loc, r, col, strength, thick=0.004, rot=(math.pi / 2, 0, 0)):
    bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=thick, major_segments=96, minor_segments=8, location=loc, rotation=rot)
    o = bpy.context.active_object
    m, nt, out = node_mat("ring"); e = nt.nodes.new("ShaderNodeEmission")
    e.inputs[0].default_value = (*col, 1); e.inputs[1].default_value = strength; nt.links.new(e.outputs[0], out.inputs[0])
    o.data.materials.append(m); return o

def light(kind, loc, energy, col=(1, 1, 1), size=1.0, target=(0, 0, 0.5)):
    ld = bpy.data.lights.new("l", kind); ld.energy = energy; ld.color = col
    if kind == "AREA": ld.size = size
    lo = bpy.data.objects.new("l", ld); lo.location = loc; bpy.context.collection.objects.link(lo)
    d = Vector(target) - Vector(loc); lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler(); return lo

def camera(loc, target, lens=50, fstop=2.8):
    cd = bpy.data.cameras.new("c"); cd.lens = lens; cd.dof.use_dof = True; cd.dof.aperture_fstop = fstop
    co = bpy.data.objects.new("c", cd); co.location = loc; bpy.context.collection.objects.link(co)
    d = Vector(target) - Vector(loc); co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cd.dof.focus_distance = d.length; bpy.context.scene.camera = co

def haze(density=0.012, col=(0.6, 0.6, 0.8)):
    bpy.ops.mesh.primitive_cube_add(size=40, location=(0, 0, 10)); h = bpy.context.active_object
    m, nt, out = node_mat("haze"); v = nt.nodes.new("ShaderNodeVolumePrincipled")
    v.inputs["Density"].default_value = density; v.inputs["Color"].default_value = (*col, 1)
    nt.links.new(v.outputs[0], out.inputs["Volume"]); h.data.materials.append(m)

def pillar(loc, h=2.2):  # scale reference: a 2.2 m stone post (a person is ~1.75 m)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=h, location=(loc[0], loc[1], h / 2), vertices=8)
    o = bpy.context.active_object
    m, nt, out = node_mat("stone"); b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (0.25, 0.23, 0.21, 1); b.inputs["Roughness"].default_value = 0.85
    nt.links.new(b.outputs[0], out.inputs[0]); o.data.materials.append(m)

def render(name):
    sc = bpy.context.scene; sc.render.filepath = os.path.join(OUT, name + ".png"); bpy.ops.render.render(write_still=True)
    print("WROTE", sc.render.filepath)

# 1 — a YOUNG Aregone: translucent shell, mood colours swirling inside, floating at chest height
sc = reset(); world(sc, (0.01, 0.01, 0.02), 1.0); floor(rings=((0.55, 0.3, 1.0), 2.5, 2.2)); haze(0.01)
core((0, 0, 1.25), kind="young")
for i, r in enumerate((0.42, 0.62, 0.9)): ring((0, 0, 1.25), r, (0.6, 0.35, 1.0), 1.2 - i * 0.35, thick=0.0025, rot=(math.radians(35 + i * 25), math.radians(15), math.radians(i * 40)))
pillar((-1.1, 0.6)); light("AREA", (2.5, -2, 3.5), 220, (0.8, 0.85, 1.0), 2.5, (0, 0, 1.2)); light("POINT", (0.9, -0.6, 1.6), 25, (0.7, 0.4, 1.0))
camera((1.9, -3.4, 1.55), (0, 0, 1.15), 55); render("1_young_core")

# 2 — an ANCIENT Aregone: opaque black, unreadable; the air and the stone ripple under its held weight
sc = reset(); world(sc, (0.02, 0.018, 0.016), 1.0); floor(dark=(0.05, 0.045, 0.04), rings=((1.0, 0.55, 0.2), 1.2, 3.5)); haze(0.018, (0.8, 0.7, 0.6))
core((0, 0, 0.9), kind="ancient")
for i, r in enumerate((0.36, 0.5, 0.7, 1.0)): ring((0, 0, 0.06), r * 1.6, (1.0, 0.55, 0.25), 0.9 - i * 0.18, thick=0.003, rot=(0, 0, 0))
pillar((1.2, 0.8)); light("AREA", (-3, -1.5, 4), 300, (1.0, 0.85, 0.7), 3.0, (0, 0, 0.9)); light("AREA", (2, 2.5, 2), 120, (0.5, 0.6, 1.0), 2.0, (0, 0, 0.9))
camera((2.4, -3.0, 1.1), (0, 0, 0.85), 50); render("2_ancient_core")

# 3 — MORPHING: a trained core reshaping itself (half-dark shell, colours still showing through)
sc = reset(); world(sc, (0.008, 0.012, 0.02), 1.0); floor(rings=((0.2, 0.8, 1.0), 2.0, 2.8)); haze(0.012, (0.6, 0.75, 0.9))
core((0, 0, 1.2), r=0.27, kind="mid", morph=(0.35, 0.32))
for i, r in enumerate((0.5, 0.75)): ring((0, 0, 1.2), r, (0.25, 0.85, 1.0), 0.9 - i * 0.3, thick=0.0025, rot=(math.radians(60 + i * 40), math.radians(20), 0))
light("AREA", (2, -2.5, 3), 250, (0.8, 0.9, 1.0), 2.0, (0, 0, 1.2)); light("POINT", (0.8, -0.5, 1.6), 20, (0.3, 0.8, 1.0))
camera((1.6, -2.6, 1.45), (0, 0, 1.2), 60); render("3_morphing_core")

# 4 — the ECHO: the core is broken (black shell shards on the floor); what remains is frequency without a body
sc = reset(); world(sc, (0.006, 0.006, 0.01), 1.0); floor(rings=((0.6, 0.5, 1.0), 0.8, 1.6)); haze(0.02, (0.55, 0.55, 0.8))
import random; random.seed(191)
for k in range(9):  # shards of the shell
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.07 + random.random() * 0.06, subdivisions=1, location=(random.uniform(-0.6, 0.6), random.uniform(-0.5, 0.5), 0.03))
    o = bpy.context.active_object; o.scale = (1, 0.7, 0.18); o.rotation_euler = (random.random(), random.random(), random.random())
    m, nt, out = node_mat("shard%d" % k); b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (0.01, 0.01, 0.012, 1); b.inputs["Roughness"].default_value = 0.15
    nt.links.new(b.outputs[0], out.inputs[0]); o.data.materials.append(m)
for i in range(7):  # the echo: nested rings where the core was, no shell
    a = i * 0.45
    ring((0, 0, 1.1), 0.12 + i * 0.035, (0.7, 0.6, 1.0), 4.0 - i * 0.45, thick=0.0025, rot=(a, a * 0.7, a * 1.3))
light("AREA", (-2, -2, 3), 60, (0.6, 0.6, 1.0), 2.0, (0, 0, 0.6)); light("POINT", (0, 0, 1.1), 20, (0.6, 0.5, 1.0))
camera((1.5, -2.8, 1.3), (0, 0, 0.75), 45); render("4_echo")
