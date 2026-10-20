# Builds res://scenes/main.tscn headless (no editor UI on the server):
#   godot --headless --path vr/iris_home res://tools/build_main_scene.tscn
# (run as a SCENE, not --script: XR Tools needs its autoloads, which --script mode skips)
# v0 (2026-10-20): open temple floor + columns, sky, XR player with hands + pickup + movement,
# a physics BALL for catch, and a placeholder "Iris" (glowing iris-coloured sphere).
extends Node

const XRT := "res://addons/godot-xr-tools/"

func _own(n: Node, root: Node) -> void:
	if n != root:
		n.owner = root
	if n != root and n.scene_file_path != "":
		return   # an instanced sub-scene: keep it as a reference, don't bake its insides
	for c in n.get_children():
		_own(c, root)

func _inst(path: String) -> Node:
	return (load(path) as PackedScene).instantiate()

func _mat(col: Color, emit := 0.0, rough := 0.85) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.roughness = rough
	if emit > 0.0:
		m.emission_enabled = true
		m.emission = col
		m.emission_energy_multiplier = emit
	return m

func _box(name: String, size: Vector3, pos: Vector3, mat: Material) -> StaticBody3D:
	var b := StaticBody3D.new()
	b.name = name
	b.position = pos
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	bm.material = mat
	mi.mesh = bm
	b.add_child(mi)
	var cs := CollisionShape3D.new()
	var sh := BoxShape3D.new()
	sh.size = size
	cs.shape = sh
	b.add_child(cs)
	return b

func _cyl(name: String, r: float, h: float, pos: Vector3, mat: Material) -> StaticBody3D:
	var b := StaticBody3D.new()
	b.name = name
	b.position = pos
	var mi := MeshInstance3D.new()
	var cm := CylinderMesh.new()
	cm.top_radius = r
	cm.bottom_radius = r * 1.08
	cm.height = h
	cm.material = mat
	mi.mesh = cm
	b.add_child(mi)
	var cs := CollisionShape3D.new()
	var sh := CylinderShape3D.new()
	sh.radius = r
	sh.height = h
	cs.shape = sh
	b.add_child(cs)
	return b

func _ready() -> void:
	var root := Node3D.new()
	root.name = "IrisHome"

	# environment: open evening sky
	var we := WorldEnvironment.new()
	var env := Environment.new()
	var sky := Sky.new()
	var psm := ProceduralSkyMaterial.new()
	psm.sky_top_color = Color(0.20, 0.16, 0.38)
	psm.sky_horizon_color = Color(0.85, 0.62, 0.70)
	psm.ground_horizon_color = Color(0.85, 0.62, 0.70)
	sky.sky_material = psm
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.glow_enabled = true
	we.environment = env
	root.add_child(we)
	var sun := DirectionalLight3D.new()
	sun.name = "Sun"
	sun.rotation_degrees = Vector3(-35, 40, 0)
	sun.light_color = Color(1.0, 0.85, 0.7)
	sun.shadow_enabled = true
	root.add_child(sun)

	# the temple: a big round stone floor ringed by columns, open to the sky
	var stone := _mat(Color(0.82, 0.78, 0.72))
	var floor := _cyl("Floor", 24.0, 0.6, Vector3(0, -0.3, 0), stone)
	root.add_child(floor)
	for i in 16:
		var a := TAU * i / 16.0
		root.add_child(_cyl("Column%d" % i, 0.7, 9.0, Vector3(cos(a) * 20.0, 4.5, sin(a) * 20.0), stone))
	root.add_child(_cyl("Dais", 3.0, 0.4, Vector3(0, 0.2, -6.0), _mat(Color(0.65, 0.6, 0.75))))

	# XR player: origin + camera + two hands with pickup, stick movement + turning, player body
	root.add_child(_inst(XRT + "xr/start_xr.tscn"))
	var origin := XROrigin3D.new()
	origin.name = "XROrigin3D"
	origin.position = Vector3(0, 0, 4)
	root.add_child(origin)
	var cam := XRCamera3D.new()
	cam.name = "XRCamera3D"
	cam.position = Vector3(0, 1.7, 0)
	origin.add_child(cam)
	for side in ["left", "right"]:
		var c := XRController3D.new()
		c.name = side.capitalize() + "Hand"
		c.tracker = &"left_hand" if side == "left" else &"right_hand"
		c.pose = &"aim"
		origin.add_child(c)
		c.add_child(_inst(XRT + "hands/scenes/lowpoly/%s_hand_low.tscn" % side))
		c.add_child(_inst(XRT + "functions/function_pickup.tscn"))
		if side == "left":
			c.add_child(_inst(XRT + "functions/movement_direct.tscn"))
		else:
			c.add_child(_inst(XRT + "functions/movement_turn.tscn"))
	origin.add_child(_inst(XRT + "player/player_body.tscn"))

	# the ball (catch): an XR Tools pickable with real physics
	var ball: Node3D = _inst(XRT + "objects/pickable.tscn")
	ball.name = "Ball"
	ball.position = Vector3(0.4, 1.2, 2.5)
	root.add_child(ball)
	for ch in ball.get_children():
		if ch is CollisionShape3D:
			var s := SphereShape3D.new()
			s.radius = 0.12
			ch.shape = s
	var bmi := MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.12
	sm.height = 0.24
	sm.material = _mat(Color(0.95, 0.45, 0.2), 0.0, 0.5)
	bmi.mesh = sm
	ball.add_child(bmi)
	if ball is RigidBody3D:
		(ball as RigidBody3D).mass = 0.4

	# Iris (placeholder until the real iris body is ported): a glowing violet orb on the dais
	var iris := Node3D.new()
	iris.name = "Iris"
	iris.position = Vector3(0, 1.6, -6.0)
	var imi := MeshInstance3D.new()
	var ism := SphereMesh.new()
	ism.radius = 0.35
	ism.height = 0.7
	ism.material = _mat(Color(0.62, 0.42, 1.0), 2.5, 0.2)
	imi.mesh = ism
	iris.add_child(imi)
	var il := OmniLight3D.new()
	il.light_color = Color(0.7, 0.5, 1.0)
	il.light_energy = 2.0
	il.omni_range = 6.0
	iris.add_child(il)
	iris.set_script(load("res://scripts/iris_presence.gd"))
	root.add_child(iris)

	_own(root, root)
	var ps := PackedScene.new()
	var err := ps.pack(root)
	if err == OK:
		err = ResourceSaver.save(ps, "res://scenes/main.tscn")
	print("[build_main_scene] save -> ", error_string(err))
	get_tree().quit(0 if err == OK else 1)
