extends SceneTree
func _init() -> void:
	await process_frame
	var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
	root.add_child(me); me.setup("bee_warrior", 1.74)
	var sk: Skeleton3D = me.skel
	var h: int = me._bone["Hips"]
	var E: float = 1.74 * 0.935
	var head := Transform3D(Basis(), Vector3(0, E, 0))
	var l := Transform3D(Basis(), Vector3(-0.2, E - 0.5, -0.3)); var r := Transform3D(Basis(), Vector3(0.2, E - 0.5, -0.3))
	for i in 5: me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame
	me.drive(head, l, r, 0.0, 1.0 / 60.0)
	print("eye-cam ", me._eye_world() - head.origin, " hips pose ", sk.get_bone_pose_position(h))
	for k in 3:
		var hp0 := sk.get_bone_pose_position(h)
		me._eyes_on_headset(head)
		print("after extra slide ", k, ": eye-cam ", me._eye_world() - head.origin, " hips pose ", sk.get_bone_pose_position(h), " moved ", sk.get_bone_pose_position(h) - hp0)
	var hg := sk.get_bone_global_pose(me._bone["Head"])
	print("head global pose origin ", sk.global_transform * hg.origin)
	sk.set_bone_pose_position(h, sk.get_bone_pose_position(h) + Vector3(0, 0, 0.05))
	print("manual +5cm z: eye-cam ", me._eye_world() - head.origin, " head ", sk.global_transform * sk.get_bone_global_pose(me._bone["Head"]).origin)
	quit()
