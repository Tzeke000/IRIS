extends SceneTree
func _init() -> void:
	await process_frame
	for id in ["jimmy", "kyle", "olivia", "anna"]:
		var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
		root.add_child(me); me.setup(id, 1.74)
		var sk: Skeleton3D = me.skel
		var E: float = 1.74 * 0.935
		var head := Transform3D(Basis(), Vector3(0, E, 0))
		var l := Transform3D(Basis(), Vector3(-0.2, E - 0.5, -0.3)); var r := Transform3D(Basis(), Vector3(0.2, E - 0.5, -0.3))
		for i in 20: me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame
		var out := {}
		var cb := func() -> void:
			var g := sk.global_transform
			var ul := sk.find_bone("LeftUpperLeg"); var ll := sk.find_bone("LeftLowerLeg"); var ft := sk.find_bone("LeftFoot")
			out["hip_joint_y"] = (g * sk.get_bone_global_pose(ul).origin).y
			out["hip_joint_rest_y"] = (g * sk.get_bone_global_rest(ul).origin).y
			out["hips_y"] = (g * sk.get_bone_global_pose(me._bone["Hips"]).origin).y
			out["hips_rest_y"] = me._rest_hips * me._scale
			out["hip_to_ankle_target"] = (g * sk.get_bone_global_pose(ul).origin).distance_to(me._t["LFoot"].global_position)
			out["leg_len"] = (me._thigh + me._shin) * me._scale
			out["foot_target"] = me._t["LFoot"].global_position
			out["hipjoint"] = g * sk.get_bone_global_pose(ul).origin
		sk.skeleton_updated.connect(cb, CONNECT_ONE_SHOT)
		me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame; await process_frame
		print(id, " ", out, " crouch=", me.crouch)
		me.queue_free(); await process_frame
	quit()
