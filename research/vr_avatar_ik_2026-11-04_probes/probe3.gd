extends SceneTree
func _init() -> void:
	await process_frame
	for id in ["bee_warrior", "kyle"]:
		var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
		root.add_child(me); me.setup(id, 1.74)
		var sk: Skeleton3D = me.skel
		var h: int = me._bone["Hips"]
		print(id, " hips parent=", sk.get_bone_parent(h), " ", sk.get_bone_name(sk.get_bone_parent(h)) if sk.get_bone_parent(h) >= 0 else "", " skel_gt=", sk.global_transform, " model_rot=", me.model.rotation, " eye_vec=", me._eye_vec, " eye_fwd=", me._eye_fwd, " eye_src=", me.eye_source)
		var E: float = 1.74 * 0.935
		var head := Transform3D(Basis(), Vector3(0, E, 0))
		var l := Transform3D(Basis(), Vector3(-0.2, E - 0.5, -0.3)); var r := Transform3D(Basis(), Vector3(0.2, E - 0.5, -0.3))
		for i in 20:
			me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame
		var before: Vector3 = me._eye_world()
		var cb := func() -> void: print("  in skeleton_updated: eye-cam=", me._eye_world() - head.origin, " hips_world=", sk.global_transform * sk.get_bone_global_pose(h).origin, " body_origin=", me.global_position)
		sk.skeleton_updated.connect(cb, CONNECT_ONE_SHOT)
		me.drive(head, l, r, 0.0, 1.0 / 60.0)
		print("  right after drive: eye-cam=", me._eye_world() - head.origin, " (before drive ", before - head.origin, ")")
		await process_frame; await process_frame
		var neck: int = me._bone["Neck"]; var hd: int = me._bone["Head"]
		print("  rest: head bone y=", sk.get_bone_global_rest(hd).origin, " neck=", sk.get_bone_global_rest(neck).origin, " hips=", sk.get_bone_global_rest(h).origin, " spine=", sk.get_bone_global_rest(sk.find_bone("Spine")).origin, " upperchest=", sk.get_bone_global_rest(sk.find_bone("UpperChest")).origin)
		me.queue_free(); await process_frame
	quit()
