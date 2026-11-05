extends SceneTree
## READ-ONLY probe 2: bone-axis conventions, rest arm pose, and the camera vs the torso surface when looking down.
const IDS := ["anna", "base_male", "bee_warrior", "bruno", "jennifer", "jimmy", "kate", "kyle", "oldmoustache", "olivia", "seed_san", "sendagaya_shino"]
const TORSO := ["Hips", "Spine", "Chest", "UpperChest", "Neck", "LeftShoulder", "RightShoulder"]

func _grab_verts(me) -> Array:     # skinned world vertices whose dominant bone is a torso bone, CPU-skinned from the CURRENT pose
	var sk: Skeleton3D = me.skel
	var tor := {}
	for n in TORSO:
		var b := sk.find_bone(n)
		if b >= 0: tor[b] = true
	var out := []
	for mi in me.model.find_children("*", "MeshInstance3D", true, false):
		var nm := str(mi.name)
		if mi.mesh == null or mi.skin == null or nm.ends_with("(Shadow)") or not mi.visible: continue
		if (mi.layers & 8) == 0: continue                      # only what MY eyes render (layer 8)
		var skn: Skin = mi.skin
		var bidx := []; var bind := []
		for i in skn.get_bind_count():
			var b := skn.get_bind_bone(i)
			if b < 0: b = sk.find_bone(skn.get_bind_name(i))
			bidx.append(b); bind.append(skn.get_bind_pose(i))
		for s in mi.mesh.get_surface_count():
			var arr: Array = mi.mesh.surface_get_arrays(s)
			var V: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
			var B = arr[Mesh.ARRAY_BONES]; var W = arr[Mesh.ARRAY_WEIGHTS]
			if B == null or W == null or V.is_empty(): continue
			var per: int = (B as PackedInt32Array).size() / V.size()
			for i in range(0, V.size(), maxi(1, V.size() / 6000)):
				var best := 0; var bw := -1.0
				for k in per:
					if W[i * per + k] > bw: bw = W[i * per + k]; best = B[i * per + k]
				if best >= bidx.size() or not tor.has(bidx[best]): continue
				var p := Vector3.ZERO
				for k in per:
					var w: float = W[i * per + k]
					if w <= 0.0: continue
					var bi: int = B[i * per + k]
					p += w * ((sk.get_bone_global_pose(bidx[bi]) * bind[bi]) * V[i])
				out.append(sk.global_transform * p)
	return out

func _init() -> void:
	await process_frame
	for id in IDS:
		var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
		root.add_child(me)
		if not me.setup(id, 1.74): continue
		var sk: Skeleton3D = me.skel
		var res := {"id": id}
		# bone +Y vs the direction to the child joint (degrees) — arm_ik/_frame assume +Y runs along the bone
		var ax := {}
		for pr in [["RightShoulder", "RightUpperArm"], ["RightUpperArm", "RightLowerArm"], ["RightLowerArm", "RightHand"], ["RightHand", "RightMiddleProximal"],
				["RightUpperLeg", "RightLowerLeg"], ["RightLowerLeg", "RightFoot"], ["RightThumbProximal", "RightThumbDistal"], ["RightIndexProximal", "RightIndexIntermediate"]]:
			var a := sk.find_bone(pr[0]); var b := sk.find_bone(pr[1])
			if a < 0 or b < 0: ax[pr[0]] = null; continue
			var ra := sk.get_bone_global_rest(a)
			ax[pr[0]] = snappedf(rad_to_deg((ra.basis.orthonormalized() * Vector3.UP).angle_to(sk.get_bone_global_rest(b).origin - ra.origin)), 0.1)
		res["boneY_vs_child_deg"] = ax
		var ua := sk.find_bone("RightUpperArm"); var la := sk.find_bone("RightLowerArm")
		var d := sk.get_bone_global_rest(la).origin - sk.get_bone_global_rest(ua).origin
		res["rest_arm_below_horizontal_deg"] = snappedf(rad_to_deg(atan2(-d.y, Vector2(d.x, d.z).length())), 0.1)
		# camera vs torso when looking down: stand, pitch the head, let the IK settle, CPU-skin the torso
		var E: float = 1.74 * 0.935
		var look := {}
		for pitch in [0.0, -45.0, -70.0]:
			var head := Transform3D(Basis(Vector3.RIGHT, deg_to_rad(pitch)), Vector3(0, E, 0))
			var l := Transform3D(Basis(), Vector3(-0.2, E - 0.5, -0.3)); var r := Transform3D(Basis(), Vector3(0.2, E - 0.5, -0.3))
			for i in 30:
				me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame
			var vv := [[]]
			var cb := func() -> void: vv[0] = _grab_verts(me)
			sk.skeleton_updated.connect(cb, CONNECT_ONE_SHOT)
			me.drive(head, l, r, 0.0, 1.0 / 60.0); await process_frame; await process_frame
			var cam := head.origin
			var fwd: Vector3 = -(me.global_transform.basis.z); fwd.y = 0; fwd = fwd.normalized()
			var front := -INF; var top := -INF; var n_ahead_above := 0; var near := INF
			for p in vv[0]:
				var q: Vector3 = p
				front = maxf(front, (q - cam).dot(fwd)); top = maxf(top, q.y)
				near = minf(near, q.distance_to(cam))
			var eye: Vector3 = me._eye_world()
			look[str(pitch)] = {"torso_front_ahead_of_cam_cm": snappedf(front * 100.0, 0.1), "torso_top_minus_cam_cm": snappedf((top - cam.y) * 100.0, 0.1),
				"nearest_torso_vertex_cm": snappedf(near * 100.0, 0.1), "avatar_eye_minus_cam_cm": [snappedf((eye - cam).x * 100, 0.1), snappedf((eye - cam).y * 100, 0.1), snappedf((eye - cam).z * 100, 0.1)], "nverts": vv[0].size()}
		res["look_down"] = look
		print("PROBE2 " + JSON.stringify(res))
		me.queue_free(); await process_frame
	quit()
