extends SceneTree
## READ-ONLY probe (helper 11-04): measures per-avatar leg/ankle/arm/thumb numbers after the real IK. Prints JSON lines.
const IDS := ["anna", "base_male", "bee_warrior", "bruno", "jennifer", "jimmy", "kate", "kyle", "oldmoustache", "olivia", "seed_san", "sendagaya_shino"]
var cap := []

func _grab(sk: Skeleton3D) -> Array:
	var out := [[]]
	var cb := func() -> void:
		var a := []
		for b in sk.get_bone_count(): a.append(sk.global_transform * sk.get_bone_global_pose(b))
		out[0] = a
	sk.skeleton_updated.connect(cb, CONNECT_ONE_SHOT)
	await process_frame
	await process_frame
	return out[0]

func _restw(me, b: int) -> Transform3D: return me.skel.global_transform * me.skel.get_bone_global_rest(b)

## rotation of child relative to parent, vs rest, as a rotation vector expressed in BODY axes (x right, y up, z back)
func _rel_rotvec(me, W: Array, parent: int, child: int) -> Vector3:
	var rp := _restw(me, parent).basis.orthonormalized(); var rc := _restw(me, child).basis.orthonormalized()
	var pp: Basis = (W[parent] as Transform3D).basis.orthonormalized(); var pc: Basis = (W[child] as Transform3D).basis.orthonormalized()
	var rel_rest := rp.inverse() * rc; var rel_now := pp.inverse() * pc
	var d := Quaternion(rel_now * rel_rest.inverse())          # in the parent's (posed) frame
	var ang := d.get_angle(); var ax := d.get_axis() if ang > 1e-5 else Vector3.ZERO
	var axw := pp * ax                                          # → world
	var bodyB: Basis = me.global_transform.basis.orthonormalized()
	return (bodyB.inverse() * axw) * rad_to_deg(ang)

func _drive(me, head: Transform3D, l: Transform3D, r: Transform3D, n := 40) -> Array:
	for i in n:
		me.drive(head, l, r, 0.0, 1.0 / 60.0)
		await process_frame
	me.drive(head, l, r, 0.0, 1.0 / 60.0)
	return await _grab(me.skel)

func _legs(me, W: Array, pre: String) -> Dictionary:
	var sk: Skeleton3D = me.skel
	var ul := sk.find_bone(pre + "UpperLeg"); var ll := sk.find_bone(pre + "LowerLeg"); var ft := sk.find_bone(pre + "Foot"); var to := sk.find_bone(pre + "Toes")
	var H: Vector3 = W[ul].origin; var K: Vector3 = W[ll].origin; var A: Vector3 = W[ft].origin
	var B: Basis = me.global_transform.basis.orthonormalized()
	var fwd := -B.z; var right := B.x
	var knee_flex := 180.0 - rad_to_deg((H - K).angle_to(A - K))
	var kv := _rel_rotvec(me, W, ul, ll); var av := _rel_rotvec(me, W, ll, ft)
	# toe direction (foot → toes) vs knee direction (where the knee points: off the hip-ankle line), horizontal
	var toe_dir: Vector3 = ((W[to].origin - A) if to >= 0 else (W[ft] as Transform3D).basis * Vector3(0, 0, 1)); toe_dir.y = 0
	var u := (A - H).normalized(); var off := (K - H) - u * (K - H).dot(u); var kd := off; kd.y = 0
	var knee_vs_toe := rad_to_deg(kd.signed_angle_to(toe_dir, Vector3.UP)) if kd.length() > 0.01 and toe_dir.length() > 0.005 else 0.0
	var shin := K - A
	var shin_fwd_tilt := rad_to_deg(atan2(shin.dot(fwd), shin.y))
	var shin_side_tilt := rad_to_deg(atan2(shin.dot(right) * (1.0 if pre == "Right" else -1.0), shin.y))
	return {"knee_flex": snappedf(knee_flex, 0.1), "ankle_y": snappedf(A.y, 0.001), "toe_y": snappedf(W[to].origin.y, 0.001) if to >= 0 else null,
		"ankle_dorsi(x)": snappedf(av.x, 0.1), "ankle_invev(z)": snappedf(av.z, 0.1), "ankle_axial(y)": snappedf(av.y, 0.1),
		"knee_flex(x)": snappedf(kv.x, 0.1), "knee_varval(z)": snappedf(kv.z, 0.1), "knee_axial(y)": snappedf(kv.y, 0.1),
		"knee_vs_toe_deg": snappedf(knee_vs_toe, 0.1), "shin_fwd_tilt": snappedf(shin_fwd_tilt, 0.1), "shin_out_tilt": snappedf(shin_side_tilt, 0.1)}

func _upper_arm_twist(me, W: Array, pre: String) -> float:
	var sk: Skeleton3D = me.skel
	var ua := sk.find_bone(pre + "UpperArm"); var p := sk.get_bone_parent(ua)
	var rp := _restw(me, p).basis.orthonormalized(); var ru := _restw(me, ua).basis.orthonormalized()
	var pp: Basis = (W[p] as Transform3D).basis.orthonormalized(); var pu: Basis = (W[ua] as Transform3D).basis.orthonormalized()
	var q := Quaternion((rp.inverse() * ru).inverse() * (pp.inverse() * pu))    # in the upper arm's rest frame (+Y along bone)
	return snappedf(rad_to_deg(wrapf(2.0 * atan2(q.y, q.w), -PI, PI)), 0.1)

func _clav(me, W: Array, pre: String) -> float:
	var c: int = me.skel.find_bone(pre + "Shoulder")
	if c < 0: return -1.0
	var p: int = me.skel.get_bone_parent(c)
	var rp := _restw(me, p).basis.orthonormalized(); var rc := _restw(me, c).basis.orthonormalized()
	var pp: Basis = (W[p] as Transform3D).basis.orthonormalized(); var pc: Basis = (W[c] as Transform3D).basis.orthonormalized()
	return snappedf(rad_to_deg(Quaternion((rp.inverse() * rc).inverse() * (pp.inverse() * pc)).get_angle()), 0.1)

func _hand_geom(me, W: Array, pre: String, curl: float) -> Dictionary:
	var sk: Skeleton3D = me.skel
	var f := func(n): return sk.find_bone(pre + n)
	var hd: int = f.call("Hand"); var ip: int = f.call("IndexProximal"); var lp: int = f.call("LittleProximal"); var mp: int = f.call("MiddleProximal")
	var mi: int = f.call("MiddleIntermediate")
	var t1: int = f.call("ThumbProximal"); var t2: int = f.call("ThumbDistal")
	if hd < 0 or ip < 0 or lp < 0 or mp < 0 or t2 < 0: return {"missing": true}
	var side := 1.0 if pre == "Left" else -1.0
	var Hh: Vector3 = W[hd].origin
	var fdir: Vector3 = (W[mp].origin - Hh).normalized()
	var across: Vector3 = (W[ip].origin - W[lp].origin)
	var pn: Vector3 = (side * fdir.cross(across)).normalized()                        # palm normal from the geometry
	var tip: Vector3 = W[t2].origin + (W[t2].origin - W[t1].origin) * 0.75
	var rel := tip - Hh
	return {"palm_n": pn, "thumb_tip_palmar_cm": snappedf(rel.dot(pn) * 100.0, 0.1), "thumb_tip_to_index_cm": snappedf(tip.distance_to(W[ip].origin) * 100.0, 0.1),
		"thumb_tip_to_mid_int_cm": snappedf(tip.distance_to(W[mi].origin) * 100.0, 0.1) if mi >= 0 else -1.0,
		"thumb_tip_along_fingers_cm": snappedf(rel.dot(fdir) * 100.0, 0.1), "thumb_tip_toward_little_cm": snappedf(rel.dot(-across.normalized()) * 100.0, 0.1)}

func _init() -> void:
	await process_frame
	var ground := StaticBody3D.new(); var gcs := CollisionShape3D.new(); var gb := BoxShape3D.new(); gb.size = Vector3(20, 0.2, 20)
	gcs.shape = gb; ground.add_child(gcs); root.add_child(ground); ground.global_position = Vector3(0, -0.1, 0)
	for id in IDS:
		var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
		root.add_child(me)
		if not me.setup(id, 1.74): print(JSON.stringify({"id": id, "setup": false})); me.queue_free(); continue
		var sk: Skeleton3D = me.skel
		var E: float = 1.74 * 0.935
		var res := {"id": id, "scale": snappedf(me._scale, 0.001), "eye_source": me.eye_source,
			"native_eye_m": snappedf(me._rest_eye * me.skel.global_transform.basis.get_scale().x / me._scale, 0.001) if false else snappedf(me._rest_eye, 0.001),
			"rest_ankle_scaled": snappedf(me._rest_ankle * me._scale, 0.003), "ik_ankle_target": snappedf(0.08 * me._scale, 0.003)}
		# hip joint half-width vs the 0.11*scale foot spacing
		var hl := _restw(me, sk.find_bone("LeftUpperLeg")).origin; var hr := _restw(me, sk.find_bone("RightUpperLeg")).origin
		res["hip_halfwidth"] = snappedf(hl.distance_to(hr) * 0.5, 0.001); res["foot_halfwidth_target"] = snappedf(0.11 * me._scale, 0.001)
		# rest palm normal vs the assumed −Y
		var Wr := []
		for b in sk.get_bone_count(): Wr.append(_restw(me, b))
		for pre in ["Left", "Right"]:
			var g := _hand_geom(me, Wr, pre, 0.0)
			if g.has("palm_n"): res[pre + "_palm_vs_down_deg"] = snappedf(rad_to_deg((g["palm_n"] as Vector3).angle_to(Vector3.DOWN)), 0.1)
		# standing + crouches, hands in front
		var crouches := {}
		for drop in [0.0, 0.25, 0.45, 0.68]:
			var head := Transform3D(Basis(), Vector3(0, E - drop, 0))
			var l := Transform3D(Basis(), head.origin + Vector3(-0.2, -0.45, -0.35)); var r := Transform3D(Basis(), head.origin + Vector3(0.2, -0.45, -0.35))
			var W: Array = await _drive(me, head, l, r)
			var c := {"L": _legs(me, W, "Left"), "body": me.crouch.duplicate()}
			crouches[str(drop)] = c
		res["crouch"] = crouches
		# arms: a few poses, roll about the fingers axis (pronation/supination)
		var arms := {}
		var head2 := Transform3D(Basis(), Vector3(0, E, 0))
		var poses := {"front": Vector3(0.2, -0.45, -0.35), "down_side": Vector3(0.25, -0.75, 0.0), "overhead": Vector3(0.2, 0.35, -0.1), "across": Vector3(-0.15, -0.4, -0.3), "face": Vector3(0.05, -0.05, -0.2)}
		for pn in poses:
			for roll in [-90.0, 0.0, 90.0]:
				var pv: Vector3 = poses[pn]
				var bR := Basis(Vector3(0, 0, 1), deg_to_rad(roll))
				if pn == "down_side": bR = Basis(Vector3(1, 0, 0), deg_to_rad(-90)) * bR      # fingers pointing down
				if pn == "overhead": bR = Basis(Vector3(1, 0, 0), deg_to_rad(90)) * bR
				var r := Transform3D(bR, head2.origin + pv)
				var l := Transform3D(Basis(), head2.origin + Vector3(-0.2, -0.45, -0.35))
				var W: Array = await _drive(me, head2, l, r, 25)
				var jr: Dictionary = {}
				var cb := func() -> void: jr.merge(me.joint_report("Right"))
				sk.skeleton_updated.connect(cb, CONNECT_ONE_SHOT)
				me.drive(head2, l, r, 0.0, 1.0 / 60.0); await process_frame; await process_frame
				jr["humeral_twist"] = _upper_arm_twist(me, W, "Right"); jr["clav_deg"] = _clav(me, W, "Right")
				arms["%s_%d" % [pn, int(roll)]] = jr
		res["arms_R"] = arms
		# thumb: open (curl 0) vs fist (curl 1)
		var head3 := Transform3D(Basis(), Vector3(0, E, 0))
		var l3 := Transform3D(Basis(), head3.origin + Vector3(-0.2, -0.45, -0.35)); var r3 := Transform3D(Basis(), head3.origin + Vector3(0.2, -0.45, -0.35))
		var th := {}
		for cv in [0.0, 0.5, 1.0]:
			me.curl = Vector4(cv, cv, cv, cv)
			var W3: Array = await _drive(me, head3, l3, r3, 8)
			var g := _hand_geom(me, W3, "Right", cv); g.erase("palm_n")
			th[str(cv)] = g
		res["thumb_R"] = th
		print("PROBE " + JSON.stringify(res))
		me.queue_free(); await process_frame
	quit()
