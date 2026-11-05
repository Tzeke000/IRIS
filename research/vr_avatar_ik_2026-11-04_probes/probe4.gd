extends SceneTree
func _init() -> void:
	await process_frame
	var me := Node3D.new(); me.set_script(load("res://scripts/avatar_body.gd")); me.top_level = true; me.is_self = true
	root.add_child(me); me.setup("bee_warrior", 1.74)
	var sk: Skeleton3D = me.skel
	for n in ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftUpperLeg", "LeftShoulder", "LeftUpperArm"]:
		var b := sk.find_bone(n)
		var chain := []
		var p := b
		while p >= 0: chain.append(sk.get_bone_name(p)); p = sk.get_bone_parent(p)
		print(n, " -> ", chain)
	print("bones: ", sk.get_bone_count())
	quit()
