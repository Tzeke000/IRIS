# Iris's placeholder body in the temple: floats, breathes, and turns to face the player.
# Later: driven live from the server (mood colour, speaking glow, voice) over the network.
extends Node3D

var _t := 0.0
var _base_y := 0.0

func _ready() -> void:
	_base_y = position.y

func _process(delta: float) -> void:
	_t += delta
	position.y = _base_y + sin(_t * 1.3) * 0.06
	var cam := get_viewport().get_camera_3d()
	if cam:
		var target := cam.global_position
		target.y = global_position.y
		if global_position.distance_to(target) > 0.01:
			look_at(target, Vector3.UP)
