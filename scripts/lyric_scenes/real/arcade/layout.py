"""Shared cabinet layout (pure python): imported by make_art.py (venv) AND build_scene.py (Blender),
so the printed control-panel art lines up with the 3-D buttons. Units: metres.

Cabinet local frame: x = right, y = BACK (the front faces -y), z = up, origin = floor centre front.
Side profile (y, z), walked from the front-bottom corner, clockwise when seen from +x."""

WIDTH = 0.66                 # outer
SIDE_T = 0.02                # side-panel thickness
INNER = WIDTH / 2 - SIDE_T   # 0.31

# front face pieces (y, z)
KICK = [(0.0, 0.0), (0.0, 0.80)]                       # kick panel / coin door
CP_UNDER = [(0.0, 0.80), (-0.26, 0.88)]
CP_LIP = [(-0.26, 0.88), (-0.26, 0.955)]
CP_TOP = [(-0.26, 0.955), (0.02, 1.03)]                # control-panel surface (front lip -> back)
BEZEL = [(0.05, 1.05), (0.215, 1.625)]                 # bezel / screen slope (bottom -> top)
SPEAKER = [(0.215, 1.625), (0.06, 1.665)]              # under the marquee
MARQUEE = [(0.06, 1.665), (0.06, 1.865)]
TOP = [(0.06, 1.865), (0.80, 1.865)]
BACK = [(0.80, 1.865), (0.80, 0.0)]

# side panel outline (slightly proud of the front pieces so they sit in a recess)
SIDE = [(-0.012, 0.0), (-0.012, 0.79), (-0.275, 0.875), (-0.275, 0.965), (0.0, 1.04),
        (0.022, 1.05), (0.19, 1.635), (0.045, 1.675), (0.045, 1.885), (0.82, 1.885), (0.82, 0.0)]

# screen (4:3) centred on the bezel slope; bezel slope length ~0.598
SCREEN_W, SCREEN_H = 0.47, 0.3525
SCREEN_BULGE = 0.022          # CRT face curvature (sagitta at the centre)

# control panel: (u, v) on the CP_TOP surface; u = x (m), v = metres along the slope from the
# front lip toward the back. CP slope length ~0.290.
JOY = (-0.165, 0.135)
BUTTONS = [  # (u, v, colour name, label)
    (0.000, 0.105, "red", "JUMP"), (0.062, 0.118, "yellow", "SMASH"), (0.124, 0.128, "blue", "BOLT"),
    (0.010, 0.175, "green", ""), (0.072, 0.188, "white", ""), (0.134, 0.198, "purple", ""),
]
START = [(-0.040, 0.245, "1P"), (0.040, 0.245, "2P")]
BTN_R = 0.0145
CP_LEN = 0.290

# coin door on the kick panel (x, z) centre + size
COIN_DOOR = (0.0, 0.47, 0.25, 0.30)

# the row of cabinets
SPACING = 0.74
N_EACH_SIDE = 3               # hero in the middle -> 7 cabinets
OTHER_GAMES = ["STAR BEAT", "NEON DRIFT", "GRID RUNNER", "VOLT KNIGHT", "PIXEL RAID", "ORBIT X"]
