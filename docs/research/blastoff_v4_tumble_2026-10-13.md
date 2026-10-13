# Blast Off v4 — a realistic near-loss-of-control tumble (research, 2026-10-13)

Zeke: "a tumble, but it has to recover and tumble realistically for a rocket flying through Earth's gravity".

## What real vehicles do (sources below)
- **Steering = thrust vector control (TVC).** Engines gimbal only a few degrees: Shuttle SSMEs up to ~10.5° (pitch/yaw, 2-DOF); F-1 ~6° (forum figure,
  unverified). Solid boosters are hard to steer (nozzle-only gimbal or fluid injection). **Roll** is usually handled by separate RCS thrusters.
- **Authority is the limit.** Firefly Alpha (Sept 2021) lost one of four engines at 15 s; it stayed stable subsonic, but going transonic "the three engine
  thrust vector control was insufficient and the vehicle tumbled" for ~10 s before the flight termination system destroyed it. So "barely
  recovers" = the disturbance must stay JUST inside the gimbal authority, and the hardest moment is near max-Q / transonic.
- **Separation upsets are real.** The NASA Shuttle SRB-separation studies note the main-engine gimbal positions can be adverse at separation, and the CG
  shifts a lot, so the engines must re-point to compensate.
- New Shepard NS-23 (2022) visibly "wobbled" through max-Q before its abort. That's the visual vocabulary: a wobble that grows, not a cartoon spin.

## Motion design for 0:56 → 1:07 (false drop → halftime drop)
1. **Impulse at the blast (0:56, on the beat):** booster 3 leaves early. The thrust from the two remaining boosters is asymmetric, and the blast
   overpressure adds a sideways shove. Model it as a step disturbance torque in yaw plus a small roll kick.
2. **Attitude dynamics:** the body angle θ follows a 2nd-order system, I·θ̈ = τ_dist(t) − K_p·θ − K_d·θ̇, with the control torque CLAMPED to the
   gimbal authority (±8° of thrust × the moment arm). While clamped (saturation) the error keeps growing. That gives the realistic overshoot to
   ~25–35° off vertical. Once inside authority it settles as a damped oscillation (period ~2.5–3.5 s for a vehicle this size, ζ ≈ 0.25), within ±2° by ~1:06.
3. **Gravity:** while the nose is off vertical, the vertical thrust component drops (cos θ). The trajectory SAGS (vertical speed dips) and drifts sideways
   (sin θ). Integrate the position from the thrust direction + g, so the arc visibly bends and then climbs again. Roll is damped faster by RCS: add
   small RCS puff sprites on the roll correction.
4. **Visual tells:** the plume bends with the gimbal (rotate the plume emitters by the commanded gimbal angle). The engine bells visibly swivel. Vapor/condensation
   streams tear off at the angle. The LEDs stay red until the oscillation is < 5°, then the rainbow reboot sweeps on the 1:07 drop.
5. **Onboard fisheye feed (idea 3):** horizon/earth swinging, plume flapping, debris streaking past, short digital-video breakups
   (block artifacts / frozen frames for 2–4 frames), then the warning HUD (B612 font).
6. **Chase cameras** must NOT follow the wobble pivot (the research #1 plan: the wobble lives on its own pivot between the rig and the stages).

## Assets fetched (2026-10-13)
- JangaFX **Mid-Air Explosion** VDB, CC0, 126 frames, grids density / flames / temperature, raw bbox ≈ 200×390×240 voxel units (scale at
  build time). `/data/iris_assets/jangafx/midair_explosion/ex/` (extracted with `unrar`; `unar` failed on RAR5 for 25 files).
- **B612 / B612 Mono** fonts (Airbus cockpit-display font), SIL OFL 1.1: `/data/iris_assets/fonts/b612/` (+ OFL.txt).

## Sources
- SJSU thesis on launch-vehicle TVC: https://www.sjsu.edu/ae/docs/project-thesis/Gia%20Hieu%20Trinh-F24.pdf
- NASA NTRS (Shuttle SRB separation / gimbal adverse positions): https://ntrs.nasa.gov/citations/19830048946 · https://ntrs.nasa.gov/api/citations/19700025223/downloads/19700025223.pdf · https://ntrs.nasa.gov/api/citations/19680014937/downloads/19680014937.pdf
- Firefly Alpha FLTA001 anomaly: https://www.fourmilab.ch/scanalyzer/archives/2021/09/firefly-aerospace-alpha-experiences-anomaly-in-flight.html · https://spacenews.com/?p=118634
- New Shepard NS-23 wobble at max-Q: https://www.cfpublic.org/2022-09-12/uncrewed-blue-origin-rocket-encounters-anomaly-during-flight-capsule-escapes
- Gimbal story: https://ing.dk/holdning/controlled-chaos-gimbal-story
