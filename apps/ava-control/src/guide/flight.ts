// How my body FLIES across the screen (2026-10-08, Zeke: "your propulsion is coming from the back of you — show
// your profile when you move left and right, so it doesn't look like you're just gliding"; + the Disney pass:
// nothing should end early). Shared by the app's guide eye and the desktop widget.
//
//   anticipation  turn side-on toward where I'm going and lean back a touch, with a tiny wind-up the other way
//   flight        smootherstep path on a gentle arc; I lean INTO the acceleration and back while braking
//   follow-through the lean overshoots as I stop, then I turn to face you again on a soft spring (it overshoots too)
//   settle        a short beat before the next action, so a move never gets cut off by the next one
export type Vec = { x: number; y: number };
export type Pose = { x: number; y: number; yaw?: number; pitch?: number; roll?: number } | undefined;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
export const smoother = (u: number) => u * u * u * (u * (u * 6 - 15) + 10);
/** smootherstep's acceleration, normalised to -1..1 (+ while speeding up, - while braking) */
const accel = (u: number) => (60 * u * (1 - u) * (1 - 2 * u)) / 5.7735;

export async function fly(from: Vec, to: Vec, io: {
  place: (p: Vec) => void | Promise<void>;          // move my body to p (page px, or screen px for the widget)
  pose: (p: Pose) => void;                          // body heading
  trail?: (p: Vec, back: Vec, speed: number) => void;
  scale?: number;                                   // px per "page px" (widget on a high-DPI screen)
}): Promise<void> {
  const k = io.scale ?? 1;
  const d = Math.hypot(to.x - from.x, to.y - from.y);
  if (d < 6 * k) { await io.place(to); return; }
  const dir = { x: (to.x - from.x) / d, y: (to.y - from.y) / d };
  const short = d < 90 * k;                          // a little hop: no full side-on turn
  const yaw = dir.x * (short ? 0.5 : 1.2), pitch = dir.y * (short ? 0.25 : 0.55);
  const look = { x: dir.x * 0.85, y: -dir.y * 0.85 };
  // anticipation: turn into the move, lean back, wind up a few px the other way
  const ANT = short ? 160 : 320, back = Math.min(10 * k, d * 0.04);
  const t0 = performance.now();
  io.pose({ ...look, yaw, pitch, roll: dir.x * 0.1 });
  for (;;) {
    const u = Math.min(1, (performance.now() - t0) / ANT);
    const e = 0.5 - 0.5 * Math.cos(u * Math.PI);
    await io.place({ x: from.x - dir.x * back * e, y: from.y - dir.y * back * e });
    if (u >= 1) break;
    await sleep(16);
  }
  // flight
  const s0 = { x: from.x - dir.x * back, y: from.y - dir.y * back };
  const dur = Math.min(2000, (650 + (d / k) * 0.7));
  const bulge = Math.min(80, (d / k) * 0.09) * k * (dir.x >= 0 ? -1 : 1);
  const t1 = performance.now();
  let prev = s0;
  for (;;) {
    const u = Math.min(1, (performance.now() - t1) / dur);
    const s = smoother(u), b = bulge * Math.sin(Math.PI * u);
    const p = { x: s0.x + (to.x - s0.x) * s - dir.y * b, y: s0.y + (to.y - s0.y) * s + dir.x * b };
    io.pose({ ...look, yaw, pitch, roll: -dir.x * 0.22 * accel(u) });
    await io.place(p);
    const sp = Math.hypot(p.x - prev.x, p.y - prev.y);
    if (io.trail && sp > 0.8 * k) io.trail(p, { x: -dir.x, y: -dir.y }, sp / k);
    prev = p;
    if (u >= 1) break;
    await sleep(16);
  }
  // follow-through: the lean swings past and back, still side-on for a beat, then turn to face you again
  io.pose({ ...look, yaw: yaw * 0.85, pitch: pitch * 0.6, roll: dir.x * 0.12 });
  await sleep(170);
  io.pose(undefined);
  await sleep(short ? 180 : 360);                    // settle — let the spring finish before the next action
}
