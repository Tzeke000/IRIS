// Cable-arms for my eye (2026-10-08). Four ribbon cables that live on a 2D overlay canvas.
//
// Physics: each arm is a Verlet chain. The root is pinned to a port on the iris rim, and the root
// direction is only a SOFT constraint, so the cable bends right where it leaves the body (Zeke: "stiff
// at the beginning"). Segments may stretch up to MAX_STRETCH when the tip is asked to reach far ("your
// arms should be able to stretch a little"). Damping is high, so it's flowing, not bouncy. Idle tips
// drift on slow, out-of-phase noise (zero-g); a reaching tip follows an animated GOAL with Disney timing:
// anticipation (a small pull-back), ease-in-out travel, slight overshoot, settle.
//
// Rendering: each arm is a ribbon of WIRES wires. Twist is drawn by rotating the ribbon's offset profile
// along its length: wires converge where the ribbon is edge-on and swap order past it.

export type Vec = { x: number; y: number };
export type ArmKey = "right" | "down" | "left" | "up";
export const ARM_KEYS: ArmKey[] = ["right", "down", "left", "up"];
const PORT_ANGLE: Record<ArmKey, number> = { right: 0, down: Math.PI / 2, left: Math.PI, up: -Math.PI / 2 };

const N = 16;               // nodes per arm
const WIRES = 6;
const MAX_STRETCH = 1.55;
const ITER = 10;

const add = (a: Vec, b: Vec): Vec => ({ x: a.x + b.x, y: a.y + b.y });
const sub = (a: Vec, b: Vec): Vec => ({ x: a.x - b.x, y: a.y - b.y });
const mul = (a: Vec, k: number): Vec => ({ x: a.x * k, y: a.y * k });
const len = (a: Vec) => Math.hypot(a.x, a.y);
const norm = (a: Vec): Vec => { const l = len(a) || 1; return { x: a.x / l, y: a.y / l }; };
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Disney-style timing curves
export const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOutBack = (t: number, s = 1.2) => 1 + (s + 1) * Math.pow(t - 1, 3) + s * Math.pow(t - 1, 2);

type Move = {
  from: Vec; to: Vec; t0: number; dur: number;   // ms
  antic: number;                                  // anticipation distance (px, opposite the move)
  press?: { at: number; done: boolean; onContact?: () => void };
};

export class Arm {
  key: ArmKey;
  p: Vec[] = [];
  prev: Vec[] = [];
  seg = 10;             // natural segment length (px), set by layout()
  stretch = 1;          // current stretch multiplier
  twist = 0;            // -1..1 → turns along the length
  twistGoal = 0;
  goal: Vec | null = null;  // where the tip is being driven (null = free drift)
  move: Move | null = null;
  phase: number;
  holding = false;
  pressDepth = 0;       // 0..1 visual "push into the screen" at the tip

  constructor(key: ArmKey, seed: number) {
    this.key = key;
    this.phase = seed * 1.7 + 0.3;
  }

  /** Re-seat the chain at a root + direction (first frame, or after a big size change). */
  reset(root: Vec, dir: Vec, seg: number) {
    this.seg = seg;
    this.p = [];
    for (let i = 0; i < N; i++) this.p.push(add(root, mul(dir, seg * i)));
    this.prev = this.p.map((q) => ({ ...q }));
  }

  /** Reach to a point with anticipation + ease + small overshoot. Optional tap at the end. */
  reachTo(target: Vec, now: number, opts: { dur?: number; press?: boolean; onContact?: () => void } = {}) {
    const tip = this.p[N - 1];
    const from = this.goal ?? { ...tip };
    const d = len(sub(target, from));
    const dur = opts.dur ?? clamp(380 + d * 0.9, 420, 1100);
    this.move = {
      from, to: target, t0: now, dur, antic: clamp(d * 0.08, 6, 22),
      press: opts.press ? { at: dur + 120, done: false, onContact: opts.onContact } : undefined,
    };
    this.holding = true;
  }

  release() {
    this.goal = null;
    this.move = null;
    this.holding = false;
  }

  private animGoal(now: number): Vec | null {
    const m = this.move;
    if (!m) return this.goal;
    const t = now - m.t0;
    const dir = norm(sub(m.to, m.from));
    const ANT = 140;
    let g: Vec;
    if (t < ANT) {                       // anticipation: ease a little backwards first
      const k = Math.sin((t / ANT) * Math.PI * 0.5);
      g = add(m.from, mul(dir, -m.antic * k));
    } else if (t < ANT + m.dur) {        // travel: ease out with a little overshoot
      const u = (t - ANT) / m.dur;
      const k = easeOutBack(u, 0.9);
      const start = add(m.from, mul(dir, -m.antic));
      g = { x: lerp(start.x, m.to.x, k), y: lerp(start.y, m.to.y, k) };
    } else {
      g = { ...m.to };
    }
    // tap: push in, fire the click at contact, come back out
    if (m.press && t >= ANT + m.dur) {
      const pt = t - (ANT + m.dur);
      const TAP = 260;
      this.pressDepth = pt < TAP ? Math.sin((pt / TAP) * Math.PI) : 0;
      if (!m.press.done && pt >= TAP * 0.5) {
        m.press.done = true;
        try { m.press.onContact?.(); } catch { /* the click handler is the app's business */ }
      }
    } else {
      this.pressDepth = 0;
    }
    this.goal = g;
    return g;
  }

  step(now: number, dt: number, root: Vec, rootDir: Vec, idleTip: Vec, scale: number) {
    if (!this.p.length) this.reset(root, rootDir, 10 * scale);
    const goal = this.animGoal(now);
    // velocity damping: high = smooth (less bouncy); a reaching arm is a bit crisper
    const damp = goal ? 0.86 : 0.9;
    const tsec = now / 1000;
    for (let i = 1; i < N; i++) {
      const cur = this.p[i];
      const v = mul(sub(cur, this.prev[i]), damp);
      this.prev[i] = { ...cur };
      // zero-g drift: slow curl-ish noise, stronger toward the tip
      const w = i / (N - 1);
      const f = {
        x: Math.sin(tsec * 0.7 + this.phase + i * 0.35) * 0.16 * scale * w,
        y: Math.cos(tsec * 0.55 + this.phase * 1.3 + i * 0.3) * 0.16 * scale * w,
      };
      this.p[i] = add(add(cur, v), mul(f, dt * 0.06));
    }
    // tip pull: toward the goal when reaching, toward the idle drift point otherwise
    const tipTarget = goal ?? idleTip;
    const tipK = goal ? 0.32 : 0.05;
    const tip = this.p[N - 1];
    this.p[N - 1] = add(tip, mul(sub(tipTarget, tip), tipK));
    // stretch: grow the segments if the goal is beyond reach, relax back otherwise
    const reach = len(sub(tipTarget, root));
    const natural = this.seg * (N - 1);
    const want = goal ? clamp(reach / natural, 1, MAX_STRETCH) : 1;
    this.stretch = lerp(this.stretch, want, 0.06);
    this.twist = lerp(this.twist, this.twistGoal, 0.05);
    const L = this.seg * this.stretch;
    // constraints
    for (let it = 0; it < ITER; it++) {
      this.p[0] = { ...root };
      // soft root direction: node 1 wants to sit along the port normal, but may bend
      const want1 = add(root, mul(rootDir, L));
      this.p[1] = { x: lerp(this.p[1].x, want1.x, 0.35), y: lerp(this.p[1].y, want1.y, 0.35) };
      for (let i = 0; i < N - 1; i++) {
        const a = this.p[i], b = this.p[i + 1];
        const d = sub(b, a);
        const dl = len(d) || 1e-6;
        const diff = (dl - L) / dl;
        const wa = i === 0 ? 0 : 0.5, wb = i === 0 ? 1 : 0.5;
        this.p[i] = add(a, mul(d, diff * wa));        // too long → pull the two ends together
        this.p[i + 1] = sub(b, mul(d, diff * wb));
      }
      if (goal) this.p[N - 1] = { x: lerp(this.p[N - 1].x, tipTarget.x, 0.25), y: lerp(this.p[N - 1].y, tipTarget.y, 0.25) };
    }
  }

  tip(): Vec { return this.p[N - 1]; }
  tipDir(): Vec { return norm(sub(this.p[N - 1], this.p[N - 3])); }
}

export type ArmStyle = { light: string; copper: string; copperDark: string; scale: number; dpr: number; glow?: boolean };

/** Smooth path through the chain (midpoint quadratic). */
function trace(ctx: CanvasRenderingContext2D, pts: Vec[]) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const m = { x: (pts[i].x + pts[i + 1].x) / 2, y: (pts[i].y + pts[i + 1].y) / 2 };
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, m.x, m.y);
  }
  const l = pts[pts.length - 1];
  ctx.lineTo(l.x, l.y);
}

export function drawArm(ctx: CanvasRenderingContext2D, arm: Arm, st: ArmStyle) {
  const P = arm.p;
  if (P.length < 2) return;
  const s = st.scale;
  const width = 3.0 * s * WIRES;         // ribbon width (px)
  // per-node normal + twist profile
  const nrm: Vec[] = P.map((_, i) => {
    const a = P[Math.max(0, i - 1)], b = P[Math.min(P.length - 1, i + 1)];
    const t = norm(sub(b, a));
    return { x: -t.y, y: t.x };
  });
  const turns = 0.25 + arm.twist * 0.9;
  const prof = P.map((_, i) => Math.cos((i / (P.length - 1)) * Math.PI * 2 * turns));
  // bind near the root and the connector (clamp + sleeve); loosen in the middle
  const loose = P.map((_, i) => { const u = i / (P.length - 1); return 0.55 + 0.45 * Math.sin(Math.PI * Math.min(1, u / 0.92)); });
  const order = [...Array(WIRES).keys()];
  // draw back-to-front: wires whose offset faces "away" first
  const wirePts = order.map((w) => {
    const off = (w - (WIRES - 1) / 2) / WIRES;
    return P.map((q, i) => add(q, mul(nrm[i], off * width * prof[i] * loose[i])));
  });
  // root collar
  ctx.save();
  // shadow/underlay for the whole ribbon (reads as depth)
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = width * 0.9;
  trace(ctx, P.map((q) => ({ x: q.x + 2 * s, y: q.y + 3 * s })));
  ctx.stroke();
  for (let w = 0; w < WIRES; w++) {
    const pts = wirePts[w];
    const isLight = w === Math.floor(WIRES / 2) - 1 || w === Math.floor(WIRES / 2);
    const lw = 2.6 * s;
    if (isLight) {
      ctx.shadowColor = st.light;
      ctx.shadowBlur = st.glow === false ? 0 : 10 * s;     // blur is the expensive part on software canvases
      ctx.strokeStyle = st.light;
      ctx.lineWidth = lw;
      trace(ctx, pts);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = lw * 0.35;
      trace(ctx, pts);
      ctx.stroke();
    } else {
      ctx.strokeStyle = st.copperDark;
      ctx.lineWidth = lw;
      trace(ctx, pts);
      ctx.stroke();
      ctx.strokeStyle = st.copper;
      ctx.lineWidth = lw * 0.62;
      trace(ctx, pts);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255,236,210,0.55)";
      ctx.lineWidth = lw * 0.2;
      trace(ctx, pts.map((q) => ({ x: q.x - 0.5 * s, y: q.y - 0.5 * s })));
      ctx.stroke();
    }
  }
  // clamp sleeve near the root
  const c1 = P[2], cd = norm(sub(P[3], P[1]));
  drawBox(ctx, c1, cd, width * 1.05, 5 * s, "#15171b", "#2a2e35");
  // connector at the tip (+ gold pins), pushed "into" the screen when pressing
  const tip = P[P.length - 1], td = arm.tipDir();
  const k = 1 - 0.18 * arm.pressDepth;
  const cc = add(tip, mul(td, 7 * s * k));
  drawBox(ctx, cc, td, width * 1.15 * k, 14 * s * k, "#1b1e24", "#3b4049");
  const pinBase = add(tip, mul(td, 15 * s * k));
  const pn = { x: -td.y, y: td.x };
  ctx.fillStyle = "#e0b04a";
  for (let i = 0; i < 7; i++) {
    const o = (i - 3) * (width * 1.0 / 7) * k;
    const q = add(pinBase, mul(pn, o));
    ctx.save();
    ctx.translate(q.x, q.y);
    ctx.rotate(Math.atan2(td.y, td.x));
    ctx.fillRect(-1.5 * s, -0.9 * s, 4 * s * k, 1.8 * s);
    ctx.restore();
  }
  if (arm.pressDepth > 0.01) {         // contact glow
    ctx.shadowColor = st.light;
    ctx.shadowBlur = 18 * s * arm.pressDepth;
    ctx.fillStyle = `rgba(255,255,255,${0.25 * arm.pressDepth})`;
    ctx.beginPath();
    ctx.arc(pinBase.x, pinBase.y, 9 * s * arm.pressDepth, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawBox(ctx: CanvasRenderingContext2D, c: Vec, dir: Vec, w: number, h: number, fill: string, edge: string) {
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.rotate(Math.atan2(dir.y, dir.x));
  const r = Math.min(3, h / 3);
  ctx.beginPath();
  // box is h long (along dir) and w wide
  const x = -h / 2, y = -w / 2;
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + h, y, x + h, y + w, r);
  ctx.arcTo(x + h, y + w, x, y + w, r);
  ctx.arcTo(x, y + w, x, y, r);
  ctx.arcTo(x, y, x + h, y, r);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = edge;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

/** Port position + outward normal on the iris rim, rotated by the body's current spin. */
export function port(center: Vec, radius: number, key: ArmKey, spin: number): { root: Vec; dir: Vec } {
  const a = PORT_ANGLE[key] + spin;
  const dir = { x: Math.cos(a), y: Math.sin(a) };
  return { root: add(center, mul(dir, radius)), dir };
}

/** Where a free arm floats: out along its port, drifting a little back and sideways (zero-g). */
export function idleTip(center: Vec, radius: number, key: ArmKey, spin: number, t: number, phase: number, reach: number): Vec {
  // swing well off the port axis so the free cables CURVE (a tip straight out reads as a stiff rod)
  const a = PORT_ANGLE[key] + spin + 0.38 * Math.sin(t * 0.19 + phase) + 0.15 * Math.sin(t * 0.47 + phase * 2.3);
  const r = radius + reach * (0.86 + 0.08 * Math.sin(t * 0.31 + phase * 1.7));
  const w = reach * 0.18;
  return { x: center.x + Math.cos(a) * r + w * Math.sin(t * 0.5 + phase), y: center.y + Math.sin(a) * r + w * Math.cos(t * 0.43 + phase) };
}

/** Which arm should reach a target: the one whose port faces it best. */
export function bestArm(center: Vec, target: Vec, spin: number, busy: Set<ArmKey>): ArmKey {
  const ang = Math.atan2(target.y - center.y, target.x - center.x);
  let best: ArmKey = "right", bd = 1e9;
  for (const k of ARM_KEYS) {
    if (busy.has(k)) continue;
    let d = Math.abs(((ang - (PORT_ANGLE[k] + spin) + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
    if (d < bd) { bd = d; best = k; }
  }
  return best;
}
