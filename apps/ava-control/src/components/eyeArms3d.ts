// Real 3D cable-arms for the cyborg eye (2026-10-08, Zeke: "your cables look 2D… on top of your body… not
// actually in those sockets… they twist onto themselves").
//
// Each arm is a 3D Verlet chain rooted INSIDE its collar (so the socket visibly swallows the cable) and rendered as
// a ribbon of real round tubes in the SAME three.js scene as the body: lit, shaded, and occluded by the ball when it
// passes behind. The ribbon's side vector is parallel-transported along the cable and kept facing the camera, so
// it never flips over itself. Physics: bendable root, bending stiffness, limited stretch, damped (not bouncy).
// The guide drives the tips in PAGE coordinates through `eyeArms` (see GuideOverlay); this module maps them in.
import * as THREE from "three";

export type ArmKey3 = "right" | "down" | "left" | "up";
const KEYS: ArmKey3[] = ["right", "down", "left", "up"];
const AXIS: Record<ArmKey3, THREE.Vector3> = {
  right: new THREE.Vector3(1, 0, 0), left: new THREE.Vector3(-1, 0, 0),
  up: new THREE.Vector3(0, 1, 0), down: new THREE.Vector3(0, -1, 0),
};
const N = 18;              // chain nodes
const SEG = 0.1;           // segment length (ball radii) → ~1.7 radii of cable
const WIRES = 6;
const WIRE_R = 0.019;      // wire radius
const RING = 6;            // tube radial segments
const SAMPLES = 30;        // tube samples along the length
const ITER = 14;
const MAX_STRETCH = 1.3;

const v = () => new THREE.Vector3();
const smoother = (u: number) => u * u * u * (u * (u * 6 - 15) + 10);

type Drive = { path: (tMs: number) => THREE.Vector3; t0: number; dur: number; press?: { atMs: number; done: boolean; onContact?: () => void } };

class Arm3D {
  key: ArmKey3;
  p: THREE.Vector3[] = [];
  q: THREE.Vector3[] = [];        // previous positions
  stretch = 1;
  drive: Drive | null = null;
  goal: THREE.Vector3 | null = null;
  holding = false;
  twistGoal = 0;
  twist = 0;
  pressDepth = 0;
  phase: number;
  group = new THREE.Group();
  wires: { geo: THREE.BufferGeometry; pos: Float32Array; nrm: Float32Array; mesh: THREE.Mesh }[] = [];
  clamp: THREE.Mesh;
  conn: THREE.Mesh;
  pins: THREE.InstancedMesh;
  private tmpM = new THREE.Matrix4();

  constructor(key: ArmKey3, seed: number, mats: { copper: THREE.Material; strip: THREE.Material; dark: THREE.Material; gun: THREE.Material; gold: THREE.Material }) {
    this.key = key;
    this.phase = seed * 1.9 + 0.4;
    for (let w = 0; w < WIRES; w++) {
      const pos = new Float32Array(SAMPLES * RING * 3), nrm = new Float32Array(SAMPLES * RING * 3);
      const idx: number[] = [];
      for (let i = 0; i < SAMPLES - 1; i++) for (let r = 0; r < RING; r++) {
        const a = i * RING + r, b = i * RING + ((r + 1) % RING), c = (i + 1) * RING + r, d = (i + 1) * RING + ((r + 1) % RING);
        idx.push(a, c, b, b, c, d);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
      geo.setIndex(idx);
      const light = w === Math.floor(WIRES / 2) - 1 || w === Math.floor(WIRES / 2);
      const mesh = new THREE.Mesh(geo, light ? mats.strip : mats.copper);
      mesh.frustumCulled = false;
      this.group.add(mesh);
      this.wires.push({ geo, pos, nrm, mesh });
    }
    this.clamp = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mats.dark);
    this.conn = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mats.gun);
    this.pins = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mats.gold, 7);
    this.group.add(this.clamp, this.conn, this.pins);
  }

  reset(root: THREE.Vector3, dir: THREE.Vector3) {
    this.p = []; this.q = [];
    for (let i = 0; i < N; i++) { const x = root.clone().addScaledVector(dir, SEG * i); this.p.push(x); this.q.push(x.clone()); }
  }

  /** Smooth reach (wind-up → accelerate → glide → decelerate, tiny settle) to a world point. */
  reachTo(target: THREE.Vector3, now: number, opts: { press?: boolean; onContact?: () => void; dur?: number } = {}) {
    const from = (this.goal ?? this.p[N - 1]).clone();
    const d = from.distanceTo(target);
    const dur = opts.dur ?? Math.max(520, Math.min(1300, 480 + d * 420));
    const dir = target.clone().sub(from).normalize();
    const back = Math.min(0.06, d * 0.05);
    const ANT = 120;
    this.drive = {
      t0: now, dur: ANT + dur + (opts.press ? 420 : 0),
      path: (t) => {
        if (t < ANT) return from.clone().addScaledVector(dir, -back * (0.5 - 0.5 * Math.cos((t / ANT) * Math.PI)));
        const u = Math.min(1, (t - ANT) / dur);
        const s0 = from.clone().addScaledVector(dir, -back);
        return s0.lerp(target, smoother(u));
      },
      press: opts.press ? { atMs: ANT + dur + 130, done: false, onContact: opts.onContact } : undefined,
    };
    this.holding = true;
  }

  follow(path: (tMs: number) => THREE.Vector3, now: number, dur: number) {
    this.drive = { path, t0: now, dur };
    this.holding = true;
  }

  release() { this.drive = null; this.goal = null; this.holding = false; }

  step(now: number, root: THREE.Vector3, dir: THREE.Vector3, idle: THREE.Vector3) {
    if (!this.p.length) this.reset(root, dir);
    // goal from the drive
    let tapZ = 0;
    if (this.drive) {
      const t = now - this.drive.t0;
      this.goal = this.drive.path(Math.min(t, this.drive.dur));
      const pr = this.drive.press;
      if (pr) {
        const pt = t - (pr.atMs - 130);
        this.pressDepth = pt > 0 && pt < 300 ? Math.sin((pt / 300) * Math.PI) : 0;
        tapZ = -0.12 * this.pressDepth;          // the tip pushes "into" the screen
        if (!pr.done && t >= pr.atMs) { pr.done = true; try { pr.onContact?.(); } catch { /* app's business */ } }
      }
    }
    const tsec = now / 1000;
    const damp = this.goal ? 0.84 : 0.9;
    for (let i = 1; i < N; i++) {
      const c = this.p[i];
      const vel = c.clone().sub(this.q[i]).multiplyScalar(damp);
      this.q[i].copy(c);
      const w = i / (N - 1);
      c.add(vel);
      c.x += Math.sin(tsec * 0.7 + this.phase + i * 0.33) * 0.0009 * w;
      c.y += Math.cos(tsec * 0.55 + this.phase * 1.3 + i * 0.29) * 0.0009 * w;
      c.z += Math.sin(tsec * 0.43 + this.phase * 0.7 + i * 0.21) * 0.0006 * w;
    }
    const tgt = this.goal ? this.goal.clone().setZ(this.goal.z + tapZ) : idle;
    const tip = this.p[N - 1];
    tip.lerp(tgt, this.goal ? 0.2 : 0.05);
    const natural = SEG * (N - 1);
    const want = this.goal ? Math.min(MAX_STRETCH, Math.max(1, root.distanceTo(tgt) / natural)) : 1;
    this.stretch += (want - this.stretch) * 0.06;
    this.twist += (this.twistGoal - this.twist) * 0.05;
    const L = SEG * this.stretch;
    const d = v(), want1 = v();
    for (let it = 0; it < ITER; it++) {
      this.p[0].copy(root);                                   // root sits INSIDE the collar
      want1.copy(root).addScaledVector(dir, L);
      this.p[1].lerp(want1, 0.6);
      const want2 = root.clone().addScaledVector(dir, L * 2);
      this.p[2].lerp(want2, 0.25);                            // the cable leaves the socket straight-ish
      for (let i = 0; i < N - 1; i++) {
        d.copy(this.p[i + 1]).sub(this.p[i]);
        const dl = d.length() || 1e-6;
        const diff = (dl - L) / dl;
        if (i === 0) this.p[1].addScaledVector(d, -diff);
        else { this.p[i].addScaledVector(d, diff * 0.5); this.p[i + 1].addScaledVector(d, -diff * 0.5); }
      }
      for (let i = 0; i < N - 2; i++) {                         // bending stiffness
        d.copy(this.p[i + 2]).sub(this.p[i]);
        const dl = d.length() || 1e-6, w2 = 2 * L * 0.93;
        if (dl < w2) {
          const diff = ((dl - w2) / dl) * 0.4;
          if (i > 0) this.p[i].addScaledVector(d, diff * 0.5);
          this.p[i + 2].addScaledVector(d, -diff * (i > 0 ? 0.5 : 1));
        }
      }
      if (this.goal) this.p[N - 1].lerp(tgt, 0.2);
    }
  }

  /** Rebuild the tube meshes from the chain (Catmull-Rom resample + parallel-transported, camera-facing frame). */
  draw(camPos: THREE.Vector3) {
    const curve = new THREE.CatmullRomCurve3(this.p, false, "centripetal");
    const pts = curve.getSpacedPoints(SAMPLES - 1);
    const tans = pts.map((_, i) => pts[Math.min(SAMPLES - 1, i + 1)].clone().sub(pts[Math.max(0, i - 1)]).normalize());
    // side vector: perpendicular to the tangent, facing the camera; transported so it never flips
    const sides: THREE.Vector3[] = [];
    for (let i = 0; i < SAMPLES; i++) {
      const view = camPos.clone().sub(pts[i]).normalize();
      let s = tans[i].clone().cross(view);
      if (s.lengthSq() < 1e-6) s = sides[i - 1]?.clone() ?? new THREE.Vector3(0, 0, 1);
      s.normalize();
      if (i > 0 && s.dot(sides[i - 1]) < 0) s.negate();          // continuity = no self-twist
      sides.push(s);
    }
    const ups = sides.map((s, i) => tans[i].clone().cross(s).normalize());
    const width = WIRE_R * 2.15 * WIRES;
    for (let w = 0; w < WIRES; w++) {
      const off0 = (w - (WIRES - 1) / 2) * WIRE_R * 2.15;
      const { pos, nrm, geo } = this.wires[w];
      for (let i = 0; i < SAMPLES; i++) {
        const u = i / (SAMPLES - 1);
        const fan = 1 + 0.25 * Math.sin(Math.PI * Math.min(1, u / 0.9));     // loosen a little mid-span
        const tw = this.twist * Math.PI * u;                                    // only when asked to twist
        const side = sides[i].clone().multiplyScalar(Math.cos(tw)).addScaledVector(ups[i], Math.sin(tw));
        const c = pts[i].clone().addScaledVector(side, off0 * fan);
        const sideN = sides[i], upN = ups[i];
        for (let r = 0; r < RING; r++) {
          const a = (r / RING) * Math.PI * 2;
          const n = sideN.clone().multiplyScalar(Math.cos(a)).addScaledVector(upN, Math.sin(a));
          const k = (i * RING + r) * 3;
          pos[k] = c.x + n.x * WIRE_R; pos[k + 1] = c.y + n.y * WIRE_R; pos[k + 2] = c.z + n.z * WIRE_R;
          nrm[k] = n.x; nrm[k + 1] = n.y; nrm[k + 2] = n.z;
        }
      }
      (geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (geo.attributes.normal as THREE.BufferAttribute).needsUpdate = true;
      geo.computeBoundingSphere();
    }
    // clamp sleeve near the root, connector + pins at the tip
    const place = (m: THREE.Object3D, at: THREE.Vector3, t: THREE.Vector3, s: THREE.Vector3, size: [number, number, number]) => {
      const up = t.clone().cross(s).normalize();
      this.tmpM.makeBasis(s, t, up).setPosition(at);
      m.matrixAutoUpdate = false;
      m.matrix.copy(this.tmpM).multiply(new THREE.Matrix4().makeScale(...size));
    };
    const ci = Math.round(SAMPLES * 0.12);
    place(this.clamp, pts[ci], tans[ci], sides[ci], [width * 1.12, 0.07, 0.07]);
    const te = tans[SAMPLES - 1], se = sides[SAMPLES - 1], tip = pts[SAMPLES - 1];
    const k = 1 - 0.15 * this.pressDepth;
    place(this.conn, tip.clone().addScaledVector(te, 0.07 * k), te, se, [width * 1.2 * k, 0.16 * k, 0.09]);
    for (let i = 0; i < 7; i++) {
      const at = tip.clone().addScaledVector(te, 0.165 * k).addScaledVector(se, (i - 3) * width * 0.15);
      const up = te.clone().cross(se).normalize();
      this.tmpM.makeBasis(se, te, up).setPosition(at).multiply(new THREE.Matrix4().makeScale(0.018, 0.045, 0.02));
      this.pins.setMatrixAt(i, this.tmpM);
    }
    this.pins.instanceMatrix.needsUpdate = true;
  }

  tipWorld() { return this.p[N - 1]; }
}

/** All four arms of one eye: physics + meshes + the page↔world mapping the guide uses. */
export class EyeArms3D {
  arms: Record<ArmKey3, Arm3D>;
  group = new THREE.Group();
  mats: { copper: THREE.MeshStandardMaterial; strip: THREE.MeshStandardMaterial; dark: THREE.MeshStandardMaterial; gun: THREE.MeshStandardMaterial; gold: THREE.MeshStandardMaterial };
  /** page px → world (z=0 plane through the ball centre), set every frame by CyborgEye */
  toWorld: (x: number, y: number) => THREE.Vector3 = (x, y) => new THREE.Vector3(x, y, 0);

  constructor() {
    this.mats = {
      copper: new THREE.MeshStandardMaterial({ color: 0xd88a55, metalness: 1, roughness: 0.32 }),
      strip: new THREE.MeshStandardMaterial({ color: 0x0a0a0a, emissive: new THREE.Color("#6aa3ff"), emissiveIntensity: 2.2, roughness: 0.4 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x15171b, metalness: 0.6, roughness: 0.45 }),
      gun: new THREE.MeshStandardMaterial({ color: 0x2b2e35, metalness: 0.9, roughness: 0.3 }),
      gold: new THREE.MeshStandardMaterial({ color: 0xe0b04a, metalness: 1, roughness: 0.25 }),
    };
    this.arms = Object.fromEntries(KEYS.map((k, i) => [k, new Arm3D(k, i, this.mats)])) as Record<ArmKey3, Arm3D>;
    KEYS.forEach((k) => this.group.add(this.arms[k].group));
  }

  /** ball: the rotating body group (ports live in its local space) */
  step(now: number, ball: THREE.Object3D, camPos: THREE.Vector3, light: THREE.Color, glow: number) {
    this.mats.strip.emissive.copy(light);
    this.mats.strip.emissiveIntensity = 2.2 * glow;
    const t = now / 1000;
    for (const k of KEYS) {
      const a = this.arms[k];
      const axis = AXIS[k];
      const root = ball.localToWorld(axis.clone().multiplyScalar(0.95));        // inside the collar
      const outer = ball.localToWorld(axis.clone().multiplyScalar(1.3));
      const dir = outer.clone().sub(root).normalize();
      // idle: float out from the port in a slow zero-g drift, a little toward/away from the camera too
      const ang = 0.4 * Math.sin(t * 0.19 + a.phase) + 0.15 * Math.sin(t * 0.47 + a.phase * 2.3);
      const perp = new THREE.Vector3(-dir.y, dir.x, 0);
      const idle = root.clone().addScaledVector(dir, 1.55 * Math.cos(ang)).addScaledVector(perp, 1.55 * Math.sin(ang))
        .add(new THREE.Vector3(0, 0, 0.25 * Math.sin(t * 0.33 + a.phase)));
      a.step(now, root, dir, idle);
      a.draw(camPos);
    }
  }

  dispose() {
    for (const k of KEYS) {
      const a = this.arms[k];
      a.wires.forEach((w) => w.geo.dispose());
      (a.clamp.geometry as THREE.BufferGeometry).dispose();
      (a.conn.geometry as THREE.BufferGeometry).dispose();
      (a.pins.geometry as THREE.BufferGeometry).dispose();
    }
    Object.values(this.mats).forEach((m) => m.dispose());
  }
}

/** What the guide talks to (page coordinates in, world coordinates inside). */
export type ArmsController = {
  reachTo: (key: ArmKey3, pageX: number, pageY: number, opts?: { press?: boolean; onContact?: () => void; dur?: number }) => void;
  follow: (key: ArmKey3, pagePath: (tMs: number) => { x: number; y: number }, dur: number) => void;
  release: (key: ArmKey3) => void;
  setTwist: (key: ArmKey3, tw: number) => void;
  holding: (key: ArmKey3) => boolean;
};
export const eyeArms = new Map<string, ArmsController>();

export function controllerFor(arms: EyeArms3D): ArmsController {
  return {
    reachTo: (key, x, y, opts) => arms.arms[key].reachTo(arms.toWorld(x, y), performance.now(), opts),
    follow: (key, path, dur) => arms.arms[key].follow((t) => { const p = path(t); return arms.toWorld(p.x, p.y); }, performance.now(), dur),
    release: (key) => arms.arms[key].release(),
    setTwist: (key, tw) => { arms.arms[key].twistGoal = tw; },
    holding: (key) => arms.arms[key].holding,
  };
}
