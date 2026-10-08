// GuideOverlay — my body above the whole app (2026-10-08).
//
// * Always: four cable-arms float around my eye on the home screen (when the panel is closed).
// * Guide mode (a script from the `app_guide` tool): my eye lifts off the home screen onto this layer,
//   which sits ABOVE the panel drawer, so I can travel anywhere, point at any control, press it (the arm
//   taps and the app clicks AT the contact frame), switch tabs, gesture, and speak each line while the
//   arm lands on the thing I'm naming. Captions show even when my voice is off.
//
// The layer is pointer-events:none, so it never blocks clicks.
import { useCallback, useEffect, useRef, useState } from "react";
import OrbCanvas from "../components/OrbCanvas";
import { eyePorts } from "../components/CyborgEye";
import { eyeArms } from "../components/eyeArms3d";
import type { IrisBodyProps } from "../components/IrisBody";
import { deriveBlendColors, getCfg } from "../components/orbShared";
import { getJson, postJson } from "../api";
import { invoke } from "@tauri-apps/api/core";
import { Arm, ARM_KEYS, ArmKey, bestArm, drawArm, easeInOutCubic, idleTip, port, Vec } from "./arms";
import { findEl, scanUi } from "./uiMap";

type Step = {
  brain?: string;               // "spin" | "overview" | "focus:<node id>" (e.g. focus:iris, focus:zeke)
  close?: "console" | "camera" | "panel" | "ssh" | "proxmox";
  via?: string;                 // with press: at contact, invoke server_open(<via>) instead of clicking (closable demo windows)
  tab?: string; move?: string | { x: number; y: number }; point?: string; press?: string; say?: string;
  gesture?: "wave" | "nod" | "shake" | "curious"; twist?: number; hold_ms?: number; emotion?: string;
};
type Script = { ok?: boolean; seq: number; steps: Step[]; stop?: boolean; silent?: boolean; issued_ts?: number };

export type GuideOverlayProps = {
  eye: IrisBodyProps;            // the same props the home-screen eye gets (emotion, colour, state…)
  activeTab: string;
  operatorOpen: boolean;
  onFloatingChange: (floating: boolean) => void;
};

// test pages open the app with ?guide_client=<name> and listen ONLY to that channel; the real app listens to
// the default channel. So my test tours never play on Zeke's live window.
const GUIDE_CLIENT = (() => { try { return new URLSearchParams(window.location.search).get("guide_client") || ""; } catch { return ""; } })();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const IRIS_FRAC = 0.546;          // iris radius / half canvas (IrisBody: 0.72 world of a 1.319 half-width)

export default function GuideOverlay({ eye, activeTab, operatorOpen, onFloatingChange }: GuideOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const arms = useRef<Record<ArmKey, Arm>>(Object.fromEntries(ARM_KEYS.map((k, i) => [k, new Arm(k, i)])) as Record<ArmKey, Arm>);
  // One handle per arm: the 3D cyborg body's real cables when it's up (CyborgEye registers them in `eyeArms`),
  // the 2D overlay arms otherwise (flat iris / classic body).
  type ArmHandle = { holding: boolean; twistGoal: number; reachTo: (p: Vec, now: number, o?: { press?: boolean; onContact?: () => void; dur?: number }) => void;
    follow: (path: (t: number) => Vec, now: number, dur: number) => void; release: () => void };
  const armH = (k: ArmKey): ArmHandle => {
    const c = eyeArms.get(live.current.floating ? "guide" : "home");
    const a2 = arms.current[k];
    if (!c) return a2 as unknown as ArmHandle;
    return {
      get holding() { return c.holding(k); },
      get twistGoal() { return 0; },
      set twistGoal(v: number) { c.setTwist(k, v); },
      reachTo: (p, _now, o) => c.reachTo(k, p.x, p.y, o),
      follow: (path, _now, dur) => c.follow(k, path, dur),
      release: () => c.release(k),
    };
  };
  const [floating, setFloating] = useState(false);
  const [caption, setCaption] = useState("");
  const [gaze, setGaze] = useState<{ x: number; y: number } | undefined>(undefined);
  const [eyePos, setEyePos] = useState<Vec>({ x: 0, y: 0 });
  const live = useRef({ floating: false, pos: { x: 0, y: 0 } as Vec, offset: { x: 0, y: 0 } as Vec, operatorOpen, activeTab,
    size: 180, stop: false, running: false, seq: -1, clicked: 0, warnedAt: 0 });
  live.current.operatorOpen = operatorOpen;
  live.current.activeTab = activeTab;

  const colors = deriveBlendColors(String(eye.emotionColor || ""), getCfg(String(eye.emotion || "calmness")));
  const colorRef = useRef(colors);
  colorRef.current = colors;

  const homeRect = () => {
    const el = document.querySelector(".eye-shell") as HTMLElement | null;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { c: { x: r.left + r.width / 2, y: r.top + r.height / 2 }, size: r.width };
  };

  // ---------------------------------------------------------------- render loop (arms)
  useEffect(() => {
    (window as unknown as { __irisGuide?: unknown }).__irisGuide = { arms: arms.current, live: live.current };
    let raf = 0, last = performance.now();
    let slowAvg = 16, glow = true;                  // drop the glow blur if this machine can't keep up
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const cv = canvasRef.current;
      if (!cv) return;
      if (document.hidden) return;          // the 24%-GPU-while-hidden scar
      const dpr = window.devicePixelRatio || 1;
      const W = window.innerWidth, H = window.innerHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      }
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const L = live.current;
      let center: Vec, size: number;
      if (L.floating) {
        center = { x: L.pos.x + L.offset.x, y: L.pos.y + L.offset.y };
        size = L.size;
      } else {
        const h = homeRect();
        if (!h) return;                      // at home the arms sit on the body's layer: the panel / camera view cover them
        center = h.c; size = h.size;
      }
      if (eyeArms.has(L.floating ? "guide" : "home")) return;   // the 3D body draws its own real cables
      // the 3D cyborg eye publishes its real collar positions (they move as the ball turns)
      const ep = eyePorts.get(L.floating ? "guide" : "home");
      const live3d = ep && performance.now() - ep.ts < 400 ? ep : null;
      if (live3d) center = live3d.center;
      const R = live3d ? live3d.radius * 0.9 : (size / 2) * IRIS_FRAC;
      const scale = Math.max(0.55, Math.min(1.6, R / 60));
      const dt = Math.max(0, Math.min(50, now - last)); last = now;
      slowAvg = slowAvg * 0.97 + dt * 0.03;
      if (glow && slowAvg > 30) glow = false;
      const t = now / 1000;
      const c = colorRef.current;
      for (const k of ARM_KEYS) {
        const a = arms.current[k];
        const pp = live3d?.ports[k];
        const { root, dir } = pp ? { root: { x: pp.x, y: pp.y }, dir: { x: pp.nx, y: pp.ny } } : port(center, R * 0.98, k, 0);
        const seg = (R * 1.7) / 15;                     // arm length ∝ eye size (≈1.7 iris radii)
        if (!a.p.length || Math.hypot(a.p[0].x - root.x, a.p[0].y - root.y) > 400) a.reset(root, dir, seg);
        a.seg = seg;
        a.step(now, dt, root, dir, idleTip(center, R, k, 0, t, a.phase, R * 1.85), scale);
      }
      // draw the far arms first (up/left), so crossings read with some depth
      for (const k of ["up", "left", "down", "right"] as ArmKey[]) {
        drawArm(ctx, arms.current[k], { light: c.lightColor, copper: "#d9894f", copperDark: "#5a2f17", scale, dpr, glow,
          width: live3d ? live3d.radius * 0.34 : undefined });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ---------------------------------------------------------------- eye travel (Disney timing)
  const travel = useCallback(async (to: Vec) => {
    // Glide: a small, soft wind-up, then accelerate → cruise → decelerate (no 0→100→0 jumps)
    const L = live.current;
    const from = { ...L.pos };
    const d = Math.hypot(to.x - from.x, to.y - from.y);
    if (d < 2) return;
    const dir = { x: (to.x - from.x) / d, y: (to.y - from.y) / d };
    const ANT = 180, dur = Math.min(1600, 650 + d * 0.75);
    const back = Math.min(8, d * 0.03);
    const t0 = performance.now();
    setGaze({ x: dir.x * 0.85, y: -dir.y * 0.85 });            // the eye looks where it's going, first
    const smoother = (u: number) => u * u * u * (u * (u * 6 - 15) + 10);   // smootherstep: zero accel at both ends
    for (;;) {
      const t = performance.now() - t0;
      if (t < ANT) {
        const k = 0.5 - 0.5 * Math.cos((t / ANT) * Math.PI);
        L.pos = { x: from.x - dir.x * back * k, y: from.y - dir.y * back * k };
      } else if (t < ANT + dur) {
        const u = smoother((t - ANT) / dur);
        const s0 = { x: from.x - dir.x * back, y: from.y - dir.y * back };
        // travel on a gentle ARC, not a ruler line (curved = friendly): bulge sideways, peaking mid-flight
        const bulge = Math.min(60, d * 0.09) * Math.sin(Math.PI * u) * (dir.x >= 0 ? -1 : 1);
        L.pos = { x: s0.x + (to.x - s0.x) * u - dir.y * bulge, y: s0.y + (to.y - s0.y) * u + dir.x * bulge };
      } else break;
      setEyePos({ ...L.pos });
      await sleep(16);
    }
    L.pos = { ...to };
    setEyePos({ ...to });
  }, []);

  const liftOff = useCallback(async () => {
    const L = live.current;
    if (L.floating) return;
    const h = homeRect();
    L.pos = h ? h.c : { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    L.size = Math.max(170, Math.min(280, (h?.size || 260) * 0.6));
    setEyePos({ ...L.pos });
    L.floating = true;
    setFloating(true);
    onFloatingChange(true);
    await sleep(60);
  }, [onFloatingChange]);

  const land = useCallback(async () => {
    const L = live.current;
    if (!L.floating) return;
    const h = homeRect();
    if (h && !L.operatorOpen) await travel(h.c);
    setGaze(undefined);
    L.floating = false;
    setFloating(false);
    onFloatingChange(false);
  }, [onFloatingChange, travel]);

  // where the eye should float to sit beside a target without covering it
  // Where my body should float to point at a rect: my favourite (right) hand toward it, my body NEVER over it.
  const bodyR = () => live.current.size * 0.47;
  const overlaps = (p: Vec, r: DOMRect, pad = 14) => {
    const nx = Math.max(r.left - pad, Math.min(p.x, r.right + pad)), ny = Math.max(r.top - pad, Math.min(p.y, r.bottom + pad));
    return Math.hypot(p.x - nx, p.y - ny) < bodyR();
  };
  const besideTarget = (r: DOMRect): Vec => {
    const L = live.current;
    const W = window.innerWidth, H = window.innerHeight;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const R = (L.size / 2) * IRIS_FRAC;
    const gap = bodyR() + R * 1.1 + 18;
    const m = L.size / 2 + 8;
    const clampP = (p: Vec) => ({ x: Math.max(m, Math.min(W - m, p.x)), y: Math.max(m, Math.min(H - m, p.y)) });
    // favourite-hand poses first (body to the LEFT so the right arm reaches right), then the rest
    const cands: Vec[] = [
      { x: r.left - gap, y: cy }, { x: r.left - gap * 0.75, y: r.bottom + gap * 0.7 },
      { x: r.left - gap * 0.75, y: r.top - gap * 0.7 }, { x: cx, y: r.bottom + gap }, { x: cx, y: r.top - gap },
      { x: r.right + gap, y: cy },
    ].map(clampP);
    for (const c of cands) if (!overlaps(c, r)) return c;
    return cands[0];
  };
  /** The point on the rect's edge facing my body, just outside it (pointing) or just inside it (pressing). */
  const edgePoint = (r: DOMRect, from: Vec, inset: number): Vec => {
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const dx = from.x - cx, dy = from.y - cy;
    const sx = dx === 0 ? Infinity : (r.width / 2) / Math.abs(dx), sy = dy === 0 ? Infinity : (r.height / 2) / Math.abs(dy);
    const k = Math.min(sx, sy);
    const ex = cx + dx * k, ey = cy + dy * k;
    const dl = Math.hypot(dx, dy) || 1;
    return { x: ex - (dx / dl) * inset, y: ey - (dy / dl) * inset };   // inset<0 = outside the edge
  };

  const targetOf = async (id: string): Promise<{ el: HTMLElement; c: Vec; r: DOMRect } | null> => {
    scanUi(live.current.activeTab);
    const el = findEl(id);
    if (!el) return null;
    let r = el.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      await sleep(450);
      r = el.getBoundingClientRect();
    }
    return { el, r, c: { x: r.left + r.width / 2, y: r.top + r.height / 2 } };
  };

  // Secondary motion on the THING, so a watcher sees what I mean: a quick squash where I tap, a soft glow where I point.
  const tapFeedback = (el: HTMLElement) => {
    try { el.animate([{ transform: "scale(1)" }, { transform: "scale(0.94)", offset: 0.35 }, { transform: "scale(1.02)", offset: 0.75 }, { transform: "scale(1)" }],
      { duration: 280, easing: "cubic-bezier(0.2, 0, 0, 1)" }); } catch { /* old webview: skip */ }
  };
  const pointGlow = (el: HTMLElement) => {
    const c = colorRef.current.lightColor || "#6aa3ff";
    try { el.animate([{ boxShadow: "0 0 0 0 transparent" }, { boxShadow: `0 0 0 3px ${c}, 0 0 18px 2px ${c}`, offset: 0.3 },
      { boxShadow: `0 0 0 2px ${c}, 0 0 12px 1px ${c}`, offset: 0.8 }, { boxShadow: "0 0 0 0 transparent" }],
      { duration: 1800, easing: "ease-in-out" }); } catch { /* skip */ }
  };

  const reachFor = async (id: string, press: boolean, twist?: number, via?: string): Promise<ArmKey | null> => {
    const tg = await targetOf(id);
    if (!tg) { report("missing", `no element ${id}`); return null; }
    const L = live.current;
    const R = (L.size / 2) * IRIS_FRAC;
    const maxReach = R + R * 1.7 * 1.3;
    const here = Math.hypot(tg.c.x - L.pos.x, tg.c.y - L.pos.y);
    const facesFav = tg.c.x > L.pos.x + 10;               // target on my favourite (right) side
    const big = tg.r.width * tg.r.height > 0.35 * window.innerWidth * window.innerHeight;   // e.g. the enlarged camera
    if (!big && (overlaps(L.pos, tg.r) || here > maxReach * 0.9 || !facesFav)) await travel(besideTarget(tg.r));
    const dx = tg.c.x - L.pos.x, dy = tg.c.y - L.pos.y, dl = Math.hypot(dx, dy) || 1;
    setGaze({ x: (dx / dl) * 0.8, y: (-dy / dl) * 0.8 });       // eye leads: look first…
    await sleep(140);                                             // …then the arm goes
    const busy = new Set(ARM_KEYS.filter((k) => armH(k).holding));
    const k = bestArm(L.pos, tg.c, 0, busy);
    const a = armH(k);
    if (twist !== undefined) a.twistGoal = twist;
    let contacted = false;
    const aim = big
      ? { x: L.pos.x + (tg.c.x - L.pos.x) * 0.25 + R * 1.4, y: L.pos.y + (tg.c.y - L.pos.y) * 0.25 }   // tap the big thing near me
      : edgePoint(tg.r, L.pos, press ? Math.min(10, Math.min(tg.r.width, tg.r.height) * 0.3) : -6);
    a.reachTo(aim, performance.now(), {
      press, onContact: () => {
        contacted = true;
        const tag = tg.el.tagName.toLowerCase();
        if (via) { void invoke("server_open", { kind: via }).catch((e) => report("note", `open ${via}: ${String(e).slice(0, 120)}`)); return; }
        if (!big) tapFeedback(tg.el);
        if (tag === "input" || tag === "textarea" || tag === "select") tg.el.focus();
        else tg.el.click();
      },
    });
    if (!press && !big) pointGlow(tg.el);
    if (press) {
      const t0 = performance.now();
      while (!contacted && performance.now() - t0 < 2500) await sleep(30);
      await sleep(220);
    } else {
      await sleep(600);
    }
    return k;
  };

  /** Returns false if someone clicked while I was saying it (I cut the line short, so the caller can warn + repeat). */
  const speak = async (text: string, emotion?: string, silent?: boolean): Promise<boolean> => {
    const L0 = live.current;
    const clicks0 = L0.clicked;
    const interrupted = () => L0.clicked !== clicks0;
    setCaption(text);
    report("say", text.slice(0, 200));
    let spoken = false;
    if (!silent) try {
      const r = await postJson<{ ok: boolean; spoken?: boolean }>("/api/v1/app/guide/say", { text, emotion });
      spoken = Boolean(r?.spoken);
    } catch { spoken = false; }
    const words = text.split(/\s+/).length;
    const est = 700 + words * 360;
    const t0 = performance.now();
    if (spoken) {
      // The mouth speaks sentence by sentence and reports "not speaking" in the gaps between them, so a short
      // quiet is NOT the end of the line (10-08: I moved on to the chat box mid-sentence about the console).
      // Done = I've been talking at least ~words x 0.3 s since I started AND it's been quiet for over a second.
      const minTalk = words * 300;
      let started = false, startedAt = 0, quietSince = 0;
      while (performance.now() - t0 < est * 2.5 + 4000) {
        await sleep(100);
        let sp = false;
        try { sp = Boolean((await getJson<{ speaking?: boolean }>("/api/v1/tts/state"))?.speaking); } catch { /* keep going */ }
        const now = performance.now();
        if (interrupted()) { void postJson("/api/v1/app/guide/hush", {}).catch(() => undefined); break; }
        if (sp) { if (!started) { started = true; startedAt = now; } quietSince = 0; }
        else if (started) {
          if (!quietSince) quietSince = now;
          if (now - quietSince > 1100 && now - startedAt > minTalk) break;
        }
        else if (now - t0 > 5000) break;                          // never started: fall back to reading time
      }
      if (!started && !interrupted()) await sleep(Math.max(0, est - (performance.now() - t0)));
    } else {
      const tEnd = t0 + est;                                     // voice off: give time to read the caption
      while (performance.now() < tEnd && !interrupted()) await sleep(80);
    }
    return !interrupted();
  };

  // Someone clicked during the tour (Zeke 10-08): stop for a sec, ask them not to, then carry on.
  const NO_CLICK_LINES = [
    "Hey, please don't click while I'm showing you around. I've got it.",
    "Oops, no clicking please. Let me do the driving.",
    "Hands off for a moment, please. I'll show you everything.",
  ];
  const warnNoClick = async (silent?: boolean) => {
    const L = live.current;
    L.warnedAt = performance.now();
    const n = Math.floor(Math.random() * NO_CLICK_LINES.length);
    const clicks0 = L.clicked;
    await gesture("shake");
    await speak(NO_CLICK_LINES[n], "calm", silent);
    if (L.clicked !== clicks0) L.clicked = clicks0;   // clicks DURING the warning don't earn a second warning
    await sleep(250);
  };

  const gesture = async (g: Step["gesture"]) => {
    const L = live.current;
    const base = { ...L.pos };
    if (g === "nod" || g === "shake") {
      const t0 = performance.now(), dur = 1100;
      while (performance.now() - t0 < dur) {
        const u = (performance.now() - t0) / dur;
        const amp = 9 * Math.sin(Math.PI * u);
        const v = Math.sin(u * Math.PI * 4) * amp;
        L.offset = g === "nod" ? { x: 0, y: v } : { x: v, y: 0 };
        setGaze(g === "nod" ? { x: 0, y: -v / 14 } : { x: v / 14, y: 0 });
        await sleep(16);
      }
      L.offset = { x: 0, y: 0 };
      setGaze(undefined);
    } else if (g === "wave") {
      // a smooth wave: the hand rises, then swings on an ARC (pendulum about a pivot), then settles back
      const a = armH("up").holding ? armH("right") : armH("up");
      const R = (L.size / 2) * IRIS_FRAC;
      const pivot = { x: base.x + R * 1.1, y: base.y - R * 1.2 };
      const rad = R * 1.35, dur = 2600;
      a.follow((t) => {
        const u = t / dur;
        const env = Math.sin(Math.PI * Math.min(1, u * 1.15)) ** 0.7;         // ease in and out of the wave
        const ang = -Math.PI / 2 + 0.55 * env * Math.sin(2 * Math.PI * 1.7 * (t / 1000));
        return { x: pivot.x + Math.cos(ang) * rad, y: pivot.y + Math.sin(ang) * rad * 0.85 };
      }, performance.now(), dur);
      await sleep(dur);
      a.release();
    } else if (g === "curious") {
      const a = armH("up");
      setGaze({ x: 0.35, y: 0.4 });
      a.twistGoal = 0.6;
      a.reachTo({ x: base.x + 30, y: base.y - (L.size / 2) * IRIS_FRAC * 2.6 }, performance.now(), { dur: 600 });
      await sleep(1100);
      a.twistGoal = 0;
      a.release();
      setGaze(undefined);
    }
  };

  // Brain tab: spin the 3D memory graph, fly to a node (me / Zeke) and point at it.
  const brainMove = async (cmd: string): Promise<ArmKey | null> => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fg = (window as any).__irisBrainGraph;
    if (!fg) { report("missing", "brain graph not mounted (open the Brain tab first)"); return null; }
    const cont = document.querySelector(".brain-canvas-wrap") as HTMLElement | null;
    if (cmd === "overview") { fg.zoomToFit(1400, 40); await sleep(1500); return null; }
    if (cmd === "spin") {
      const p0 = fg.cameraPosition();
      const d = Math.hypot(p0.x, p0.z) || 600, y = p0.y;
      const a0 = Math.atan2(p0.x, p0.z), t0 = performance.now(), dur = 4200;
      for (;;) {
        const u = Math.min(1, (performance.now() - t0) / dur);
        const k = u * u * (3 - 2 * u);                          // ease in/out of the spin
        const a = a0 + k * Math.PI * 2;
        fg.cameraPosition({ x: d * Math.sin(a), y, z: d * Math.cos(a) }, { x: 0, y: 0, z: 0 });   // circle ME (pinned at the origin)
        if (u >= 1) break;
        await sleep(16);
      }
      return null;
    }
    if (cmd.startsWith("focus:")) {
      const id = cmd.slice(6);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const node = (fg.graphData().nodes as any[]).find((n) => String(n.id) === id);
      if (!node) { report("missing", `no brain node ${id}`); return null; }
      // Come in toward the node from where the camera already is (10-08: "me" is pinned at the origin, so the old
      // "push out along the node's own direction" put the camera INSIDE my node and the Brain tab went blank).
      const nx = Number(node.x || 0), ny = Number(node.y || 0), nz = Number(node.z || 0);
      const cam = fg.cameraPosition();
      let vx = cam.x - nx, vy = cam.y - ny, vz = cam.z - nz;
      let vl = Math.hypot(vx, vy, vz);
      if (vl < 1) { vx = 0; vy = 0; vz = 1; vl = 1; }
      const dist = 160;
      fg.cameraPosition({ x: nx + (vx / vl) * dist, y: ny + (vy / vl) * dist, z: nz + (vz / vl) * dist }, { x: nx, y: ny, z: nz }, 1800);
      await sleep(1900);
      if (!cont) return null;
      const sc = fg.graph2ScreenCoords(nx, ny, nz);
      const r = cont.getBoundingClientRect();
      const pt = { x: r.left + sc.x, y: r.top + sc.y };
      const box = new DOMRect(pt.x - 14, pt.y - 14, 28, 28);
      const L = live.current;
      if (overlaps(L.pos, box) || Math.hypot(pt.x - L.pos.x, pt.y - L.pos.y) > (L.size / 2) * IRIS_FRAC * 3.1 || pt.x < L.pos.x)
        await travel(besideTarget(box));
      const dx = pt.x - L.pos.x, dy = pt.y - L.pos.y, dl = Math.hypot(dx, dy) || 1;
      setGaze({ x: (dx / dl) * 0.8, y: (-dy / dl) * 0.8 });
      await sleep(140);
      const busy = new Set(ARM_KEYS.filter((k) => armH(k).holding));
      const k = bestArm(L.pos, pt, 0, busy);
      armH(k).reachTo(edgePoint(box, L.pos, -4), performance.now());
      await sleep(700);
      return k;
    }
    return null;
  };

  const seqRef = useRef(0);
  const report = (status: string, note = "", index = -1) => {
    void postJson("/api/v1/app/guide/progress", { seq: seqRef.current, index, status, note }).catch(() => undefined);
  };

  const runScript = useCallback(async (s: Script) => {
    const L = live.current;
    L.stop = false;
    L.running = true;
    seqRef.current = s.seq;
    report("started", `${s.steps.length} steps`);
    let openedPanel = false;
    try {
      await liftOff();
      for (let i = 0; i < s.steps.length; i++) {
        if (L.stop) { report("stopped", "", i); break; }
        const st = s.steps[i];
        const clicksAtStep = L.clicked;
        report("step", st.say?.slice(0, 80) || st.point || st.press || st.tab || "", i);
        if (st.tab) {
          if (!L.operatorOpen) { await reachFor("panel", true); openedPanel = true; await sleep(500); }
          await reachFor(`tab:${st.tab}`, true);
          await sleep(350);
          for (const k of ARM_KEYS) armH(k).release();
        }
        if (st.move) {
          let to: Vec | null = null;
          if (typeof st.move === "object") to = { x: st.move.x * window.innerWidth, y: st.move.y * window.innerHeight };
          else if (st.move === "home") { const h = homeRect(); to = h ? h.c : null; }
          else if (st.move === "center") to = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
          else { const tg = await targetOf(st.move); if (tg) to = besideTarget(tg.r); }
          if (to) await travel(to);
        }
        let used: ArmKey | null = null;
        if (st.press) used = await reachFor(st.press, true, st.twist, st.via);
        else if (st.point) used = await reachFor(st.point, false, st.twist);
        if (st.brain) used = (await brainMove(st.brain)) ?? used;
        if (st.gesture) await gesture(st.gesture);
        if (st.close === "camera") await reachFor("camera-overlay", true);
        if (st.close === "panel" && L.operatorOpen) await reachFor("panel-close", true);
        if (st.close === "console" || st.close === "ssh" || st.close === "proxmox") {   // their own windows: close them myself
          try { await invoke("server_close", { kind: st.close }); } catch (e) { report("note", `close ${st.close}: ${String(e).slice(0, 120)}`); }
        }
        if (st.say) {
          for (let tries = 0; tries < 3; tries++) {
            const done = await speak(st.say, st.emotion, s.silent);
            if (done || L.stop) break;
            report("clicked", st.say.slice(0, 80), i);
            await warnNoClick(s.silent);                    // then say the line again, from the top
            setCaption(st.say);
          }
        }
        await sleep(st.hold_ms ?? 400);
        if (L.clicked !== clicksAtStep && !L.stop) { report("clicked", "between lines", i); await warnNoClick(s.silent); }
        for (const k of ARM_KEYS) { const h = armH(k); h.release(); h.twistGoal = 0; }
        void used;
        setCaption("");
      }
      if (openedPanel && L.operatorOpen && !L.stop) {   // leave the app the way I found it
        await reachFor("panel-close", true);
        await sleep(450);
      }
      if (!L.stop) report("done", "", s.steps.length);
    } catch (e) {
      report("error", String(e).slice(0, 200));
    } finally {
      for (const k of ARM_KEYS) armH(k).release();
      setCaption("");
      await land();
      L.running = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liftOff, land, travel]);

  // ---------------------------------------------------------------- no clicking during the tour
  // My own presses are el.click() (isTrusted = false) and pass straight through; a REAL person's click is
  // swallowed (so it can't knock the tour off course) and counted, and the runner asks them not to.
  useEffect(() => {
    const block = (e: Event) => {
      const L = live.current;
      if (!L.running || !e.isTrusted) return;
      e.preventDefault();
      e.stopPropagation();
      (e as Event & { stopImmediatePropagation: () => void }).stopImmediatePropagation();
      if (e.type === "pointerdown" && performance.now() - L.warnedAt > 2500) L.clicked += 1;
    };
    const types = ["pointerdown", "mousedown", "pointerup", "mouseup", "click", "dblclick", "contextmenu"];
    types.forEach((t) => window.addEventListener(t, block, { capture: true }));
    return () => types.forEach((t) => window.removeEventListener(t, block, { capture: true }));
  }, []);

  // ---------------------------------------------------------------- poll the script
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getJson<Script>(`/api/v1/app/guide${GUIDE_CLIENT ? `?client=${encodeURIComponent(GUIDE_CLIENT)}` : ""}`);
        const L = live.current;
        if (!alive || !s || typeof s.seq !== "number") return;
        if (L.seq < 0) {                                          // never replay an OLD script at startup…
          L.seq = s.seq;
          const fresh = s.issued_ts && Date.now() / 1000 - s.issued_ts < 20 && s.steps?.length && !s.stop;
          if (!fresh) return;                                     // …but a script issued seconds ago is meant for me
          console.info("[guide] running fresh script at startup", s.seq);
          void runScript(s);
          return;
        }
        if (s.seq > L.seq) {
          console.info("[guide] new script", s.seq, "had", L.seq);
          L.seq = s.seq;
          if (s.stop || !s.steps?.length) { L.stop = true; return; }
          if (L.running) { L.stop = true; while (L.running) await sleep(50); }
          void runScript(s);
        }
      } catch { /* backend without the guide routes yet: stay idle */ }
    };
    const id = window.setInterval(() => void tick(), 600);
    void tick();
    return () => { alive = false; window.clearInterval(id); };
  }, [runScript]);

  // ---------------------------------------------------------------- report the layout so I know where things are
  useEffect(() => {
    const send = () => {
      if (document.hidden) return;
      const items = scanUi(live.current.activeTab);
      void postJson("/api/v1/app/guide/ui_map", {
        tab: live.current.operatorOpen ? live.current.activeTab : "home", window: "main",
        viewport: { w: window.innerWidth, h: window.innerHeight }, items,
      }).catch(() => undefined);
    };
    const id = window.setInterval(send, 4000);
    const t = window.setTimeout(send, 800);
    return () => { window.clearInterval(id); window.clearTimeout(t); };
  }, [activeTab, operatorOpen]);

  const L = live.current;
  return (
    <div className={`guide-layer${floating ? " floating" : ""}`} aria-hidden="true">
      <canvas ref={canvasRef} className="guide-canvas" />
      {floating && (
        <div className="guide-eye" style={{
          width: L.size, height: L.size,
          transform: `translate(${eyePos.x + L.offset.x - L.size / 2}px, ${eyePos.y + L.offset.y - L.size / 2}px)`,
        }}>
          <OrbCanvas {...eye} size={L.size} gaze={gaze} portsKey="guide" />
        </div>
      )}
      {caption && (
        <div className="guide-caption" style={{
          left: Math.max(16, Math.min(window.innerWidth - 376, eyePos.x - 180)),
          top: eyePos.y + L.size / 2 + 14 > window.innerHeight - 90 ? eyePos.y - L.size / 2 - 74 : eyePos.y + L.size / 2 + 14,
        }}>{caption}</div>
      )}
    </div>
  );
}
