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
import type { IrisBodyProps } from "../components/IrisBody";
import { deriveBlendColors, getCfg } from "../components/orbShared";
import { getJson, postJson } from "../api";
import { Arm, ARM_KEYS, ArmKey, bestArm, drawArm, easeInOutCubic, idleTip, port, Vec } from "./arms";
import { findEl, scanUi } from "./uiMap";

type Step = {
  tab?: string; move?: string | { x: number; y: number }; point?: string; press?: string; say?: string;
  gesture?: "wave" | "nod" | "shake" | "curious"; twist?: number; hold_ms?: number; emotion?: string;
};
type Script = { ok?: boolean; seq: number; steps: Step[]; stop?: boolean; silent?: boolean };

export type GuideOverlayProps = {
  eye: IrisBodyProps;            // the same props the home-screen eye gets (emotion, colour, state…)
  activeTab: string;
  operatorOpen: boolean;
  onFloatingChange: (floating: boolean) => void;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const IRIS_FRAC = 0.546;          // iris radius / half canvas (IrisBody: 0.72 world of a 1.319 half-width)

export default function GuideOverlay({ eye, activeTab, operatorOpen, onFloatingChange }: GuideOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const arms = useRef<Record<ArmKey, Arm>>(Object.fromEntries(ARM_KEYS.map((k, i) => [k, new Arm(k, i)])) as Record<ArmKey, Arm>);
  const [floating, setFloating] = useState(false);
  const [caption, setCaption] = useState("");
  const [gaze, setGaze] = useState<{ x: number; y: number } | undefined>(undefined);
  const [eyePos, setEyePos] = useState<Vec>({ x: 0, y: 0 });
  const live = useRef({ floating: false, pos: { x: 0, y: 0 } as Vec, offset: { x: 0, y: 0 } as Vec, operatorOpen, activeTab,
    size: 180, stop: false, running: false, seq: -1 });
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
        if (!h || L.operatorOpen) return;    // the panel covers the home eye: no free-floating arms
        center = h.c; size = h.size;
      }
      // the 3D cyborg eye publishes its real collar positions (they move as the ball turns)
      const ep = eyePorts.get(L.floating ? "guide" : "home");
      const live3d = ep && performance.now() - ep.ts < 400 ? ep : null;
      if (live3d) center = live3d.center;
      const R = live3d ? live3d.radius * 0.9 : (size / 2) * IRIS_FRAC;
      const scale = Math.max(0.55, Math.min(1.6, R / 60));
      const dt = Math.min(50, now - last); last = now;
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
        a.step(now, dt, root, dir, idleTip(center, R, k, 0, t, a.phase, R * 1.6), scale);
      }
      // draw the far arms first (up/left), so crossings read with some depth
      for (const k of ["up", "left", "down", "right"] as ArmKey[]) {
        drawArm(ctx, arms.current[k], { light: c.lightColor, copper: "#d9894f", copperDark: "#5a2f17", scale, dpr, glow });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ---------------------------------------------------------------- eye travel (Disney timing)
  const travel = useCallback(async (to: Vec) => {
    const L = live.current;
    const from = { ...L.pos };
    const d = Math.hypot(to.x - from.x, to.y - from.y);
    if (d < 2) return;
    const dir = { x: (to.x - from.x) / d, y: (to.y - from.y) / d };
    const ANT = 160, dur = Math.min(1100, 420 + d * 0.55);
    const t0 = performance.now();
    setGaze({ x: dir.x * 0.85, y: -dir.y * 0.85 });            // the eye looks where it's going, first
    for (;;) {
      const t = performance.now() - t0;
      if (t < ANT) {                                              // anticipation: dip back a little
        const k = Math.sin((t / ANT) * Math.PI * 0.5);
        L.pos = { x: from.x - dir.x * 10 * k, y: from.y - dir.y * 10 * k };
      } else if (t < ANT + dur) {
        const u = easeInOutCubic((t - ANT) / dur);
        const s = { x: from.x - dir.x * 10, y: from.y - dir.y * 10 };
        L.pos = { x: s.x + (to.x - s.x) * u, y: s.y + (to.y - s.y) * u };
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
  const besideTarget = (r: DOMRect): Vec => {
    const L = live.current;
    const W = window.innerWidth, H = window.innerHeight;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const gap = L.size * 0.9 + 40;
    const cands: Vec[] = [
      { x: cx - r.width / 2 - gap, y: cy }, { x: cx + r.width / 2 + gap, y: cy },
      { x: cx, y: cy + r.height / 2 + gap }, { x: cx, y: cy - r.height / 2 - gap },
    ];
    const m = L.size / 2 + 10;
    const ok = cands.filter((p) => p.x > m && p.x < W - m && p.y > m && p.y < H - m);
    const pick = (ok.length ? ok : cands).sort((a, b) =>
      Math.hypot(a.x - L.pos.x, a.y - L.pos.y) - Math.hypot(b.x - L.pos.x, b.y - L.pos.y))[0];
    return { x: Math.max(m, Math.min(W - m, pick.x)), y: Math.max(m, Math.min(H - m, pick.y)) };
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

  const reachFor = async (id: string, press: boolean, twist?: number): Promise<ArmKey | null> => {
    const tg = await targetOf(id);
    if (!tg) { report("missing", `no element ${id}`); return null; }
    const L = live.current;
    const R = (L.size / 2) * IRIS_FRAC;
    const scale = Math.max(0.55, Math.min(1.6, R / 60));
    const maxReach = R + R * 1.7 * 1.45;
    if (Math.hypot(tg.c.x - L.pos.x, tg.c.y - L.pos.y) > maxReach * 0.92) await travel(besideTarget(tg.r));
    const dx = tg.c.x - L.pos.x, dy = tg.c.y - L.pos.y, dl = Math.hypot(dx, dy) || 1;
    setGaze({ x: (dx / dl) * 0.8, y: (-dy / dl) * 0.8 });       // eye leads: look first…
    await sleep(140);                                             // …then the arm goes
    const busy = new Set(ARM_KEYS.filter((k) => arms.current[k].holding));
    const k = bestArm(L.pos, tg.c, 0, busy);
    const a = arms.current[k];
    if (twist !== undefined) a.twistGoal = twist;
    let contacted = false;
    a.reachTo(tg.c, performance.now(), {
      press, onContact: () => {
        contacted = true;
        const tag = tg.el.tagName.toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") tg.el.focus();
        else tg.el.click();
      },
    });
    if (press) {
      const t0 = performance.now();
      while (!contacted && performance.now() - t0 < 2500) await sleep(30);
      await sleep(220);
    } else {
      await sleep(600);
    }
    return k;
  };

  const speak = async (text: string, emotion?: string, silent?: boolean) => {
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
      let started = false, quietSince = 0;
      while (performance.now() - t0 < est * 2.2 + 3000) {
        await sleep(100);
        let sp = false;
        try { sp = Boolean((await getJson<{ speaking?: boolean }>("/api/v1/tts/state"))?.speaking); } catch { /* keep going */ }
        if (sp) { started = true; quietSince = 0; }
        else if (started) { if (!quietSince) quietSince = performance.now(); if (performance.now() - quietSince > 450) break; }
        else if (performance.now() - t0 > 4000) break;           // never started: fall back to reading time
      }
      if (!started) await sleep(Math.max(0, est - (performance.now() - t0)));
    } else {
      await sleep(est);                                          // voice off: give time to read the caption
    }
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
      const a = arms.current["up"].holding ? arms.current["right"] : arms.current["up"];
      const R = (L.size / 2) * IRIS_FRAC;
      const hand = { x: base.x + R * 1.6, y: base.y - R * 2.4 };
      a.reachTo(hand, performance.now(), { dur: 420 });
      await sleep(560);
      for (let i = 0; i < 4; i++) {
        a.reachTo({ x: hand.x + (i % 2 ? -1 : 1) * R * 0.7, y: hand.y + (i % 2 ? 6 : -6) }, performance.now(), { dur: 260 });
        await sleep(300);
      }
      a.release();
    } else if (g === "curious") {
      const a = arms.current["up"];
      setGaze({ x: 0.35, y: 0.4 });
      a.twistGoal = 0.6;
      a.reachTo({ x: base.x + 30, y: base.y - (L.size / 2) * IRIS_FRAC * 2.6 }, performance.now(), { dur: 600 });
      await sleep(1100);
      a.twistGoal = 0;
      a.release();
      setGaze(undefined);
    }
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
        report("step", st.say?.slice(0, 80) || st.point || st.press || st.tab || "", i);
        if (st.tab) {
          if (!L.operatorOpen) { await reachFor("panel", true); openedPanel = true; await sleep(500); }
          await reachFor(`tab:${st.tab}`, true);
          await sleep(350);
          for (const k of ARM_KEYS) arms.current[k].release();
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
        if (st.press) used = await reachFor(st.press, true, st.twist);
        else if (st.point) used = await reachFor(st.point, false, st.twist);
        if (st.gesture) await gesture(st.gesture);
        if (st.say) await speak(st.say, st.emotion, s.silent);
        await sleep(st.hold_ms ?? 400);
        if (used) { arms.current[used].release(); arms.current[used].twistGoal = 0; }
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
      for (const k of ARM_KEYS) arms.current[k].release();
      setCaption("");
      await land();
      L.running = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liftOff, land, travel]);

  // ---------------------------------------------------------------- poll the script
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getJson<Script>("/api/v1/app/guide");
        const L = live.current;
        if (!alive || !s || typeof s.seq !== "number") return;
        if (L.seq < 0) { L.seq = s.seq; return; }                // never replay an old script at startup
        if (s.seq > L.seq) {
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
    <div className="guide-layer" aria-hidden="true">
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
