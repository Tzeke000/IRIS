// WidgetGuide — the "what I can do" tour, run from the WIDGET (2026-10-08).
//
// Zeke: "start with minimizing the app … you'd auto flip to the widget like normal and then go through what you
// can do: open apps, open Chrome, type anywhere a human can see the screen, maneuver apps — searching a website
// and scrolling on it — moving the mouse around, then your head and eyes: bring it up and show you can look up,
// down, left, right, show you can recognise objects like your server…"
//
// The main window's tour ends with `minimize`; the server hands the rest (`then`) to channel "widget", which this
// runner polls. Steps (all optional, run in this order):
//   travel {x,y}|"home"   glide the widget window across the primary monitor (x,y = where my EYE lands)
//   look {x,y}|null       gaze
//   head {pan_deg,tilt_deg}|"home"   my real PTZ head
//   camera show|hide · image <name>|hide   a panel beside the eye (live camera / a picture)
//   desk {do,...}         the real mouse + keyboard (Tauri desk_* commands) — runs WHILE the line is spoken
//   gesture · say · wait_ms · hold_ms
import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getJson, postJson } from "../api";
import { eyeArms } from "../components/eyeArms3d";
import { speakLine } from "./speech";
import { fly, type Pose } from "./flight";
import { Trail } from "./trail";

type Vec = { x: number; y: number };
type Desk = { do: string; x?: number; y?: number; w?: number; h?: number; ms?: number; r?: number; text?: string; combo?: string;
  amount?: number; app?: string; arg?: string; pattern?: string; button?: string; double?: boolean };
type WStep = {
  travel?: Vec | "home" | "center"; look?: Vec | null; head?: { pan_deg: number; tilt_deg: number } | "home";
  camera?: "show" | "hide"; image?: string; desk?: Desk; gesture?: "wave" | "nod" | "shake" | "curious";
  say?: string; emotion?: string; wait_ms?: number; hold_ms?: number;
};
type Script = { seq: number; steps: WStep[]; stop?: boolean; silent?: boolean; issued_ts?: number };

export type WidgetLayout = { w: number; h: number; eye: Vec; eyeSize: number };
export const WIDGET_BASE: WidgetLayout = { w: 300, h: 380, eye: { x: 150, y: 150 }, eyeSize: 120 };
export const PANEL = { w: 420, h: 236 };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const smoother = (u: number) => u * u * u * (u * (u * 6 - 15) + 10);
const HEAD_HOME = { pan_deg: 0, tilt_deg: 10 };

export function useWidgetGuide() {
  const [caption, setCaption] = useState("");
  const [gaze, setGaze] = useState<Pose>(undefined);
  const trail = useRef(new Trail());
  const winRef = useRef({ x: 0, y: 0, k: 1 });            // where my window is (physical px) — the trail draws relative to it
  const [panel, setPanel] = useState<{ kind: "camera" | "image"; name?: string } | null>(null);
  const [panelSide, setPanelSide] = useState<"right" | "left">("right");
  const [running, setRunning] = useState(false);
  const live = useRef({ seq: -1, running: false, stop: false, home: null as Vec | null, panelSide: "right" as "right" | "left" });

  const report = (seq: number, status: string, note = "", index = -1) => {
    void postJson("/api/v1/app/guide/progress", { seq, index, status, note, client: "widget" }).catch(() => undefined);
  };

  // ------------------------------------------------------------- window geometry (physical px)
  const win = async () => (await import("@tauri-apps/api/window"));
  const monitor = async () => {
    const w = await win();
    const m = (await w.primaryMonitor()) || (await w.currentMonitor());
    return m ? { x: m.position.x, y: m.position.y, w: m.size.width, h: m.size.height, k: m.scaleFactor || 1 }
             : { x: 0, y: 0, w: 1920, h: 1080, k: 1 };
  };
  /** Where my eye is on the screen right now (physical px). */
  const eyeScreen = async (): Promise<Vec> => {
    const w = await win();
    const cw = w.getCurrentWindow();
    const p = await cw.outerPosition();
    const k = await cw.scaleFactor();
    const offX = live.current.panelSide === "left" && panelRef.current ? PANEL.w : 0;
    return { x: p.x + (WIDGET_BASE.eye.x + offX) * k, y: p.y + WIDGET_BASE.eye.y * k };
  };
  const panelRef = useRef(false);
  const placeEye = async (eye: Vec) => {
    const w = await win();
    const cw = w.getCurrentWindow();
    const k = await cw.scaleFactor();
    const offX = live.current.panelSide === "left" && panelRef.current ? PANEL.w : 0;
    const x = Math.round(eye.x - (WIDGET_BASE.eye.x + offX) * k), y = Math.round(eye.y - WIDGET_BASE.eye.y * k);
    winRef.current = { x, y, k };
    await cw.setPosition(new w.PhysicalPosition(x, y));
  };

  /** FLY my eye to a screen point: side-on, leaning into it, thrust from the back (guide/flight.ts). */
  const travelTo = async (to: Vec) => {
    const from = await eyeScreen();
    const k = await (await win()).getCurrentWindow().scaleFactor();
    await fly(from, to, {
      place: placeEye,
      pose: (g) => setGaze(g),
      trail: (p, back, sp) => trail.current.emit(p, back, sp, WIDGET_BASE.eyeSize * 0.42 * k),
      scale: k,
    });
  };

  // ------------------------------------------------------------- the real mouse
  const glideMouse = async (to: Vec, ms: number) => {
    let from: Vec;
    try { const [x, y] = await invoke<[number, number]>("desk_cursor"); from = { x, y }; } catch { from = to; }
    const d = Math.hypot(to.x - from.x, to.y - from.y) || 1;
    const bulge = Math.min(120, d * 0.12);
    const t0 = performance.now();
    for (;;) {
      const u = Math.min(1, (performance.now() - t0) / ms);
      const s = smoother(u), b = bulge * Math.sin(Math.PI * u);
      const x = from.x + (to.x - from.x) * s - ((to.y - from.y) / d) * b;
      const y = from.y + (to.y - from.y) * s + ((to.x - from.x) / d) * b;
      await invoke("desk_move", { x: Math.round(x), y: Math.round(y) });
      if (u >= 1) break;
      await sleep(12);
    }
  };

  const doDesk = async (seq: number, dk: Desk) => {
    const m = await monitor();
    const at = (fx?: number, fy?: number): Vec | null =>
      fx === undefined || fy === undefined ? null : { x: Math.round(m.x + fx * m.w), y: Math.round(m.y + fy * m.h) };
    try {
      switch (dk.do) {
        case "move": { const p = at(dk.x, dk.y); if (p) await glideMouse(p, dk.ms ?? 900); break; }
        case "circle": {                                   // loop the cursor around a point (default: around me)
          const c = at(dk.x, dk.y) ?? await eyeScreen();
          const r = dk.r ?? 140, dur = dk.ms ?? 2400;
          await glideMouse({ x: c.x + r, y: c.y }, 500);
          const t0 = performance.now();
          for (;;) {
            const u = Math.min(1, (performance.now() - t0) / dur);
            const a = smoother(u) * Math.PI * 2;
            await invoke("desk_move", { x: Math.round(c.x + r * Math.cos(a)), y: Math.round(c.y + r * Math.sin(a)) });
            if (u >= 1) break;
            await sleep(12);
          }
          break;
        }
        case "click": {
          const p = at(dk.x, dk.y);
          if (p) await glideMouse(p, dk.ms ?? 700);
          await invoke("desk_click", { button: dk.button ?? "left", double: Boolean(dk.double) });
          break;
        }
        case "type": await invoke("desk_type", { text: dk.text ?? "", cps: 14 }); break;
        case "keys": await invoke("desk_keys", { combo: dk.combo ?? "" }); break;
        case "scroll": {
          const n = dk.amount ?? 5, steps = Math.abs(n);
          for (let i = 0; i < steps; i++) { await invoke("desk_scroll", { amount: Math.sign(n) }); await sleep(dk.ms ?? 220); }
          break;
        }
        case "open": await invoke("desk_open", { app: dk.app ?? "", arg: dk.arg ?? null }); break;
        case "snap": await invoke("desk_snap"); break;
        case "place": {                                     // put the window I just opened where I know it is
          const p = at(dk.x ?? 0, dk.y ?? 0)!;
          await invoke("desk_place", { pattern: dk.pattern ?? ".", x: p.x, y: p.y,
            w: Math.round((dk.w ?? 0.5) * m.w), h: Math.round((dk.h ?? 0.6) * m.h) });
          break;
        }
        case "close_new": await invoke("desk_close_new", { pattern: dk.pattern ?? "^$" }); break;
      }
    } catch (e) {
      report(seq, "note", `desk ${dk.do}: ${String(e).slice(0, 160)}`);
    }
  };

  // ------------------------------------------------------------- panel (camera / picture) beside the eye
  const showPanel = async (p: { kind: "camera" | "image"; name?: string } | null) => {
    const w = await win();
    const cw = w.getCurrentWindow();
    const k = await cw.scaleFactor();
    const eye = await eyeScreen();
    const m = await monitor();
    if (p) {
      const side: "right" | "left" = eye.x + (WIDGET_BASE.w / 2 + PANEL.w + 20) * k > m.x + m.w ? "left" : "right";
      live.current.panelSide = side;
      setPanelSide(side);
      panelRef.current = true;
      await cw.setSize(new w.LogicalSize(WIDGET_BASE.w + PANEL.w, WIDGET_BASE.h));
      await placeEye(eye);                                  // the eye stays put; the window grows beside it
      setPanel(p);
    } else {
      setPanel(null);
      panelRef.current = false;
      await cw.setSize(new w.LogicalSize(WIDGET_BASE.w, WIDGET_BASE.h));
      await placeEye(eye);
    }
  };

  // ------------------------------------------------------------- arms (the widget body's real 3D cables)
  const gesture = async (g: WStep["gesture"]) => {
    const arms = eyeArms.get("widget");
    const c = WIDGET_BASE.eye, R = WIDGET_BASE.eyeSize * 0.42;
    if (g === "wave" && arms) {
      const pivot = { x: c.x + R * 0.9, y: c.y - R * 1.1 }, rad = R * 1.1, dur = 2400;
      arms.follow("up", (t) => {
        const u = t / dur, env = Math.sin(Math.PI * Math.min(1, u * 1.15)) ** 0.7;
        const ang = -Math.PI / 2 + 0.55 * env * Math.sin(2 * Math.PI * 1.7 * (t / 1000));
        return { x: pivot.x + Math.cos(ang) * rad, y: pivot.y + Math.sin(ang) * rad * 0.85 };
      }, dur);
      await sleep(dur);
      arms.release("up");
    } else if (g === "nod" || g === "shake") {
      const t0 = performance.now(), dur = 1100;
      while (performance.now() - t0 < dur) {
        const u = (performance.now() - t0) / dur, v = Math.sin(u * Math.PI * 4) * 0.6 * Math.sin(Math.PI * u);
        setGaze(g === "nod" ? { x: 0, y: v } : { x: v, y: 0 });
        await sleep(16);
      }
      setGaze(undefined);
    } else if (g === "curious") {
      setGaze({ x: 0.35, y: 0.4 });
      if (arms) { arms.setTwist("up", 0.6); arms.reachTo("up", c.x + 20, c.y - R * 2.2, { dur: 600 }); }
      await sleep(1100);
      if (arms) { arms.setTwist("up", 0); arms.release("up"); }
      setGaze(undefined);
    }
  };

  /** Reach an arm toward a screen point (it can only reach within my window, so it leans that way). */
  const reachToward = async (target: Vec) => {
    const arms = eyeArms.get("widget");
    const eye = await eyeScreen();
    const dx = target.x - eye.x, dy = target.y - eye.y, dl = Math.hypot(dx, dy) || 1;
    setGaze({ x: (dx / dl) * 0.85, y: (-dy / dl) * 0.85 });
    if (!arms) return null;
    const c = WIDGET_BASE.eye, reach = WIDGET_BASE.eyeSize * 1.05;
    const key = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : (dy > 0 ? "down" : "up");
    arms.reachTo(key, c.x + (dx / dl) * reach, c.y + (dy / dl) * reach);
    return key;
  };

  // ------------------------------------------------------------- the runner
  const run = useCallback(async (s: Script) => {
    const L = live.current;
    L.stop = false; L.running = true; setRunning(true);
    report(s.seq, "started", `${s.steps.length} steps`);
    try {
      const w = await win();
      const cw = w.getCurrentWindow();
      await cw.setIgnoreCursorEvents(true);               // nobody can grab me mid-tour
      L.home = await eyeScreen();
      const m = await monitor();
      for (let i = 0; i < s.steps.length; i++) {
        if (L.stop) { report(s.seq, "stopped", "", i); break; }
        const st = s.steps[i];
        report(s.seq, "step", st.say?.slice(0, 80) || st.desk?.do || "", i);
        if (st.travel) {
          const to = st.travel === "home" ? L.home! : st.travel === "center" ? { x: m.x + m.w / 2, y: m.y + m.h / 2 }
            : { x: m.x + st.travel.x * m.w, y: m.y + st.travel.y * m.h };
          await travelTo(to);
        }
        if (st.look !== undefined) setGaze(st.look ?? undefined);
        if (st.head) void postJson("/api/v1/app/guide/head", st.head === "home" ? HEAD_HOME : st.head).catch(() => undefined);
        if (st.camera) await showPanel(st.camera === "show" ? { kind: "camera" } : null);
        if (st.image) await showPanel(st.image === "hide" ? null : { kind: "image", name: st.image });
        if (st.gesture) await gesture(st.gesture);
        let armKey: string | null = null;
        if (st.desk && st.desk.x !== undefined && st.desk.y !== undefined && ["move", "click"].includes(st.desk.do))
          armKey = await reachToward({ x: m.x + st.desk.x * m.w, y: m.y + st.desk.y * m.h });
        const deskP = st.desk ? doDesk(s.seq, st.desk) : Promise.resolve();
        if (st.say) {
          setCaption(st.say);
          report(s.seq, "say", st.say.slice(0, 200));
          await speakLine(st.say, { emotion: st.emotion, silent: s.silent, interrupted: () => L.stop });
        }
        await deskP;
        if (st.wait_ms) await sleep(st.wait_ms);
        await sleep(st.hold_ms ?? 350);
        if (armKey) eyeArms.get("widget")?.release(armKey as "up");
        if (st.look === undefined) setGaze(undefined);
        setCaption("");
      }
      if (!L.stop) report(s.seq, "done", "", s.steps.length);
    } catch (e) {
      report(s.seq, "error", String(e).slice(0, 200));
    } finally {
      setCaption("");
      setGaze(undefined);
      try { if (panelRef.current) await showPanel(null); } catch { /* fine */ }
      try { if (live.current.home) await travelTo(live.current.home); } catch { /* fine */ }
      try { const w = await win(); await w.getCurrentWindow().setIgnoreCursorEvents(false); } catch { /* fine */ }
      L.running = false; setRunning(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // poll channel "widget"
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getJson<Script>("/api/v1/app/guide?client=widget");
        const L = live.current;
        if (!alive || !s || typeof s.seq !== "number") return;
        if (L.seq < 0) {                                        // never replay an OLD script at startup
          L.seq = s.seq;
          const fresh = s.issued_ts && Date.now() / 1000 - s.issued_ts < 20 && s.steps?.length && !s.stop;
          if (fresh) void run(s);
          return;
        }
        if (s.seq > L.seq) {
          L.seq = s.seq;
          if (s.stop || !s.steps?.length) { L.stop = true; return; }
          if (L.running) { L.stop = true; while (L.running) await sleep(50); }
          void run(s);
        }
      } catch { /* older backend: idle */ }
    };
    const id = window.setInterval(() => void tick(), 600);
    void tick();
    return () => { alive = false; window.clearInterval(id); };
  }, [run]);

  return { caption, gaze, panel, panelSide, running, trail, winRef };
}
