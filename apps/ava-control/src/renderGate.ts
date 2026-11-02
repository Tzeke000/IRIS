// One frame budget for every render loop in a window (2026-11-02). Measured on the tower while Zeke gamed: the app
// burned ~35-57% of the whole CPU — the 3D body, its hidden iris, the guide canvas and the camera thumb all ran at
// the full vsync rate, even with Discord maximized over the window (WebView2 only reports `document.hidden` for a
// minimize, never for a covered window).
//   - PAUSED while nobody can see the window: document.hidden, minimized, hidden, or fully covered (main_occluded,
//     a Rust hit-test of the main window). Loops stop asking for frames and are resumed by this module.
//   - otherwise CAPPED: ACTIVE_FPS while I'm talking/listening/thinking, just changed mood, or the window has
//     focus; IDLE_FPS otherwise. The eye moves slowly at rest, so 20 fps reads the same as 60.
// My VOICE is not in here: it plays through the separate audio_sink process, never through this webview.
// A/B knobs (localStorage "iris.perf", JSON): {"activeFps":30,"idleFps":20,"occlusion":true}.

import { useEffect, useState } from "react";

type Perf = { activeFps: number; idleFps: number; occlusion: boolean };
const DEFAULTS: Perf = { activeFps: 30, idleFps: 20, occlusion: true };
function readPerf(): Perf {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("iris.perf") || "{}") }; } catch { return DEFAULTS; }
}
let perf = readPerf();

let visible = typeof document === "undefined" ? true : !document.hidden;
let winVisible = true;                 // from the Tauri poll: not minimized, not hidden, not covered
let focused = typeof document === "undefined" ? true : document.hasFocus();
let activeUntil = 0;
let activeNow = false;
const resumers = new Set<() => void>();
const watchers = new Set<(v: boolean) => void>();

function update() {
  const was = visible;
  visible = !document.hidden && winVisible;
  if (visible === was) return;
  if (visible) resumers.forEach((f) => { try { f(); } catch { /* a dead loop */ } });
  watchers.forEach((f) => { try { f(visible); } catch { /* unmounted */ } });
}

/** Can anyone see this window right now? Loops stop scheduling frames when this is false. */
export function renderVisible(): boolean { return visible; }

/** Called when the window becomes visible again — restart a stopped loop here. Returns an unsubscribe. */
export function onRenderResume(fn: () => void): () => void {
  resumers.add(fn);
  return () => { resumers.delete(fn); };
}

/** React: re-renders when the window becomes visible / invisible (e.g. to stop a camera stream nobody sees). */
export function useRenderVisible(): boolean {
  const [v, setV] = useState(visible);
  useEffect(() => {
    setV(visible);
    watchers.add(setV);
    return () => { watchers.delete(setV); };
  }, []);
  return v;
}

const ACTIVE_STATES = new Set(["speaking", "listening", "attentive", "thinking", "deep", "waking", "pointing"]);
/** React: feed the gate from the body's state + emotion (full rate while I'm in a conversation or just changed). */
export function useRenderActivity(state: string, emotion: string) {
  useEffect(() => { setRenderActive(ACTIVE_STATES.has(state)); }, [state]);
  useEffect(() => { bumpRenderActive(); }, [state, emotion]);
}

/** I'm speaking / listening / thinking (true) or at rest (false). */
export function setRenderActive(active: boolean) { activeNow = active; }
/** Run at the active rate for a moment (a mood change, a blink, a gesture). */
export function bumpRenderActive(ms = 3000) { activeUntil = Math.max(activeUntil, performance.now() + ms); }

export function targetFps(): number {
  return activeNow || focused || performance.now() < activeUntil ? perf.activeFps : perf.idleFps;
}

/** A per-loop limiter: call with the rAF timestamp, render only when it returns true. */
export function frameLimiter(): (now: number) => boolean {
  let last = -Infinity;
  return (now: number) => {
    const iv = 1000 / targetFps();
    if (now - last < iv - 2) return false;        // 2 ms slack so a 60 Hz vsync lands every 2nd frame at 30 fps
    last = now;
    return true;
  };
}

if (typeof window !== "undefined") {
  document.addEventListener("visibilitychange", update);
  window.addEventListener("focus", () => { focused = true; if (!winVisible) { winVisible = true; update(); } });
  window.addEventListener("blur", () => { focused = false; });
  window.addEventListener("storage", (e) => { if (e.key === "iris.perf") perf = readPerf(); });
  (async () => {
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const { invoke } = await import("@tauri-apps/api/core");
      const win = getCurrentWindow();
      const isMain = win.label === "main";
      const poll = async () => {
        try {
          let v = !(await win.isMinimized()) && (await win.isVisible());
          if (v && isMain && perf.occlusion) v = !(await invoke<boolean>("main_occluded"));
          if (v !== winVisible) { winVisible = v; update(); }
        } catch { /* window gone mid-poll / older backend without main_occluded: stay as we are */ }
      };
      void poll();
      setInterval(poll, 1000);
    } catch { /* not under Tauri (dev browser): document.hidden still applies */ }
  })();
}
