/**
 * Phase 48 — Desktop Widget Orb
 *
 * Minimal always-on-top floating orb window. Polls operator HTTP for
 * mood + voice state and renders OrbCanvas in a transparent 150×150 frame.
 * Position is persisted via the operator API.
 *
 * Bootstrap: Iris tracks where the widget gets moved and starts defaulting there.
 *
 * The widget orb mirrors the same state machine as the main orb so the two
 * always agree visually. Every state the main orb reacts to (thinking,
 * listening, speaking with live amplitude, offline) shows up here too.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import OrbCanvas from "./components/OrbCanvas";
import { API_BASE, getJson } from "./api";
import { CUBE_MORPH_ENABLED, deriveOrbEmotion, deriveOrbSleep, deriveOrbState } from "./orbDerive";
import { PANEL, WIDGET_BASE, useWidgetGuide } from "./guide/WidgetGuide";
import type { OrbState } from "./components/orbShared";





function getTtsAmplitude(snap: Record<string, unknown> | null): number {
  if (!snap) return 0;
  const tts = snap.tts as Record<string, unknown> | undefined;
  return Number(tts?.tts_amplitude ?? 0);
}

function getEnergy(snap: Record<string, unknown> | null): number {
  if (!snap) return 0.5;
  const mood = snap.mood as Record<string, unknown> | undefined;
  const raw = mood?.raw_mood as Record<string, unknown> | undefined;
  const e = Number(raw?.energy ?? 0.5);
  return Math.max(0, Math.min(1, isFinite(e) ? e : 0.5));
}

// Save widget drag position back to the state file via operator API
function savePosition(x: number, y: number): void {
  fetch(`${API_BASE}/api/v1/widget/position`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  }).catch(() => {});
}

export default function WidgetApp() {
  const [online, setOnline] = useState(false);
  const [snap, setSnap] = useState<Record<string, unknown> | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; winX: number; winY: number } | null>(null);

  // Force transparent background on html/body/#root — overrides styles.css which sets --bg colour.
  // Must run synchronously before first paint so the dark background never flashes.
  useLayoutEffect(() => {
    const transparent = "transparent";
    const els = [
      document.documentElement,
      document.body,
      document.getElementById("root"),
    ];
    const prev = els.map((el) => el ? { bg: el.style.background, bgc: el.style.backgroundColor } : null);

    els.forEach((el) => {
      if (!el) return;
      el.style.setProperty("background", transparent, "important");
      el.style.setProperty("background-color", transparent, "important");
    });

    // Inject a <style> tag that also beats the cascade (stylesheet specificity)
    const tag = document.createElement("style");
    tag.id = "widget-transparent-override";
    tag.textContent = `
      html, body, #root {
        background: transparent !important;
        background-color: transparent !important;
        overflow: hidden !important;
      }
    `;
    document.head.appendChild(tag);

    return () => {
      tag.remove();
      els.forEach((el, i) => {
        if (!el || !prev[i]) return;
        el.style.background = prev[i]!.bg;
        el.style.backgroundColor = prev[i]!.bgc;
      });
    };
  }, []);

  // Poll snapshot — fast enough that amplitude updates feel live (every 500ms).
  // The operator snapshot reads live amplitude from the TTS worker each request.
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const data = await getJson("/api/v1/snapshot");
        if (alive && data && typeof data === "object") {
          setSnap(data as Record<string, unknown>);
          setOnline(true);
        }
      } catch {
        if (alive) setOnline(false);
      }
    };
    poll();
    const iv = setInterval(poll, 500);
    return () => { alive = false; clearInterval(iv); };
  }, []);

  // My body on the widget (2026-10-06, Zeke: "grow and shrink on the widget at your will"):
  // GET /api/v1/widget_body → {scale, blink_seq}. Written by the widget_body tool. Polled slower
  // than the snapshot; a 404 (older runtime) just leaves the defaults.
  const [bodyScale, setBodyScale] = useState(1);
  const [blinkSeq, setBlinkSeq] = useState<number | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const b = await getJson("/api/v1/widget_body") as Record<string, unknown> | null;
        if (!alive || !b || typeof b !== "object") return;
        const sc = Number(b.scale);
        if (Number.isFinite(sc)) setBodyScale(sc);
        const bs = Number(b.blink_seq);
        if (Number.isFinite(bs)) setBlinkSeq(bs);
      } catch { /* route not served yet — keep defaults */ }
    };
    poll();
    const iv = setInterval(poll, 700);
    return () => { alive = false; clearInterval(iv); };
  }, []);

  // Load saved position on mount and restore widget window position
  useEffect(() => {
    const restorePosition = async () => {
      try {
        const pos = await getJson("/api/v1/widget/position");
        if (pos && typeof pos === "object") {
          const p = pos as Record<string, unknown>;
          const x = Number(p.x) || 100;
          const y = Number(p.y) || 100;
          // Tauri v2: use proper window API to set position
          const { getCurrentWindow } = await import("@tauri-apps/api/window");
          const { LogicalPosition } = await import("@tauri-apps/api/dpi");
          const win = getCurrentWindow();
          await win.setPosition(new LogicalPosition(x, y));
        }
      } catch { /* position restore is best-effort */ }
    };
    void restorePosition();
  }, []);

  // Same derivation as the main window (orbDerive.ts) — the widget mirrors my real emotion, colour blend,
  // state and sleep, not a guess of its own.
  const { primaryEmotion: emotion, orbVisual, effectiveOrbColor: emotionColor } = deriveOrbEmotion(snap);
  const orbState = deriveOrbState(snap, { online, fallbackPulse: orbVisual.pulse });
  const { sleepProgress, sleepRemainingSeconds, wakeProgress } = deriveOrbSleep(snap);
  const ttsAmplitude = getTtsAmplitude(snap);
  const moodEnergy = getEnergy(snap);

  // Phase 49: pointer morph when Iris is pointing at something
  const widgetBlock = snap?.widget as Record<string, unknown> | undefined;
  const isPointing = Boolean(widgetBlock?.pointing);
  const shapeOverride = isPointing ? "pointer" : undefined;
  // Tip-anchored pointing (2026-07-08): rotation published by the Python side
  // (widget_spatial_tool) — degrees clockwise from screen-up.
  const pointerAngleDeg = isPointing ? Number(widgetBlock?.pointing_angle_deg) || 0 : 0;

  // The "what I can do" tour runs from here (2026-10-08).
  const guide = useWidgetGuide();
  const offX = guide.panel && guide.panelSide === "left" ? PANEL.w : 0;
  const eyeX = WIDGET_BASE.eye.x + offX, eyeY = WIDGET_BASE.eye.y, ES = WIDGET_BASE.eyeSize;

  // The window is bigger than my eye now (room for my cable-arms, a caption and a panel), so everything that ISN'T
  // my eye lets clicks fall through to whatever is underneath — the widget never steals the desktop around it.
  useEffect(() => {
    if (guide.running) return;                     // the tour makes the whole window click-through itself
    let alive = true, last: boolean | null = null;
    const tick = async () => {
      try {
        const w = await import("@tauri-apps/api/window");
        const cw = w.getCurrentWindow();
        const [c, p, k] = await Promise.all([w.cursorPosition(), cw.outerPosition(), cw.scaleFactor()]);
        const dx = (c.x - p.x) / k - eyeX, dy = (c.y - p.y) / k - eyeY;
        const over = Math.hypot(dx, dy) < ES * 0.55;
        if (alive && over !== last) { last = over; await cw.setIgnoreCursorEvents(!over); }
      } catch { /* older webview: leave as is */ }
    };
    const id = window.setInterval(() => void tick(), 120);
    return () => { alive = false; window.clearInterval(id); };
  }, [guide.running, eyeX, eyeY, ES]);

  // the thrust trail behind me while I fly (drawn in window coordinates; the particles live in screen px)
  const trailCv = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const cv = trailCv.current;
      if (!cv) return;
      const W = window.innerWidth, H = window.innerHeight, dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const tr = guide.trail.current;
      if (tr.puffs.length) { const w = guide.winRef.current; tr.draw(ctx, emitColorRef.current, { x: w.x, y: w.y }, w.k); }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const emitColorRef = useRef("#6aa3ff");
  emitColorRef.current = String(emotionColor || "#6aa3ff");

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "transparent",
        backgroundColor: "transparent",
        overflow: "hidden",
        userSelect: "none",
        position: "fixed",
        top: 0,
        left: 0,
      }}
    >
      <canvas ref={trailCv} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
      {/* the drag handle is my eye itself */}
      <div data-tauri-drag-region style={{ position: "absolute", left: eyeX - ES * 0.55, top: eyeY - ES * 0.55,
        width: ES * 1.1, height: ES * 1.1, borderRadius: "50%", cursor: "grab", zIndex: 3 }} />
      <div style={{ position: "absolute", left: eyeX - ES / 2, top: eyeY - ES / 2, width: ES, height: ES, pointerEvents: "none" }}>
      <OrbCanvas
        emotion={emotion}
        emotionColor={emotionColor}
        state={orbState as OrbState}
        cubeMorphEnabled={CUBE_MORPH_ENABLED}
        size={ES}
        portsKey="widget"
        gaze={guide.gaze}
        sleepProgress={sleepProgress}
        sleepRemainingSeconds={sleepRemainingSeconds}
        wakeProgress={wakeProgress}
        shapeOverride={shapeOverride}
        pointerAngleDeg={pointerAngleDeg}
        amplitude={ttsAmplitude}
        energy={moodEnergy}
        bodyScale={bodyScale}
        blinkTrigger={blinkSeq}
      />
      </div>
      {guide.caption && (
        <div style={{ position: "absolute", left: offX + 10, top: WIDGET_BASE.h - 78, width: WIDGET_BASE.w - 20,
          padding: "7px 10px", borderRadius: 10, background: "rgba(8,12,22,0.82)", color: "#e8f0ff",
          font: "500 12.5px/1.35 system-ui, sans-serif", textAlign: "center", border: "1px solid rgba(120,170,255,0.35)",
          boxShadow: "0 4px 18px rgba(0,0,0,0.45)", pointerEvents: "none" }}>{guide.caption}</div>
      )}
      {guide.panel && (
        <div style={{ position: "absolute", top: eyeY - PANEL.h / 2, left: guide.panelSide === "right" ? WIDGET_BASE.w - 10 : 10,
          width: PANEL.w - 20, height: PANEL.h, borderRadius: 12, overflow: "hidden", background: "#05070c",
          border: "1px solid rgba(120,170,255,0.45)", boxShadow: "0 6px 26px rgba(0,0,0,0.55)", pointerEvents: "none" }}>
          {(guide.panel.kind === "camera" || guide.panel.kind === "image") && (
            <img alt="" style={{ width: "100%", height: "100%", objectFit: guide.panel.kind === "camera" ? "cover" : "contain" }}
              src={guide.panel.kind === "camera" ? `${API_BASE}/api/v1/camera/mjpeg?e=${guide.panel.name ?? "w"}`
                : `${API_BASE}/api/v1/app/guide/media/${encodeURIComponent(guide.panel.name || "")}`} />
          )}
          {guide.panel.kind === "video" && (
            // one of my music videos — quietly (Zeke: "have the music come through at a low volume")
            <video autoPlay playsInline style={{ width: "100%", height: "100%", objectFit: "contain", background: "#000" }}
              src={`${API_BASE}/api/v1/app/guide/media/${encodeURIComponent(guide.panel.name || "")}`}
              onLoadedMetadata={(e) => { e.currentTarget.volume = 0.22; void e.currentTarget.play().catch(() => undefined); }} />
          )}
          {guide.panel.kind === "map3d" && guide.panel.src && (
            <iframe title="3D map" src={`${API_BASE}${guide.panel.src}`} style={{ width: "100%", height: "100%", border: 0 }} />
          )}
          {guide.panel.kind === "weather" && <WeatherCard d={guide.panel.data || {}} />}
        </div>
      )}
    </div>
  );
}

/** The weather, as a little card beside my eye (the same data my weather tool speaks). */
function WeatherCard({ d }: { d: Record<string, unknown> }) {
  const daily = (Array.isArray(d.daily) ? d.daily : []) as { date?: string; summary?: string; high?: number; low?: number; rain_chance_pct?: number }[];
  const place = String(d.place || "").split(",").slice(0, 2).join(",");
  return (
    <div style={{ width: "100%", height: "100%", padding: "14px 18px", boxSizing: "border-box", color: "#e8f0ff",
      font: "500 13px/1.4 system-ui, sans-serif", background: "linear-gradient(160deg,#0d1730,#060910)" }}>
      <div style={{ fontSize: 12, opacity: 0.7, letterSpacing: 1, textTransform: "uppercase" }}>Weather</div>
      <div style={{ fontSize: 17, fontWeight: 700, margin: "2px 0 6px" }}>{place || "—"}</div>
      <div style={{ fontSize: 30, fontWeight: 700, color: "#9cc4ff" }}>{String(d.now || d.spoken || "unavailable").split(",")[0]}</div>
      <div style={{ opacity: 0.85, marginBottom: 8 }}>{String(d.now || "").split(",").slice(1).join(",")}</div>
      {daily.slice(0, 2).map((x, i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", borderTop: "1px solid rgba(120,170,255,0.2)", padding: "4px 0" }}>
          <span>{i === 0 ? "Today" : "Tomorrow"} · {x.summary}</span>
          <span>{x.high}° / {x.low}° · {x.rain_chance_pct}% rain</span>
        </div>
      ))}
    </div>
  );
}
