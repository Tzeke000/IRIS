// Live words (full-duplex step e, 2026-10-31): what Zeke is saying RIGHT NOW, while he says it -
// including while my mouth is busy (the barge path publishes partials too). Polls the read-only
// /api/v1/voice/live_partial (orb_http, 10-29) ~4x a second; shows a partial while final=false, holds
// the final words for a moment, clears a dropped turn (my own echo, unclear audio) at once. Display
// only: nothing here can start, stop or change a turn.
import { useEffect, useRef, useState } from "react";
import { getJson } from "../api";

type Live = { utterance_id?: string; source?: string; text?: string; final?: boolean; ts?: number;
              mouth_busy?: boolean; dropped?: string | null };
type Resp = { ok?: boolean; now?: number; room?: Live | null; vr?: Record<string, Live> | Live | null };

const HOLD_FINAL_S = 4;     // keep the final words up this long after he stops
const STALE_S = 12;         // a partial nobody updated for this long is not "right now" any more

function pick(r: Resp): { who: string; live: Live } | null {
  const cands: { who: string; live: Live }[] = [];
  if (r.room) cands.push({ who: "room", live: r.room });
  const vr = r.vr as Record<string, Live> | null | undefined;
  if (vr && typeof vr === "object") {
    if (typeof (vr as Live).text === "string") cands.push({ who: "VR", live: vr as Live });
    else for (const [k, v] of Object.entries(vr)) if (v && typeof v === "object") cands.push({ who: k, live: v });
  }
  cands.sort((a, b) => Number(b.live.ts ?? 0) - Number(a.live.ts ?? 0));
  return cands[0] ?? null;
}

export default function LiveWords({ intervalMs = 250 }: { intervalMs?: number }) {
  const [shown, setShown] = useState<{ who: string; text: string; final: boolean; overMe: boolean } | null>(null);
  const failures = useRef(0);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await getJson<Resp>("/api/v1/voice/live_partial");
        failures.current = 0;
        const now = Number(r.now ?? Date.now() / 1000);
        const p = pick(r);
        if (!alive) return;
        if (!p) { setShown(null); return; }
        const age = now - Number(p.live.ts ?? 0);
        const text = String(p.live.text ?? "").trim();
        if (!text || p.live.dropped || age > (p.live.final ? HOLD_FINAL_S : STALE_S)) { setShown(null); return; }
        setShown({ who: p.who, text, final: Boolean(p.live.final), overMe: Boolean(p.live.mouth_busy) });
      } catch {
        failures.current += 1;               // an old runtime without the route: stay quiet, poll slower
      }
    };
    void tick();
    const iv = setInterval(() => { if (failures.current < 20 || Math.random() < 0.1) void tick(); }, intervalMs);
    return () => { alive = false; clearInterval(iv); };
  }, [intervalMs]);
  if (!shown) return null;
  return (
    <div className={`live-words ${shown.final ? "final" : "partial"}`} aria-live="polite"
         title={shown.overMe ? "heard while I was talking" : "heard"}>
      <span className="live-words-who">{shown.who === "room" ? "you" : shown.who}{shown.overMe ? " (over me)" : ""}:</span>{" "}
      <span className="live-words-text">{shown.text}{shown.final ? "" : " …"}</span>
    </div>
  );
}
