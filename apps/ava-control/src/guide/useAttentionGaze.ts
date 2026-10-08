// The home eye looks at the person I'm following (2026-10-08, Zeke: "when your eye on the desktop is following
// the person, the entire body should move"). Reads the attention system's PTZ bearing + in-frame offset.
import { useEffect, useState } from "react";
import { getJson } from "../api";

type Att = { ok?: boolean; status?: string; pan_deg?: number | null; tilt_deg?: number | null; dx?: number | null; dy?: number | null; age_s?: number };

// Camera and screen both face the person. A person to the camera's right sits on the screen's own right, which is
// the viewer's LEFT — so the eye (x = viewer's right) looks the opposite way. Flip here if it ever looks away.
const PAN_TO_GAZE = 1 / 40;       // per degree of bearing — sign VERIFIED by Zeke 10-08 (−1 turned away from him)
const TILT_TO_GAZE = 1 / 40;
const HFOV_HALF_DEG = 40;         // in-frame offset dx (−1..1) → degrees

export function useAttentionGaze(): { x: number; y: number } | undefined {
  const [g, setG] = useState<{ x: number; y: number } | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const a = await getJson<Att>("/api/v1/app/attention");
        if (!alive) return;
        const fresh = typeof a.age_s === "number" && a.age_s < 3;
        if (a.status === "locked" && fresh && typeof a.pan_deg === "number") {
          const ang = a.pan_deg + (a.dx ?? 0) * HFOV_HALF_DEG;
          const til = (a.tilt_deg ?? 0) - (a.dy ?? 0) * 25;
          const clamp = (v: number) => Math.max(-1, Math.min(1, v));
          setG({ x: clamp(ang * PAN_TO_GAZE), y: clamp(til * TILT_TO_GAZE) });
        } else setG(undefined);
      } catch { if (alive) setG(undefined); }
    };
    const id = window.setInterval(() => void tick(), 300);
    void tick();
    return () => { alive = false; window.clearInterval(id); };
  }, []);
  return g;
}
