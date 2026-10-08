// One spoken line for the guide (main window AND widget), 2026-10-08.
// The mouth speaks sentence by sentence and reports "not speaking" in the gaps between them, so a short quiet is
// NOT the end of the line (10-08: I moved on to the chat box mid-sentence about the console). Done = I've been
// talking at least ~words x 0.3 s since I started AND it's been quiet for over a second.
import { getJson, postJson } from "../api";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Speak (or just caption, when silent / voice off) and wait until the line is really over.
 *  Returns false if `interrupted()` turned true part-way (the line is cut short at the mouth). */
export async function speakLine(text: string, opts: { emotion?: string; silent?: boolean; interrupted?: () => boolean } = {}): Promise<boolean> {
  const interrupted = opts.interrupted ?? (() => false);
  let spoken = false;
  if (!opts.silent) try {
    const r = await postJson<{ ok: boolean; spoken?: boolean }>("/api/v1/app/guide/say", { text, emotion: opts.emotion });
    spoken = Boolean(r?.spoken);
  } catch { spoken = false; }
  const words = text.split(/\s+/).length;
  const est = 700 + words * 360;
  const t0 = performance.now();
  if (spoken) {
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
}
