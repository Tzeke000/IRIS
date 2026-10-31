// Which me this app talks to (Zeke 2026-10-06): the tower's runtime, or the one on the server
// (iris-home, VM 100) after the cutover. Stored in localStorage so the main window and the widget
// agree; switching RELOADS both windows (sockets + polls rebind cleanly). The Server tab flips it.
export type Backend = "tower" | "server";
export const BACKENDS: Record<Backend, string> = {
  tower: "http://127.0.0.1:5876",
  // From the git-ignored apps/ava-control/.env.local (VITE_IRIS_SERVER_URL) — the repo is public.
  server: import.meta.env.VITE_IRIS_SERVER_URL || "",
};
export const BACKEND_KEY = "iris.backend";
export function readBackend(): Backend {
  // Default when nothing is stored: the build's VITE_DEFAULT_BACKEND (the Zorin build sets "server" —
  // there is no tower runtime on that machine), else the tower.
  const dflt: Backend = import.meta.env.VITE_DEFAULT_BACKEND === "server" ? "server" : "tower";
  try {
    const v = localStorage.getItem(BACKEND_KEY);
    return v === "server" ? "server" : v === "tower" ? "tower" : dflt;
  } catch { return dflt; }
}
export function setBackend(b: Backend): void {
  try { localStorage.setItem(BACKEND_KEY, b); } catch { /* storage unavailable */ }
  location.reload();
}
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => { if (e.key === BACKEND_KEY) location.reload(); });
}
const API_BASE: string = import.meta.env.VITE_OPERATOR_API || BACKENDS[readBackend()];

// :5876 bearer token (code review 10-29): from the git-ignored .env.local (VITE_IRIS_ORB_TOKEN, written by
// scripts/server/orb_token.sh env-line) — never in the repo. Empty = no token sent (the server's default).
// Headers where fetch() is used; ?token= on URLs the webview loads itself (<img>, <iframe>, WebSocket).
const ORB_TOKEN: string = import.meta.env.VITE_IRIS_ORB_TOKEN || "";
export function authHeaders(h: Record<string, string> = {}): Record<string, string> {
  return ORB_TOKEN ? { ...h, Authorization: `Bearer ${ORB_TOKEN}` } : h;
}
export function withToken(url: string): string {
  if (!ORB_TOKEN) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(ORB_TOKEN);
}

/** Emitted after each completed HTTP API response (before JSON throw on error). */
export type ApiLogEntry = {
  timestamp: string;
  endpoint: string;
  status: number;
  responseBody: string;
};

type ApiLogger = (entry: ApiLogEntry) => void;

let apiLogger: ApiLogger | null = null;

export function registerApiLogger(fn: ApiLogger | null): void {
  apiLogger = fn;
}

function emitLog(method: string, path: string, status: number, rawBody: string): void {
  let pretty = rawBody;
  try {
    pretty = JSON.stringify(JSON.parse(rawBody), null, 2);
  } catch {
    /* plain text or empty */
  }
  apiLogger?.({
    timestamp: new Date().toISOString(),
    endpoint: `${method.toUpperCase()} ${path}`,
    status,
    responseBody: pretty,
  });
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "GET", headers: authHeaders() });
  const txt = await res.text();
  emitLog("GET", path, res.status, txt);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return JSON.parse(txt) as T;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const txt = await res.text();
  emitLog("POST", path, res.status, txt);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return JSON.parse(txt) as T;
}

export async function getText(path: string): Promise<string> {
  const res = await fetch(`${API_BASE}${path}`, { method: "GET", headers: authHeaders() });
  const txt = await res.text();
  emitLog("GET", path, res.status, txt);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return txt;
}

export { API_BASE };
