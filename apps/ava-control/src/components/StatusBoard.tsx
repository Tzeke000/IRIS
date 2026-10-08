// Status board (Zeke 2026-10-07): "some of that server tab should be on the main page … where you're
// running, what model, CPU, RAM, GPU, SSD storage + read/write speed, network … along the top with your
// mood … bottom right the server status and server reaching the tower stuff and the open my console
// button … make sure it all good and cant false say things are good or vise versa".
//
// HONESTY RULES — the backend (brain/app_status_board.py) does real round-trips and reports ok as
// true / false / null; this file must not undo that:
//   * null / missing / stale  -> GREY with the reason, never green.
//   * a number that failed     -> "—", never 0.
//   * the whole board is stale  -> every chip greys (the runtime stopped answering).
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getJson, readBackend } from "../api";

export type Link = { label: string; ok: boolean | null; detail: string; checked_ts: number | null; age_s?: number; every_s: number };
export type Board = {
  ok: boolean; ts: number;
  host: { role: "server" | "tower"; hostname: string };
  model: string | null;
  sample_age_s: number | null;
  cpu: { percent: number | null; cores: number; temp_c: number | null } | null;
  host_cpus?: { socket: number; threads?: number; percent: number | null; temp_c: number | null }[] | null;
  host_cpus_error?: string | null;
  ram: { used_gb: number; total_gb: number; percent: number } | null;
  gpu: { name: string; util_pct: number; mem_used_mb: number; mem_total_mb: number; temp_c: number }[] | null;
  disk: { path: string; used_gb: number; total_gb: number; percent: number; read_mb_s: number | null; write_mb_s: number | null } | null;
  net: { iface: string | null; rx_mbps: number | null; tx_mbps: number | null } | null;
  camera_fps: number | null;
  links: Record<string, Link>;
};
type Reach = { proxmox: boolean; iris_home_ssh: boolean; iris_home_runtime: boolean };

const STALE_S = 10; // board older than this (wall clock) = not live

/** One poller for the whole page. `board` is null until the first answer; `err` says why it's missing. */
export function useStatusBoard() {
  const [board, setBoard] = useState<Board | null>(null);
  const [fetchedAt, setFetchedAt] = useState(0);
  const [err, setErr] = useState<string>("");
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const b = await getJson<Board>("/api/v1/app/status_board");
        if (!alive) return;
        if (b && b.ok) { setBoard(b); setFetchedAt(Date.now()); setErr(""); }
        else setErr("status board answered not-ok");
      } catch (e) {
        if (alive) setErr(String(e).slice(0, 120));
      }
    };
    void tick();
    const iv = setInterval(() => void tick(), 2000);
    const clock = setInterval(() => setNow(Date.now()), 1000);
    return () => { alive = false; clearInterval(iv); clearInterval(clock); };
  }, []);
  const stale = !board || (now - fetchedAt) / 1000 > STALE_S || (board.sample_age_s ?? 99) > STALE_S;
  return { board, stale, err };
}

const fmt = (v: number | null | undefined, d = 0, unit = "") =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(d)}${unit}`;
const rate = (mb: number | null | undefined) =>
  mb === null || mb === undefined ? "—" : mb >= 100 ? `${mb.toFixed(0)}` : mb >= 1 ? mb.toFixed(1) : mb.toFixed(2);

function Chip({ k, v, sub, title, dim }: { k: string; v: string; sub?: string; title?: string; dim?: boolean }) {
  return (
    <div className={`sb-chip${dim ? " sb-dim" : ""}`} title={title}>
      <span className="sb-k">{k}</span>
      <span className="sb-v">{v}</span>
      {sub ? <span className="sb-sub">{sub}</span> : null}
    </div>
  );
}

export function StatusTopBar({ sb, mood, moodSub, moodColor }: {
  sb: ReturnType<typeof useStatusBoard>; mood: string; moodSub?: string; moodColor?: string;
}) {
  const { board: b, stale, err } = sb;
  const where = b ? (b.host.role === "server" ? "Server" : "Tower") : readBackend() === "server" ? "Server?" : "Tower?";
  const dim = stale;
  const why = stale ? (err ? `not live: ${err}` : "not live — no fresh reading") : undefined;
  const g = b?.gpu?.[0];
  const cpus = b?.host_cpus;
  // Order (Zeke 10-07): my VM's CPU, then the server's two physical CPUs to its right, before RAM;
  // mood sits in the MIDDLE of the same row.
  const moodChip = (
    <div className="sb-mood" style={{ color: moodColor }} key="mood">
      <span className="sb-k">Mood</span>
      <span className="sb-v">{mood || "—"}</span>
      {moodSub ? <span className="sb-sub">with a little {moodSub}</span> : null}
    </div>
  );
  const left = [
    <Chip key="where" k="Running on" v={where} sub={b?.host.hostname} dim={dim} title={why} />,
    <Chip key="model" k="Model" v={b?.model ?? "—"} dim={dim} title={why ?? "read from the live process's --model flag"} />,
    <Chip key="cpu" k={b?.host.role === "server" ? "CPU (mine)" : "CPU"} v={fmt(b?.cpu?.percent, 0, "%")}
      sub={b?.cpu ? `${b.cpu.cores} vCPU${b.cpu.temp_c !== null ? ` · ${fmt(b.cpu.temp_c, 0, "°C")}` : ""}` : undefined}
      dim={dim} title={why ?? (b?.host.role === "server" ? "my VM's share of the server" : undefined)} />,
    ...(b?.host.role === "server"
      ? (cpus && cpus.length > 0
        ? cpus.map((c) => (
          <Chip key={`sock${c.socket}`} k={`Server CPU ${c.socket + 1}`} v={fmt(c.percent, 0, "%")}
            sub={`${c.temp_c !== null ? fmt(c.temp_c, 0, "°C") : "temp —"}${c.threads ? ` · ${c.threads} thr` : ""}`}
            dim={dim} title={why ?? "physical Xeon on the R740 (read from the Proxmox host)"} />))
        : [<Chip key="sockna" k="Server CPUs" v="—" dim title={b?.host_cpus_error ?? "no reading from the Proxmox host"} />])
      : []),
  ];
  const right = [
    <Chip key="ram" k="RAM" v={b?.ram ? `${fmt(b.ram.used_gb, 1)}/${fmt(b.ram.total_gb, 0)} GB` : "—"} dim={dim} title={why} />,
    <Chip key="gpu" k="GPU" v={g ? `${fmt(g.util_pct, 0, "%")}` : "—"}
      sub={g ? `${fmt(g.mem_used_mb / 1024, 1)}/${fmt(g.mem_total_mb / 1024, 0)} GB · ${fmt(g.temp_c, 0, "°C")}` : undefined}
      dim={dim} title={why ?? g?.name} />,
    <Chip key="ssd" k="SSD" v={b?.disk ? `${fmt(b.disk.used_gb, 0)}/${fmt(b.disk.total_gb, 0)} GB` : "—"}
      sub={b?.disk ? `R ${rate(b.disk.read_mb_s)} · W ${rate(b.disk.write_mb_s)} MB/s` : undefined} dim={dim} title={why} />,
    <Chip key="net" k="Network" v={b?.net ? `↓${rate(b.net.rx_mbps)} ↑${rate(b.net.tx_mbps)}` : "—"} sub={b?.net ? `Mb/s · ${b.net.iface ?? "?"}` : undefined} dim={dim} title={why} />,
  ];
  return (
    <div className="sb-top" role="status">
      <div className="sb-side sb-left">{left}</div>
      {moodChip}
      <div className="sb-side sb-right">{right}</div>
    </div>
  );
}

const Dot = ({ ok }: { ok: boolean | null | undefined }) => (
  <span className="sb-dot" style={{ background: ok === true ? "#3ee68f" : ok === false ? "#e5534b" : "#556" }} />
);
const ago = (s?: number | null) => (s === undefined || s === null ? "" : s < 90 ? `${Math.round(s)}s ago` : `${Math.round(s / 60)} min ago`);

export function StatusCorner({ sb }: { sb: ReturnType<typeof useStatusBoard> }) {
  const { board: b, stale, err } = sb;
  const [reach, setReach] = useState<Reach | null>(null);
  const [reachErr, setReachErr] = useState("");
  const [msg, setMsg] = useState("");
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const r = JSON.parse(await invoke<string>("server_reach")) as Reach; if (alive) { setReach(r); setReachErr(""); } }
      catch (e) { if (alive) { setReach(null); setReachErr(String(e).slice(0, 100)); } }
    };
    void tick();
    const iv = setInterval(() => void tick(), 10000);
    return () => { alive = false; clearInterval(iv); };
  }, []);
  const openConsole = async () => {
    try { await invoke("server_open", { kind: "console" }); setMsg(""); } catch (e) { setMsg(String(e)); }
  };
  const order = ["tower_ssh", "desktop_bridge", "wake_on_lan", "post_office", "mouth", "ears"];
  const links = b ? order.filter((k) => b.links[k]).map((k) => [k, b.links[k]] as const) : [];
  const rdetail = reachErr ? `the app couldn't check: ${reachErr}` : undefined;
  return (
    <div className="sb-corner">
      <div className="sb-corner-h">Server</div>
      <div className="sb-row" title={rdetail ?? "TCP connect from this PC to the Proxmox web port"}><Dot ok={reach ? reach.proxmox : null} />Proxmox host</div>
      <div className="sb-row" title={rdetail ?? "TCP connect from this PC to iris-home's SSH port"}><Dot ok={reach ? reach.iris_home_ssh : null} />iris-home reachable</div>
      <div className="sb-row" title={stale ? (err || "no fresh status from me") : `my status board answered${b?.camera_fps ? ` · camera ${b.camera_fps} fps` : ""}`}>
        <Dot ok={stale ? (b ? false : null) : b?.host.role === "server" ? true : null} />
        {b?.host.role === "tower" && !stale ? "Me: running on the tower" : "Me running on the server"}
      </div>
      <div className="sb-corner-h">Server ↔ tower</div>
      {links.length === 0 && <div className="sb-row sb-muted"><Dot ok={null} />{stale ? "no fresh status" : "not checked from this machine"}</div>}
      {links.map(([k, l]) => (
        <div key={k} className="sb-row" title={`${l.detail}${l.checked_ts ? ` · checked ${ago(l.age_s)}` : ""}`}>
          <Dot ok={stale ? null : l.ok} />{l.label}
          {!stale && l.ok === false ? <span className="sb-why"> · {l.detail.slice(0, 60)}</span> : null}
        </div>
      ))}
      <button type="button" className="btn primary sb-console" onClick={() => void openConsole()}>Open my console</button>
      {msg && <div className="sb-msg">{msg}</div>}
    </div>
  );
}
