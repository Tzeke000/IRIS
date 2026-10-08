#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::time::Duration;
use tauri::Manager;

// ── Server panel (Zeke 2026-10-06): the tower app stays the front door after my cognition moves to
// the R740. Bring the VMs up/down through the Proxmox API (its OWN token: iris-tower@pve!panel, role
// IrisLauncher = VM.PowerMgmt + VM.Audit on /vms/100-102 only), open an SSH session to iris-home, or
// attach to my live console there (tmux session `iris`, via ~/iris_console.sh on iris-home).
//
// The token never touches a command line or this binary: curl reads the Authorization header from a
// file (`-H @file`). Default path below; override with IRIS_PVE_HEADER.
//
// Host addresses come from the git-ignored config/private.local.json (2026-10-07: the repo is
// PUBLIC, so the home network layout stays out of source). Override the path with IRIS_PRIVATE_CONFIG.
// LINUX (2026-10-07, the Zorin build — Zeke: "make the iris app on zorin as well"): the same
// two git-ignored files live in ~/.config/iris/ (chmod 600). No voice player there: my mouth streams
// to ONE sink, the tower's.
#[cfg(windows)]
const PRIVATE_CFG: &str = r"D:\Wren-Companion\config\private.local.json";
#[cfg(windows)]
const DEFAULT_HEADER: &str = r"D:\Wren-Companion\state\secrets\pve_tower.hdr";

fn home_cfg(name: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/.config/iris/{name}")
}
fn private_cfg_path() -> String {
    #[cfg(windows)]
    { PRIVATE_CFG.to_string() }
    #[cfg(not(windows))]
    { home_cfg("private.local.json") }
}
fn default_header_path() -> String {
    #[cfg(windows)]
    { DEFAULT_HEADER.to_string() }
    #[cfg(not(windows))]
    { home_cfg("pve_panel.hdr") }
}

/// Background helpers must never flash a console over his game (Windows); a no-op elsewhere.
trait Quiet {
    fn quiet(&mut self) -> &mut Self;
}
impl Quiet for Command {
    #[cfg(windows)]
    fn quiet(&mut self) -> &mut Self {
        self.creation_flags(CREATE_NO_WINDOW)
    }
    #[cfg(not(windows))]
    fn quiet(&mut self) -> &mut Self {
        self
    }
}
#[cfg(windows)]
const SSH: &str = r"C:\Windows\System32\OpenSSH\ssh.exe";
#[cfg(not(windows))]
const SSH: &str = "ssh";
#[cfg(windows)]
const CURL: &str = "curl.exe";
#[cfg(not(windows))]
const CURL: &str = "curl";
const ALLOWED_VMS: [u32; 3] = [100, 101, 102];
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000; // background helpers must never flash a console over his game

// ── My voice plays through THIS app (Zeke 2026-10-07: "only through the app", "the app should stay up
// if you're on"). My mouth runs on the server's V100 and streams PCM to TCP 8775; the app owns the
// player for exactly its own lifetime: scripts/audio_sink.py starts hidden with the app and is killed
// (process TREE - the venv pythonw stub re-execs a child) when the app exits. Allowlist + firewall
// rule keep it server-only.
#[cfg(windows)]
const SINK_PYTHONW: &str = r"D:\Wren-Companion\.venv\Scripts\pythonw.exe";
#[cfg(windows)]
const SINK_SCRIPT: &str = r"D:\Wren-Companion\scripts\audio_sink.py";

struct VoicePlayer(std::sync::Mutex<Option<std::process::Child>>);

#[cfg(windows)]
fn start_voice_player() -> Option<std::process::Child> {
    Command::new(SINK_PYTHONW)
        .arg(SINK_SCRIPT)
        .current_dir(r"D:\Wren-Companion")
        .quiet()
        .spawn()
        .ok()
}
#[cfg(not(windows))]
fn start_voice_player() -> Option<std::process::Child> {
    None
}

fn stop_voice_player(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill.exe")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .quiet()
            .output();
    }
    #[cfg(not(windows))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}

/// `"key": "value"` from the private config — a tiny flat-JSON read, no serde dependency needed.
fn private(key: &str) -> String {
    let path = std::env::var("IRIS_PRIVATE_CONFIG").unwrap_or_else(|_| private_cfg_path());
    let txt = std::fs::read_to_string(path).unwrap_or_default();
    let pat = format!("\"{key}\"");
    txt.find(&pat)
        .and_then(|i| txt[i + pat.len()..].split('"').nth(1))
        .unwrap_or("")
        .to_string()
}

fn pve_host() -> String {
    private("proxmox_host")
}

fn iris_home() -> String {
    format!("iris@{}", private("iris_home_host"))
}

fn header_file() -> String {
    std::env::var("IRIS_PVE_HEADER").unwrap_or_else(|_| default_header_path())
}

fn pve(method: &str, path: &str) -> Result<String, String> {
    let hdr = header_file();
    if !std::path::Path::new(&hdr).is_file() {
        return Err(format!("Proxmox token file missing ({hdr})"));
    }
    let url = format!("https://{}:8006/api2/json{path}", pve_host());
    let out = Command::new(CURL)
        .args(["-sk", "-m", "12", "-X", method, "-H", &format!("@{hdr}"), "-w", "\n%{http_code}", &url])
        .quiet()
        .output()
        .map_err(|e| format!("curl failed to start: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let (body, code) = text.rsplit_once('\n').unwrap_or((&text, "000"));
    if code.trim() != "200" {
        return Err(format!("Proxmox answered HTTP {} {}", code.trim(), body.chars().take(200).collect::<String>()));
    }
    Ok(body.to_string())
}

/// VM list (vmid, name, status, node) — raw Proxmox JSON for the panel to parse.
#[tauri::command]
fn server_vms() -> Result<String, String> {
    pve("GET", "/cluster/resources?type=vm")
}

/// start | shutdown (graceful) one of VMs 100-102. The node is looked up, never assumed.
#[tauri::command]
fn server_vm_power(vmid: u32, action: String) -> Result<String, String> {
    if !ALLOWED_VMS.contains(&vmid) {
        return Err(format!("VM {vmid} is not on the panel"));
    }
    if action != "start" && action != "shutdown" {
        return Err(format!("unknown action {action}"));
    }
    let list = pve("GET", "/cluster/resources?type=vm")?;
    let needle = format!("\"vmid\":{vmid}");
    let node = list
        .split('{')
        .find(|chunk| chunk.contains(&needle))
        .and_then(|chunk| chunk.split("\"node\":\"").nth(1))
        .and_then(|rest| rest.split('"').next())
        .ok_or_else(|| format!("VM {vmid} not found on the cluster"))?
        .to_string();
    pve("POST", &format!("/nodes/{node}/qemu/{vmid}/status/{action}"))
}

fn reach(addr: &str) -> bool {
    addr.parse::<SocketAddr>()
        .ok()
        .map(|a| TcpStream::connect_timeout(&a, Duration::from_millis(900)).is_ok())
        .unwrap_or(false)
}

/// Quick TCP reachability: Proxmox UI, iris-home SSH, and whether a me is serving on iris-home.
#[tauri::command]
fn server_reach() -> String {
    format!(
        "{{\"proxmox\":{},\"iris_home_ssh\":{},\"iris_home_runtime\":{}}}",
        reach(&format!("{}:8006", pve_host())),
        reach(&format!("{}:22", private("iris_home_host"))),
        reach(&format!("{}:5876", private("iris_home_host")))
    )
}

#[cfg(windows)]
const CHROME: &str = r"C:\Program Files\Google\Chrome\Application\chrome.exe";

/// Chrome's visible top-level windows (2026-10-08, the guide's Proxmox demo). `snap` remembers which windows
/// exist BEFORE I open mine; `close` posts WM_CLOSE only to windows that are NEW since the snap AND whose title
/// is the Proxmox page (or its certificate warning) — so a Chrome window Zeke had open is never touched.
#[cfg(windows)]
fn chrome_windows(mode: &str) -> Result<(), String> {
    const PS: &str = r#"param($mode)
$snap = Join-Path $env:TEMP 'iris_pve_guide_before.txt'
Add-Type -TypeDefinition 'using System;using System.Text;using System.Runtime.InteropServices;public static class IrisW{public delegate bool P(IntPtr h,IntPtr l);[DllImport("user32.dll")]public static extern bool EnumWindows(P p,IntPtr l);[DllImport("user32.dll",CharSet=CharSet.Unicode)]public static extern int GetWindowText(IntPtr h,StringBuilder s,int n);[DllImport("user32.dll")]public static extern bool IsWindowVisible(IntPtr h);[DllImport("user32.dll")]public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);[DllImport("user32.dll")]public static extern bool PostMessage(IntPtr h,uint m,IntPtr w,IntPtr l);}'
$ids = @(Get-Process chrome -ErrorAction SilentlyContinue | ForEach-Object { [uint32]$_.Id })
$global:wins = @()
$cb = [IrisW+P]{ param($h, $l)
  if ([IrisW]::IsWindowVisible($h)) {
    $p = [uint32]0; [void][IrisW]::GetWindowThreadProcessId($h, [ref]$p)
    if ($ids -contains $p) { $sb = New-Object System.Text.StringBuilder 512; [void][IrisW]::GetWindowText($h, $sb, 512)
      if ($sb.Length -gt 0) { $global:wins += [pscustomobject]@{ h = [int64]$h; t = $sb.ToString() } } } }
  $true }
[void][IrisW]::EnumWindows($cb, [IntPtr]::Zero)
if ($mode -eq 'list') { $global:wins | ForEach-Object { "$($_.h) $($_.t)" }; exit 0 }
if ($mode -eq 'snap') { ($global:wins | ForEach-Object { $_.h }) -join "`n" | Set-Content -Path $snap -Encoding ascii; exit 0 }
$before = @(); if (Test-Path $snap) { $before = @(Get-Content $snap | Where-Object { $_ } | ForEach-Object { [int64]$_ }) }
foreach ($w in $global:wins) {
  if (($before -notcontains $w.h) -and ($w.t -match 'Proxmox|Privacy error|not private|:8006')) { [void][IrisW]::PostMessage([IntPtr]$w.h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) }
}
"#;
    let tmp = std::env::var("TEMP").unwrap_or_else(|_| r"C:\Windows\Temp".into());
    let script = format!(r"{tmp}\iris_chrome_windows.ps1");
    std::fs::write(&script, PS).map_err(|e| format!("could not write helper: {e}"))?;
    Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script.as_str(), mode])
        .quiet()
        .status()
        .map(|_| ())
        .map_err(|e| format!("chrome windows {mode}: {e}"))
}

/// Close a window that server_open opened (2026-10-08: the guide opens my console to show it, then closes it
/// itself). console/ssh: kill the local ssh client that window runs (marker on its command line), and the
/// terminal window goes with it. My session on the server is untouched (the console only ATTACHES to it).
#[tauri::command]
fn server_close(kind: String) -> Result<(), String> {
    // what to match on the process command line (each window server_open made carries one)
    let (win_proc, pat) = match kind.as_str() {
        "console" => ("ssh.exe", "*iris_console.sh*"),
        "ssh" => ("ssh.exe", "*ServerAliveInterval=29*"),
        "proxmox" => ("chrome.exe", ""),
        _ => return Err(format!("can't close {kind}")),
    };
    #[cfg(windows)]
    {
        if kind == "proxmox" {
            return chrome_windows("close");
        }
        let ps = format!("Get-CimInstance Win32_Process -Filter \"Name='{win_proc}'\" | Where-Object {{ $_.CommandLine -like '{pat}' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}");
        Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", ps.as_str()])
            .quiet()
            .status()
            .map(|_| ())
            .map_err(|e| format!("could not close {kind}: {e}"))
    }
    #[cfg(not(windows))]
    {
        let _ = win_proc;
        let _ = pat;
        if kind == "proxmox" {   // a window in his normal browser: close it by title if wmctrl is around, else leave it
            let _ = Command::new("wmctrl").args(["-c", "Proxmox"]).status();
            return Ok(());
        }
        let lin = match kind.as_str() { "console" => "iris_console.sh", _ => "ServerAliveInterval=29" };
        Command::new("pkill").args(["-f", lin]).status()
            .map(|_| ())
            .map_err(|e| format!("could not close {kind}: {e}"))
    }
}

/// ssh = a shell on iris-home · console = attach to my live session there · proxmox = the web UI.
#[tauri::command]
fn server_open(kind: String) -> Result<(), String> {
    let home = iris_home();
    let pve_ui = format!("https://{}:8006/", pve_host());
    #[cfg(windows)]
    {
        let mut cmd = Command::new("cmd.exe");
        match kind.as_str() {
            // ServerAliveInterval=29 is a harmless MARKER so server_close("ssh") can find exactly this window's ssh
            "ssh" => cmd.args(["/c", "start", "Iris - server shell", SSH, "-o", "ServerAliveInterval=29", "-t", home.as_str()]),
            "console" => cmd.args(["/c", "start", "Iris - my console", SSH, "-t", home.as_str(), "~/iris_console.sh"]),
            "proxmox" => cmd.args(["/c", "start", "", pve_ui.as_str()]),
            // the guide's demo (Zeke 10-08): a normal window in the Chrome he already runs (not an app window,
            // not a throwaway profile). Snapshot Chrome's windows FIRST, so server_close closes only the one I made.
            "proxmox_guide" => {
                let _ = chrome_windows("snap");
                if std::path::Path::new(CHROME).exists() {
                    cmd.args(["/c", "start", "", CHROME, "--new-window", pve_ui.as_str()])
                } else {
                    cmd.args(["/c", "start", "", pve_ui.as_str()])
                }
            }
            _ => return Err(format!("unknown target {kind}")),
        };
        // `start` opens the visible window we WANT; the helper cmd itself stays hidden.
        cmd.quiet()
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("could not open {kind}: {e}"))
    }
    #[cfg(not(windows))]
    {
        if kind == "proxmox" {
            return Command::new("xdg-open").arg(&pve_ui).spawn().map(|_| ())
                .map_err(|e| format!("could not open the Proxmox page: {e}"));
        }
        if kind == "proxmox_guide" {   // his normal browser, like the real button
            return Command::new("xdg-open").arg(&pve_ui).spawn().map(|_| ())
                .map_err(|e| format!("could not open the Proxmox page: {e}"));
        }
        let remote: Vec<&str> = match kind.as_str() {
            "ssh" => vec![SSH, "-o", "ServerAliveInterval=29", "-t", home.as_str()],
            "console" => vec![SSH, "-t", home.as_str(), "~/iris_console.sh"],
            _ => return Err(format!("unknown target {kind}")),
        };
        // first terminal that exists wins (Zorin ships gnome-terminal)
        for term in ["gnome-terminal", "x-terminal-emulator", "konsole", "xterm"] {
            let mut cmd = Command::new(term);
            if term == "gnome-terminal" { cmd.arg("--"); } else { cmd.arg("-e"); }
            if cmd.args(&remote).spawn().is_ok() {
                return Ok(());
            }
        }
        Err("no terminal emulator found".to_string())
    }
}

/// Start me ON THE SERVER (Zeke 2026-10-07: buttons in the app "so that I won't even have to log
/// into Zorin"). mode = cli | opus | fable. Runs scripts/server/iris_start_detached.sh over SSH:
/// the ONE-OF-ME gate decides first and its verdict text comes back to the panel either way.
#[tauri::command]
fn server_start_iris(mode: String) -> Result<String, String> {
    if !["cli", "opus", "fable"].contains(&mode.as_str()) {
        return Err(format!("unknown mode {mode}"));
    }
    let remote = format!("~/IRIS/scripts/server/iris_start_detached.sh {mode}");
    let out = Command::new(SSH)
        .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=6", iris_home().as_str(), remote.as_str()])
        .quiet()
        .output()
        .map_err(|e| format!("ssh failed to start: {e}"))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let text = text.trim().to_string();
    if out.status.success() { Ok(text) } else { Err(text) }
}

fn main() {
    tauri::Builder::default()
        // Single-instance guard (MUST be the first plugin registered, per Tauri).
        // On a second launch the new process hands off to the running one and exits;
        // we focus/restore the existing "main" window instead of stacking another
        // orphan. Root cause of the 9-instances-broken-app diagnosis 2026-06-29.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        // start the player in setup - AFTER the single-instance plugin has turned a second launch into
        // a hand-off, so a relaunch never spawns (and orphans) a second player
        .setup(|app| {
            app.manage(VoicePlayer(std::sync::Mutex::new(start_voice_player())));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![server_vms, server_vm_power, server_reach, server_open, server_close, server_start_iris])
        .build(tauri::generate_context!())
        .expect("error while building Iris Control")
        .run(|app, event| {
            // Closing the MAIN window ends the whole app (Zeke 2026-10-07: "when you exit the app it
            // closes all things about the app"). Before this, the hidden always-on-top widget window
            // kept the process alive invisibly after he closed the app, and the single-instance guard
            // then swallowed every relaunch — the app "wouldn't open", even as admin.
            if let tauri::RunEvent::WindowEvent { label, event: tauri::WindowEvent::CloseRequested { .. }, .. } = &event {
                if label == "main" {
                    app.exit(0);
                }
            }
            if let tauri::RunEvent::Exit = event {
                if let Some(vp) = app.try_state::<VoicePlayer>() {
                    if let Ok(mut g) = vp.0.lock() {
                        if let Some(mut child) = g.take() {
                            stop_voice_player(&mut child);
                        }
                    }
                }
            }
        });
}
