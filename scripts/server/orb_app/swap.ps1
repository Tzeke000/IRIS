# swap.ps1 [EXE] — replace the running orb app with EXE (default: the fresh release build) WITHOUT disturbing Zeke:
# graceful WM_CLOSE (the app's exit handler stops its audio sink), leftover sinks killed, new exe launched, window put
# back where it was (monitor 2, maximized, BEHIND whatever covered it) and the foreground handed back to whatever had it.
# Runs in his session: scp to D:\Wren-Companion\.tmp\orbcpu\ then  remote_ps.sh tower run_interactive.ps1 swap.ps1 25
# Log: D:\Wren-Companion\.tmp\orbcpu\swap.log. Rollback = pass the .prev.exe path (e.g. iris-control.prev.exe).
# ⚠ The monitor-2 rectangle (1920,1 1920x1040) is Zeke's current layout — adjust if his screens change.
Add-Type @"
using System; using System.Text; using System.Runtime.InteropServices;
public class U {
 public delegate bool EP(IntPtr h, IntPtr l);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EP f, IntPtr l);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
 [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr h);
 [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
 [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
 [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
 [StructLayout(LayoutKind.Sequential)] public struct R { public int L,T,Rr,B; }
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
 public static IntPtr Find(uint pid, string title){ IntPtr found=IntPtr.Zero; EnumWindows((h,l)=>{ uint p; GetWindowThreadProcessId(h,out p); if(p==pid){ var sb=new StringBuilder(256); GetWindowText(h,sb,256); if(sb.ToString()==title){found=h; return false;} } return true; }, IntPtr.Zero); return found; }
}
"@
$HWND_BOTTOM=[IntPtr]1; $HWND_TOP=[IntPtr]0
$SWP_NOSIZE=0x1; $SWP_NOMOVE=0x2; $SWP_NOACTIVATE=0x10
function Log($m){ Add-Content D:\Wren-Companion\.tmp\orbcpu\swap.log ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss.fff), $m) }
function FgName(){ $fp=0; [void][U]::GetWindowThreadProcessId([U]::GetForegroundWindow(),[ref]$fp); (Get-Process -Id $fp -EA SilentlyContinue).ProcessName }
$exe = $args[0]; if (-not $exe) { $exe = "D:\Wren-Companion\apps\ava-control\src-tauri\target\release\iris-control.exe" }
$fg0 = [U]::GetForegroundWindow()
Log "=== swap -> $exe ; foreground=$(FgName)"
$old = Get-Process iris-control -EA SilentlyContinue | select -First 1
$zoom = $true; $rect = $null
if ($old) {
  $h = [U]::Find([uint32]$old.Id, "Iris")
  $r = New-Object U+R; [void][U]::GetWindowRect($h, [ref]$r); $zoom = [U]::IsZoomed($h)
  Log "old pid $($old.Id) hwnd $h zoomed=$zoom rect=$($r.L),$($r.T),$($r.Rr),$($r.B)"
  [void][U]::PostMessage($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)     # WM_CLOSE -> app exits + stops its sink
  if (-not $old.WaitForExit(10000)) { Log "graceful close timed out -> kill"; Stop-Process -Id $old.Id -Force }
  else { Log "old app exited gracefully" }
}
Start-Sleep -Milliseconds 800
$sinks = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | ? { $_.CommandLine -match 'audio_sink\.py' }
if ($sinks) { Log "leftover sink pids $($sinks.ProcessId -join ',') -> killing"; $sinks | % { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue } ; Start-Sleep -Milliseconds 500 }
$p = Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) -PassThru
Log "launched pid $($p.Id)"
$h=[IntPtr]::Zero; $sw=[Diagnostics.Stopwatch]::StartNew()
while ($sw.ElapsedMilliseconds -lt 15000) { $h=[U]::Find([uint32]$p.Id,"Iris"); if ($h -ne [IntPtr]::Zero -and [U]::IsWindowVisible($h)) { break }; Start-Sleep -Milliseconds 5 }
Log "main window $h visible after $($sw.ElapsedMilliseconds) ms; foreground=$(FgName)"
# back where it was: monitor 2 work area, maximized, BEHIND the window that covered it, never activated
[void][U]::SetWindowPos($h, $HWND_BOTTOM, 1920, 1, 1920, 1040, $SWP_NOACTIVATE)
if ($zoom) { [void][U]::ShowWindow($h, 3) }      # SW_MAXIMIZE (maximizes on the monitor it's on now)
[void][U]::SetWindowPos($h, $HWND_BOTTOM, 0, 0, 0, 0, $SWP_NOMOVE -bor $SWP_NOSIZE -bor $SWP_NOACTIVATE)
$r = New-Object U+R; [void][U]::GetWindowRect($h, [ref]$r)
Log "placed: rect=$($r.L),$($r.T),$($r.Rr),$($r.B) zoomed=$([U]::IsZoomed($h)) foreground=$(FgName)"
# 11-02: the new window took the foreground from his game despite the no-activate placement — hand it back
if ((FgName) -eq 'iris-control' -and $fg0 -ne [IntPtr]::Zero -and $fg0 -ne $h) {
  [U]::keybd_event(0x12,0,0,[UIntPtr]::Zero); [U]::keybd_event(0x12,0,2,[UIntPtr]::Zero)   # ALT tap unlocks SetForegroundWindow
  $ok = [U]::SetForegroundWindow($fg0); Start-Sleep -Milliseconds 200
  Log "gave the foreground back: $ok -> $(FgName)"
}
