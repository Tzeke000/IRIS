# measure.ps1 — CPU of the orb app (iris-control + its WebView2 children) as % of the WHOLE tower CPU, 3 x 10 s.
# Run from the server: scripts/server/remote_ps.sh tower scripts/server/orb_app/measure.ps1  (2026-11-02, orb-cpu work)
$n=[Environment]::ProcessorCount
$app = Get-CimInstance Win32_Process -Filter "Name='iris-control.exe'"
"app: " + ($app | % { "$($_.ProcessId) $($_.ExecutablePath)" })
$wv = Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'" | ? { $_.CommandLine -match 'com.iris.control' }
$ids = @($app.ProcessId) + @($wv.ProcessId)
$types=@{}; foreach($w in $wv){ if($w.CommandLine -match '--type=([\w-]+)'){$types[$w.ProcessId]=$matches[1]}else{$types[$w.ProcessId]='browser'} }
foreach($a in $app){$types[$a.ProcessId]='APP'}
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W { [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h); [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);}
"@
foreach($p in (Get-Process -Id $app.ProcessId)){ "window: '$($p.MainWindowTitle)' hwnd=$($p.MainWindowHandle) iconic=$([W]::IsIconic($p.MainWindowHandle)) visible=$([W]::IsWindowVisible($p.MainWindowHandle))" }
$dur = 10
for($s=1;$s -le 3;$s++){
  $a=@{}; foreach($i in $ids){ $p=Get-Process -Id $i -EA SilentlyContinue; if($p){$a[$i]=$p.TotalProcessorTime.TotalMilliseconds} }
  Start-Sleep -Seconds $dur
  $tot=0; $line=@()
  foreach($i in $ids){ $p=Get-Process -Id $i -EA SilentlyContinue; if($p -and $a.ContainsKey($i)){ $pct=($p.TotalProcessorTime.TotalMilliseconds-$a[$i])/($dur*1000)/$n*100; $tot+=$pct; if($pct -gt 0.05){$line+= ("{0}:{1}={2:N1}%" -f $types[$i],$i,$pct)} } }
  "sample $s total={0:N1}%  {1}" -f $tot, ($line -join ' ')
}
