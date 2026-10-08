# mic_watchdog.ps1 — keep my EARS' link alive (2026-10-08, Zeke: "can we create a health check for that, so you will know?")
#
# The tower's scripts/mic_source.py streams his headset mic to my ears on the server (port 8776). On 10-08 it died
# silently at 13:14 after a PortAudio error and the Iris-Mic-Source task's restart-on-failure never brought it back,
# so I was deaf all afternoon. This runs every minute (task Iris-Mic-Watchdog, via run_hidden.vbs = no console flash):
# if nothing is LISTENING on 8776 it clears any stuck copies and re-runs Iris-Mic-Source, then logs what it did.
# state\mic_watchdog.json = the latest verdict (the server's status board reads it).
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'state\mic_watchdog.log'
$stateFile = Join-Path $root 'state\mic_watchdog.json'
function Say($m) { "[mic_watchdog $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Out-File -FilePath $log -Append -Encoding utf8 }
function Listening { [bool](Get-NetTCPConnection -LocalPort 8776 -State Listen -ErrorAction SilentlyContinue) }

$prev = $null
if (Test-Path $stateFile) { try { $prev = Get-Content $stateFile -Raw | ConvertFrom-Json } catch { } }
$restarts = if ($prev -and $prev.restarts) { [int]$prev.restarts } else { 0 }
$lastRestart = if ($prev) { $prev.last_restart } else { $null }

if (Listening) {
    $ok = $true; $detail = 'listening on 8776'
} else {
    Say 'mic link DOWN (nothing listening on 8776) - restarting Iris-Mic-Source'
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*mic_source.py*' } |
        ForEach-Object { Say "  clearing stuck copy pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
    schtasks /run /tn Iris-Mic-Source | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 10; $i++) { Start-Sleep -Seconds 1; if (Listening) { $ok = $true; break } }
    $restarts++; $lastRestart = (Get-Date).ToString('o')
    $detail = if ($ok) { 'was down - restarted, listening again' } else { 'was down - restart did NOT bring it back' }
    Say $detail
}
$streaming = [bool](Get-NetTCPConnection -LocalPort 8776 -State Established -ErrorAction SilentlyContinue)
@{ ts = (Get-Date).ToString('o'); epoch = [DateTimeOffset]::Now.ToUnixTimeSeconds(); ok = $ok; detail = $detail; streaming = $streaming; restarts = $restarts;
   last_restart = $lastRestart } | ConvertTo-Json | Set-Content -Path $stateFile -Encoding ascii
