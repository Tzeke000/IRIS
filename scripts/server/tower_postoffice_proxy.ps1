# scripts/server/tower_postoffice_proxy.ps1 - run ON THE TOWER (elevated / admin SSH session).
# The post-office moved to the server 2026-10-07 (scripts/server/install_postoffice.sh). Wren's
# laptop still sends letters to the TOWER's :5877, so the tower forwards that port to the server
# until Wren's own URL is switched. Address comes in as a parameter - never tracked in the repo.
#   powershell -NoProfile -File tower_postoffice_proxy.ps1 -Server <server address>
#   powershell -NoProfile -File tower_postoffice_proxy.ps1 -Remove
param([string]$Server = "", [switch]$Remove)
$ErrorActionPreference = "Stop"
netsh interface portproxy delete v4tov4 listenport=5877 listenaddress=0.0.0.0 2>$null | Out-Null
if ($Remove) {
    Remove-NetFirewallRule -DisplayName "Iris post-office proxy 5877" -ErrorAction SilentlyContinue
    "portproxy removed"
    exit 0
}
if (-not $Server) { throw "-Server <address> is required" }
# a leftover tower post-office would hold the port; it must not run beside the proxy
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match "sibling_postoffice\.py" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped tower post-office pid $($_.ProcessId)" }
Set-Service iphlpsvc -StartupType Automatic
Start-Service iphlpsvc
netsh interface portproxy add v4tov4 listenport=5877 listenaddress=0.0.0.0 connectport=5877 connectaddress=$Server | Out-Null
if (-not (Get-NetFirewallRule -DisplayName "Iris post-office proxy 5877" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "Iris post-office proxy 5877" -Direction Inbound -Protocol TCP -LocalPort 5877 -Action Allow -Profile Any | Out-Null
}
"portproxy 0.0.0.0:5877 -> server:5877 installed"
