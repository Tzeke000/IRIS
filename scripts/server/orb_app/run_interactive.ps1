# run_interactive.ps1 NAME.ps1 [wait_s] — run D:\Wren-Companion\.tmp\orbcpu\NAME.ps1 in the LOGGED-ON user's session
# (ssh lands in session 0, which cannot see or touch his windows) via a one-shot scheduled task + run_hidden.vbs,
# so no console ever flashes over his game. Copy the script there first (scp ... tower:D:/Wren-Companion/.tmp/orbcpu/).
# args: <name of ps1 already in D:\Wren-Companion\.tmp\orbcpu\>  — runs it in the logged-on user's session, no window
$ps1 = "D:\Wren-Companion\.tmp\orbcpu\" + $args[0]
$tn = "IrisOrbCpu_" + [IO.Path]::GetFileNameWithoutExtension($ps1)
$a = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "//B //NoLogo `"D:\Wren-Companion\scripts\run_hidden.vbs`" `"$ps1`""
$p = New-ScheduledTaskPrincipal -UserId (whoami) -LogonType Interactive
Register-ScheduledTask -TaskName $tn -Action $a -Principal $p -Force | Out-Null
Start-ScheduledTask -TaskName $tn
Start-Sleep -Seconds ([int]($args[1] | ForEach-Object { if($_){$_}else{6} }))
Unregister-ScheduledTask -TaskName $tn -Confirm:$false
"ran $tn"
