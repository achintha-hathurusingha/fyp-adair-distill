# Run a long job while holding a wake lock, so the host cannot sleep under it.
#
#   powershell -ExecutionPolicy Bypass -File .\run_with_wakelock.ps1 -Command ".\scripts\run_cache.bat"
#
# WHY THIS EXISTS. On 2026-08-01 the teacher-cache job died silently at 21.9%:
# Kernel-Power event 42 ("system is entering sleep") at 12:42:53, one second
# after the last cached file was written. Sleep is set to "never" on AC but
# 10 minutes on battery, and a running Python job registers NO power request --
# `powercfg /requests` reports SYSTEM: None while training. So the machine
# sleeps out from under any unattended run whenever it is on battery.
#
# SetThreadExecutionState with ES_SYSTEM_REQUIRED | ES_CONTINUOUS tells Windows
# the system is in use for as long as this script lives, which survives being on
# battery. The flag is per-thread and is released automatically when the process
# exits, so a crash cannot leave the machine permanently unable to sleep.
#
# ES_DISPLAY_REQUIRED is deliberately NOT set: the screen may blank, only the
# system must stay awake.

param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$WorkingDir = "C:\Users\User\Documents\FYP\fyp-adair-distill"
)

$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$power = Add-Type -MemberDefinition $sig -Name PowerUtil -Namespace Win32 -PassThru

$ES_CONTINUOUS      = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001

$prev = $power::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
if ($prev -eq 0) {
    Write-Host "FAILED to acquire wake lock; refusing to start an unattended run." -ForegroundColor Red
    exit 1
}
Write-Host "wake lock acquired -- system will not sleep while this runs" -ForegroundColor Green
Write-Host "verify with:  powercfg /requests   (expect SYSTEM: [PROCESS] powershell.exe)"

Set-Location $WorkingDir
$started = Get-Date
Write-Host "started $($started.ToString('HH:mm:ss')): $Command"

try {
    & cmd.exe /c $Command
    $code = $LASTEXITCODE
} finally {
    # Always release, even on Ctrl+C or an exception -- otherwise the machine
    # keeps a stale lock until reboot.
    [void]$power::SetThreadExecutionState($ES_CONTINUOUS)
    $dur = (Get-Date) - $started
    Write-Host ("wake lock released after {0:hh\:mm\:ss}" -f $dur) -ForegroundColor Yellow
}

Write-Host "exit code: $code"
exit $code
