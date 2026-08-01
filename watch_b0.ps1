# Live view of the B0 run.
#
# The trainer only logs at validation (every 5000 iterations, ~50 min), so
# tailing the log alone looks dead most of the time. This estimates the current
# iteration from the measured step rate and shows GPU/RAM health alongside it.
#
#   powershell -ExecutionPolicy Bypass -File .\watch_b0.ps1
#
# Ctrl+C to stop. Read-only -- it never touches the run.

param(
    [string]$RunDir = "runs\b0\B0\B0_seed0_20260801_094314",
    [int]$RefreshSeconds = 20
)

$ErrorActionPreference = "Continue"
$logPath = Join-Path $RunDir "train.log"

if (-not (Test-Path $logPath)) {
    Write-Host "No log at $logPath" -ForegroundColor Red
    exit 1
}

function Get-ItLines {
    param([string]$Path)
    # Copy first: the trainer holds the log open for writing.
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        Copy-Item $Path $tmp -Force
        Get-Content $tmp | Where-Object { $_ -match '\|\s+train\s+\|\s+it\s+\d+' }
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Parse-Row {
    param([string]$Line)
    $ts = $null
    if ($Line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})') {
        $ts = [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', $null)
    }
    $it = $null
    if ($Line -match '\bit\s+(\d+)') { $it = [int]$Matches[1] }
    [PSCustomObject]@{ Time = $ts; Iter = $it; Text = $Line }
}

while ($true) {
    Clear-Host
    $now = Get-Date
    Write-Host "B0 seed 0 -- live  ($($now.ToString('HH:mm:ss')))" -ForegroundColor Cyan
    Write-Host ("=" * 78)

    $rows = @(Get-ItLines -Path $logPath | ForEach-Object { Parse-Row $_ })
    $total = 300000

    if ($rows.Count -eq 0) {
        Write-Host "No validation logged yet (first lands at iteration 5000)." -ForegroundColor Yellow
        $lastIter = 0
        $rate = $null
    } else {
        $last = $rows[-1]
        $lastIter = $last.Iter
        Write-Host "last validation:" -ForegroundColor Green
        Write-Host "  $($last.Text)"

        # Step rate from the two most recent validations; that interval is
        # steady-state, unlike the first one which includes worker startup.
        $rate = $null
        if ($rows.Count -ge 2) {
            $prev = $rows[-2]
            $dIt = $last.Iter - $prev.Iter
            $dT = ($last.Time - $prev.Time).TotalSeconds
            if ($dT -gt 0) { $rate = $dIt / $dT }
        }
    }

    Write-Host ""
    if ($rate) {
        $elapsed = ($now - $rows[-1].Time).TotalSeconds
        $est = [math]::Min($total, $lastIter + [int]($rate * $elapsed))
        $remain = ($total - $est) / $rate
        $eta = $now.AddSeconds($remain)
        $pct = 100.0 * $est / $total
        Write-Host ("progress : ~{0:N0} / {1:N0}  ({2:N1}%)" -f $est, $total, $pct) -ForegroundColor White
        Write-Host ("rate     : {0:N2} steps/s   (measured, last interval)" -f $rate)
        Write-Host ("remaining: {0:N1} h" -f ($remain / 3600))
        Write-Host ("ETA      : {0}" -f $eta.ToString('ddd HH:mm, dd MMM')) -ForegroundColor White
        $bars = [int]($pct / 2)
        Write-Host ("[" + ("#" * $bars) + ("." * (50 - $bars)) + "]")
    } else {
        Write-Host ("progress : {0:N0} / {1:N0} logged; need a second validation for a rate" -f $lastIter, $total)
    }

    Write-Host ""
    Write-Host "health:" -ForegroundColor Green
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader) 2>$null
    Write-Host "  gpu     : $gpu"

    $py = Get-Process python -ErrorAction SilentlyContinue
    if ($py) {
        $priv = ($py | Measure-Object PM -Sum).Sum / 1GB
        Write-Host ("  python  : {0} procs, {1:N2} GB private" -f $py.Count, $priv)
    } else {
        Write-Host "  python  : NO TRAINER RUNNING" -ForegroundColor Red
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $free = $os.FreePhysicalMemory / 1MB
    $colour = "Gray"
    if ($free -lt 1.0) { $colour = "Red" } elseif ($free -lt 2.0) { $colour = "Yellow" }
    Write-Host ("  ram free: {0:N2} GB of {1:N2} GB" -f $free, ($os.TotalVisibleMemorySize / 1MB)) -ForegroundColor $colour

    # Anything that would end the run.
    $tmp = [System.IO.Path]::GetTempFileName()
    Copy-Item $logPath $tmp -Force
    $bad = Get-Content $tmp | Where-Object { $_ -match 'diverged|Traceback|MemoryError|CUDA out of memory' }
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    if ($bad) {
        Write-Host ""
        Write-Host "PROBLEM:" -ForegroundColor Red
        $bad | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }

    Write-Host ""
    Write-Host "refreshing every ${RefreshSeconds}s -- Ctrl+C to stop" -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
