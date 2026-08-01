# Live view of the teacher-pair caching job.
#
#   powershell -ExecutionPolicy Bypass -File .\watch_cache.ps1
#
# Progress comes from counting cached files, not from the log: the job buffers
# stdout, so runs\teacher_cache.log stays empty for long stretches even while
# work is happening. Rate is measured across this script's own samples, so the
# ETA reflects the job as it is running now.
#
# Ctrl+C to stop. Read-only -- it never touches the job.

param(
    [string]$PairsDir = "data\pairs",
    [int]$Target = 15632,          # derain 200 + denoise 5144 x 3 sigmas
    [int]$RefreshSeconds = 30
)

$ErrorActionPreference = "Continue"

function Get-Counts {
    param([string]$Dir)
    $out = [ordered]@{}
    $total = 0
    Get-ChildItem $Dir -Directory -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
        $n = @(Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
        $out[$_.Name] = $n
        $total += $n
    }
    [PSCustomObject]@{ PerTask = $out; Total = $total }
}

$startTime = Get-Date
$startCount = (Get-Counts -Dir $PairsDir).Total
$prevTime = $startTime
$prevCount = $startCount

while ($true) {
    Clear-Host
    $now = Get-Date
    $c = Get-Counts -Dir $PairsDir

    Write-Host "Teacher cache -- live  ($($now.ToString('HH:mm:ss')))" -ForegroundColor Cyan
    Write-Host ("=" * 74)

    $pct = 100.0 * $c.Total / $Target
    Write-Host ("progress : {0:N0} / {1:N0}  ({2:N1}%)" -f $c.Total, $Target, $pct) -ForegroundColor White
    $bars = [int]($pct / 2)
    Write-Host ("[" + ("#" * $bars) + ("." * (50 - $bars)) + "]")
    Write-Host ""

    Write-Host "by task:" -ForegroundColor Green
    foreach ($k in $c.PerTask.Keys) {
        Write-Host ("  {0,-14} {1,6:N0}" -f $k, $c.PerTask[$k])
    }
    Write-Host ""

    # Instantaneous rate (last interval) and average since this script started.
    $dN = $c.Total - $prevCount
    $dT = ($now - $prevTime).TotalMinutes
    $inst = $null
    if ($dT -gt 0 -and $dN -ge 0) { $inst = $dN / $dT }

    $aN = $c.Total - $startCount
    $aT = ($now - $startTime).TotalMinutes
    $avg = $null
    if ($aT -gt 0.5 -and $aN -gt 0) { $avg = $aN / $aT }

    Write-Host "rate:" -ForegroundColor Green
    if ($inst -ne $null) { Write-Host ("  last {0:N0}s : {1:N1} items/min" -f $RefreshSeconds, $inst) }
    if ($avg -ne $null) {
        Write-Host ("  average  : {0:N1} items/min  (since {1})" -f $avg, $startTime.ToString('HH:mm'))
        $remain = ($Target - $c.Total) / $avg
        $eta = $now.AddMinutes($remain)
        Write-Host ""
        Write-Host ("remaining: {0:N1} h" -f ($remain / 60)) -ForegroundColor White
        Write-Host ("ETA      : {0}" -f $eta.ToString('ddd HH:mm, dd MMM')) -ForegroundColor White
    } else {
        Write-Host "  average  : need ~1 min of samples for an ETA"
    }

    Write-Host ""
    Write-Host "health:" -ForegroundColor Green
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader) 2>$null
    Write-Host "  gpu   : $gpu"
    $py = Get-Process python -ErrorAction SilentlyContinue
    if ($py) {
        Write-Host ("  python: {0} proc(s)" -f $py.Count)
    } else {
        Write-Host "  python: NO CACHE JOB RUNNING" -ForegroundColor Red
    }
    $sz = (Get-ChildItem $PairsDir -Recurse -File -ErrorAction SilentlyContinue |
           Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("  on disk: {0:N2} GB" -f $sz)

    $prevTime = $now
    $prevCount = $c.Total

    if ($c.Total -ge $Target) {
        Write-Host ""
        Write-Host "COMPLETE" -ForegroundColor Green
        break
    }

    Write-Host ""
    Write-Host "refreshing every ${RefreshSeconds}s -- Ctrl+C to stop" -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
