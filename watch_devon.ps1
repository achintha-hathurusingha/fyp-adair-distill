# Live view of the two B0 divergence experiments running on devon.
#
#   powershell -ExecutionPolicy Bypass -File .\watch_devon.ps1
#
# The question these runs answer is NOT "did it reach 300k" -- it is whether the
# max gradient norm curve FLATTENS or merely shifts right. The pre-divergence
# trend doubled every ~5k steps (0.708 -> 1.439 -> 2.803 -> 6.5e7), so maxgn is
# printed and colour-coded rather than buried in the log line.
#
#   QA-control  full LayerNorm2d, clip 8.0, from scratch -> is the growth N-F-specific?
#   NF-retry    N-F, clip 1.0, resumed from 20k          -> does a tighter clip fix it?
#
# The query lives on devon as status.sh rather than being sent inline: a
# PowerShell here-string carries CRLF line endings, which bash rejects
# ("cd: $'...': No such file or directory"), and the quoting does not survive
# argument passing intact. Calling a file on the remote avoids both problems.
#
# Ctrl+C to stop. Read-only over SSH; it never touches the runs.

param(
    [string]$KeyPath = "C:\Users\User\Documents\FYP\Achintha",
    [string]$Target  = "minura@192.248.10.68",
    [int]$RefreshSeconds = 60
)

$ErrorActionPreference = "Continue"

while ($true) {
    Clear-Host
    $now = Get-Date
    Write-Host "devon -- B0 divergence experiments  ($($now.ToString('HH:mm:ss')))" -ForegroundColor Cyan
    Write-Host ("=" * 92)

    $out = ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -i $KeyPath $Target "bash ~/fyp-adair-distill/status.sh" 2>&1

    $section = ""
    foreach ($line in $out) {
        $t = "$line"

        if ($t.StartsWith("###")) {
            $section = $t.Substring(3)
            if ($section -ne "GPU" -and $section -ne "PROC") {
                Write-Host ""
                Write-Host $section -ForegroundColor Green
            }
            continue
        }

        if ($section -eq "GPU")  { Write-Host ""; Write-Host "GPU  : $t" -ForegroundColor DarkGray; continue }
        if ($section -eq "PROC") { Write-Host "procs: $t python" -ForegroundColor DarkGray; continue }

        if ($t -match 'diverged')     { Write-Host "  $t" -ForegroundColor Red;      continue }
        if ($t -match 'non-finite')   { Write-Host "  $t" -ForegroundColor Yellow;   continue }
        if ($t -match 'resumed from') { Write-Host "  $t" -ForegroundColor DarkGray; continue }

        # Reformat a validation line around the maxgn trend.
        if ($t -match 'maxgn') {
            $itNum = 0; $gn = 0.0; $loss = ""; $psnr = ""; $skip = 0
            if ($t -match '\bit\s+(\d+)')         { $itNum = [int]$Matches[1] }
            if ($t -match 'maxgn\s+([0-9.eE+]+)') { $gn    = [double]$Matches[1] }
            if ($t -match 'loss\s+([0-9.]+)')     { $loss  = $Matches[1] }
            if ($t -match 'psnr\s+([0-9.]+)')     { $psnr  = $Matches[1] }
            if ($t -match 'skip\s+(\d+)')         { $skip  = [int]$Matches[1] }

            $colour = "Gray"
            if ($gn -gt 100)   { $colour = "Red" }
            elseif ($gn -gt 5) { $colour = "Yellow" }
            elseif ($gn -gt 1) { $colour = "White" }

            $msg = "  it {0,7}   loss {1,-9}  psnr {2,-8}  maxgn {3}" -f $itNum, $loss, $psnr, $gn
            if ($skip -gt 0) { $msg += "   SKIPPED $skip" }
            Write-Host $msg -ForegroundColor $colour
            continue
        }

        if ($t.Trim()) { Write-Host "  $t" }
    }

    Write-Host ""
    Write-Host "maxgn: white >1, yellow >5, RED >100   |   pre-fix trend 0.708 -> 1.439 -> 2.803 -> 6.5e7" -ForegroundColor DarkGray
    Write-Host "refreshing every ${RefreshSeconds}s -- Ctrl+C to stop" -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
