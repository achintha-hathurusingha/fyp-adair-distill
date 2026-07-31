@echo off
REM M spot-check runner — launched detached via PowerShell Start-Process so it
REM survives the parent shell exiting. Earlier attempts used `nohup ... &` from
REM a Bash tool call, and every one of them died when that session ended.
REM
REM Runs M-F (resuming if a checkpoint exists) then M-A, each alone on the GPU.

cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
call C:\ProgramData\anaconda3\Scripts\activate.bat adair

echo === M-F start %TIME% >> reports\spotcheck_runner.log

REM Resume M-F from its last checkpoint if one exists, else start fresh.
set MF_CKPT=
for /f "delims=" %%d in ('dir /b /ad /o-n runs\1p5b\M-F 2^>nul') do (
    if exist "runs\1p5b\M-F\%%d\last.pth" set MF_CKPT=runs\1p5b\M-F\%%d\last.pth
)

if defined MF_CKPT (
    echo resuming M-F from %MF_CKPT% >> reports\spotcheck_runner.log
    python -m src.train.train --arm M-F --iters 10000 --batch-size 16 --patch-size 128 --num-workers 6 --resume %MF_CKPT% > runs\1p5b_M-F.log 2>&1
) else (
    python -m src.train.train --arm M-F --iters 10000 --batch-size 16 --patch-size 128 --num-workers 6 > runs\1p5b_M-F.log 2>&1
)
echo === M-F done %TIME% exit=%ERRORLEVEL% >> reports\spotcheck_runner.log

REM Let CUDA fully release before the next arm — back-to-back launches OOM'd.
timeout /t 30 /nobreak > nul

echo === M-A start %TIME% >> reports\spotcheck_runner.log
python -m src.train.train --arm M-A --iters 10000 --batch-size 16 --patch-size 128 --num-workers 6 > runs\1p5b_M-A.log 2>&1
echo === M-A done %TIME% exit=%ERRORLEVEL% >> reports\spotcheck_runner.log

echo M_SPOTCHECK_BOTH_DONE >> reports\spotcheck_runner.log
