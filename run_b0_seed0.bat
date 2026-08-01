@echo off
REM B0 seed 0 -- launched ALONE first, per the staged-launch protocol. Seeds 1
REM and 2 wait until this one is confirmed healthy at ~20k iterations.
REM
REM Launch detached (a background job started from a tool call does not survive
REM the session -- see reports/overnight_report.md, process lessons):
REM   powershell -Command "Start-Process -FilePath '.\run_b0_seed0.bat' -WindowStyle Hidden"
REM
REM Every run setting comes from configs/train/b0_baseline.yaml; the CLI cannot
REM override them. --iters/--batch-size are deliberately not passed.
call C:\ProgramData\anaconda3\Scripts\activate.bat adair
cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
python -m src.train.train --arm B0 --seed 0 --out-root runs/b0 >> runs\b0_seed0.log 2>&1
