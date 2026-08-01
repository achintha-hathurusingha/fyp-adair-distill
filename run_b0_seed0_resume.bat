@echo off
REM Resume B0 seed 0 from its last checkpoint, picking up the bounded image
REM cache (data.cache_budget_gb). The original launch ran an unbounded cache
REM that would have exhausted RAM before the run finished.
REM
REM Launch detached:
REM   powershell -Command "Start-Process -FilePath '.\run_b0_seed0_resume.bat' -WindowStyle Hidden"
call C:\ProgramData\anaconda3\Scripts\activate.bat adair
cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
python -m src.train.train --arm B0 --seed 0 --out-root runs/b0 --resume runs/b0/B0/B0_seed0_20260801_094314/last.pth >> runs\b0_seed0.log 2>&1
