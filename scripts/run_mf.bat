@echo off
cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
call C:\ProgramData\anaconda3\Scripts\activate.bat adair
echo === M-F resume2 start %TIME% >> reports\spotcheck_runner.log
python -m src.train.train --arm M-F --iters 10000 --batch-size 16 --patch-size 128 --num-workers 6 --resume "runs\1p5b\M-F\M-F_seed0_20260801_002343\last.pth" > runs\1p5b_M-F.log 2>&1
echo === M-F resume2 done %TIME% exit=%ERRORLEVEL% >> reports\spotcheck_runner.log
