@echo off
REM Resume devon's clean 20k B0 checkpoint ON THE LAPTOP.
REM
REM devon diverged twice near iteration 22-26k with a gradient norm reaching
REM 6.5e7. This decides whether that is devon's degraded CPU or a real B0
REM instability: identical checkpoint, identical code, known-good hardware.
call C:\ProgramData\anaconda3\Scripts\activate.bat adair
cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
python -m src.train.train --arm B0 --seed 0 --out-root runs/divtest_laptop_run --resume runs/divtest_laptop/last.pth --resume-reason "hardware-vs-instability test: devon diverged at 21967 and 25582 from this checkpoint" >> runs\divtest_laptop.log 2>&1
