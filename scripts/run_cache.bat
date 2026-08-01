@echo off
cd /d C:\Users\User\Documents\FYP\fyp-adair-distill
call C:\ProgramData\anaconda3\Scripts\activate.bat adair
echo === CACHE resume %TIME% >> reports\spotcheck_runner.log
python -m src.cache.precompute_teacher --tasks derain denoise --device cuda --budget-gb 60 > runs\teacher_cache.log 2>&1
echo === CACHE done %TIME% exit=%ERRORLEVEL% >> reports\spotcheck_runner.log
