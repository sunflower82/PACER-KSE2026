@echo off
REM scripts/run_clothing.bat — DAMPS-MMHCL on Amazon Clothing (Windows)
cd /d "%~dp0\..\src"

if "%SEED%"=="" set SEED=42

python train.py ^
    --dataset Clothing ^
    --seed %SEED% ^
    --rebuild_R 5 ^
    --damps_apc 1 ^
    --damps_avrf 1 ^
    --damps_imcf 1 ^
    --damps_soft_routing 1 ^
    --damps_momentum 1 ^
    --damps_data_driven_prior 1 ^
    --use_amp 1 ^
    --epoch 250 ^
    --verbose 5 ^
    --early_stopping_patience 5 ^
    --ablation_target damps_full
