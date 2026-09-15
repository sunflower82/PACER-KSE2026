@echo off
REM scripts/run_permutation_fft_ablation.bat
REM ---------------------------------------------------------------------------
REM Falsifiability ablation (Spec Section 6, Item 8 / compliance check INFO 4).
REM Runs the canonical DAMPS-MMHCL pipeline twice per seed: once with the
REM standard 1-D FFT, once with a fixed random permutation FFT. The paired
REM scores feed scripts\_aggregate_permutation_fft.py which performs a
REM paired t-test on Recall@20 / NDCG@20.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0\..\src"

if "%DATASET%"=="" set DATASET=Clothing
if "%EPOCH%"==""   set EPOCH=250
if "%N_SEEDS%"=="" set N_SEEDS=3
if "%SEED_BASE%"=="" set SEED_BASE=42

set /a UPPER=%N_SEEDS% - 1
for /L %%I in (0,1,%UPPER%) do (
    set /a SEED=%SEED_BASE% + %%I
    echo ============================================================
    echo [ablation] perm_fft_off  seed=!SEED!
    echo ============================================================
    python train.py ^
        --dataset %DATASET% --seed !SEED! --epoch %EPOCH% ^
        --ablation_target perm_fft_off_seed!SEED! ^
        --damps_apc 1 --damps_avrf 1 --damps_imcf 1 ^
        --damps_soft_routing 1 --damps_momentum 1 ^
        --damps_data_driven_prior 1 --use_amp 1 ^
        --damps_permutation_fft 0
    if errorlevel 1 goto :error

    echo ============================================================
    echo [ablation] perm_fft_on   seed=!SEED!
    echo ============================================================
    python train.py ^
        --dataset %DATASET% --seed !SEED! --epoch %EPOCH% ^
        --ablation_target perm_fft_on_seed!SEED! ^
        --damps_apc 1 --damps_avrf 1 --damps_imcf 1 ^
        --damps_soft_routing 1 --damps_momentum 1 ^
        --damps_data_driven_prior 1 --use_amp 1 ^
        --damps_permutation_fft 1
    if errorlevel 1 goto :error
)

echo.
echo Now run: python scripts\_aggregate_permutation_fft.py
goto :eof

:error
echo [FATAL] training step exited with non-zero status.
exit /b 1
