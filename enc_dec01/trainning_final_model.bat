@echo off

REM ============================================================
REM Final Model Training
REM Usage:
REM     train_final.bat vocab1.json 25
REM ============================================================

set "TOKENIZER_PATH=%~1"
set "MODEL_IDX=%~2"

if not defined TOKENIZER_PATH (
    echo ERROR: Tokenizer path is required.
    echo Usage: train_final.bat vocab1.json MODEL_IDX
    exit /b 1
)

if not defined MODEL_IDX (
    echo ERROR: Model index is required.
    echo Usage: train_final.bat vocab1.json MODEL_IDX
    exit /b 1
)

REM Get tokenizer filename without extension
set "VOCAB_NAME=%~n1"

REM ============================================================
REM Paths
REM ============================================================

set "FINAL_DIR=trained_models\final"
set "LOG_DIR=logs\%VOCAB_NAME%\final"

if not exist "%FINAL_DIR%" mkdir "%FINAL_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "LOG_FILE=%LOG_DIR%\final_training.log"
set "ERROR_LOG=%LOG_DIR%\errors.log"

REM ============================================================
REM Start log
REM ============================================================

echo ============================================================ >> "%LOG_FILE%"
echo Final model training started: %date% %time% >> "%LOG_FILE%"
echo Tokenizer: %TOKENIZER_PATH% >> "%LOG_FILE%"
echo Model index: %MODEL_IDX% >> "%LOG_FILE%"
echo Output directory: %FINAL_DIR% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

echo.
echo ============================================================
echo Final Model Training
echo ============================================================
echo Tokenizer : %TOKENIZER_PATH%
echo Model     : %MODEL_IDX%
echo Output    : %FINAL_DIR%
echo Log       : %LOG_FILE%
echo ============================================================
echo.

REM ============================================================
REM Train final model
REM ============================================================

python -u trainning_final.py --model_idx %MODEL_IDX% -b 128 --epochs 15 --cuda 1 -lr 0.001 --final 1 --tokenizer_path "%TOKENIZER_PATH%" 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"

REM ============================================================
REM Check result
REM ============================================================

if errorlevel 1 (
    echo.
    echo ============================================================
    echo MODEL %MODEL_IDX% FAILED
    echo ============================================================

    echo MODEL %MODEL_IDX% FAILED: %date% %time% >> "%ERROR_LOG%"

    echo.
    echo Training failed.
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo MODEL %MODEL_IDX% COMPLETED SUCCESSFULLY
    echo ============================================================

    echo MODEL %MODEL_IDX% FINISHED SUCCESSFULLY: %date% %time% >> "%LOG_FILE%"
)

echo.
echo Final model training finished.