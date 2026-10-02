@echo off

set "TOKENIZER_PATH=%~1"
if not defined TOKENIZER_PATH set "TOKENIZER_PATH=None"
set START_MODEL=4
set END_MODEL=29
set "VOCAB_NAME=%~n1"
if not defined VOCAB_NAME set "VOCAB_NAME=vocab"

if not exist logs mkdir logs

echo ============================================================ >> logs\%VOCAB_NAME%\training.log
echo Training run started: %date% %time% >> logs\%VOCAB_NAME%\training.log
echo ============================================================ >> logs\%VOCAB_NAME%\training.log

for /L %%M in (%START_MODEL%,1,%END_MODEL%) do (

    echo.
    echo ============================================================
    echo Starting model %%M
    echo ============================================================

    echo. >> logs\%VOCAB_NAME%\training.log
    echo ============================================================ >> logs\%VOCAB_NAME%\training.log
    echo MODEL %%M STARTED: %date% %time% >> logs\%VOCAB_NAME%\training.log
    echo ============================================================ >> logs\%VOCAB_NAME%\training.log

    python -u trainning.py --model_idx %%M -b 128 --batch_ratio 0.4 --epochs 10 --cuda 1 -lr 0.009 --tokenizer_path %TOKENIZER_PATH%  2>&1 | powershell -Command "$input | Tee-Object -FilePath 'logs\%VOCAB_NAME%\training.log'"

    if errorlevel 1 (
        echo Model %%M FAILED. Continuing...
        echo MODEL %%M FAILED: %date% %time% >> logs\%VOCAB_NAME%\errors.log
    ) else (
        echo Model %%M completed successfully.
        echo MODEL %%M FINISHED SUCCESSFULLY: %date% %time% >> logs\%VOCAB_NAME%\training.log
    )
)

echo.
echo ============================================================
echo All models finished.
echo ============================================================