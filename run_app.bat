@echo off
echo Starting ZULF Signal Selection Tool...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo Error occurred. Press any key to exit.
    pause
)
