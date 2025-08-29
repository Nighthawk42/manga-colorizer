@echo off
title Manga Colorizer Training
echo =======================================
echo   Manga Colorizer - Training Launcher
echo =======================================

cd /d "C:\Users\Nighthawk\Desktop\manga_colorize"

echo.
echo [1/1] Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b
)

echo.
echo =======================================
echo   Launching Training Script...
echo =======================================
echo.

python train_model.py

echo.
echo Training script has exited. Press any key to close this window.
pause >nul
