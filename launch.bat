@echo off
echo =======================================
echo  Manga Colorize Launcher
echo =======================================

cd /d "C:\Users\Nighthawk\Desktop\manga_colorize"

echo.
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b
)

echo.
echo [2/3] Installing dependencies from requirements.txt...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b
)

echo.
echo [3/3] Installing PyTorch for CUDA...
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b
)

echo.
echo =======================================
echo  Launching Application...
echo =======================================
echo.

REM --- THIS IS THE KEY FIX ---
python main.py

echo.
echo Application has closed. Press any key to exit the window.
pause