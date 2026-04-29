@echo off
chcp 65001 >nul
title TrueVision — Installer
color 0B

echo.
echo  ████████╗██████╗ ██╗   ██╗███████╗██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
echo  ╚══██╔══╝██╔══██╗██║   ██║██╔════╝██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
echo     ██║   ██████╔╝██║   ██║█████╗  ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
echo     ██║   ██╔══██╗██║   ██║██╔══╝  ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
echo     ██║   ██║  ██║╚██████╔╝███████╗ ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
echo     ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
echo.
echo  Real-Time Deepfake Detection — Installer
echo  ──────────────────────────────────────────
echo.

:: ── Check Python ──────────────────────────────────────────
echo  [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
python --version
echo  Python OK.
echo.

:: ── Check Node.js ─────────────────────────────────────────
echo  [2/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found. Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)
node --version
echo  Node.js OK.
echo.

:: ── Install Python dependencies ───────────────────────────
echo  [3/4] Installing Python dependencies...
echo  (This may take a few minutes for PyTorch...)
echo.
pip install -r "%~dp0backend\requirements.txt"
if errorlevel 1 (
    echo.
    echo  [ERROR] Python dependency installation failed.
    echo  Try running manually: pip install -r backend\requirements.txt
    pause
    exit /b 1
)
echo.
echo  Python dependencies installed.
echo.

:: ── Install Node dependencies ─────────────────────────────
echo  [4/4] Installing Node.js dependencies (Electron)...
cd /d "%~dp0frontend"
npm install
if errorlevel 1 (
    echo.
    echo  [ERROR] Node.js dependency installation failed.
    echo  Try running manually: cd frontend ^&^& npm install
    pause
    exit /b 1
)
cd /d "%~dp0"
echo.
echo  Node.js dependencies installed.
echo.

:: ── Check model weights ───────────────────────────────────
echo  ──────────────────────────────────────────
echo  Checking model weight files...
echo.

set MISSING=0

if not exist "%~dp0backend\FaceForensics.pth" (
    echo  [MISSING] backend\FaceForensics.pth
    set MISSING=1
)
if not exist "%~dp0backend\FaceDetector_PP\FaceDetector_PP\pth_fiels\FaceForensics_PP.pth" (
    echo  [MISSING] backend\FaceDetector_PP\FaceDetector_PP\pth_fiels\FaceForensics_PP.pth
    set MISSING=1
)

if "%MISSING%"=="1" (
    echo.
    echo  [!] One or more model weight files are missing.
    echo      Download them and place them in the correct folders.
    echo      See README.md for download links.
    echo.
) else (
    echo  All model weight files found.
    echo.
)

:: ── Done ──────────────────────────────────────────────────
echo  ══════════════════════════════════════════
echo  Installation complete!
echo  Run start.bat to launch TrueVision.
echo  ══════════════════════════════════════════
echo.
pause
