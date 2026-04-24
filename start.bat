@echo off
chcp 65001 >nul
title DeepGuard Launcher
color 0A

echo.
echo  ██████╗ ███████╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
echo  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
echo  ██║  ██║█████╗  █████╗  ██████╔╝██║  ███╗██║   ██║███████║██████╔╝██║  ██║
echo  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
echo  ██████╔╝███████╗███████╗██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
echo  ╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
echo.
echo  Real-Time Deepfake Detection Overlay
echo  ─────────────────────────────────────
echo.

:: ── Start Backend ──────────────────────────────────────────
echo  [1/2] Starting backend (Flask)...
start "DeepGuard Backend" cmd /k "cd /d %~dp0backend && python api.py"

:: Wait for backend to initialize
echo  Waiting for backend to start...
timeout /t 5 /nobreak >nul

:: ── Start Frontend ─────────────────────────────────────────
echo  [2/2] Starting frontend (Electron)...
start "DeepGuard Frontend" cmd /k "cd /d %~dp0frontend && npm start"

echo.
echo  DeepGuard is running!
echo  Backend : http://localhost:5000
echo  Close this window or press any key to exit launcher.
echo.
pause >nul
