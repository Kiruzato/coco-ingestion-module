@echo off
title CoCo Ingestion Module
cd /d "%~dp0"

:: Use the local virtual environment created by setup.ps1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
    goto :end
)

echo.
echo ERROR: Virtual environment not found.
echo Please run setup.ps1 first:
echo   powershell -ExecutionPolicy Bypass -File setup.ps1
echo.
pause

:end
