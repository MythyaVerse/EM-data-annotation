@echo off
REM CVAT Integration Installation Script for Windows
REM This script installs all dependencies needed for CVAT integration

echo ========================================
echo CVAT Integration - Installation Script
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.8+ required. Please upgrade Python.
    pause
    exit /b 1
)
echo      OK - Python version is compatible

echo.
echo [2/4] Installing CVAT dependencies...
pip install -r requirements_cvat.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo      OK - Dependencies installed

echo.
echo [3/4] Verifying CVAT SDK installation...
python -c "import cvat_sdk; print('     OK - CVAT SDK installed')"
if %errorlevel% neq 0 (
    echo ERROR: CVAT SDK verification failed
    pause
    exit /b 1
)

echo.
echo [4/4] Verifying integration package...
python -c "from cvat_integration import format_converter; print('     OK - Integration package ready')"
if %errorlevel% neq 0 (
    echo ERROR: Integration package verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Start CVAT server (see CVAT_SETUP.md)
echo 2. Configure cvat_config.json with your credentials
echo 3. Run your first workflow (see CVAT_QUICKSTART.md)
echo.
echo Documentation:
echo - CVAT_SETUP.md       : Detailed setup guide
echo - CVAT_QUICKSTART.md  : Quick start examples
echo - README_CVAT_INTEGRATION.md : Complete reference
echo.
pause
