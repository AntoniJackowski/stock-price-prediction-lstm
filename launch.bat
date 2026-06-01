@echo off
setlocal enabledelayedexpansion

echo =========================================
echo Launching Gold Forecast AI Project...
echo =========================================
echo.

set "PYTHON_EXE="

REM 1. Search through ALL installed Python versions in the system PATH
echo [INFO] Scanning system for Python 3.10...
for /f "delims=" %%i in ('where python 2^>nul') do (
    "%%i" -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 10 else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=%%i"
        goto :python_found
    )
)

:python_found
REM 2. Verify if a compatible version was actually found
if "%PYTHON_EXE%"=="" (
    echo [ERROR] Could not find Python 3.10 among your installed versions.
    echo Please make sure Python 3.10 is installed and "Add to PATH" is checked.
    echo Download: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    pause
    exit /b
)

echo [INFO] Successfully detected compatible Python at: "%PYTHON_EXE%"
echo.

REM 3. Check if the virtual environment already exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating a new one...
    "%PYTHON_EXE%" -m venv .venv
)

REM 4. Activate the virtual environment
call .venv\Scripts\activate.bat

REM 5. Install required dependencies
echo [INFO] Verifying required dependencies...
pip install -r requirements.txt

REM 6. Launch the graphical user interface
echo [INFO] Starting the application GUI...
python src\desktop_app.py

pause
