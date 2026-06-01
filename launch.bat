@echo off
echo =========================================
echo Launching Gold Forecast AI Project...
echo =========================================
echo.

REM 1. Check if Python 3.10 is installed using the Windows Launcher
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10 is not found on your system.
    echo This AI project requires EXACTLY Python 3.10.x due to TensorFlow dependencies.
    echo Please download and install Python 3.10.11 directly from:
    echo https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    pause
    exit /b
)

REM 2. Check if the virtual environment already exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating a new one using Python 3.10...
    py -3.10 -m venv .venv
)

REM 3. Activate the virtual environment
call .venv\Scripts\activate.bat

REM 4. Install required dependencies
echo [INFO] Verifying required dependencies...
pip install -r requirements.txt

REM 5. Launch the graphical user interface
echo [INFO] Starting the application GUI...
python src\desktop_app.py

pause