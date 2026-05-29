@echo off
echo =========================================
echo Launching Gold Forecast AI Project...
echo =========================================
echo.

REM 1. Check if Python is installed and available in the system PATH
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to the system PATH.
    echo Please download and install Python 3.10.11 directly from:
    echo https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    echo IMPORTANT: Make sure to check the "Add Python to PATH" box during installation!
    echo.
    pause
    exit /b
)

REM 2. Check if Python version is EXACTLY 3.10
python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 10 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Incompatible Python version detected.
    echo This AI project requires EXACTLY Python 3.10.x due to TensorFlow dependencies.
    echo Newer versions ^(like 3.11+^) may not be supported yet.
    echo Please download the correct 64-bit installer here:
    echo https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    pause
    exit /b
)

REM 3. Check if the virtual environment already exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating a new one...
    python -m venv .venv
)

REM 4. Activate the virtual environment
call .venv\Scripts\activate.bat

REM 5. Install required dependencies (will skip if already satisfied)
echo [INFO] Verifying required dependencies...
pip install -r requirements.txt

REM 6. Launch the graphical user interface
echo [INFO] Starting the application GUI...
python src\desktop_app.py

pause