@echo off
echo =========================================
echo Launching Gold Forecast AI Project...
echo =========================================
echo.

REM 1. Check if Python is exactly 3.10.x using standard python command
python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 10 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Incompatible Python version detected.
    echo This AI project requires EXACTLY Python 3.10.x.
    echo Please make sure Python 3.10 is installed and added to PATH.
    echo.
    pause
    exit /b
)

REM 2. Check if the virtual environment already exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating a new one...
    python -m venv .venv
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
