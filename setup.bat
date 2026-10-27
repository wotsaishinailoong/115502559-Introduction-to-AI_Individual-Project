@echo off
setlocal
cd /d "%~dp0"
py -3.11 -c "import struct; assert struct.calcsize('P') == 8"
if errorlevel 1 (
    echo Install Python 3.11 64-bit from python.org, including the Python launcher.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed. Check the error above, then run setup.bat again.
    pause
    exit /b 1
)
echo Setup complete. Run run.bat to start.
pause
