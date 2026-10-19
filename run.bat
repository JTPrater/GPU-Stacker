@echo off
rem GPU Stacker - double-click to start. Optional argument: a Siril .seq file to open.
rem First run creates .venv here and installs the packages (needs Python 3.11+ and an NVIDIA driver).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up the Python environment, this takes a minute...
    python -m venv .venv || py -3 -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)
start "" ".venv\Scripts\pythonw.exe" -m gpustacker %*
exit /b 0

:error
echo.
echo Setup failed. Make sure Python 3.11+ is installed (python.org) and on PATH, then run this again.
pause
exit /b 1
