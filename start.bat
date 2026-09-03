@echo off
rem =====================================================================
rem  One-click launcher for Windows (double-click or run from CMD/PowerShell)
rem  Calls the cross-platform start.py so behaviour is identical everywhere.
rem =====================================================================
cd /d "%~dp0"

rem --- find a Python interpreter ----------------------------------------
where python >nul 2>nul
if %errorlevel%==0 (
    python start.py
    goto :end
)
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 start.py
    goto :end
)
echo [start.bat] Python was not found on PATH.
echo Please install Python 3.9+ from https://www.python.org/downloads/
pause
:end
