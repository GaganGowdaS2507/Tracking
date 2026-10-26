@echo off
title Entry Exit Tracking - Setup
cd /d "%~dp0"
echo ============================================
echo   Entry / Exit Tracking - Setup
echo ============================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.10 or 3.11 from python.org and enable Add Python to PATH.
  pause
  exit /b 1
)
py -3.11 --version >nul 2>nul
if not errorlevel 1 (set PY=py -3.11) else (
  py -3.10 --version >nul 2>nul
  if not errorlevel 1 (set PY=py -3.10) else (set PY=py)
)
if not exist .venv (
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Setup complete. Run start.bat to launch the app.
echo InsightFace may download its pretrained model files on first use.
pause
exit /b 0
:fail
echo.
echo Setup failed. Check the error above. Python 3.10 or 3.11 is recommended.
pause
exit /b 1
