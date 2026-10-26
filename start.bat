@echo off
title Entry Exit Tracking
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo App environment not found. Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
pause
