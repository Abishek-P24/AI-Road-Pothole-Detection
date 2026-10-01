@echo off
title AI Road Pothole Detection
echo ========================================================
echo   AI-BASED ROAD POTHOLE DETECTION (COMPUTER VISION)
echo ========================================================
echo.
echo Starting FastAPI Backend Server on http://127.0.0.1:8000...
echo.

:: Automatically open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Start Uvicorn Server
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

pause
