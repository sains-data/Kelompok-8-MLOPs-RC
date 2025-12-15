@echo off
echo ================================
echo   MENJALANKAN FASTAPI SERVER
echo ================================
echo.

cd /d "%~dp0"
uvicorn api.main:app --reload

pause
