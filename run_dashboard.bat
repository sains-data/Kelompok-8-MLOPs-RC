@echo off
echo ================================
echo     MENJALANKAN DASHBOARD
echo ================================
echo.

cd /d "%~dp0"
streamlit run monitoring/dashboard/app.py

pause
