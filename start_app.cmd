@echo off
setlocal
set "PYTHON=%~dp0.venv312\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Die Python-3.12-Umgebung fehlt: %PYTHON%
  echo Bitte requirements.txt installieren.
  pause
  exit /b 1
)
"%PYTHON%" -m streamlit run "%~dp0app\main.py"
