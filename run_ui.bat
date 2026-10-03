@echo off
REM Launches the Action Detection web UI at http://localhost:8000
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv .venv || exit /b 1
    .venv\Scripts\python -m pip install -r webapp\requirements.txt || exit /b 1
)
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
