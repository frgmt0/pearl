@echo off
REM Setup script for Terminal Chess on Windows

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo uv is not installed. Installing uv...
    pip install uv
)

REM Create a virtual environment
echo Creating virtual environment...
uv venv

REM Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
uv pip install -r requirements.txt

echo Setup complete! You can now run the game with:
echo python main.py 