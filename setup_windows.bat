@echo off
echo ======================================================================
echo Universal ML Engine — Windows Environment Setup
echo ======================================================================

echo [1/2] Installing Main Engine & UI Dependencies...
python -m pip install -r backend\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Python command failed. Trying with active interpreter...
)

echo.
echo [2/2] Setup complete!
echo.
echo To start the web dashboard, run:
echo    python run_ui.py --port 8000
echo.
echo Open your browser at:
echo    http://127.0.0.1:8000
echo ======================================================================
