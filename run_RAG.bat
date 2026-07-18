@echo off
setlocal

cd /d "%~dp0"

call "D:\Anaconda\condabin\conda.bat" activate RAG
if errorlevel 1 (
    echo Failed to activate Conda environment RAG.
    pause
    exit /b 1
)

if /i "%~1"=="--check" (
    echo Conda environment activated successfully.
    echo Project directory: %CD%
    python --version
    exit /b 0
)

echo Starting Course Material Assistant...
echo The knowledge base will continue loading in the application window.
python app.py
if errorlevel 1 pause

endlocal
