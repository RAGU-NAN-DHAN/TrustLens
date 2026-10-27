@echo off
REM TrustLens full local run: backend with M1 self-consistency + trust model, and the website.
REM First run installs the embedding model (CPU PyTorch + sentence-transformers, a few hundred MB). Later runs start in seconds.
setlocal
cd /d "%~dp0"

echo.
echo === TrustLens local run ===
echo.

REM ---- backend virtual environment ----
if not exist "backend\.venv\Scripts\python.exe" (
    echo [1/4] Creating the Python environment...
    py -3.13 -m venv backend\.venv || python -m venv backend\.venv || goto :fail
    backend\.venv\Scripts\python -m pip install -q -r backend\requirements.txt || goto :fail
    backend\.venv\Scripts\python -m pip install -q .\m1 --no-deps || goto :fail
) else (
    echo [1/4] Python environment found.
)

if not exist "backend\.env" (
    echo.
    echo backend\.env is missing. Copy backend\.env.example to backend\.env and put your API keys in it.
    goto :fail
)

REM ---- embedding model for self-consistency and the trained trust model ----
backend\.venv\Scripts\python -c "import sentence_transformers" 2>nul
if errorlevel 1 (
    echo [2/4] Installing the embedding model ^(one time, CPU only^)...
    backend\.venv\Scripts\python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
    backend\.venv\Scripts\python -m pip install -q sentence-transformers || goto :fail
) else (
    echo [2/4] Embedding model installed.
)
echo       Loading the model once so the first question is fast...
backend\.venv\Scripts\python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || goto :fail

REM ---- website dependencies ----
if not exist "frontend\node_modules" (
    echo [3/4] Installing website packages...
    pushd frontend
    call npm install || (popd & goto :fail)
    popd
) else (
    echo [3/4] Website packages found.
)

REM the local website talks to the local backend
> frontend\.env.development.local (
    echo VITE_USE_MOCK=false
    echo VITE_API_BASE_URL=http://localhost:8000
)

echo [4/4] Starting the backend ^(port 8000^) and the website ^(port 5180^)...
start "TrustLens backend" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python -m uvicorn app.main:app --port 8000"
start "TrustLens website" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM give both a moment, then open the site
timeout /t 8 /nobreak >nul
start "" http://localhost:5180

echo.
echo Running. Close the two "TrustLens" windows to stop.
echo The answer box now says "Mean P(supported) from M1's logistic-regression model" when the trust model is used.
exit /b 0

:fail
echo.
echo Something failed above. Read the message, fix it, and run this file again.
pause
exit /b 1
