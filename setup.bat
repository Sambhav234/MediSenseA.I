@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM MediSense AI v3 — Windows Setup & Launch
REM Usage:  setup.bat          (full setup + start)
REM         setup.bat train    (train model only)
REM         setup.bat backend  (start server only)
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  ============================================
echo   MediSense AI v3 - XGBoost + CNN + Maps
echo  ============================================
echo.

SET MODE=%1
IF "%MODE%"=="" SET MODE=all

REM ── Create .env if missing ──────────────────────────────────────────────────
IF NOT EXIST backend\.env (
  IF EXIST backend\.env.example (
    copy backend\.env.example backend\.env >nul
    echo [INFO] Created backend\.env — add your API keys there.
  )
)

REM ── Install ──────────────────────────────────────────────────────────────────
IF "%MODE%"=="all" GOTO install
IF "%MODE%"=="install" GOTO install
GOTO skip_install
:install
echo [1/3] Installing dependencies...
py -m pip install -r requirements.txt -q --disable-pip-version-check
IF ERRORLEVEL 1 (python -m pip install -r requirements.txt -q)
echo     Done.
:skip_install

REM ── Train ────────────────────────────────────────────────────────────────────
IF "%MODE%"=="all" GOTO train
IF "%MODE%"=="train" GOTO train
GOTO skip_train
:train
echo [2/3] Training XGBoost model...
IF NOT EXIST ml\data\dataset.csv (
  echo ERROR: ml\data\dataset.csv not found. Copy your 4 CSV files into ml\data\
  pause & exit /b 1
)
py ml\train_model.py
IF ERRORLEVEL 1 (python ml\train_model.py)
echo     Model saved to ml\models\
:skip_train

REM ── Start backend ────────────────────────────────────────────────────────────
IF "%MODE%"=="all" GOTO backend
IF "%MODE%"=="backend" GOTO backend
GOTO end
:backend
echo [3/3] Starting Flask backend...
IF NOT EXIST ml\models\xgb_model.pkl (
  echo Model not found — training first...
  py ml\train_model.py
)
echo.
echo  Backend  : http://localhost:5000
echo  Frontend : open frontend\index.html in your browser
echo.
py backend\app.py
IF ERRORLEVEL 1 (python backend\app.py)
:end
