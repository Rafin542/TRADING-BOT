@echo off
echo ================================================
echo   🧠 Market Prediction AI — Setup Script
echo ================================================
echo.

:: Check Python
echo 🔍 Checking Python installation...
python --version 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ❌ Python is NOT installed!
    echo.
    echo 👉 Download Python from: https://www.python.org/downloads/
    echo 👉 During installation, CHECK "Add Python to PATH"
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)
echo ✅ Python found!
echo.

:: Install dependencies
echo 📦 Installing dependencies (this may take a few minutes)...
echo.
pip install numpy pandas scipy scikit-learn matplotlib tqdm joblib pyyaml requests
echo.
pip install ccxt
echo.
pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.
pip install pytorch-lightning
echo.
pip install lightgbm xgboost
echo.
pip install stable-baselines3 gymnasium
echo.
pip install deap
echo.
pip install hmmlearn statsmodels
echo.
pip install fastapi uvicorn
echo.
pip install shap plotly
echo.

echo.
echo ================================================
echo   ✅ Setup Complete!
echo ================================================
echo.
echo 📋 How to use:
echo.
echo   1. START DASHBOARD:
echo      python server.py
echo      Then open: http://localhost:8000
echo.
echo   2. TRAIN MODELS (optional, for better accuracy):
echo      python main.py train --symbol BTC/USDT --timeframe 4h
echo.
echo   3. PREDICT from command line:
echo      python main.py predict --symbol BTC/USDT --timeframe 4h
echo.
echo ================================================
pause
