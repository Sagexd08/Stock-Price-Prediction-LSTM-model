@echo off
echo ===================================================
echo Advanced Stock Price Prediction System
echo ===================================================
echo.

if "%1"=="" (
    echo Usage: run.bat [command] [options]
    echo.
    echo Available commands:
    echo   setup         - Set up directories
    echo   download      - Download stock data
    echo   process       - Process raw data
    echo   train         - Train a model
    echo   evaluate      - Evaluate a model
    echo   predict       - Make predictions
    echo   deploy-api    - Deploy model as API
    echo   deploy-dash   - Deploy model as dashboard
    echo.
    echo Examples:
    echo   run.bat setup
    echo   run.bat download --ticker AAPL --start-date 2018-01-01
    echo   run.bat train --input-file data/processed/AAPL_processed.csv --model-type lstm_attention --target-col Target_5
    echo.
    exit /b
)

echo Running command: %*
echo.

python main.py %*

echo.
echo Command completed.
