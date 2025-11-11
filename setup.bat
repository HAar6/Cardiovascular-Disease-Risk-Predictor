@echo off
REM Quick setup script for Heart Disease Prediction App
REM Run this once to set up the project

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ====================================
echo Setup complete!
echo ====================================
echo.
echo To start the app, run:
echo   python -m streamlit run streamlit_app.py
echo.
echo The app will open at http://localhost:8501
echo.
pause
