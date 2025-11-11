#!/usr/bin/env bash
# Quick setup script for Heart Disease Prediction App (Linux/Mac)

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "===================================="
echo "Setup complete!"
echo "===================================="
echo ""
echo "To start the app, run:"
echo "  python -m streamlit run streamlit_app.py"
echo ""
echo "The app will open at http://localhost:8501"
echo ""
