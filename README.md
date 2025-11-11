# Heart Disease Prediction Web Application

A Streamlit-based web application for predicting cardiovascular disease risk using a Keras neural network trained on the UCI Heart Disease dataset.

## Features

- 🫀 **Neural Network Model**: Binary classification using a 3-layer Keras Sequential model
- 📊 **13 Patient Input Features**: Age, sex, chest pain type, blood pressure, cholesterol, and more
- 🚀 **Fast Predictions**: Auto-trains on first run (~1 minute), instant predictions thereafter
- 🎨 **Interactive UI**: Clean, user-friendly Streamlit interface with real-time feedback
- ⚕️ **Medical Interpretation**: Provides risk assessment and clinical recommendations

## Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. **Clone/Navigate to project directory**:
```bash
cd c:\Users\harsh\OneDrive\Desktop\Major
```

2. **Create and activate virtual environment**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the App

```powershell
python -m streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
Major/
├── streamlit_app.py          # Main Streamlit application
├── backend/
│   ├── __init__.py           # Package marker
│   ├── model.py              # Keras model loader and predictor
│   └── binary_model.h5       # Trained model (auto-generated on first run)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

1. Fill in the 13 patient health parameters
2. Click **"Predict Disease Risk"**
3. View the prediction result with risk percentage and clinical interpretation
4. First prediction takes ~30-60 seconds (model training); subsequent predictions are instant

## Model Details

- **Architecture**: Input(13) → Dense(8, relu) → Dense(4, relu) → Dense(1, sigmoid)
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam (lr=0.001)
- **Training**: 100 epochs, batch size 10
- **Dataset**: UCI Heart Disease (Cleveland clinic, ~297 samples)
- **Target**: Binary classification (0=no disease, 1=disease present)

## Input Features

1. Age (1-120)
2. Sex (Male/Female)
3. Chest Pain Type (0-3)
4. Resting Blood Pressure (50-250 mm Hg)
5. Cholesterol (50-600 mg/dl)
6. Fasting Blood Sugar > 120 (Yes/No)
7. Resting ECG (0-2)
8. Max Heart Rate (60-220)
9. Exercise Induced Angina (Yes/No)
10. ST Depression (0.0-10.0)
11. ST Segment Slope (0-2)
12. Major Vessels Count (0-3)
13. Thal Type (0-3)

## Dependencies

- **streamlit**: Web application framework
- **tensorflow/keras**: Neural network model
- **pandas**: Data manipulation
- **scikit-learn**: Model training utilities
- **numpy**: Numerical computations

## Important Notes

⚠️ **Medical Disclaimer**: This model is for educational purposes only and should NOT be used for actual medical diagnosis. Always consult with a qualified healthcare professional for proper diagnosis and treatment.

## Troubleshooting

**"Module not found" errors**: Make sure you activated the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

**Model training takes too long**: This is normal on first run. The model downloads the UCI dataset and trains from scratch. Subsequent runs use the cached model.

**Port 8501 already in use**: Run on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Future Enhancements

- [ ] CSV batch upload for multiple predictions
- [ ] Model performance metrics dashboard
- [ ] Database integration for prediction history
- [ ] Improved preprocessing and feature engineering
- [ ] Model retraining pipeline

## License

Educational use

## Author

Created using Keras/TensorFlow and Streamlit
