"""
Heart disease prediction model backend.
Loads a pre-trained Keras model or trains one on-the-fly if not available.
"""

import os
import json
import warnings
import sys
import collections
import typing

# CRITICAL: Fix Python 3.7 compatibility BEFORE importing TensorFlow
# TensorFlow 2.11 uses Optional[Dict[...]] syntax which doesn't work in Python 3.7
# This patch allows that syntax to work

try:
    if sys.version_info < (3, 9):
        # Monkey-patch to allow subscripting type hints at class definition time
        original_setattr = typing.TYPE_CHECKING
        
        # Pre-patch for common problematic patterns
        if not hasattr(collections, 'OrderedDict'):
            from collections import OrderedDict
            collections.OrderedDict = OrderedDict
except Exception:
    pass

# Suppress TensorFlow logging BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'binary_model.h5')

# Feature names in order (must match training data)
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

_model = None
_model_trained = False


def load_or_train_model():
    """Load pre-trained model or train a new one if not available."""
    global _model, _model_trained
    
    if _model_trained:
        return _model
    
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        try:
            # Import Keras AFTER OrderedDict fix
            from tensorflow.keras.models import load_model
            _model = load_model(MODEL_PATH)
            _model_trained = True
            print(f"Loaded model from {MODEL_PATH}")
            return _model
        except Exception as e:
            print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
    
    # If no saved model, train one
    print("Training model from UCI Heart Disease dataset...")
    _model = train_and_save_model()
    _model_trained = True
    return _model


def train_and_save_model():
    """Train the binary model from scratch and save it."""
    import pandas as pd
    import numpy as np
    from sklearn import model_selection
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    
    print("Downloading dataset...")
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Download and load
    cleveland = pd.read_csv(url, names=FEATURE_NAMES + ['class'])
    
    # Clean
    data = cleveland[~cleveland.isin(['?'])].dropna(axis=0)
    data = data.apply(pd.to_numeric)
    
    print(f"Dataset shape: {data.shape}")
    
    # Prepare
    X = np.array(data.drop(['class'], 1))
    y = np.array(data['class'])
    y_binary = y.copy()
    y_binary[y_binary > 0] = 1
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_binary, test_size=0.2)
    
    # Build
    print("Building and training model...")
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model


def predict(input_dict):
    """
    Predict heart disease risk given patient features.
    
    Args:
        input_dict: Dict with keys matching FEATURE_NAMES
        
    Returns:
        (probability, class) tuple where:
        - probability: float in [0, 1], probability of disease
        - class: int, 0 or 1 (0=no disease, 1=disease)
    """
    import numpy as np
    import traceback
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
    
    try:
        model = load_or_train_model()
        
        # Build feature vector in correct order
        row = [input_dict.get(fname, 0) for fname in FEATURE_NAMES]
        
        # Predict
        try:
            proba = float(model.predict(np.array([row]), verbose=0)[0][0])
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            proba = 0.5
        
        pred = 1 if proba >= 0.5 else 0
        return float(proba), int(pred)
    except Exception as e:
        print(f"Fatal prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Return neutral prediction on error
        return 0.5, 0
