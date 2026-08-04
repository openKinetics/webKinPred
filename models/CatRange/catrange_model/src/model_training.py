"""
Model I/O for CatRange

Provides model loading and prediction for inference.
Training logic lives in scripts/cv_train.py.
"""

import joblib
import numpy as np
from pathlib import Path


def load_model(model_path):
    """Load a trained XGBoost model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_with_model(model, X: np.ndarray):
    """Run prediction with a loaded XGBoost model.

    Returns:
        (predictions, probabilities_or_None)
    """
    predictions = model.predict(X)
    try:
        probabilities = model.predict_proba(X)
    except Exception:
        probabilities = None
    return predictions, probabilities
