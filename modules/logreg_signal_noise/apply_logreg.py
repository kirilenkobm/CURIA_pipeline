import json
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_logreg_model(model_path='logreg_signal_noise/logreg_noise_model.json'):
    """Load the trained Logistic Regression model from JSON and reconstruct the sklearn object."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_logreg.py first.")
    with open(model_path, "r") as f:
        data = json.load(f)
    required = {"coefficients", "intercept", "classes"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in model JSON: {sorted(missing)}")

    model = LogisticRegression()
    model.coef_ = np.array([data["coefficients"]])
    model.intercept_ = np.array([data["intercept"]])
    model.classes_ = np.array(data["classes"])
    return model


def score_embeddings(embeddings_16, model=None):
    """
    Score PCA-reduced RNA-FM embeddings using the Logistic Regression model.

    NOTE: The GPU executor already applies PCA projection from 640->16 dimensions.
    This function expects the 16-dimensional embeddings directly.

    Args:
        embeddings_16: numpy array of shape (N, 16) - already PCA-reduced
        model: pre-loaded LogisticRegression model (optional)

    Returns:
        probs: probability of being "proper RNA"
        status: 'noise' if prob < 0.5, 'signal' if prob >= 0.5
    """
    if model is None:
        model = load_logreg_model()

    probs = model.predict_proba(embeddings_16)[:, 1]
    preds = model.predict(embeddings_16)
    status = np.where(preds == 1, 'signal', 'noise')

    return probs, status


if __name__ == "__main__":
    print("LogReg model applier script.")
