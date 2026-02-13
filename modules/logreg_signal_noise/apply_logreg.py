import numpy as np
import joblib
import os
import sys

# Add project root to path to import local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_logreg_model(model_path='logreg_signal_noise/logreg_noise_model.pkl'):
    """Load the trained Logistic Regression model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_logreg.py first.")
    return joblib.load(model_path)

def score_embeddings(embeddings_16, model=None, pca_model=None):
    """
    Score PCA-reduced RNA-FM embeddings using the Logistic Regression model.

    NOTE: The GPU executor already applies PCA projection from 640->16 dimensions.
    This function expects the 16-dimensional embeddings directly.

    Args:
        embeddings_16: numpy array of shape (N, 16) - already PCA-reduced
        model: pre-loaded LogisticRegression model (optional)
        pca_model: DEPRECATED - kept for API compatibility, not used

    Returns:
        probs: probability of being "proper RNA"
        status: 'trash' if prob < 0.5 (noise), 'keep' if prob >= 0.5 (proper)
    """
    if model is None:
        model = load_logreg_model()

    # Embeddings are already PCA-reduced by GPU executor, use directly
    X16 = embeddings_16

    # Get probabilities for class 1 (proper RNA)
    probs = model.predict_proba(X16)[:, 1]

    # Decision: 1 (proper) -> 'keep', 0 (noise) -> 'trash'
    # By default, LogisticRegression.predict uses 0.5 threshold
    preds = model.predict(X16)
    status = np.where(preds == 1, 'keep', 'trash')

    return probs, status

if __name__ == "__main__":
    # Example usage (requires embeddings)
    print("LogReg model applier script.")
    # test_embs = np.random.rand(5, 640)
    # p, s = score_embeddings(test_embs)
    # print(p, s)
