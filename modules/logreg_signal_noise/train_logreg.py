#!/usr/bin/env python3
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model(data_path=None, model_path=None):
    if data_path is None:
        data_path = os.path.join(SCRIPT_DIR, 'train.npz')
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, 'logreg_noise_model.json')
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"Training Logistic Regression (Train size: {len(X_train)})...")
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    logreg.fit(X_train, y_train)
    
    # Evaluate
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['noise', 'proper']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    model_dict = {
        "coefficients": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
        "classes": logreg.classes_.tolist(),
        "feature_dim": X_train.shape[1],
    }

    print(f"\nSaving model to {model_path}...")
    with open(model_path, "w") as f:
        json.dump(model_dict, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    # If paths need to be relative to project root
    train_model()
