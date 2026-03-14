#!/usr/bin/env python3
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_model(data_path='logreg_signal_noise/train.npz', model_path='logreg_signal_noise/logreg_noise_model.pkl'):
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
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(logreg, model_path)
    print("Done.")

if __name__ == "__main__":
    # If paths need to be relative to project root
    train_model()
