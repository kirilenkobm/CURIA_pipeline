#!/usr/bin/env python3
"""Minimal test to reproduce the segfault."""
import sys
import os

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

# Set threading env vars BEFORE importing anything
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import joblib
import xgboost
import numpy as np

print(f"joblib: {joblib.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"numpy: {np.__version__}")

model_path = "../modules/TOGA_mini/chain_classification_models/se_model.dat"
print(f"\nAttempting to load: {model_path}")
print(f"File exists: {os.path.isfile(model_path)}")
print(f"File size: {os.path.getsize(model_path)} bytes")
sys.stdout.flush()

try:
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully!")
    print(f"  Type: {type(model)}")
    print(f"  Attributes: {dir(model)[:5]}")
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
