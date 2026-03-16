#!/usr/bin/env python3
"""Download RNA-FM pretrained model."""
import os
# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Add RNA-FM to path
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR / "RNA-FM"))

import torch
import fm

print("Downloading RNA-FM pretrained model (~500MB)...")
print("This may take a few minutes depending on your connection.")

try:
    model, alphabet = fm.pretrained.rna_fm_t12()
except RuntimeError as e:
    if "failed finding central directory" in str(e) or "PytorchStreamReader failed" in str(e):
        # Corrupted model file - remove and retry
        cached_file = Path(torch.hub.get_dir()) / "checkpoints" / "RNA-FM_pretrained.pth"
        if cached_file.exists():
            print(f"✗ Corrupted file detected at {cached_file}")
            print(f"  Removing and re-downloading...")
            os.remove(cached_file)
            model, alphabet = fm.pretrained.rna_fm_t12()
        else:
            raise
    else:
        raise

print("✓ Model downloaded successfully!")
