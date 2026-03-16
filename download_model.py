#!/usr/bin/env python3
"""Download RNA-FM pretrained model."""
import sys
from pathlib import Path

# Add RNA-FM to path
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR / "RNA-FM"))

import fm

print("Downloading RNA-FM pretrained model (~500MB)...")
print("This may take a few minutes depending on your connection.")
model, alphabet = fm.pretrained.rna_fm_t12()
print("✓ Model downloaded successfully!")
