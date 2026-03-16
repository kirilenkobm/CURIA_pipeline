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

print("Downloading RNA-FM pretrained model (~1.1GB)...")
print("This may take several minutes depending on your connection.")
print("If download is interrupted, just re-run this script.\n")

def try_load_model():
    """Try loading model, remove corrupted cache if needed."""
    try:
        return fm.pretrained.rna_fm_t12()
    except (RuntimeError, KeyboardInterrupt) as e:
        cached_file = Path(torch.hub.get_dir()) / "checkpoints" / "RNA-FM_pretrained.pth"
        if cached_file.exists():
            # Check if file size is wrong (should be ~1.1GB)
            size_mb = cached_file.stat().st_size / (1024 * 1024)
            if size_mb < 1000:  # Less than 1GB = incomplete
                print(f"\n✗ Incomplete download detected ({size_mb:.1f}MB)")
                print(f"  Removing {cached_file}")
                os.remove(cached_file)
                print("  Please re-run this script to retry download.\n")
                sys.exit(1)
        if isinstance(e, KeyboardInterrupt):
            print("\n✗ Download interrupted")
            if cached_file.exists():
                os.remove(cached_file)
            sys.exit(1)
        raise

model, alphabet = try_load_model()
print("✓ Model downloaded and validated successfully!")
