#!/usr/bin/env python3
"""Download RNA-FM pretrained model."""
import argparse
import os
# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch

MODEL_DIR = Path(torch.hub.get_dir()) / "checkpoints"
MODEL_FILE = MODEL_DIR / "RNA-FM_pretrained.pth"

# Add RNA-FM to path
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR / "RNA-FM"))


def try_load_model():
    """Try loading model, remove corrupted cache if needed."""
    import fm

    try:
        return fm.pretrained.rna_fm_t12()
    except (RuntimeError, KeyboardInterrupt) as e:
        if MODEL_FILE.exists():
            size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
            if size_mb < 1000:
                print(f"\n✗ Incomplete download detected ({size_mb:.1f}MB)")
                print(f"  Removing {MODEL_FILE}")
                os.remove(MODEL_FILE)
                print("  Please re-run this script to retry download.\n")
                sys.exit(1)
        if isinstance(e, KeyboardInterrupt):
            print("\n✗ Download interrupted")
            if MODEL_FILE.exists():
                os.remove(MODEL_FILE)
            sys.exit(1)
        raise


def main():
    parser = argparse.ArgumentParser(description="Download RNA-FM pretrained model.")
    parser.add_argument(
        "--show-dir",
        action="store_true",
        help="Print the expected model path and exit (useful for manual placement).",
    )
    args = parser.parse_args()

    if args.show_dir:
        print(MODEL_FILE)
        return

    print("Downloading RNA-FM pretrained model (~1.1GB)...")
    print("This may take several minutes depending on your connection.")
    print("If download is interrupted, just re-run this script.\n")

    model, alphabet = try_load_model()
    print("✓ Model downloaded and validated successfully!")


if __name__ == "__main__":
    main()
