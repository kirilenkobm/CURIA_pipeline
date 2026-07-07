#!/usr/bin/env python3
"""Download the RiNALMo pretrained model (default model, ~2.6 GB).

RiNALMo is the default RNA foundation model. Weights are cached at
~/.cache/rinalmo_pretrained/<model>.pt (fetched from Google Drive via gdown).
The RNA-FM counterpart (deprecated, comparison only) is download_rnafm_model.py.
"""
import argparse
import os
# Fix macOS OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Add RiNALMo to path
MODULES_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULES_DIR / "RiNALMo"))

from rinalmo.pretrained import get_pretrained_model, DEFAULT_CACHE_DIR

DEFAULT_MODEL = "giga-v1"
MIN_SIZE_MB = 2000  # a complete giga-v1 checkpoint is ~2.6 GB


def try_load_model(model_name: str):
    """Try downloading/loading the model, removing a corrupted cache if needed."""
    model_file = DEFAULT_CACHE_DIR / f"{model_name}.pt"
    try:
        return get_pretrained_model(model_name=model_name)
    except (RuntimeError, KeyboardInterrupt) as e:
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            if isinstance(e, KeyboardInterrupt) or size_mb < MIN_SIZE_MB:
                reason = "interrupted" if isinstance(e, KeyboardInterrupt) else f"incomplete ({size_mb:.1f}MB)"
                print(f"\n✗ Download {reason}. Removing {model_file}")
                os.remove(model_file)
                print("  Please re-run this script to retry download.\n")
                sys.exit(1)
        raise


def main():
    parser = argparse.ArgumentParser(description="Download the RiNALMo pretrained model.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"RiNALMo model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--show-dir", action="store_true",
                        help="Print the expected model path and exit (useful for manual placement).")
    args = parser.parse_args()

    model_file = DEFAULT_CACHE_DIR / f"{args.model}.pt"
    if args.show_dir:
        print(model_file)
        return

    if model_file.exists():
        print(f"✓ RiNALMo {args.model} already present at {model_file}")
        return

    print(f"Downloading RiNALMo {args.model} (~2.6 GB)...")
    print("This may take several minutes depending on your connection.")
    print("If download is interrupted, just re-run this script.\n")

    try_load_model(args.model)
    print("✓ Model downloaded and validated successfully!")


if __name__ == "__main__":
    main()
