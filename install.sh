#!/usr/bin/env bash
#
# install.sh — set up the CURIA pipeline environment and download model weights.
#
# Default model is RiNALMo (giga-v1, ~2.6 GB). RNA-FM (~1.1 GB) is optional and
# kept only for comparison (deprecated).
#
# Usage:
#   ./install.sh                 # env + RiNALMo weights
#   ./install.sh --with-rnafm    # also fetch RNA-FM weights (comparison only)
#   ./install.sh --no-weights    # env only, skip weight downloads
#
set -euo pipefail

WITH_RNAFM=0
DOWNLOAD_WEIGHTS=1
for arg in "$@"; do
  case "$arg" in
    --with-rnafm) WITH_RNAFM=1 ;;
    --no-weights) DOWNLOAD_WEIGHTS=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- macOS OpenMP prerequisite -------------------------------------------
if [[ "$(uname)" == "Darwin" ]]; then
  if ! brew list libomp >/dev/null 2>&1; then
    echo "# macOS detected: installing libomp (required by scikit-learn / numerical libs)"
    brew install libomp || echo "# WARNING: 'brew install libomp' failed — install it manually if you hit OpenMP errors."
  fi
  export KMP_DUPLICATE_LIB_OK=TRUE
fi

# --- uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "# Installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "# Syncing Python environment (uv sync)"
uv sync

# --- model weights --------------------------------------------------------
if [[ "$DOWNLOAD_WEIGHTS" == "1" ]]; then
  echo "# Downloading RiNALMo giga-v1 weights (~2.6 GB, cached at ~/.cache/rinalmo_pretrained)"
  uv run python -c "
import sys
sys.path.insert(0, 'modules/RiNALMo')
from rinalmo.pretrained import get_pretrained_model
get_pretrained_model(model_name='giga-v1')
print('# RiNALMo weights ready.')
"
  if [[ "$WITH_RNAFM" == "1" ]]; then
    echo "# Downloading RNA-FM weights (~1.1 GB, comparison only)"
    uv run python download_rnafm_model.py || echo "# WARNING: RNA-FM download failed."
  fi
fi

echo ""
echo "# Done. Activate the environment with:  source .venv/bin/activate"
echo "# Smoke test:  see 'Quick Start' in README.md"
