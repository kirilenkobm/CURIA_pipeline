"""
apply_pca.py: Apply PCA transformation to RNA foundation model embeddings.

Supports any model registered in model_registry (RNA-FM, RiNALMo, etc.).
Dimensions are read from the .npz file — no hardcoded assumptions.

Usage:
    from modules.global_PCA.apply_pca import apply_pca, load_pca

    # Option 1: Auto-resolve via model name
    pca_emb = apply_pca(embeddings, model_name="rinalmo")

    # Option 2: Explicit path
    pca_model = load_pca("/path/to/pca.npz")
    pca_emb = apply_pca(embeddings, pca_model=pca_model)

    # Option 3: Default (RNA-FM)
    pca_emb = apply_pca(embeddings)
"""

import numpy as np
import torch
from pathlib import Path

# Global cache: keyed by resolved path so different models don't collide
_PCA_CACHE = {}


def load_pca(pca_path=None, model_name=None):
    """
    Load PCA model from disk.

    Args:
        pca_path: Explicit path to .npz file (takes priority)
        model_name: Model name from registry (e.g. "rnafm", "rinalmo").
                    Used to resolve pca_path if pca_path is None.

    Returns:
        dict with 'mean', 'components', 'explained_variance_ratio', 'n_components'
    """
    if pca_path is None:
        from modules.model_registry import get_pca_path
        pca_path = get_pca_path(model_name)

    pca_path = Path(pca_path).resolve()
    data = np.load(pca_path)
    return {
        'mean': data['mean'],
        'components': data['components'],
        'explained_variance_ratio': data['explained_variance_ratio'],
        'n_components': int(data['n_components']),
    }


def apply_pca(embeddings, pca_model=None, model_name=None):
    """
    Apply PCA transformation to embeddings.

    Args:
        embeddings: numpy array or torch tensor, shape (L, D) or (D,)
        pca_model: Pre-loaded PCA model dict (optional)
        model_name: Model name for auto-loading (optional, default from registry)

    Returns:
        PCA-transformed embeddings, same format as input:
            (L, D) -> (L, k)  or  (D,) -> (k,)
    """
    global _PCA_CACHE

    if pca_model is None:
        from modules.model_registry import get_pca_path
        cache_key = str(get_pca_path(model_name))
        if cache_key not in _PCA_CACHE:
            _PCA_CACHE[cache_key] = load_pca(model_name=model_name)
        pca_model = _PCA_CACHE[cache_key]

    is_torch = isinstance(embeddings, torch.Tensor)
    if is_torch:
        device = embeddings.device
        dtype = embeddings.dtype
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    squeeze_output = False
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)
        squeeze_output = True

    centered = embeddings_np - pca_model['mean']
    pca_result = centered @ pca_model['components'].T

    if squeeze_output:
        pca_result = pca_result.squeeze(0)

    if is_torch:
        pca_result = torch.from_numpy(pca_result).to(dtype).to(device)

    return pca_result
