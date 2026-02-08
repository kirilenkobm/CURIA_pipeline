"""
apply_pca.py: Simple interface to apply RNA-FM PCA transformation

Usage:
    from pca.apply_pca import apply_pca, load_pca

    # Option 1: Automatic loading
    pca_emb = apply_pca(rna_fm_embeddings)

    # Option 2: Load once, apply many times
    pca_model = load_pca()
    pca_emb = apply_pca(rna_fm_embeddings, pca_model)
"""

import numpy as np
import torch
from pathlib import Path

# Global cache for PCA model
_PCA_MODEL = None


def load_pca(pca_path=None):
    """
    Load PCA model from disk.

    Args:
        pca_path: Path to .npz file (default: pca/rnafm_pca_k16.npz)

    Returns:
        dict with 'mean' and 'components' arrays
    """
    if pca_path is None:
        pca_path = Path(__file__).parent / "rnafm_pca_k16.npz"

    data = np.load(pca_path)
    return {
        'mean': data['mean'],
        'components': data['components'],
        'explained_variance_ratio': data['explained_variance_ratio'],
        'n_components': int(data['n_components']),
    }


def apply_pca(embeddings, pca_model=None):
    """
    Apply PCA transformation to RNA-FM embeddings.

    Args:
        embeddings: RNA-FM embeddings, can be:
            - numpy array: (L, 640) or (640,)
            - torch tensor: (L, 640) or (640,)
        pca_model: Pre-loaded PCA model (optional, will auto-load if None)

    Returns:
        PCA-transformed embeddings in same format as input:
            - If input is (L, 640) -> output is (L, 16)
            - If input is (640,) -> output is (16,)
            - If input is torch tensor, output is torch tensor on same device
    """
    global _PCA_MODEL

    # Load PCA model if needed
    if pca_model is None:
        if _PCA_MODEL is None:
            _PCA_MODEL = load_pca()
        pca_model = _PCA_MODEL

    # Handle torch tensors
    is_torch = isinstance(embeddings, torch.Tensor)
    if is_torch:
        device = embeddings.device
        dtype = embeddings.dtype
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    # Handle shape
    original_shape = embeddings_np.shape
    if embeddings_np.ndim == 1:
        # Single vector (640,) -> reshape to (1, 640)
        embeddings_np = embeddings_np.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False

    # Apply PCA: center then project
    centered = embeddings_np - pca_model['mean']
    pca_result = centered @ pca_model['components'].T

    # Restore original shape if needed
    if squeeze_output:
        pca_result = pca_result.squeeze(0)

    # Convert back to torch if needed
    if is_torch:
        pca_result = torch.from_numpy(pca_result).to(dtype).to(device)

    return pca_result


# Convenience function for quick testing
def test():
    """Test PCA application with random data."""
    print("Testing PCA application...")

    # Load model
    model = load_pca()
    print(f"Loaded PCA model: {model['n_components']} components")
    print(f"Explained variance: {model['explained_variance_ratio'].sum():.2%}")

    # Test with numpy array (L, 640)
    emb_np = np.random.randn(100, 640).astype(np.float32)
    pca_np = apply_pca(emb_np, model)
    print(f"\nNumPy (L, 640): {emb_np.shape} -> {pca_np.shape}")

    # Test with numpy vector (640,)
    vec_np = np.random.randn(640).astype(np.float32)
    pca_vec_np = apply_pca(vec_np, model)
    print(f"NumPy (640,): {vec_np.shape} -> {pca_vec_np.shape}")

    # Test with torch tensor
    emb_torch = torch.randn(100, 640)
    pca_torch = apply_pca(emb_torch, model)
    print(f"\nTorch (L, 640): {emb_torch.shape} -> {pca_torch.shape}")
    print(f"Device: {pca_torch.device}, dtype: {pca_torch.dtype}")

    # Test with torch vector
    vec_torch = torch.randn(640)
    pca_vec_torch = apply_pca(vec_torch, model)
    print(f"Torch (640,): {vec_torch.shape} -> {pca_vec_torch.shape}")

    # Test with GPU if available
    if torch.cuda.is_available():
        emb_cuda = torch.randn(100, 640).cuda()
        pca_cuda = apply_pca(emb_cuda, model)
        print(f"\nCUDA (L, 640): {emb_cuda.shape} -> {pca_cuda.shape}")
        print(f"Device: {pca_cuda.device}")
    elif torch.backends.mps.is_available():
        emb_mps = torch.randn(100, 640).to('mps')
        pca_mps = apply_pca(emb_mps, model)
        print(f"\nMPS (L, 640): {emb_mps.shape} -> {pca_mps.shape}")
        print(f"Device: {pca_mps.device}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test()
