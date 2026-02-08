# RNA-FM PCA Model (k=16)

Dimensionality reduction for RNA-FM embeddings: 640 → 16 dimensions.

## Training Data

**Total samples**: ~100,000 position embeddings
**Input dimension**: 640 (RNA-FM layer 12)
**Output dimension**: 16 (PCA components)

### Composition

- **30% Genomic Noise**: Random intergenic regions from hg38 + mm10
- **70% ncRNA Windows**: Various non-coding RNA types
  - lncRNA (long non-coding RNA)
  - snoRNA (small nucleolar RNA)
  - miRNA (microRNA)
  - snRNA (small nuclear RNA)
  - misc_RNA (miscellaneous RNA)
  - scaRNA (small Cajal body-specific RNA)

### Window Sizes

Variable window sizes sampled uniformly: **48, 64, 80, 96, 128, 160, 192, 224, 256 nt**

This ensures the PCA model works well across different RNA lengths.

## Model File

**File**: `rnafm_pca_k16.npz`
**Size**: ~20-40 KB (compressed)

### Contents
```python
{
    'mean': (640,),                    # PCA centering mean
    'components': (16, 640),           # PCA projection matrix
    'explained_variance': (16,),       # Variance per component
    'explained_variance_ratio': (16,), # Fraction of variance explained
    'n_components': 16,                # Number of components
    'n_samples': int,                  # Training samples used
    'input_dim': 640                   # Input feature dimension
}
```

## Usage

### Load PCA Model

```python
import numpy as np

pca_data = np.load('pca/rnafm_pca_k16.npz')
mean = pca_data['mean']
components = pca_data['components']
```

### Apply PCA Transform

```python
def apply_pca(embeddings):
    """
    Transform RNA-FM embeddings to 16D space.

    Args:
        embeddings: (L, 640) numpy array from RNA-FM layer 12

    Returns:
        (L, 16) PCA-transformed embeddings
    """
    centered = embeddings - mean
    return centered @ components.T
```

### Complete Example

```python
import sys
import numpy as np
import torch

# Load RNA-FM
sys.path.insert(0, '/path/to/RNA-FM')
import fm

model, alphabet = fm.pretrained.rna_fm_t12()
model = model.to('cuda')  # or 'mps' or 'cpu'
model.eval()
batch_converter = alphabet.get_batch_converter()

# Load PCA
pca_data = np.load('pca/rnafm_pca_k16.npz')
mean = pca_data['mean']
components = pca_data['components']

# Get RNA-FM embeddings
def get_rnafm_embeddings(seq):
    seq = seq.upper().replace('T', 'U')
    data = [('seq', seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to('cuda')

    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
        emb = results['representations'][12][0, 1:-1, :].cpu().numpy()

    return emb  # shape: (L, 640)

# Apply PCA
seq = "ACGUACGUACGU..."
emb = get_rnafm_embeddings(seq)           # (L, 640)
pca_emb = (emb - mean) @ components.T     # (L, 16)

print(f"Original: {emb.shape}")      # (L, 640)
print(f"PCA: {pca_emb.shape}")       # (L, 16)
```

## GPU Acceleration

After PCA transformation, the 16D embeddings are small enough for efficient GPU operations:

```python
# Convert to torch tensor for GPU matmul
pca_emb_gpu = torch.from_numpy(pca_emb).float().to('cuda')

# Now you can do fast computations
# e.g., distance matrices, MMD, alignment scoring, etc.
```

## Reproducibility

- **Random seed**: 42
- **Genomes**: hg38 (human), mm10 (mouse)
- **RNA-FM version**: pretrained.rna_fm_t12()
- **Training script**: `fit_pca.py`

## Benefits

1. **Memory efficiency**: 40× reduction (640 → 16 dimensions)
2. **Speed**: Faster distance computations, MMD calculations
3. **GPU-friendly**: Small vectors fit in GPU memory easily
4. **Noise reduction**: Removes low-variance dimensions
5. **Interpretability**: Top PCs capture major structural patterns

## Explained Variance

Run `fit_pca.py` to see the explained variance ratio for each component. Typically:

- **PC1-8**: ~60-70% of total variance
- **PC1-16**: ~75-85% of total variance

This means 16 components capture most of the structural information while being 40× more compact.

## Retraining

To retrain with different parameters:

```bash
# Edit parameters in fit_pca.py
# - TARGET_EMBEDDINGS: 100_000 (or up to 1_000_000)
# - N_PCA_COMPONENTS: 16
# - NOISE_RATIO: 0.3
# - WINDOW_SIZES: [48, 64, ..., 256]

python3 pca/fit_pca.py
```

## Citation

If you use this PCA model, please cite:

- **RNA-FM**: [Chen et al. 2022](https://github.com/ml4bio/RNA-FM)
- **This work**: CURIA RNA structure analysis pipeline
