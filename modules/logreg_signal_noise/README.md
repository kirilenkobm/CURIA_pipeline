t# Logistic Regression Signal/Noise Classifier

This module provides a trained logistic regression model for classifying RNA-FM embeddings as either "proper RNA" (signal) or "noise".

## Files

- `train_logreg.py` - Script to train the logistic regression model on labeled data
- `apply_logreg.py` - Functions to apply the trained model to RNA-FM embeddings
- `logreg_noise_model.json` - Trained logistic regression model (binary classifier)

## Usage

### Training a Model

```python
from logreg_signal_noise.train_logreg import train_model

# Assumes you have labeled training data in the specified format
train_model(
    data_path='path/to/train.npz',
    model_path='logreg_noise_model.json'
)
```

### Applying the Model

```python
from logreg_signal_noise.apply_logreg import score_embeddings, load_logreg_model
from pca.apply_pca import load_pca

# Load models
logreg_model = load_logreg_model('logreg_noise_model.json')
pca_model = load_pca()

# Score embeddings (shape: N x 640)
probs, status = score_embeddings(
    embeddings_640,
    model=logreg_model
)

# probs: probability of being "proper RNA" (0-1)
# status: 'signal' (proper RNA) or 'noise'
```

## Model Details

- **Input**: RNA-FM embeddings (640 dimensions) transformed via PCA to 16 dimensions
- **Output**: Probability that the embedding represents "proper RNA" (class 1) vs "noise" (class 0)
- **Threshold**: Default threshold of 0.5 for binary classification
- **Use Case**: Filtering noisy sequences and identifying functional RNA regions

## Integration

This module is used by:
- `modules/utils/query_islands_scanner.py` - For scanning orthologous query regions and identifying functional "islands" that may contain lncRNAs
- Analysis notebooks for quality control and region identification

Note: The islands identified are candidate functional regions in the query genome. The actual lncRNA annotation happens in subsequent pipeline steps.
