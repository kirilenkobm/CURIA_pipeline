#!/usr/bin/env python3
"""
fit_pca.py: Train PCA on RNA-FM embeddings

Collects embeddings from:
- 30% genomic noise (intergenic regions)
- 70% ncRNAs (lncRNA, snoRNA, miRNA, snRNA, misc_RNA, scaRNA)

Window sizes: 48-256 nt
Target: ~100k-1M embeddings (640-dim) -> PCA(16)
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import random
import gc
import torch

# Paths
REF_BED = "input_data/reference_annotations/hg38.input.bed"
REF_2BIT = "input_data/2bit/hg38.2bit"
MM_2BIT = "input_data/2bit/mm10.2bit"
BIOMART_PATH = "input_data/biomart/hg38.biomart.data.tsv"
RNA_FM_PATH = "/Users/Bogdan.Kirilenko/PycharmProjects/RNA-FM"

# Parameters
N_PCA_COMPONENTS = 16
TARGET_EMBEDDINGS = 100_000  # Start with 100k, can increase to 1M
NOISE_RATIO = 0.3
WINDOW_SIZES = [48, 64, 80, 96, 128, 160, 192, 224, 256]
RANDOM_SEED = 42
BATCH_SIZE = 128  # Process sequences in batches for speed

# ncRNA biotypes to include
NCRNA_BIOTYPES = ['lncRNA', 'snoRNA', 'miRNA', 'snRNA', 'misc_RNA', 'scaRNA']

print(f"Target embeddings: {TARGET_EMBEDDINGS:,}")
print(f"Noise ratio: {NOISE_RATIO:.1%}")
print(f"Window sizes: {WINDOW_SIZES}")
print(f"ncRNA biotypes: {NCRNA_BIOTYPES}")

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load RNA-FM
sys.path.insert(0, RNA_FM_PATH)
import fm
model, alphabet = fm.pretrained.rna_fm_t12()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()
print("RNA-FM loaded\n")

# Load pyrion
import pyrion
from pyrion import TwoBitAccessor


def clear_memory():
    """Clear GPU/MPS memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_rna_fm_embedding(seq: str) -> np.ndarray:
    """Get RNA-FM positional embeddings (L, 640)."""
    seq = seq.upper().replace('T', 'U')
    data = [('seq', seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
        emb = results['representations'][12][0, 1:-1, :].cpu().numpy()

    del results, tokens
    return emb


def get_rna_fm_embeddings_batch(sequences: list) -> list:
    """Get RNA-FM embeddings for a batch of sequences (faster)."""
    if not sequences:
        return []

    # Prepare batch
    sequences_clean = [s.upper().replace('T', 'U') for s in sequences]
    data = [(f'seq_{i}', s) for i, s in enumerate(sequences_clean)]

    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    embeddings = []
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
        representations = results['representations'][12]

        # Extract embeddings for each sequence (skip BOS/EOS tokens)
        for i in range(len(sequences_clean)):
            seq_len = len(sequences_clean[i])
            emb = representations[i, 1:seq_len+1, :].cpu().numpy().copy()
            embeddings.append(emb)

        # Explicitly delete tensors
        del representations
        del results

    del tokens
    del data
    del sequences_clean

    return embeddings


def extract_genomic_noise_windows(accessor: TwoBitAccessor, chrom_sizes: dict,
                                  n_windows: int, window_sizes: list) -> list:
    """Extract random genomic windows (intergenic noise)."""
    windows = []
    chroms = [c for c in chrom_sizes.keys() if c.startswith('chr') and '_' not in c and c != 'chrM']

    for _ in range(n_windows):
        chrom = random.choice(chroms)
        window_size = random.choice(window_sizes)
        max_start = chrom_sizes[chrom] - window_size
        if max_start <= 0:
            continue
        start = random.randint(0, max_start)
        end = start + window_size

        try:
            seq = str(accessor.fetch(chrom, start, end)).upper().replace('T', 'U')
            if 'N' not in seq and len(seq) == window_size:
                windows.append(seq)
        except:
            continue

    return windows


def get_transcript_sequence(transcript, accessor: TwoBitAccessor) -> str:
    """Get spliced transcript sequence as RNA."""
    seq_parts = [str(accessor.fetch(transcript.chrom, int(b[0]), int(b[1]))).upper()
                 for b in transcript.blocks]
    seq = ''.join(seq_parts)

    if transcript.strand == -1:
        comp = {'A': 'U', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        seq = ''.join(comp.get(b, 'N') for b in reversed(seq))
    else:
        seq = seq.replace('T', 'U')

    return seq


def extract_transcript_windows(transcript, accessor: TwoBitAccessor,
                               window_sizes: list, max_windows: int = 5) -> list:
    """Extract sliding windows from a transcript."""
    seq = get_transcript_sequence(transcript, accessor)
    if 'N' in seq or len(seq) < min(window_sizes):
        return []

    windows = []
    for window_size in window_sizes:
        if len(seq) < window_size:
            continue
        # Take up to max_windows per size
        step = max(1, (len(seq) - window_size) // max_windows)
        for start in range(0, len(seq) - window_size + 1, step):
            windows.append(seq[start:start + window_size])
            if len(windows) >= max_windows:
                break

    return windows


# Main collection
print("=" * 60)
print("STEP 1: Loading reference data")
print("=" * 60)

transcripts = pyrion.io.read_bed12_file(REF_BED)
biodata = pyrion.io.read_gene_data(BIOMART_PATH, gene_column=1, transcript_id_column=2,
                                    gene_name_column=3, transcript_type_column=4)
transcripts.bind_gene_data(biodata)
print(f"Loaded {len(transcripts)} transcripts")

# Get chromosome sizes
hg38_accessor = TwoBitAccessor(REF_2BIT)
chrom_sizes = {}
with open("input_data/reference_annotations/hg38.chrom.sizes") as f:
    for line in f:
        if line.strip():
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
print(f"Loaded {len(chrom_sizes)} chromosomes")

# Try to load mouse data
try:
    mm_accessor = TwoBitAccessor(MM_2BIT)
    print("Mouse genome loaded")
    use_mouse = True
except:
    print("Mouse genome not available")
    mm_accessor = None
    use_mouse = False

print("\n" + "=" * 60)
print("STEP 2: Collecting embeddings")
print("=" * 60)

n_noise = int(TARGET_EMBEDDINGS * NOISE_RATIO)
n_ncrna = TARGET_EMBEDDINGS - n_noise

print(f"Target noise embeddings: {n_noise:,}")
print(f"Target ncRNA embeddings: {n_ncrna:,}")

# Collect genomic noise
print("\nCollecting genomic noise...")
noise_windows_hg38 = extract_genomic_noise_windows(hg38_accessor, chrom_sizes,
                                                    n_noise // 2 if use_mouse else n_noise,
                                                    WINDOW_SIZES)
print(f"Collected {len(noise_windows_hg38):,} hg38 noise windows")

if use_mouse:
    mm_chrom_sizes = {f'chr{i}': 200_000_000 for i in range(1, 20)}  # approximate
    noise_windows_mm = extract_genomic_noise_windows(mm_accessor, mm_chrom_sizes,
                                                      n_noise // 2, WINDOW_SIZES)
    print(f"Collected {len(noise_windows_mm):,} mm10 noise windows")
    all_noise_windows = noise_windows_hg38 + noise_windows_mm
else:
    all_noise_windows = noise_windows_hg38

print(f"Total noise windows: {len(all_noise_windows):,}")

# Collect ncRNA windows
print("\nCollecting ncRNA windows...")
ncrna_transcripts = [t for t in transcripts if t.biotype in NCRNA_BIOTYPES]
print(f"Found {len(ncrna_transcripts):,} ncRNA transcripts")

# Group by biotype
by_biotype = {}
for t in ncrna_transcripts:
    if t.biotype not in by_biotype:
        by_biotype[t.biotype] = []
    by_biotype[t.biotype].append(t)

for bt, tlist in by_biotype.items():
    print(f"  {bt}: {len(tlist):,}")

# Sample transcripts and extract windows
random.seed(RANDOM_SEED)
ncrna_windows = []
max_windows_per_tx = 5

for biotype, tlist in by_biotype.items():
    random.shuffle(tlist)
    for t in tqdm(tlist, desc=f"Extracting {biotype}"):
        windows = extract_transcript_windows(t, hg38_accessor, WINDOW_SIZES, max_windows_per_tx)
        ncrna_windows.extend(windows)
        if len(ncrna_windows) >= n_ncrna:
            break
    if len(ncrna_windows) >= n_ncrna:
        break

print(f"Collected {len(ncrna_windows):,} ncRNA windows")

# Combine and shuffle
all_sequences = all_noise_windows + ncrna_windows[:n_ncrna]
random.shuffle(all_sequences)
print(f"\nTotal sequences: {len(all_sequences):,}")

# Compute embeddings (with batching)
print("\nComputing RNA-FM embeddings...")
all_embeddings = []

# Process in batches for speed
for i in tqdm(range(0, len(all_sequences), BATCH_SIZE), desc="RNA-FM batches"):
    batch = all_sequences[i:i+BATCH_SIZE]
    try:
        batch_embs = get_rna_fm_embeddings_batch(batch)
        all_embeddings.extend(batch_embs)
        del batch_embs

        # Clear memory more aggressively every 5 batches
        if i % (BATCH_SIZE * 5) == 0:
            clear_memory()
    except Exception as e:
        print(f"\nBatch error at {i}, falling back to single processing: {e}")
        # Fallback to single sequence processing
        for seq in batch:
            try:
                emb = get_rna_fm_embedding(seq)
                all_embeddings.append(emb)
            except:
                continue
        clear_memory()

    del batch

print(f"Computed {len(all_embeddings):,} embeddings")

# Stack embeddings
print("\nStacking embeddings...")
embedding_matrix = np.vstack(all_embeddings)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

del all_embeddings
clear_memory()

# Train PCA
print("\n" + "=" * 60)
print("STEP 3: Training PCA")
print("=" * 60)

pca = PCA(n_components=N_PCA_COMPONENTS)
pca.fit(embedding_matrix)

print(f"\nPCA trained with {N_PCA_COMPONENTS} components")
print(f"Total samples: {embedding_matrix.shape[0]:,}")
print(f"Input dimension: {embedding_matrix.shape[1]}")
print(f"\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    print(f"  PC{i:2d}: {var:.4f} ({var*100:.2f}%)")

cumulative = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative explained variance:")
print(f"  PC1-8:  {cumulative[7]:.4f} ({cumulative[7]*100:.2f}%)")
print(f"  PC1-16: {cumulative[15]:.4f} ({cumulative[15]*100:.2f}%)")

# Save PCA
print("\n" + "=" * 60)
print("STEP 4: Saving PCA model")
print("=" * 60)

output_file = "pca/rnafm_pca_k16.npz"
np.savez_compressed(
    output_file,
    mean=pca.mean_,
    components=pca.components_,
    explained_variance=pca.explained_variance_,
    explained_variance_ratio=pca.explained_variance_ratio_,
    n_components=N_PCA_COMPONENTS,
    n_samples=embedding_matrix.shape[0],
    input_dim=embedding_matrix.shape[1]
)

print(f"Saved PCA model to {output_file}")

import os
file_size_kb = os.path.getsize(output_file) / 1024
print(f"File size: {file_size_kb:.2f} KB")

# Show usage example
print("\n" + "=" * 60)
print("USAGE EXAMPLE")
print("=" * 60)
print("""
import numpy as np

# Load PCA model
pca_data = np.load('pca/rnafm_pca_k16.npz')
mean = pca_data['mean']
components = pca_data['components']

# Apply PCA to RNA-FM embeddings
def apply_pca(embeddings):
    '''
    embeddings: (L, 640) numpy array from RNA-FM
    returns: (L, 16) PCA-transformed embeddings
    '''
    centered = embeddings - mean
    return centered @ components.T

# Example
# emb = get_rna_fm_embedding(sequence)  # shape: (L, 640)
# pca_emb = apply_pca(emb)              # shape: (L, 16)
""")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
