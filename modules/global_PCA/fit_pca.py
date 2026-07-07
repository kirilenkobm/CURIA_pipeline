#!/usr/bin/env python3
"""
fit_pca.py: Train PCA on RNA foundation model embeddings.

Collects embeddings from:
- 30% genomic noise (intergenic regions)
- 70% ncRNAs (lncRNA, snoRNA, miRNA, snRNA, misc_RNA, scaRNA)

Window sizes: 48-256 nt.  Target: ~100k position-level embeddings -> PCA(k).

This fits the MATCHING PCA (per-token, k from registry: 16). The finding
scanner uses a separate PCA built by build_rinalmo_finding.py, not this script.

Usage:
    python fit_pca.py                       # default: RiNALMo, k=16 (matching PCA)
    python fit_pca.py --model rnafm         # RNA-FM (deprecated), k=16
    python fit_pca.py --model rinalmo --pca-components 32
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import random
import gc
import torch
from pathlib import Path

# Make modules importable
MODULES_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.model_registry import get_model_config, load_model

# Paths
REF_BED = "input_data/reference_annotations/hg38.input.bed"
REF_2BIT = "input_data/2bit/hg38.2bit"
MM_2BIT = "input_data/2bit/mm10.2bit"
BIOMART_PATH = "input_data/biomart/hg38.biomart.data.tsv"

# Parameters
TARGET_EMBEDDINGS = 100_000
NOISE_RATIO = 0.3
WINDOW_SIZES = [48, 64, 80, 96, 128, 160, 192, 224, 256]
RANDOM_SEED = 42
BATCH_SIZE = 128
NCRNA_BIOTYPES = ['lncRNA', 'snoRNA', 'miRNA', 'snRNA', 'misc_RNA', 'scaRNA']


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCA on RNA model embeddings")
    parser.add_argument("--model", default=None,
                        help="Model name from registry (default: rinalmo)")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="Number of PCA components (default: from registry)")
    parser.add_argument("--target-embeddings", type=int, default=TARGET_EMBEDDINGS,
                        help=f"Target number of position-level embeddings (default: {TARGET_EMBEDDINGS})")
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_embedding_single(seq, model, tokenize_fn, extract_fn):
    """Get per-position embeddings (L, D) for a single sequence."""
    seq = seq.upper().replace('T', 'U')
    tokens = tokenize_fn([seq])
    with torch.no_grad():
        reps = extract_fn(model, tokens)
    return reps[0, 1:1 + len(seq), :].cpu().numpy()


def get_embeddings_batch(sequences, model, tokenize_fn, extract_fn):
    """Get per-position embeddings for a batch of sequences."""
    if not sequences:
        return []
    sequences_clean = [s.upper().replace('T', 'U') for s in sequences]
    tokens = tokenize_fn(sequences_clean)
    with torch.no_grad():
        reps = extract_fn(model, tokens)
    embeddings = []
    for i, seq in enumerate(sequences_clean):
        emb = reps[i, 1:1 + len(seq), :].cpu().numpy().copy()
        embeddings.append(emb)
    return embeddings


def extract_genomic_noise_windows(accessor, chrom_sizes, n_windows, window_sizes):
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
        except Exception:
            continue
    return windows


def get_transcript_sequence(transcript, accessor):
    seq_parts = [str(accessor.fetch(transcript.chrom, int(b[0]), int(b[1]))).upper()
                 for b in transcript.blocks]
    seq = ''.join(seq_parts)
    if transcript.strand == -1:
        comp = {'A': 'U', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        seq = ''.join(comp.get(b, 'N') for b in reversed(seq))
    else:
        seq = seq.replace('T', 'U')
    return seq


def extract_transcript_windows(transcript, accessor, window_sizes, max_windows=5):
    seq = get_transcript_sequence(transcript, accessor)
    if 'N' in seq or len(seq) < min(window_sizes):
        return []
    windows = []
    for window_size in window_sizes:
        if len(seq) < window_size:
            continue
        step = max(1, (len(seq) - window_size) // max_windows)
        for start in range(0, len(seq) - window_size + 1, step):
            windows.append(seq[start:start + window_size])
            if len(windows) >= max_windows:
                break
    return windows


def main():
    args = parse_args()

    model_cfg = get_model_config(args.model)
    model_name = args.model or "rinalmo"  # match get_model_config's default (DEFAULT_MODEL)
    n_components = args.pca_components or model_cfg["pca_components"]
    target_embeddings = args.target_embeddings

    print(f"Model: {model_name} (dim={model_cfg['emb_dim']})")
    print(f"PCA components: {n_components}")
    print(f"Target embeddings: {target_embeddings:,}")
    print(f"Noise ratio: {NOISE_RATIO:.1%}")
    print(f"Window sizes: {WINDOW_SIZES}")

    device = get_device()
    print(f"Using device: {device}")

    # Load model via registry
    model, tokenize_fn, extract_fn = load_model(model_name, device)
    print(f"{model_name} loaded\n")

    # Load pyrion
    import pyrion
    from pyrion import TwoBitAccessor

    print("=" * 60)
    print("STEP 1: Loading reference data")
    print("=" * 60)

    transcripts = pyrion.io.read_bed12_file(REF_BED)
    biodata = pyrion.io.read_gene_data(BIOMART_PATH, gene_column=1, transcript_id_column=2,
                                        gene_name_column=3, transcript_type_column=4)
    transcripts.bind_gene_data(biodata)
    print(f"Loaded {len(transcripts)} transcripts")

    hg38_accessor = TwoBitAccessor(REF_2BIT)
    chrom_sizes = {}
    with open("input_data/reference_annotations/hg38.chrom.sizes") as f:
        for line in f:
            if line.strip():
                chrom, size = line.strip().split()
                chrom_sizes[chrom] = int(size)
    print(f"Loaded {len(chrom_sizes)} chromosomes")

    try:
        mm_accessor = TwoBitAccessor(MM_2BIT)
        print("Mouse genome loaded")
        use_mouse = True
    except Exception:
        print("Mouse genome not available")
        mm_accessor = None
        use_mouse = False

    print("\n" + "=" * 60)
    print("STEP 2: Collecting embeddings")
    print("=" * 60)

    n_noise = int(target_embeddings * NOISE_RATIO)
    n_ncrna = target_embeddings - n_noise

    print(f"Target noise embeddings: {n_noise:,}")
    print(f"Target ncRNA embeddings: {n_ncrna:,}")

    # Collect genomic noise
    print("\nCollecting genomic noise...")
    noise_windows_hg38 = extract_genomic_noise_windows(hg38_accessor, chrom_sizes,
                                                        n_noise // 2 if use_mouse else n_noise,
                                                        WINDOW_SIZES)
    print(f"Collected {len(noise_windows_hg38):,} hg38 noise windows")

    if use_mouse:
        mm_chrom_sizes = {f'chr{i}': 200_000_000 for i in range(1, 20)}
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

    by_biotype = {}
    for t in ncrna_transcripts:
        if t.biotype not in by_biotype:
            by_biotype[t.biotype] = []
        by_biotype[t.biotype].append(t)

    for bt, tlist in by_biotype.items():
        print(f"  {bt}: {len(tlist):,}")

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
    print(f"\nComputing {model_name} embeddings...")
    all_embeddings = []

    for i in tqdm(range(0, len(all_sequences), BATCH_SIZE), desc=f"{model_name} batches"):
        batch = all_sequences[i:i+BATCH_SIZE]
        try:
            batch_embs = get_embeddings_batch(batch, model, tokenize_fn, extract_fn)
            all_embeddings.extend(batch_embs)
            del batch_embs
            if i % (BATCH_SIZE * 5) == 0:
                clear_memory()
        except Exception as e:
            print(f"\nBatch error at {i}, falling back to single processing: {e}")
            for seq in batch:
                try:
                    emb = get_embedding_single(seq, model, tokenize_fn, extract_fn)
                    all_embeddings.append(emb)
                except Exception:
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

    pca = PCA(n_components=n_components)
    pca.fit(embedding_matrix)

    print(f"\nPCA trained with {n_components} components")
    print(f"Total samples: {embedding_matrix.shape[0]:,}")
    print(f"Input dimension: {embedding_matrix.shape[1]}")
    print(f"\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i:2d}: {var:.4f} ({var*100:.2f}%)")

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nCumulative explained variance:")
    for k in [8, min(n_components, 16), n_components]:
        if k <= len(cumulative):
            print(f"  PC1-{k}: {cumulative[k-1]:.4f} ({cumulative[k-1]*100:.2f}%)")

    # Save PCA
    print("\n" + "=" * 60)
    print("STEP 4: Saving PCA model")
    print("=" * 60)

    output_file = str(MODULES_DIR / "global_PCA" / model_cfg["pca_file"])
    np.savez_compressed(
        output_file,
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=n_components,
        n_samples=embedding_matrix.shape[0],
        input_dim=embedding_matrix.shape[1],
    )

    print(f"Saved PCA model to {output_file}")
    file_size_kb = os.path.getsize(output_file) / 1024
    print(f"File size: {file_size_kb:.2f} KB")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
