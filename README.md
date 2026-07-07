# CURIA

**Cross-species Unified ncRNA Inference and Annotation**

CURIA is a research prototype for cross-species ncRNA annotation using genome alignment chains, orthology-guided locus projection, and RNA foundation model embeddings.

It supports:
- short ncRNA refinement by local matching in embedding space
- long ncRNA analysis via localized structured subregions (“islands”)
- genome-scale analysis by restricting search to syntenic candidate loci

For methodological details, validation, and limitations, see the accompanying preprint.

## Status

Research prototype. Preprint in preparation.

## Installation

The quickest path is the bundled installer, which sets up the environment (via
[uv](https://docs.astral.sh/uv/)), handles the macOS OpenMP prerequisite, and downloads
the default RiNALMo weights:

```bash
git clone --recurse-submodules git@github.com:kirilenkobm/curia_pipeline.git
cd curia_pipeline
./install.sh                 # env + RiNALMo weights (~2.6 GB)
# ./install.sh --with-rnafm  # also fetch RNA-FM weights (comparison only)
# ./install.sh --no-weights  # environment only
source .venv/bin/activate
```

Or do it manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync
source .venv/bin/activate
```

> **macOS note:** OpenMP is required for scikit-learn and other numerical libraries.
> Install it once with: `brew install libomp`

**Model weights:**
- **RiNALMo** `giga-v1` (~2.6 GB, default) downloads automatically on first run
  (cached at `~/.cache/rinalmo_pretrained`), or manually via `./download_rinalmo_model.py`.
- **RNA-FM** (~1.1 GB, comparison only / deprecated) downloads automatically when
  `--model rnafm` is used, or manually via `./download_rnafm_model.py`.

## Model choice

CURIA defaults to **RiNALMo** (1280-dim, 650M params). RNA-FM is retained behind
`--model rnafm` for comparison only and is **deprecated**.

Why RiNALMo is the default:
- **Context stability.** RiNALMo embeddings are essentially invariant to flanking
  sequence (≈0 MMD drift), whereas RNA-FM embeddings shift substantially when input
  boundaries move. This is the root cause of RNA-FM's expensive per-window re-embedding.
- **Embed once, align at nucleotide resolution.** Because RiNALMo is context-stable, island
  alignment embeds each island **once** and matches them by Smith-Waterman on the per-token
  cosine dotplot — no per-window re-embedding. Dramatically faster and more accurate on true
  matches (see `notebooks/matching_benchmark.ipynb`).

Per-model parameters (PCA files, signal/noise classifier, scan/match thresholds,
embedding strategy) live in [`modules/model_registry.py`](modules/model_registry.py) and
are selected automatically by `--model`.

```bash
./curia.py ... --model rinalmo   # default
./curia.py ... --model rnafm     # deprecated, comparison only
```

**Artifact provenance** (committed under `modules/`):
- `global_PCA/rinalmo_pca_k16.npz` — matching PCA (position-level embeddings).
- `global_PCA/rinalmo_pca_find_k64.npz` — island-finding PCA.
- `logreg_signal_noise/logreg_noise_model_rinalmo.json` — signal/noise (island-finding)
  classifier. The PCA and classifier are built together by
  `logreg_signal_noise/build_rinalmo_finding.py` (`--from-cache` re-fits from the cached
  features); sweep in `notebooks/finding_benchmark.ipynb`.

---

## Quick Start

```bash
# Optional: benchmark optimal GPU batch size for your hardware
python modules/GPU_executor/benchmark_batch_size.py

# Run smoke test (<1 minute on strong machines)
./curia.py \
  --ref-bed12 input_data/reference_annotation/smoke_test.bed \
  --reference-metadata input_data/reference_annotation/smoke_test.metadata.tsv \
  --chain input_data/chains/smoke_test.chain.gz \
  --ref-2bit input_data/2bit/hg38.test.subset.2bit \
  --query-2bit input_data/2bit/mm39.test.subset.2bit \
  --output-dir smoke_test_output \
  --cpu-max-workers 12 \
  --gpu-min-batch 4 \
  --gpu-max-batch 16 \
  --gpu-logger

# For a more comprehensive test (~20 minutes)
./curia.py \
  --ref-bed12 input_data/reference_annotation/test_sample.bed \
  --reference-metadata input_data/reference_annotation/test_sample.metadata.tsv \
  --chain input_data/chains/test_sample.chain.gz \
  --ref-2bit input_data/2bit/hg38.test.subset.2bit \
  --query-2bit input_data/2bit/mm39.test.subset.2bit \
  --output-dir test_output
```

---

## Requirements

**Input files:**
- Reference annotation (BED12)
- Reference metadata (TSV with gene name and biotype mappings; can be downloaded from Ensembl BioMart with attributes: transcript ID, gene name, and transcript biotype)
- Query and reference genomes (2bit format)
- Genome alignment chains

**Compute:**
- CPU required for RNA TOGA and sequence processing
- GPU optional but recommended for foundation-model embeddings
- Tested on macOS (MPS) and Linux (CUDA)

---

## Usage

```bash
./curia.py \
  --ref-bed12 $REFERENCE_BED12 \
  --reference-metadata $REFERENCE_METADATA \
  --chain $ALIGNMENT_CHAINS \
  --ref-2bit $REF_2BIT \
  --query-2bit $QUERY_2BIT \
  --output-dir $OUTPUT_DIR \
  --cpu-max-workers 128 \               # max concurrent async workers (default: 128)
  --gpu-max-batch 160 \                 # max GPU batch size (default: 160, tune with benchmark script)
  --gpu-min-batch 32 \                  # min batch size before timeout (default: 32)
  --no-cleanup                          # optional: keep all intermediate files (SQLite DBs, joblists)
```

**Performance tuning:**
- `--cpu-max-workers` controls concurrent async I/O workers (not threads), allowing high parallelism for GPU-bound tasks
- `--gpu-max-batch` sets maximum batch size sent to GPU; use `python modules/GPU_executor/benchmark_batch_size.py` to find optimal value for your hardware
- `--gpu-min-batch` sets minimum batch size before GPU executor times out and processes incomplete batch

---

## Output

By default, CURIA automatically cleans up and organizes outputs into a user-friendly structure:

```
output_dir/
├── query_annotation/
│   ├── short_ncRNA.bed              # Short ncRNA annotations (≤160bp)
│   ├── short_ncRNA_details.tsv      # Detailed short ncRNA results
│   ├── aligned_query_islands.bed     # Aligned lncRNA islands in query
│   ├── aligned_reference_islands.bed # Matching reference islands
│   ├── raw_reference_islands.bed    # All reference islands (QC)
│   └── raw_query_islands.bed        # All query islands (QC)
├── island_alignment_results.tsv     # Island alignment scores
├── preprocessed_reference_data.json # Reusable reference data
├── reference_union_transcripts.bed  # Collapsed reference isoforms
├── reference_union_transcripts_metadata.tsv
├── mappings/
│   ├── union_to_isoforms.json       # Transcript → isoforms mapping
│   ├── union_to_query.json          # Transcript → query regions mapping
│   └── query_regions_clusters.json  # Merged query regions
└── toga_results/
    ├── rna_orthologous_regions.tsv          # RNA orthology predictions
    ├── toga_orthologous_regions.tsv         # Original TOGA output
    └── original_toga_classification_table.tsv # TOGA classification scores
```

**Cleanup options:**
- Use `--no-cleanup` to keep all intermediate files (SQLite DBs, joblists, etc.)

See [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) for detailed file descriptions.

---

## Validation

Evaluated on several mammalian genome pairs, including human–mouse.

For full evaluation details and limitations, see the preprint.

---

## Citation

If you use CURIA, please cite:

Kirilenko, B.M. (2026).  
Cross-species ncRNA annotation using synteny-constrained embedding similarity.  
bioRxiv (preprint).

---

## References

- **TOGA:** Kirilenko et al., *Integrating gene annotation with orthology inference at scale*, Science (2023)
- **RiNALMo:** Penić et al., *RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks*, arXiv (2024)
- **RNA-FM:** Chen et al., *Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions*, arXiv (2022)
