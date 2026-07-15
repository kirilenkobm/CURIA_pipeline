# CURIA

**Cross-species Unified ncRNA Inference and Annotation**

CURIA is a research prototype for cross-species ncRNA correspondence analysis
using genome alignment chains, chain-based candidate-locus classification, and
RNA foundation-model embeddings.

It supports:
- compact ncRNA matching using embedding-distance comparison
- long ncRNA analysis through localized embedding-positive subregions (“islands”)
- genome-scale analysis by restricting searches to chain-supported candidate loci

For methodological details, validation, and limitations, see the accompanying preprint.

## Status

Research prototype. A preprint is in preparation. The exact per-species result
snapshot used for the manuscript is archived on Zenodo:

https://doi.org/10.5281/zenodo.21383175

## Installation

The quickest path is the bundled installer, which sets up the environment (via
[uv](https://docs.astral.sh/uv/)), handles the macOS OpenMP prerequisite, and downloads
the default RiNALMo weights:

```bash
git clone --recurse-submodules https://github.com/kirilenkobm/CURIA_pipeline.git
cd CURIA_pipeline
./install.sh                 # env + RiNALMo weights (~2.6 GB)
# ./install.sh --with-rnafm  # also fetch RNA-FM weights (comparison only)
# ./install.sh --no-weights  # environment only
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

# Run smoke test (a couple of minutes on strong machines)
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
- Attention uses PyTorch's built-in `scaled_dot_product_attention`, allowing
  memory-efficient or Flash Attention backends when supported by the installed
  PyTorch/CUDA configuration. No separate `flash-attn` package is required.

---

## Usage

```bash
./curia.py \
  --ref-bed12 "$REFERENCE_BED12" \
  --reference-metadata "$REFERENCE_METADATA" \
  --chain "$ALIGNMENT_CHAINS" \
  --ref-2bit "$REF_2BIT" \
  --query-2bit "$QUERY_2BIT" \
  --output-dir "$OUTPUT_DIR" \
  --cpu-max-workers 128 \
  --gpu-max-batch 160 \
  --gpu-min-batch 32 \
  --ref-islands-db hg38_ref_islands.db \
  --no-cleanup
```

**Performance tuning:**
- `--cpu-max-workers` controls concurrent async I/O workers (not threads), allowing high parallelism for GPU-bound tasks
- `--gpu-max-batch` sets maximum batch size sent to GPU; use `python modules/GPU_executor/benchmark_batch_size.py` to find optimal value for your hardware
- `--gpu-min-batch` sets minimum batch size before GPU executor times out and processes incomplete batch

**Running many species against one reference (`--ref-islands-db`):**
Reference-transcript island scanning is *species-independent* — a transcript's
islands depend only on its exonic sequence, the model, and the scan parameters,
not on the query genome. When you align many query species to the same reference,
pass a shared `--ref-islands-db <path>` (a SQLite file, created if absent): the
first run populates it, and every subsequent species **restores** already-scanned
reference transcripts instead of re-embedding them, scanning only the transcripts
new to its set. "No islands" is cached too. Entries are keyed by model + scan
parameters + exon blocks, so changing the model/params or the reference annotation
transparently recomputes. If you run several pipelines **concurrently**, give each
lane its own DB (`--ref-islands-db lane1.db`, `lane2.db`, …) — one cache per lane.

**Search-space mode (`--projection-mode`):**
Default `orthologous` uses chain--transcript pairs classified as ORTH by the
chain-based candidate classifier. `--projection-mode best-chain` additionally
retains the top-scoring chain for transcripts without an accepted ORTH candidate,
providing a more permissive search-space option for deeply diverged query
genomes. Existing ORTH candidates are retained unchanged.

---

## Output

By default, CURIA automatically cleans up and organizes outputs into a user-friendly structure:

```
output_dir/
├── query_annotation/
│   ├── short_ncRNA.bed              # Accepted compact-locus predictions (≤256 nt)
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
    ├── rna_orthologous_regions.tsv          # Chain-supported candidate-region table
    ├── toga_orthologous_regions.tsv         # Original TOGA output
    └── original_toga_classification_table.tsv # TOGA classification scores
```

**Cleanup options:**
- Use `--no-cleanup` to keep all intermediate files (SQLite DBs, joblists, etc.)

See [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) for detailed file descriptions.

---

## Validation

CURIA was evaluated across 19 human-to-query mammalian genome comparisons, with
detailed sequence-baseline and annotation-supported analyses for human--mouse
and human--cow. The deposited result snapshot is available on Zenodo:
https://doi.org/10.5281/zenodo.21383175

For the full evaluation design, numerical results, and limitations, see the
accompanying manuscript.

---

## Citation

A preprint describing CURIA is in preparation. Until it is available, please cite
the archived result dataset:

> Kirilenko, Bogdan M. (2026). *CURIA cross-species ncRNA correspondence
> predictions across 19 mammalian genomes* (Version 1.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.21383175

The manuscript citation and BibTeX entry will be added here after the preprint is
published.

---

## References

- **TOGA:** Kirilenko et al., *Integrating gene annotation with orthology inference at scale*, Science (2023)
- **RiNALMo:** Penić et al., *RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks*, arXiv (2024)
- **RNA-FM:** Chen et al., *Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions*, arXiv (2022)
