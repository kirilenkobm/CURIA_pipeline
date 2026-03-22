# CURIA

**Cross-species Unified ncRNA Inference and Annotation**

CURIA identifies conserved non-coding RNAs across distantly related species by combining genome alignments, orthology prediction, and RNA foundation model embeddings.

---

## Installation

If you don't have [uv](https://docs.astral.sh/uv/) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
git clone --recurse-submodules git@github.com:kirilenkobm/curia_pipeline.git
cd curia_pipeline
uv sync
source .venv/bin/activate
```

> **macOS note:** OpenMP is required for scikit-learn and other numerical libraries.
> Install it once with: `brew install libomp`

The RNA-FM model (~1.1GB) downloads automatically on first run, or use `./download_rnafm_model.py` to download manually.
If you already have the weights elsewhere, run `python download_rnafm_model.py --show-dir` to see where to place them.

---

## Quick Start

```bash
# Optional: benchmark optimal GPU batch size for your hardware
python modules/GPU_executor/benchmark_batch_size.py

# Run smoke test (~1-2 minutes on strong machines)
./curia.py \
  --ref-bed12 input_data/reference_annotation/smoke_test.bed \
  --biomart-tsv input_data/reference_annotation/smoke_test.metadata.tsv \
  --chain input_data/chains/smoke_test.chain.gz \
  --ref-2bit input_data/2bit/hg38.test.subset.2bit \
  --query-2bit input_data/2bit/mm39.test.subset.2bit \
  --output-dir smoke_test_output \
  --gpu-logger

# For a more comprehensive test (~20 minutes)
./curia.py \
  --ref-bed12 input_data/reference_annotation/test_sample.bed \
  --biomart-tsv input_data/reference_annotation/test_sample.metadata.tsv \
  --chain input_data/chains/test_sample.chain.gz \
  --ref-2bit input_data/2bit/hg38.test.subset.2bit \
  --query-2bit input_data/2bit/mm39.test.subset.2bit \
  --output-dir test_output
```

---

## Requirements

**Input files:**
- Reference annotation (BED12) and metadata (TSV with transcript-to-biotype mappings)
- Query and reference genomes (2bit format)
- Genome alignment chains

**Compute:**
- CPU required for RNA TOGA and sequence processing
- GPU optional but recommended for RNA-FM embeddings
- Tested on macOS (MPS) and Linux (CUDA)

---

## Usage

```bash
./curia.py \
  --ref-bed12 $REFERENCE_BED12 \
  --biomart-tsv $REFERENCE_METADATA \
  --chain $ALIGNMENT_CHAINS \
  --ref-2bit $REF_2BIT \
  --query-2bit $QUERY_2BIT \
  --output-dir $OUTPUT_DIR \
  --ref-preprocessed $REFERENCE_DATA \  # optional: reuse preprocessed reference
  --cpu-max-workers 128 \               # max concurrent async workers (default: 128)
  --gpu-max-batch 160 \                 # max GPU batch size (default: 160, tune with benchmark script)
  --gpu-min-batch 32 \                  # min batch size before timeout (default: 32)
  --no-cleanup                          # optional: keep all intermediate files (SQLite DBs, joblists)
```

**Performance tuning:**
- `--cpu-max-workers` controls concurrent async I/O workers (not threads), allowing high parallelism for GPU-bound tasks
- `--gpu-max-batch` sets maximum batch size sent to GPU; use `python modules/GPU_executor/benchmark_batch_size.py` to find optimal value for your hardware
- `--gpu-min-batch` sets minimum batch size before GPU executor times out and processes incomplete batch
- `--ref-preprocessed` directory contains reference data that only needs to be computed once per reference species and can be reused across multiple query genomes

---

## Output

By default, CURIA automatically cleans up and organizes outputs into a user-friendly structure:

```
output_dir/
├── query_annotation/
│   ├── short_ncRNA.bed              # Short ncRNA annotations (≤160bp)
│   ├── lncRNA_islands.bed           # Aligned lncRNA islands
│   ├── reference_islands.bed        # Reference islands (QC)
│   └── query_islands.bed            # Query islands (QC)
├── island_alignment_results.tsv     # Island alignment scores
├── preprocessed_reference.json      # Reusable reference data
├── reference_union_transcripts.bed  # Collapsed reference isoforms
├── reference_union_transcripts_metadata.tsv
├── mappings/
│   ├── union_to_isoforms.json       # Transcript → isoforms mapping
│   └── union_to_query.json          # Transcript → query regions mapping
└── toga_results/
    ├── rna_orthologous_regions.tsv  # RNA orthology predictions
    └── toga_orthologous_regions.tsv # Original TOGA output
```

**Cleanup options:**
- Use `--no-cleanup` to keep all intermediate files (SQLite DBs, joblists, etc.)
- Run `./cleanup_outputs.py <output_dir>` to cleanup existing pipeline outputs

---

## How It Works

1. **Orthologous loci prediction** — RNA TOGA identifies candidate ncRNA regions in the query genome
2. **Short ncRNA annotation** — Dedicated MMD-based pipeline for ncRNAs ≤160bp (miRNA, tRNA, snoRNA, etc.)
3. **Reference island scanning** — RNA-FM identifies functional islands in reference lncRNAs where embeddings separate signal from background; results are reusable across queries
4. **Query island scanning** — RNA-FM scans orthologous query lncRNA loci for islands (computational bottleneck)
5. **Island alignment** — Windowed MMD matching of reference and query islands based on RNA-FM embedding similarity

---

## Validation

CURIA has been validated on human–mouse, human–dog, and human–cow comparisons, successfully recovering known conserved lncRNAs and structured ncRNAs. Full validation details are provided in the preprint (in preparation).

---

## Citation

If you use CURIA, please cite:

> Kirilenko, B.M., *CURIA: Cross-species Unified ncRNA Inference and Annotation*, preprint in preparation, 2026.

---

## References

- **TOGA:** Kirilenko et al., *Integrating gene annotation with orthology inference at scale*, Science (2023)
- **RNA-FM:** Chen et al., *Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions*, arXiv (2022)
