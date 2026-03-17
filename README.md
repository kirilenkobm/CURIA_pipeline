# CURIA

**Cross-species Unified ncRNA Inference and Annotation**

CURIA identifies conserved non-coding RNAs across distantly related species by combining genome alignments, orthology prediction, and RNA foundation model embeddings.

---

## Installation

```bash
git clone --recurse-submodules git@github.com:kirilenkobm/curia_pipeline.git
cd curia_pipeline
conda env create -f environment.yaml
conda activate CURIA_pipeline
```

The RNA-FM model (~500MB) downloads automatically on first run, or use `./download_model.py` to download manually.

---

## Quick Start

```bash
# Optional: benchmark optimal GPU batch size for your hardware
python modules/GPU_executor/benchmark_batch_size.py

# Run test dataset
./curia.py \
  --ref-bed12 input_data/reference_annotation/test_sample.bed \
  --biomart-tsv input_data/reference_annotation/test_sample.metadata.tsv \
  --chain input_data/chains/test_sample.chain.gz \
  --ref-2bit input_data/2bit/hg38.test.subset.2bit \
  --query-2bit input_data/2bit/mm39.test.subset.2bit \
  --output-dir quick_test
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
  --ref-preprocessed $REFERENCE_DATA  # optional: reuse preprocessed reference
```

The `--ref-preprocessed` directory contains reference data that only needs to be computed once per reference species and can be reused across multiple query genomes.

---

## Output

- **BED12 annotation** of conserved ncRNA loci in the query genome
- **Per-locus quality scores** based on structural conservation
- **Intermediate files** for downstream analysis (RNA-FM alignments, island predictions)

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

> Kirilenko et al., *CURIA: Cross-species Unified ncRNA Inference and Annotation*, preprint in preparation, 2026.

---

## References

- **TOGA:** Kirilenko et al., *Integrating gene annotation with orthology inference at scale*, Science (2023)
- **RNA-FM:** Chen et al., *Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions*, arXiv (2022)
