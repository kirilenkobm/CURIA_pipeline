# CURIA

CURIA is a cross-species pipeline for orthology-aware annotation of conserved non-coding RNAs.
It combines genome alignments, orthologous locus detection, RNA foundation model embeddings,
and secondary structure validation to identify conserved ncRNA loci across distantly related species.

# Usage

```bash
git clone --recurse-submodules git@github.com:kirilenkobm/curia_pipeline.git
cd curia_pipeline
conda env create -f environment.yaml
conda activate CURIA_pipeline
```

**Note:** RNA-FM pretrained model (~500MB) downloads automatically on first run from `proj.cse.cuhk.edu.hk`

## Quick test

```bash 
./curia.py --ref-bed12 input_data/reference_annotation/test_sample.bed \
--biomart-tsv input_data/reference_annotation/test_sample.metadata.tsv \
--chain input_data/chains/test_sample.chain.gz \
--ref-2bit input_data/2bit/hg38.test.subset.2bit \
--query-2bit input_data/2bit/mm39.test.subset.2bit \
--gpu-max-batch 128 --short-max-workers 100 \
--gpu-logger --output-dir quick_test
```

## Compute requirements

- CPU: required for RNA TOGA and ViennaRNA steps
- GPU: optional, recommended for RNA-FM embeddings
- Disk: genome alignments and intermediate embeddings may require tens of GB

Tested on MacOS with MPS backend and Linux with CUDA.

## Prerequisites

### Reference and query genome sequences in 2bit format.

Referenced below as `$REF_2BIT` and `$QUERY_2BIT`.

### Genome alignment chains.

Referenced below as `$ALIGNMENT_CHAINS`.

### Comprehensive reference annotation in bed12 format.

Note: protein-coding genes will be not projected in the final results, 
but they increase the accuracy of the orthologous region prediction.

The annotation is referenced below as `$REFERENCE_BED12`.

### Annotation metadata (biomart format for example)

Ideally, mappings between reference transcript ID and biotype.
Then, reference transcript ID to gene ID.

Metadata is referenced below as `$REFERENCE_METADATA`.

## Outputs

- BED12 annotation of predicted ncRNA loci
- Per-locus QC scores
- Intermediate files for RNA-FM alignment and ViennaRNA analysis (optional)

## Running pipeline.

### Activate conda environment

```bash

```

### Run pipeline

```bash
./curia.py -r $REF_2BIT -q $QUERY_2BIT -c $ALIGNMENT_CHAINS -a $REFERENCE_BED12 -m $REFERENCE_METADATA -r $REFERENCE_DATA -o $OUTPUT_DIR 
```

`$REFERENCE_DATA` contains preprocessed reference annotation - will be created once per reference species.
`$OUTPUT_DIR` contains final annotation files for the given query genome.

# Pipeline structure

## Step 0: prefilter lncRNA cores (run once per reference species)

## Step 1: predict potential orthologous regions using RNA TOGA

## Step 2: RNA-FM embeddings-based realignment of reference ncRNAs in the orthologous regions

## Step 3: Vienna-RNA BPPM-based QA of the predicted ncRNAs

## Step 4: produce final annotation files

# Validation

CURIA was validated on human–mouse, human–dog, and human–cow genome pairs,
recovering known conserved lncRNAs and structured ncRNAs.
See preprint for details.

# Related work

- TOGA: Integrating gene annotation with orthology inference at scale  
  Kirilenko et al., Science (2023)

- RNA-FM: A foundation model for RNA sequence representation  
  (paper + repo)

# Citation

If you use CURIA, please cite:

Kirilenko et al., *CURIA: Cross-species Unified ncRNA Inference and Annotation*, preprint, 2026.
(preprint link coming soon)
