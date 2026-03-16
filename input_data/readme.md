# Input Data

To run the pipeline, the following files are required:
(1) Genome sequences for the reference and query genomes in 2bit format.
(2) Reference annotation: bed12 format and annotation metadata (transcript - gene - biotype mapping)
(3) Genome alignment chains.

## Smoke Test

Quick end-to-end validation (~2 min) using 3 chrX-chrX chains and a small gene set:

```bash
./curia.py --ref-bed12 input_data/reference_annotation/smoke_test.bed \
--biomart-tsv input_data/reference_annotation/smoke_test.metadata.tsv \
--chain input_data/chains/smoke_test.chain.gz \
--ref-2bit input_data/2bit/hg38.2bit \
--query-2bit input_data/2bit/mm39.2bit \
--output-dir smoke_test_output
```

To regenerate the smoke test data, see `create_smoke_test_data.ipynb`.

TODO: provide convenient way to convert fasta to 2bit.
TODO: provide recipe to generate alignment chains.
