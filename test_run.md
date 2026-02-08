In general, test run:

```bash
./curia.py --ref-bed12 input_data/reference_annotation/hg38.input.w.tRNA.bed --biomart-tsv input_data/reference_annotation/hg38.transcript_metadata.tsv --chain input_data/chains/hg38.mm39.allfilled.chain.gz --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit --gpu-max-batch 128 --output-dir sample_output
```
