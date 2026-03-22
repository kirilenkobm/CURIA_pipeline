# CURIA Output Structure

## Final Output Structure (after cleanup)

```
output_dir/
├── query_annotation/
│   ├── short_ncRNA.bed              # Short ncRNA annotations (miRNA, tRNA, snoRNA, etc. ≤160bp)
│   ├── short_ncRNA_details.tsv      # Detailed short ncRNA results with scores and metadata
│   ├── lncRNA_islands.bed           # Aligned lncRNA functional islands in query genome
│   ├── lncRNA_islands_reference.bed # Matching reference islands for aligned pairs
│   ├── reference_islands.bed        # All reference islands (for QC/visualization)
│   └── query_islands.bed            # All query islands before alignment (for QC)
│
├── island_alignment_results.tsv     # Detailed island alignment results with MMD scores
├── preprocessed_reference_data.json # Reference islands data (reusable across queries)
│
├── reference_union_transcripts.bed             # Collapsed reference isoforms (union transcripts)
├── reference_union_transcripts_metadata.tsv    # Biotype and gene metadata
│
├── mappings/
│   ├── union_to_isoforms.json       # Maps union transcript ID → original isoform IDs
│   ├── union_to_query.json          # Maps union transcript ID → query genomic regions
│   └── query_regions_clusters.json  # Merged query regions from island liftover
│
└── toga_results/
    ├── rna_orthologous_regions.tsv          # RNA-specific orthology predictions
    ├── toga_orthologous_regions.tsv         # Original TOGA output (all biotypes)
    └── original_toga_classification_table.tsv # TOGA classification scores
```

---

## File Descriptions

### Primary Outputs (what most users need)

| File | Description | Use Case |
|------|-------------|----------|
| `query_annotation/short_ncRNA.bed` | Short structured ncRNA annotations | Load in genome browser, functional analysis |
| `query_annotation/short_ncRNA_details.tsv` | Detailed short ncRNA results with scores and metadata | Filter by quality, inspect individual predictions |
| `query_annotation/lncRNA_islands.bed` | Conserved lncRNA functional islands in query | Identify conserved regulatory elements |
| `query_annotation/lncRNA_islands_reference.bed` | Matching reference islands for aligned pairs | Compare ref vs query island pairs in browser |
| `island_alignment_results.tsv` | Alignment scores and coordinates | Filter by quality, downstream analysis |

### Reusable Data

| File | Description | Use Case |
|------|-------------|----------|
| `preprocessed_reference_data.json` | Reference island embeddings and coordinates | Can be reused across query species with `--skip-completed` |

### Traceability & QC

| File | Description | Use Case |
|------|-------------|----------|
| `reference_union_transcripts.bed` | Reference transcripts (collapsed isoforms) | Understand what was analyzed |
| `mappings/union_to_isoforms.json` | Transcript ID mappings | Trace back to original annotation |
| `mappings/union_to_query.json` | Reference → query region mappings | Link reference to query coordinates |
| `mappings/query_regions_clusters.json` | Merged query regions from island liftover | Inspect how islands were projected |
| `toga_results/rna_orthologous_regions.tsv` | Orthology predictions | See which RNAs have orthologs |
| `toga_results/original_toga_classification_table.tsv` | TOGA classification scores | Inspect raw TOGA output |
| `query_annotation/reference_islands.bed` | All reference functional islands | Compare ref vs query in browser |
| `query_annotation/query_islands.bed` | All detected query islands | QC: what was scanned before alignment |

---

## Removed Files (not in final output)

These are automatically removed during cleanup to save space:

### Temporary Files
- `joblists/*.txt` — Internal task scheduling (no scientific value)
- `intermediate_sqlite_dbs/*.sqlite` — Async processing cache (redundant)
- `toga_results/reference_chrom_sizes.tsv` — Technical file
- `mappings/query_islands.json` — Redundant with `query_islands.bed`
- `temp_shortrna_results.tsv` — Temporary processing artifact

---

## Cleanup Options

### During Pipeline Execution
```bash
# Default: automatic cleanup
./curia.py --ref-bed12 ... --output-dir my_output

# Keep all intermediate files
./curia.py --ref-bed12 ... --output-dir my_output --no-cleanup
```

### Post-hoc Cleanup

Cleanup is integrated into the pipeline and runs automatically.
Use `--no-cleanup` to keep all intermediate files for debugging.

---

## File Formats

### BED12 files
Standard UCSC BED12 format with 12 columns:
```
chrom  start  end  name  score  strand  thickStart  thickEnd  itemRgb  blockCount  blockSizes  blockStarts
```

### TSV files
Tab-separated with header row. Key files:
- `island_alignment_results.tsv` — Reference/query island pairs with alignment scores
- `rna_orthologous_regions.tsv` — TOGA orthology calls with classification
- `*_metadata.tsv` — Gene/transcript metadata (biotypes, names, etc.)

### JSON files
- `preprocessed_reference_data.json` — Reference island data (embeddings, coordinates, metadata)
- `mappings/*.json` — ID mapping dictionaries for traceability
