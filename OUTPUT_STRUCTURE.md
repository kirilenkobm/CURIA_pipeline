# CURIA Output Structure

## Final Output Structure (after cleanup)

```
output_dir/
├── query_annotation/
│   ├── short_ncRNA.bed              # Short ncRNA annotations (miRNA, tRNA, snoRNA, etc. ≤160bp)
│   ├── lncRNA_islands.bed           # Aligned lncRNA functional islands in query genome
│   ├── reference_islands.bed        # Reference islands (for QC/visualization)
│   └── query_islands.bed            # All query islands before alignment (for QC)
│
├── island_alignment_results.tsv     # Detailed island alignment results with MMD scores
├── preprocessed_reference.json      # Reference islands data (reusable across queries)
│
├── reference_union_transcripts.bed             # Collapsed reference isoforms (union transcripts)
├── reference_union_transcripts_metadata.tsv    # Biotype and gene metadata
│
├── mappings/
│   ├── union_to_isoforms.json       # Maps union transcript ID → original isoform IDs
│   └── union_to_query.json          # Maps union transcript ID → query genomic regions
│
└── toga_results/
    ├── rna_orthologous_regions.tsv  # RNA-specific orthology predictions
    └── toga_orthologous_regions.tsv # Original TOGA output (all biotypes)
```

---

## File Descriptions

### Primary Outputs (what most users need)

| File | Description | Use Case |
|------|-------------|----------|
| `query_annotation/short_ncRNA.bed` | Short structured ncRNA annotations | Load in genome browser, functional analysis |
| `query_annotation/lncRNA_islands.bed` | Conserved lncRNA functional islands | Identify conserved regulatory elements |
| `island_alignment_results.tsv` | Alignment scores and coordinates | Filter by quality, downstream analysis |

### Reusable Data

| File | Description | Use Case |
|------|-------------|----------|
| `preprocessed_reference.json` | Reference island embeddings and coordinates | **Reuse** with `--ref-preprocessed` for other query species |

### Traceability & QC

| File | Description | Use Case |
|------|-------------|----------|
| `reference_union_transcripts.bed` | Reference transcripts (collapsed isoforms) | Understand what was analyzed |
| `mappings/union_to_isoforms.json` | Transcript ID mappings | Trace back to original annotation |
| `mappings/union_to_query.json` | Reference → query region mappings | Link reference to query coordinates |
| `toga_results/rna_orthologous_regions.tsv` | Orthology predictions | See which RNAs have orthologs |
| `query_annotation/reference_islands.bed` | Reference functional islands | Compare ref vs query in browser |
| `query_annotation/query_islands.bed` | All detected query islands | QC: what was scanned before alignment |

---

## Removed Files (not in final output)

These are automatically removed during cleanup to save space:

### Temporary Files
- `joblists/*.txt` — Internal task scheduling (no scientific value)
- `intermediate_sqlite_dbs/*.sqlite` — Async processing cache (redundant)
- `toga_mini_results/reference_chrom_sizes.tsv` — Technical file
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
```bash
# Cleanup existing output directory
./cleanup_outputs.py my_output

# Quiet mode
./cleanup_outputs.py -q my_output
```

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
- `preprocessed_reference.json` — Reference island data (embeddings, coordinates, metadata)
- `mappings/*.json` — ID mapping dictionaries for traceability
