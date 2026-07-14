# Recurrent lncRNA core tables

These tables are generated from the accepted island-match rows in the completed
human-to-query CURIA runs. Run `make -C paper cores` from the repository root to
recreate them.

## Broad and strict sets

The broad set contains lncRNA cores with an accepted match in at least
17 of 19 query assemblies and a
mean reference-island length of at least 120 bp. The pipeline
accepts island matches at embedding-SW distance `d <= 0.10`; consequently, the
broad criterion primarily measures recurrence at the pipeline acceptance ceiling.
Strict quorum columns count assemblies in which the best accepted match for that
core also satisfies the indicated per-species cutoff. Lower distance is better.

The current broad export contains 1,813 cores in
1,693 annotated human lncRNA loci, including
379 loci more than 5 kb from and not overlapping a
protein-coding gene.

## Files

* `recurrent_gene_summary.tsv`: one row per broad-set versionless ENSG locus.
  `best_mean_dist` and `best_median_dist` are minima across that gene's cores;
  `mean_species_per_core` is the mean broad-acceptance count across its cores.
* `recurrent_core_summary.tsv`: one row per overlap-aggregated human-reference
  core, including coordinates, accepted and strict species counts, distance and
  aligned-length summaries, and the assemblies with an accepted match.
* `recurrent_core_matches.tsv`: one row per core x query assembly, retaining the
  lowest-distance accepted match when multiple accepted rows exist.
* `distance_sensitivity.tsv`: recurrent-core counts at a fixed >=17-species quorum
  while varying the per-species distance cutoff.
* `quorum_sensitivity_d_le_0.02.tsv`: counts at fixed `d <= 0.02` while varying
  the required number of query assemblies.

## Definitions

`core_id`
: Stable public identifier assigned by genomic order within a versionless ENSG
  locus. A gene can contain multiple non-overlapping cores.

Coordinates
: All reference and query coordinates are BED-style, 0-based, half-open. Reference
  coordinates use hg38. Query assembly is given in the `assembly` column.

Core reconstruction
: Accepted reference-island intervals from all species runs are pooled within each
  source gene and merged by overlap in hg38 coordinates, matching Figure 6.

`embedding_sw_distance`
: CURIA cosine-dotplot Smith-Waterman distance `d = 1/(1+s)`, where lower values
  indicate a stronger retained match. The saved match tables contain accepted rows
  only (`d <= 0.10`) and therefore have a restricted score range.

`effective_aligned_length`
: Reconstructed from the saved per-chain reference and query spans using the same
  integer formula as the matcher.

`overlap_category`
: Relationship of the complete human lncRNA gene locus to protein-coding gene
  loci: `intergenic`, `near_coding_5kb`, `antisense`, `sense_overlapping`, or
  `antisense+sense`.

Not stored in the completed island-match tables
: Query strand, nucleotide Smith-Waterman score, nucleotide aligned identity,
  source chain ID, and rank among rejected candidates were not persisted by the
  pipeline and are therefore not included or inferred here.

## Query assemblies

rheMac10, mm39, rn7, HLoryCun3, bosTau9, susScr11, HLbalEde1, HLcamDro2, equCab3, felCat9, canFam5, HLmanJav2, HLpteVam2, eriEur2, dasNov3, HLeleMax1, monDom5, HLdidVir1, HLnotEug3
