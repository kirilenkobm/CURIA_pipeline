# CURIA scientific consistency audit

This audit records issues verified against the current manuscript, implementation,
and committed result summaries. Corrections marked **fixed** have been applied to
the manuscript. Items marked **unresolved** require either analysis of existing
large outputs or a scientific choice; no result was invented.

## IMPORTANT

### MMD estimator and reported quantity — fixed

- **Previous text:** described the unbiased MMD² equation but called the output
  MMD and stated that zero always denoted identical empirical distributions.
- **Issue:** `modules/pipeline/short_ncrna.py::_compute_mmd_with_ref` calculates
  the unbiased finite-sample MMD² estimator, clips negative estimates to zero,
  and returns the square root. Zero can therefore result from clipping and does
  not uniquely imply identical samples.
- **Classification:** textual and mathematical correction.
- **Correction:** Methods now defines the estimator and the reported
  `sqrt(max(MMD²_u, 0))` value explicitly.

### Short-ncRNA overlap definitions — fixed

- **Previous text:** called the 50% metric reciprocal overlap and described it as
  coverage of the annotated transcript.
- **Issue:** `analysis/compute_fig4_data.py::_annotation_overlap` divides the
  maximum exonic intersection by predicted-locus length only. The 50% and 99%
  criteria are one-directional prediction coverage.
- **Classification:** implementation-description correction.
- **Correction:** Methods and Results now state the actual denominator and avoid
  “reciprocal.” The numerical values did not change.

### Meaning of the approximately 230 loci — fixed

- **Previous text:** implied that these were ncRNA elements absent from current
  annotations.
- **Issue:** `analysis/make_conserved_cores.py` starts from annotated human lncRNA
  genes and classifies their proximity/overlap with human protein-coding genes.
  The committed table defines “intergenic” as more than 5 kb from coding genes;
  it does not demonstrate absence from ncRNA annotation.
- **Classification:** claim and terminology correction.
- **Correction:** the manuscript now describes annotated human lncRNA loci and
  reports the coding-proximity classification precisely.

### Genome/species denominator — fixed

- **Previous text:** “19 mammalian genomes” while listing human plus 19 queries.
- **Issue:** the analysis comprises 20 genomes total, 19 query genomes, and 19
  human-to-query pairs.
- **Classification:** denominator correction.
- **Correction:** abstract and benchmark setup now distinguish these quantities.

## SHOULD FIX

### Recurrence threshold and genome quality — partly fixed, unresolved analysis

- **Current criterion:** a core recovered in at least 18 of 19 query genomes,
  yielding 1,107 annotated human lncRNA loci; the committed table reports 234
  loci in its intergenic (>5 kb) category, rounded to approximately 230 in text.
- **Issue:** failure to recover a core can reflect assembly gaps, chain coverage,
  or detection failure and is not evidence of biological absence.
- **Classification:** operational heuristic and genuine limitation.
- **Correction:** the manuscript calls 18/19 a deliberately stringent operational
  criterion, allows one biological or technical non-detection, and does not infer
  lineage-specific absence from a failed call.
- **TODO:** compute 17/19, 18/19, and 19/19 counts from the full existing core
  table and summarize species-specific missingness. If technically assessable
  loci can be defined without a new benchmark, report recovered/assessable support.

### Context reuse — fixed conservatively

- **Previous text:** treated RiNALMo context stability as an exact property.
- **Evidence:** `notebooks/context_dependency.ipynb` contains direct context
  perturbation comparisons.
- **Classification:** wording correction; quantitative reporting remains optional.
- **Correction:** context reuse is now described as an engineering approximation
  motivated by comparative stability under the tested perturbations.
- **TODO:** if this notebook is retained as formal evidence, report its sample
  size and summary metric in the manuscript or supplement.

### Match-specificity null analyses — resolved for two genome pairs

- **Previous text:** concluded broadly that recurrent cores were unlikely to be
  best-of-many artifacts.
- **Issue:** the text did not report tested pairs, locus count, alternative-window
  construction, effect size, or uncertainty.
- **Classification:** overstated conclusion and missing methodological detail.
- **Correction:** Methods now document the cross-locus, dinucleotide-shuffle, and
  within-locus controls. Results report the exact sample counts and effect sizes
  for hg38--bosTau9 and hg38--mm39, while limiting the interpretation to the
  tested artifact models rather than biological function.

### Nucleotide-resolution wording — fixed

- **Previous text:** “matched at nucleotide resolution” could imply established
  nucleotide homology.
- **Issue:** the matcher performs Smith–Waterman local alignment over a per-token
  cosine-similarity matrix.
- **Classification:** terminology correction.
- **Correction:** manuscript now uses “token-level” or “nucleotide-indexed local
  alignment path.”

### Novelty scope — fixed

- **Previous text:** “A key missing component is the integration of synteny with
  embedding-based similarity.”
- **Issue:** broader than supported without a systematic literature review.
- **Classification:** conservative wording correction.
- **Correction:** novelty is scoped to pretrained RNA-language-model
  representations used as the similarity signal within syntenically constrained
  genomic search regions.

## MINOR / REMAINING VERIFICATION

- The pooled MMD/annotation association remains an empirical benchmark trend,
  not a universal calibrated confidence scale; Results now says this explicitly.
- The compact 257–320 nt whole-exon route is now described as an engineering
  heuristic rather than evidence of end-to-end conservation.
- The Rfam AUC 0.993 should be read as a matcher sanity check unless family,
  sampling, duplicate, and split details are documented from
  `notebooks/matching_benchmark.ipynb`.
- A complete machine-readable provenance table for every manuscript percentage
  remains to be generated from the full result directory. Existing committed
  summaries include `analysis/data/fig4_mmd.json` and
  `analysis/data/pair_numbers_hg38_vs_mm39.json`.

## Numerical changes in this correction pass

No computed numerical result was changed. The panel denominator was clarified as
20 genomes total / 19 query genomes / 19 genome pairs. The manuscript retains
1,107 loci and rounds the committed intergenic count of 234 to approximately 230.

## Author-comment pass (2026-07-13)

The manuscript was revised after checking the repository before considering new
analyses. Existing committed artifacts answered the PCA, detector-training, Rfam
matcher-benchmark, context-dependence, and detector-generalization questions.

- The abstract no longer includes context-reuse details, the total-genome
  parenthetical, or the coding-proximity count; islands are introduced as
  RNA-like, potentially structured cores.
- Methods now distinguishes the 16-component matching PCA (10,368 samples) from
  the 64-component finding PCA and reports the committed detector cache size
  (18,827 windows: 6,538 positive and 12,289 negative).
- The Rfam AUC is now explained as a controlled planted-core matcher sanity check,
  not a genome-wide performance estimate.
- Discussion now states the detector-blindness failure mode supported by
  `notebooks/rfam_held_out_family.ipynb`, and Results includes assembly/chain
  quality as a failure source.
- Figure 3 and Figure 6 layout requests remain manual/generated-figure tasks and
  were not changed in this pass.

### Recurrent-core sensitivity and identity --- resolved from existing outputs

The ignored `preprint_results/hg38_vs_*` directories contain all 19 completed
per-species island-alignment tables. Re-running the existing summary script gave:

- at least 17/19: 1,693 lncRNA genes (379 intergenic);
- at least 18/19: 1,107 lncRNA genes (234 intergenic);
- 19/19: 524 lncRNA genes (106 intergenic).

The manuscript and supplementary summary now use 17/19. Core identity is based on
overlap of human reference coordinates within each gene. Thus non-overlapping
pieces matched in different species remain different cores. A direct geometry
check of all 1,813 cores passing the 17/19 core-level filters found zero cores
containing pairwise-disjoint member intervals joined through transitive overlap.

## External red-flag review

- The deployed candidate-region model was verified as GBM-7 with global threshold
  0.5 and lncRNA threshold 0.3. Methods now uses the seven stored feature names.
- Island projection and scanning parameters were recovered from
  `reference_islands_liftover.py`, `query_islands_scanner.py`, and the model
  registry and are now stated explicitly.
- Coding proximity is implemented in `analysis/make_conserved_cores.py`; the
  17/19 table contains 1,693 loci, of which 379 are intergenic by the stated
  >5-kb definition. Methods and Results now document this analysis.
- The recurrent-core result is now presented as a full funnel: 34,846 annotated
  lncRNA union loci; 34,532 processed by the island branch in at least one pair;
  18,364 with a reference island; 11,328 projectable in at least 17 species;
  1,761 genes with a 17-species recurrent core before length/distance filters;
  1,693 after filtering; and 379 intergenic genes.
- Short-branch coverage is reported from the 9,247-locus geometry denominator.
  `analysis/short_exact_name_recall.py` additionally gives a conservative
  exact-name proxy: 373/446 (83.6%) named human--mouse counterparts recovered.
- Figure 6C was checked against all 259,514 match rows. Species median distance
  and effective aligned length both vary across the panel; the caption now treats
  the panel as descriptive and explicitly rejects a strictly monotonic trend.
- `analysis/island_null_test.py` and `analysis/island_searchspace_null.py`
  implement the requested cross-locus, dinucleotide-shuffle, and within-locus
  controls. Complete JSON/CSV and text reports for hg38--bosTau9 and hg38--mm39
  are now present under `analysis/scratch`, and their exact results are reported
  in Methods and Results.
