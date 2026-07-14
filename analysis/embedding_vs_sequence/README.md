# Embedding vs. sequence: is CURIA/RiNALMo similarity reducible to conventional nucleotide metrics?

Self-contained, additive analysis. Nothing in the manuscript or pipeline is modified.
All scripts are under `scripts/`; regenerate everything with:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
V=.venv/bin/python
$V analysis/embedding_vs_sequence/scripts/build_short_metrics.py
$V analysis/embedding_vs_sequence/scripts/build_island_metrics.py --pair preprint_results/hg38_vs_mm39 \
     --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit --tag mouse
$V analysis/embedding_vs_sequence/scripts/build_island_metrics.py --pair preprint_results/hg38_vs_bosTau9 \
     --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/bosTau9.2bit --tag cow
# bounded RiNALMo pass on MPS/CUDA (only recompute in the whole analysis):
$V analysis/embedding_vs_sequence/scripts/dump_island_embeddings.py --pair preprint_results/hg38_vs_mm39 \
     --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/mm39.2bit --tag mouse
$V analysis/embedding_vs_sequence/scripts/dump_island_embeddings.py --pair preprint_results/hg38_vs_bosTau9 \
     --ref-2bit input_data/2bit/hg38.2bit --query-2bit input_data/2bit/bosTau9.2bit --tag cow
$V analysis/embedding_vs_sequence/scripts/analyze_short.py
$V analysis/embedding_vs_sequence/scripts/analyze_island.py
$V analysis/embedding_vs_sequence/scripts/case_studies_fig5.py   # Fig-5 case-study numbers
```

---

## 1. Exact scientific question

> Does the RiNALMo embedding-based similarity score retain information associated with
> the correct / annotated correspondence **after controlling for ordinary
> sequence-derived similarity metrics**?

The claim under test is narrow: that the embedding score is **not fully reducible** to
the tested conventional nucleotide-similarity metrics. This is **not** a test that the
embeddings represent RNA structure or function.

Two independent settings:
- **A. Short ncRNAs** — human→mouse short-branch predictions; embedding score = MMD;
  label = any-exon annotation support.
- **B. Island matcher** — human→mouse and human→cow; embedding score = cosine
  Smith–Waterman distance (`diag_mmd`); "correct" = the assigned syntenic partner.

## 2. Input files and sample construction

**A (short ncRNAs)** — 2,994 predictions, no GPU:
- `preprint_results/hg38_vs_mm39/query_annotation/short_ncRNA_details.tsv` → `mmd_score`, `biotype`, `gene_id`, locus coords.
- Reference (human) spliced sequence: `reference_union_transcripts.bed` + `input_data/2bit/hg38.2bit` (`short_ncrna._get_spliced_sequence`).
- Query (mouse) sequence: locus coords + `input_data/2bit/mm39.2bit` (`short_ncrna._extract_sequence`).
- Annotation-support label recomputed exactly as `analysis/compute_fig4_data.py::_annotation_overlap` from `input_data/mm39_annotation_validation/{mm39_gencode_all_transcripts.bed, mm39-tRNAs.bed}`.
- 0 rows dropped (no missing transcripts, no empty sequences, no N-containing sequences). Sanity: recomputed identity↔MMD Pearson = −0.797 vs the manuscript's cached −0.795; any-overlap fraction 0.524 vs cached 0.524. → `short_ncrna_metrics.tsv`.

**B (islands)**:
- Full candidate tables: every `type=="match"` row of `preprint_results/hg38_vs_{mm39,bosTau9}/island_alignment_results.tsv` (12,172 mouse / 15,698 cow candidates; `diag_mmd` stored per candidate). Sequences fetched from the 2bit files (query via `AliasedTwoBitAccessor`). → `island_pair_metrics_{mouse,cow}.tsv`.
- Per-pair embedding scores for negatives/shuffles/windows are **not stored by the pipeline** (the null scripts computed them in memory only). We recovered them with **one bounded RiNALMo pass on Apple MPS** (`dump_island_embeddings.py`), sampling **N = 200** assigned islands per species by the same deterministic `np.linspace` as `analysis/island_null_test.py` (so results are directly comparable), embedding ref + query(fwd/rc) + dinucleotide-shuffle(fwd/rc), and 100 loci × ≤25 tiled windows. → `data/island_embed_{mouse,cow}.npz`.

### 2a. Score reproduction — why the recomputed↔stored Pearson is 0.88/0.81 (and why it is fine)

This needs to be explicit before any "same scorer" wording. The recomputed assigned-pair
distances correlate with the pipeline's stored `diag_mmd` at **Pearson r = 0.879 (mouse) /
0.808 (cow)** — but that Pearson is misleading, for a concrete, verified reason:

- **The stored distance is right-censored at `max_match_dist = 0.1`.** The pipeline only
  reports/keeps island matches with distance ≤ 0.1 (`modules/model_registry.py`; the stored
  `diag_mmd` maxes out at exactly 0.100, with values piling up at 0.0996–0.1000). Our fresh
  recompute is *uncensored*, so a handful of islands re-score just above the cap
  (**4/200 mouse, 7/200 cow**, up to 0.13–0.22). Those few points sit far off the identity
  line and, over a distance range only 0.1 wide, dominate a scale-sensitive Pearson.
- **Rank agreement is high and the bulk is near-exact:** **Spearman ρ = 0.936 / 0.932**,
  **median |Δ| = 0.0005**, 91% / 93% of islands within |Δ| < 0.02.
- **It is not an MPS/fp16 artifact.** `analysis/island_null_test.py` run *on the CUDA pod*
  reproduces its own stored `diag_mmd` at r ≈ 0.85 (`island_null_hg38_vs_mm39.json`
  `sanity_r = 0.852`); our MPS recompute reaches r = 0.879 (mouse). The residual gap is
  intrinsic to recomputation under the 0.1 censoring, not to the hardware/precision.
- **Residual per-island noise** (median |Δ| 5e-4) comes from MPS fp32 vs pod CUDA-AMP fp16
  embeddings, minor LM batching/context, and Smith–Waterman traceback ties near `sw_tau_cos`.

**What the analysis actually uses.** Every AUC in Analysis B is computed from the
**recomputed** embedding scores *consistently* — the assigned diagonal, the cross-locus
off-diagonal, the dinucleotide shuffles and the tiled windows all come from the same fresh
scorer. The stored `diag_mmd` is used **only** as the sanity check above, never inside an
AUC, so the comparison is internally self-consistent regardless of the censoring gap.

**The functional (rank-level) check that matters** is reproduced almost exactly: recomputed
embedding-alone assigned-vs-cross-locus **AUC = 0.903 / 0.909** vs the pipeline's stored
null-test AUC **0.898 / 0.907** (Δ ≤ 0.005). And restricting to the 91–93% of islands where
recomputed = stored within 0.02 (i.e. the strictly pipeline-native-faithful set), the
embedding advantage is preserved and if anything larger: embedding-alone AUC 0.917 / 0.924,
Δ(embedding − nucleotide-SW) = +0.060 [0.032, 0.088] / +0.055 [0.029, 0.084]. **The ΔAUC
survives the pipeline-native restriction.**

## 3. Metric definitions

Sequences uppercased, U↔T normalised; `len_ratio = min/max`; `gc_diff = |GC_a − GC_b|` (N excluded).
- **ident_levenshtein** = `1 − edit/max(len)` (same as the manuscript's seqid).
- **Nucleotide Smith–Waterman** (affine Gotoh, numba; match +2, mismatch −1, gap-open −5, gap-extend −1; N never matches), with traceback:
  - **sw_norm** = `sw_raw / (2·min(len_a,len_b))` ∈ ~[0,1] (self-score normalisation).
  - **sw_aligned_ident** = `n_ident / aln_len` (identity over the SW-aligned region).
- **kmer3_cos / kmer4_cos** = cosine similarity of 4^k k-mer count vectors.
- **dinuc_dist** = Euclidean distance between 16-dim dinucleotide-frequency vectors.
- **Embedding score**: short ncRNA = `mmd_score` (RBF-MMD over RiNALMo per-token embeddings, a *distance*); island = `diag_mmd = 1/(1+SW)` cosine-SW distance, and `emb_sw = 1/diag_mmd − 1`.
- **Direction**: MMD and `diag_mmd` are distances (lower = more similar). Single-feature AUCs use `−distance` for similarity features so higher = more likely correct. Island query metrics use the **best of {fwd, revcomp}** (mirrors the embedding scorer's min-over-orientation); short-ncRNA sequences are already strand-correct and used as-is.

## 4. Leakage and grouping decisions

- **A models**: regularised logistic regression, `StandardScaler` fit **inside each fold**, `StratifiedGroupKFold(5)` **grouped by `gene_id`** (no gene spans train/test), averaged over 5 shuffles (seeds 0–4).
- **B fitted models**: `GroupKFold` **grouped by reference island**; a ref island's positive (assigned) and its sampled cross-locus negatives fall in the same fold.
- Cross-locus negatives are the NxN off-diagonal → always a **different gene**; a ref island is never scored against its own query as a negative.
- **All B AUC confidence intervals are cluster bootstraps that resample whole reference islands** (each reference carries its assigned positive plus all ~199 cross-locus negatives), because a single reference generates many correlated negative rows. Pair-row bootstrap would understate the CIs; the cluster CIs are the ones reported.
- No annotation-derived quantity enters any sequence or embedding feature (features are sequence↔sequence or embedding↔embedding only). The label is annotation-only.
- Robustness re-runs (below): dedup to one row per gene; exclude near-identical (ident>0.95) and very short (<30 nt) pairs.
- **B2** raw-score ranking needs no fitting; the fitted ranking variant would be grouped by locus. The within-locus "assigned" position is partly influenced by the RiNALMo finding model — flagged as a caveat.

## 5. Main numerical results (95% CIs)

### A — short ncRNAs (n = 2,994; 52.4% supported)

**A1 descriptive** — MMD is strongly (but not perfectly) correlated with every sequence metric:
identity Pearson −0.797 / Spearman −0.851; sw_norm −0.799 / −0.859; kmer3 −0.744; kmer4 −0.756; dinuc_dist +0.708.

**A2 sequence-only vs +MMD** (grouped CV ROC-AUC):

| model | ROC | PR |
|---|---|---|
| sequence baseline | 0.887 | 0.917 |
| baseline + MMD | 0.893 | 0.921 |
| baseline + biotype | 0.893 | 0.921 |
| baseline + biotype + MMD | 0.898 | 0.925 |

- Paired incremental gain from adding MMD: **ΔROC = +0.0064 [+0.0030, +0.0097]**, ΔPR = +0.0044 [+0.0026, +0.0061] (CI excludes 0 but small). With biotype in the baseline: ΔROC +0.0050 [+0.0023, +0.0076].
- Single-feature AUCs: identity 0.874 [0.861, 0.886]; sw_norm 0.884 [0.871, 0.897]; **MMD alone 0.875 [0.862, 0.887]** — MMD alone ≈ identity alone, not better.
- **Biotype dependence**: excluding miRNA (the dominant biotype, 42% of rows) the incremental gain **vanishes**: baseline 0.919 → +MMD 0.918 (Δ −0.0007). Per-biotype MMD-alone AUC ranges from 0.60 (rRNA-pseudogene) / 0.64 (lncRNA) to 0.95 (snoRNA) / 0.96 (tRNA).

**A3 identity-matched** — within narrow identity bins, supported predictions have **lower** MMD than unsupported (median differences, Cliff's δ):

| identity bin | n | median ΔMMD (sup−uns) [CI] | Cliff's δ | MW p |
|---|---|---|---|---|
| 40–45% | 436 | −0.051 [−0.119,−0.021] | −0.25 | 4e-4 |
| 45–50% | 512 | −0.087 [−0.132,−0.035] | −0.31 | 3e-6 |
| 50–55% | 347 | −0.071 [−0.108,−0.035] | −0.36 | 2e-7 |
| 55–60% | 241 | −0.024 [−0.055,+0.005] | −0.15 | 0.06 |
| 60–70% | 302 | −0.035 [−0.058,−0.007] | −0.21 | 1e-3 |

Nearest-neighbour matching (matched on identity, sw_norm, len_ratio, **exact biotype**; 574 pairs): supported MMD 0.161 vs matched-unsupported 0.208, **paired median Δ −0.039 [−0.053, −0.029]**, Wilcoxon p = 1e-16, Cliff's δ −0.19 (median identity gap between matched pairs = 0.014).

**A4 residual** — sequence metrics linearly explain **R² = 0.68** of MMD. The residual MMD is still lower for supported loci (Cliff's δ −0.10, Cohen's d −0.25, MW p = 2e-6); residual-only support AUC = 0.55 [0.53, 0.57] (descriptive, full-data fit).

**A robustness** — dedup to one row/gene (n=2,948): Δ +0.0068. Exclude near-identical/short (n=2,953): Δ +0.0064. Sign stable.

### B — island matcher (N = 200 sampled islands each; MPS pass)

**B1 assigned vs cross-locus** (ROC-AUC; all CIs are **cluster bootstraps resampling whole
reference islands**, so the ~199 negatives sharing a reference move together):

| | mouse | cow |
|---|---|---|
| embedding-SW alone | 0.903 [0.887,0.919] | 0.909 [0.892,0.924] |
| nucleotide-SW alone | 0.844 [0.814,0.876] | 0.868 [0.836,0.897] |
| nucleotide-identity alone | 0.828 | 0.848 |
| **Δ(embedding − nucleotide-SW), alone** | **+0.059 [+0.031,+0.087]** | **+0.041 [+0.014,+0.070]** |
| sequence-feature model (GroupKFold by ref island) | 0.897 | 0.912 |
| **sequence + embedding-SW (CV)** | **0.935** | **0.952** |
| **ΔROC (adding embedding-SW)** | **+0.038 [+0.022,+0.056]** | **+0.040 [+0.026,+0.055]** |

Embedding-alone AUC reproduces the stored null-test values (0.898 mouse / 0.907 cow) to
within 0.005. The ROC increment from adding embedding-SW to the full sequence baseline is
clear and its cluster-bootstrap CI excludes 0 in both species (the PR increment is near
zero and its CI includes 0: −0.005 [−0.034,+0.024] mouse / +0.012 [−0.017,+0.040] cow — ROC
is the comparable metric given the extreme diagonal/off-diagonal imbalance). Restricting the
assigned set to the pipeline-native-faithful islands (|recomputed−stored| < 0.02; §2a) gives
the same conclusion: Δ(embedding − nucleotide-SW) = +0.060 [0.032,0.088] / +0.055 [0.029,0.084].

**B2 within-locus ranking** (tiled same-length windows; loci with ≥2 windows, mouse n=87 / cow n=86):

| | mouse emb | mouse nt-SW | cow emb | cow nt-SW |
|---|---|---|---|---|
| assigned ranked #1 | 0.31 | 0.18 | 0.24 | 0.12 |
| MRR | 0.578 | 0.472 | 0.531 | 0.444 |
| median rank-percentile | 0.111 | 0.286 | 0.163 | 0.222 |

Embedding-SW localises the assigned window first ~1.7–2.1× as often as nucleotide-SW.

**B3 low-sequence-similarity subset** (thresholds pre-registered: bottom-quartile `sw_norm`
at 0.205; and `sw_aligned_ident < 0.5`). All AUC CIs are cluster bootstraps over reference
islands; ΔAUC is the paired embedding − nucleotide-SW difference on the same resample:

| stratum | n | embedding AUC [CI] | nucleotide-SW AUC [CI] | Δ(emb−nt) [CI] | frac(real<shuf) [CI] |
|---|---|---|---|---|---|
| mouse, bottom-quartile sw_norm | 50 | 0.798 [0.766,0.830] | 0.615 [0.554,0.678] | **+0.184 [0.119,0.249]** | 0.84 [0.74,0.94] |
| mouse, aligned-ident < 0.5 | 13 | 0.758 [0.679,0.829] | 0.544 [0.422,0.661] | **+0.214 [0.063,0.358]** | 0.92 [0.77,1.00] |
| cow, bottom-quartile sw_norm | 50 | 0.796 [0.762,0.829] | 0.616 [0.552,0.678] | **+0.180 [0.110,0.254]** | 0.76 [0.64,0.88] |
| cow, aligned-ident < 0.5 | 11 | 0.815 [0.765,0.864] | 0.433 [0.356,0.505] | **+0.382 [0.282,0.484]** | 0.91 [0.73,1.00] |

Where nucleotide similarity is weakest, embedding-SW retains strong discrimination while
nucleotide-SW collapses toward chance; every Δ(emb−nt) cluster-bootstrap CI excludes 0
(the aligned-ident<0.5 strata have n = 11–13 and correspondingly wide CIs — treat as
indicative). Assigned-vs-dinucleotide-shuffle overall: embedding d_real 0.045 ≪ d_shuf 0.096
(frac 0.92 [0.88,0.96], mouse) — stronger separation than the nucleotide-SW shuffle contrast
(real 0.274 vs shuf 0.173, frac 0.835).

## 6. Negative / inconclusive findings

- **Short-ncRNA incremental predictive value is small and biotype-concentrated.** ΔROC ≈ +0.006 over the full sequence baseline, and it disappears entirely once miRNAs are removed (Δ −0.0007). MMD alone does not beat sw_norm/identity alone.
- **PR-AUC increment for islands is not resolved** (CI includes 0), even though the ROC increment is.
- **B3 aligned-ident<0.5 stratum is small** (n = 11–13); the Δ(emb−nt) CI still excludes 0 but is wide — treat as indicative, not definitive.
- Cross-locus negatives are "easy" (different genes); the harder within-locus test (B2) shows a smaller but consistent embedding advantage.
- **Recomputed↔stored embedding-distance Pearson is only 0.88/0.81**, which would be a red flag for a "same scorer" claim — but it is fully explained by right-censoring of the stored distance at 0.1 (§2a): Spearman ρ = 0.93, median |Δ| = 5e-4, AUC reproduced to ≤0.005, and the ΔAUC is unchanged on the pipeline-native-faithful subset. Not an outstanding concern, but it must be stated rather than glossed as "same scorer".

## 7. Three distinct evidentiary levels

- **(i) Beyond pairwise identity — YES (both settings).** MMD/embedding-SW is not a deterministic function of pairwise identity: A3 identity-matched pairs and A4 residual analysis both show significant support-associated MMD structure after conditioning on identity; islands add discrimination where identity is low (B3).
- **(ii) Beyond the full tested sequence baseline — settings differ.** Islands: **clear** (ΔROC +0.04, CI excludes 0, both species; dominates in the low-similarity subset). Short ncRNAs: **modest** (ΔROC +0.006, CI excludes 0 but concentrated in one biotype).
- **(iii) Evidence for RNA structure or function — NOT established.** No structure- or function-based test was performed here. Any structural interpretation is outside this analysis.

## 8. Recommended manuscript wording (supported)

- "Embedding similarity is not a deterministic function of pairwise nucleotide identity: sequence metrics explain 68% of the short-ncRNA MMD variance, and after matching predictions on nucleotide identity, sequence-SW, length and biotype, annotation-supported loci retain significantly lower MMD (paired median Δ = −0.039, 95% CI [−0.053, −0.029])."
- "For the island matcher, adding embedding-SW to a full conventional sequence-similarity baseline improves assigned-vs-cross-locus discrimination (ΔROC-AUC +0.038 [0.021, 0.057] human–mouse; +0.040 [0.025, 0.057] human–cow)."
- "In the lowest-nucleotide-similarity quartile, embedding-SW discriminates the correct syntenic partner (AUC ≈ 0.80) where nucleotide Smith–Waterman is near chance (AUC ≈ 0.61)."
- "Embedding-SW localises the assigned window first ~2× as often as nucleotide-SW within the same syntenic locus."

## 9. Wording that would remain UNSUPPORTED

- That RiNALMo embeddings "encode conserved RNA structure / secondary structure / function" (no structural test performed).
- That the embedding score adds *large* predictive value for short-ncRNA annotation support (increment is small and miRNA-driven).
- That embedding similarity is "independent of" sequence similarity (they are strongly correlated, r ≈ 0.8).
- Any claim generalised beyond the human–mouse and human–cow pairs and N=200 island sample tested here.

## 10. Case studies (Fig. 5)

`scripts/case_studies_fig5.py` → `case_studies_fig5.tsv` reports the eight requested fields
for the three loci in the case-study figure, reusing `short_ncrna_metrics.tsv` (identical
metric definitions) and computing the overlapped mm39 gene from GENCODE+tRNA exons:

| Fig-5 locus | ref gene id | query coords (mm39) | ref/qry len | edit-ident | sw_norm | SW aln-ident | MMD | overlapped mm39 gene |
|---|---|---|---|---|---|---|---|---|
| SNORD57 (chain 7) | ENSG00000226572 | chr2:130,119,932–130,120,004 | 72/72 | 0.736 | 0.667 | 0.797 | 0.000 | Snord57 (Nop56 host) |
| RNU6-7 (chain 1) | ENSG00000201654 | chr12:52,650,566–52,650,673 | 107/107 | 0.925 | 0.963 | 1.000 | 0.000 | Gm24859 |
| vault RNA (chain 57) | ENSG00000199990 | chr18:36,934,960–36,935,061 | 99/101 | 0.594 | 0.348 | 0.632 | 0.137 | Vaultrc5 |

The vault case is the informative one: correct localisation to Vaultrc5 at only 59% edit
identity / 0.35 sw_norm (MMD 0.14) — the same low-nucleotide-similarity regime where the
island analysis (§B3) shows the largest embedding advantage.

---

## Conclusion (per the required rubric)

Selecting strictly from the results, and not forcing conclusion A:

- **Island matcher → A (strong incremental evidence):** *"Embedding similarity improves correspondence discrimination after conditioning on conventional nucleotide sequence-similarity metrics."* (ΔROC +0.04, CI excludes 0 in both species; embedding AUC ≈ 0.80 vs nucleotide ≈ 0.61 in the low-similarity subset; better within-locus ranking.)
- **Short ncRNAs → B (limited incremental evidence):** *"Embedding similarity is not fully determined by pairwise identity and provides modest additional information beyond the tested sequence metrics."* (Identity-matched and residual analyses are significant; the predictive increment over the full baseline is small and concentrated in miRNAs.)

**Conclusion C is rejected in both settings.** The overall, conservative single statement supported end-to-end is **B**, strengthening to **A** in the island-matching regime and specifically in the low-nucleotide-similarity subset.
