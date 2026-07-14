# Manuscript-ready material — embedding vs. sequence baseline

Draft text with numbers inserted from this analysis. **No manuscript source files are
modified here.** All figures referenced live in `analysis/embedding_vs_sequence/figures/`.
The language deliberately does **not** claim that RiNALMo embeddings encode conserved RNA
structure or function, because no structure-based test was performed.

---

## Methods (one paragraph)

> To test whether the RiNALMo embedding similarity carries correspondence information
> beyond conventional nucleotide similarity, we compared it against a panel of
> sequence-only metrics computed from the same reference and query sequences: Levenshtein
> identity, a normalised affine-gap Smith–Waterman score (match +2, mismatch −1, gap
> −5/−1; normalised by 2·min-length), Smith–Waterman aligned identity, 3-mer and 4-mer
> cosine similarity, length ratio, GC-content difference, and dinucleotide-composition
> distance. For the short-ncRNA benchmark (human→mouse, n = 2,994) the embedding score was
> the MMD used in the pipeline and the label was any-exon annotation support; we fit
> L2-regularised logistic regression with standardisation inside each fold and 5-fold
> StratifiedGroupKFold grouped by gene (5 shuffles), comparing a sequence-only baseline to
> the same features plus MMD, and reported paired bootstrap AUC differences. For the island
> matcher (human→mouse and human→cow) we recovered the per-pair embedding cosine
> Smith–Waterman scores for assigned pairs, cross-locus negatives, dinucleotide-preserving
> shuffles, and tiled within-locus windows with a single bounded RiNALMo pass over N = 200
> sampled islands (recomputed-vs-stored distance Pearson r = 0.88/0.81), and computed the
> matching nucleotide metrics on identical pairs. All island scores used in AUC
> computations were the freshly recomputed values (assigned, cross-locus, shuffle and
> tiled-window scores from one scorer); the pipeline's stored distances, which are
> right-censored at the acceptance threshold, served only as a reproduction check
> (recomputed-vs-stored Spearman ρ = 0.93, median |Δ| = 5×10⁻⁴). Discrimination was
> quantified with ROC/PR-AUC (models fit with GroupKFold grouped by reference island), with
> confidence intervals from a cluster bootstrap that resamples whole reference islands,
> identity-matched effect sizes, and a pre-registered low-nucleotide-similarity subset
> (bottom sw_norm quartile; aligned identity < 50%).

## Results (one paragraph)

> Embedding similarity was strongly but not perfectly correlated with nucleotide similarity
> (short-ncRNA MMD vs identity Pearson r = −0.80), and sequence metrics linearly explained
> 68% of MMD variance. Conditioning on sequence similarity did not remove the embedding
> signal: within nearest-neighbour pairs matched on identity, sequence-SW, length and
> biotype, annotation-supported short ncRNAs retained lower MMD (paired median Δ = −0.039,
> 95% CI [−0.053, −0.029]; Wilcoxon p = 1×10⁻¹⁶), and residual MMD after regressing out the
> sequence metrics still separated supported from unsupported loci (Cliff's δ = −0.10,
> p = 2×10⁻⁶). The added predictive value of MMD over the full sequence baseline was small,
> however (ΔROC-AUC = +0.006, 95% CI [+0.003, +0.010]) and was concentrated in miRNAs
> (Δ = −0.001 with miRNAs excluded). The island matcher showed a clearer effect: adding
> embedding-SW to the full sequence baseline improved assigned-vs-cross-locus discrimination
> from ROC-AUC 0.90→0.94 (human–mouse, Δ +0.038 [+0.021, +0.057]) and 0.91→0.95
> (human–cow, Δ +0.040 [+0.026, +0.055]; cluster-bootstrap CIs over reference islands), and
> in the lowest nucleotide-similarity quartile embedding-SW retained AUC 0.80 [0.77, 0.83]
> while nucleotide Smith–Waterman fell to 0.62 [0.55, 0.68] (paired Δ +0.18 [+0.11, +0.25]).
> Within a single syntenic locus, embedding-SW ranked the assigned window first roughly twice
> as often as nucleotide-SW (top-1 0.31 vs 0.18 human–mouse; 0.24 vs 0.12 human–cow).

## Discussion (one cautious sentence)

> These results indicate that the embedding similarity is not fully reducible to
> conventional nucleotide-similarity metrics — most clearly for island matching in the
> low-sequence-similarity regime — although the present analysis does not establish that
> this residual information reflects conserved RNA structure or function, which would
> require a direct structure-based test.

## Suggested figure captions

- **Fig. (short ncRNA), `figures/A1_mmd_vs_sequence.png`.** RiNALMo MMD versus nucleotide
  identity and normalised Smith–Waterman score for 2,994 human→mouse short-ncRNA
  predictions, coloured by annotation support (blue = supported, red = unsupported); a
  40–60% identity zoom and per-identity-bin median MMD by support are also shown. MMD tracks
  sequence similarity (r ≈ −0.8) but supported loci sit at systematically lower MMD within
  each identity bin.
- **Fig. (short ncRNA residual), `figures/A4_residual_mmd.png`.** Distribution of residual
  MMD (MMD minus its sequence-metric linear prediction; sequence R² = 0.68) for supported
  vs unsupported loci; the supported distribution remains shifted to lower residual MMD
  (Cliff's δ = −0.10, p = 2×10⁻⁶).
- **Fig. (islands), `figures/B1_mouse.png` / `figures/B1_cow.png`.** Assigned vs cross-locus
  score distributions for embedding-SW and nucleotide-SW, and assigned-vs-dinucleotide-
  shuffle scatter. Embedding-SW separates assigned from cross-locus pairs more sharply than
  nucleotide-SW (ROC-AUC 0.90/0.91 vs 0.84/0.87), and the sequence+embedding model reaches
  0.94/0.95.

## Exact numbers (for insertion / traceability)

All below trace to `short_model_results.tsv`, `island_model_results.tsv`,
`short_analysis_summary.json`, `island_analysis_summary.json`.

- Short ncRNA: n = 2,994; support 52.4%; MMD↔identity r = −0.797; seq R²(MMD) = 0.68.
- Short A2: baseline ROC 0.887 / +MMD 0.893 / ΔROC +0.0064 [0.0030, 0.0097]; no-miRNA Δ −0.0007.
- Short single-feature ROC: identity 0.874, sw_norm 0.884, MMD 0.875.
- Short A3 matched (574 pairs): ΔMMD median −0.039 [−0.053, −0.029], Cliff's δ −0.19.
- Short A4: residual Cliff's δ −0.10 (p = 2e-6), residual-only AUC 0.55.
- Islands score reproduction: recomputed↔stored Pearson 0.879/0.808 but Spearman 0.936/0.932, median |Δ| 5e-4/7e-4, only 4/7 of 200 above the 0.1 acceptance cap; emb-alone AUC 0.903/0.909 reproduces stored null-test 0.898/0.907. ΔAUC on |Δ|<0.02 subset: emb-alone 0.917/0.924, Δ(emb−nt) +0.060/+0.055.
- Islands B1 ROC (cluster-bootstrap CIs): emb-alone 0.903 [0.887,0.919] / 0.909 [0.892,0.924]; nt-SW-alone 0.844 [0.814,0.876] / 0.868 [0.836,0.897]; seq 0.897/0.912; seq+emb 0.935/0.952; Δ(add emb) +0.038 [0.022,0.056] / +0.040 [0.026,0.055]; Δ(emb−nt alone) +0.059 [0.031,0.087] / +0.041 [0.014,0.070].
- Islands B2 (multi-candidate loci): top-1 emb 0.31/0.24 vs nt 0.18/0.12; MRR 0.578/0.531 vs 0.472/0.444.
- Islands B3 (bottom-quartile sw_norm, n=50, cluster CI): emb AUC 0.80 [0.77,0.83] / 0.80 [0.76,0.83] vs nt 0.62 [0.55,0.68] / 0.62 [0.55,0.68]; paired Δ +0.18 [0.12,0.25] / +0.18 [0.11,0.25]. (aligned-ident<0.5, n=13/11: Δ +0.21 [0.06,0.36] / +0.38 [0.28,0.48].)

## Case studies (Fig. 5) — per-locus numbers

Human→mouse, from `case_studies_fig5.tsv` (metrics identical to Analysis A). Reference and
query lengths in nt; identity/SW as defined in §3 of the README; MMD is the pipeline score.

| Fig-5 locus | ref gene id | ref coords (hg38) | query coords (mm39) | ref/qry len | edit-identity | sw_norm | SW aln-identity | MMD | overlapped mm39 gene |
|---|---|---|---|---|---|---|---|---|---|
| SNORD57 (chain 7) | ENSG00000226572 | chr20:2,656,938–2,657,010 (+) | chr2:130,119,932–130,120,004 (+) | 72 / 72 | 0.736 | 0.667 | 0.797 | 0.000 | Snord57 (69 bp; Nop56 host 72 bp) |
| RNU6-7 (chain 1) | ENSG00000201654 | chr14:32,202,044–32,202,151 (+) | chr12:52,650,566–52,650,673 (+) | 107 / 107 | 0.925 | 0.963 | 1.000 | 0.000 | Gm24859 (103 bp) |
| vault RNA (chain 57) | ENSG00000199990 | chr5:140,711,274–140,711,373 (+) | chr18:36,934,960–36,935,061 (+) | 99 / 101 | 0.594 | 0.348 | 0.632 | 0.137 | Vaultrc5 (101 bp) |

Suggested sentence to append to the case-study paragraph:
> Quantitatively, the SNORD57 projection recovers annotated mouse Snord57 at 74% edit
> identity (MMD 0.00); RNU6-7 recovers Gm24859 at 93% identity (MMD 0.00); and the vault-RNA
> projection localises mouse Vaultrc5 despite only 59% edit identity and 35% normalised
> Smith–Waterman (MMD 0.14), illustrating embedding-guided localisation where nucleotide
> similarity is weak — consistent with the low-similarity island result.

## Wording checklist

**Supported:**
- Embedding similarity is not a deterministic function of pairwise identity.
- Adding embedding-SW improves island correspondence discrimination beyond a full sequence baseline (ΔROC +0.04, CI excludes 0, two species).
- Embedding-SW discriminates where nucleotide Smith–Waterman is near chance (low-similarity quartile: 0.80 vs 0.61).
- Embedding-SW ranks the correct within-locus window first ~2× as often as nucleotide-SW.

**Unsupported (do not write):**
- RiNALMo embeddings "encode conserved RNA structure/function" (no structural test).
- Embeddings add *large* value for short-ncRNA annotation support (increment small, miRNA-driven).
- Embedding similarity is "independent of" sequence similarity (r ≈ 0.8).
- Generalisation beyond the tested pairs / N=200 island sample.
- That the recomputed scores are *identical* to the pipeline's. Correct phrasing: "the same
  cosine Smith–Waterman scorer, recomputed" — and note the stored distances are censored at
  the acceptance threshold, so agreement is rank-level (Spearman 0.93; AUC reproduced to
  ≤0.005), not value-identical (Pearson 0.88/0.81). Do not claim value-level identity.
