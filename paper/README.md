# Paper

LaTeX source for the CURIA preprint.

```
paper/
  main.tex              # top-level: pulls in sections, refs, bib style
  CURIA_Preprint.bib    # bibliography (auto-exported from Zotero, see below)
  Makefile              # build commands
  sections/*.tex        # 00_abstract .. 05_availability
  figures/*.pdf         # figures (Affinity export or analysis/make_figures.py)
  tables/*.tex          # tables (analysis/make_tables.py)
analysis/
  make_figures.py       # writes figures/*.pdf
  make_tables.py        # writes tables/*.tex
  make_conserved_cores.py  # writes paper/lncRNAs_with_conserved_cores.tsv
```

## Recurrent-core table package

The recurrent-core export contains gene-level, core-level, and core-by-assembly
TSV tables plus distance/quorum sensitivity tables. Regenerate the package with:

```bash
make recurrent-tables PYTHON=../.venv/bin/python  # from paper/
# or directly, from repo root:
.venv/bin/python analysis/make_conserved_cores.py --results-dir preprint_results \
    --out-dir paper/recurrent_core_tables \
    --legacy-out paper/lncRNAs_with_conserved_cores.tsv
```

It reuses Figure 6's core definition (`make_figures._load_islands` /
`_cluster_cores`) and classifies each gene by proximity to protein-coding loci.
The default export uses the completed 19-assembly panel, a quorum of at least 17
assemblies, and a mean reference-island length of at least 120 bp. Strict
per-species distance counts are reported in the sensitivity tables and do not
require a new embedding run. See `paper/recurrent_core_tables/README.md` for
field definitions and provenance.

## Build

```bash
cd paper
make            # -> main.pdf (needs latexmk + pdflatex + bibtex)
make assets     # regenerate programmatic figures + tables
make all        # assets, then main.pdf
make watch      # rebuild on save
make clean      # drop aux files
```

If you don't have TeX locally, install MacTeX (`brew install --cask mactex-no-gui`)
or use Overleaf (push this folder to a Git-backed Overleaf project).

## Figure pipeline

Figures are built three different ways depending on their content. `make figures`
regenerates only the **scripted** ones; the others are hand-made assets committed
in `figures/`.

| Fig | How it's made | File in `figures/` |
|-----|---------------|--------------------|
| 1 embeddings   | **scripted** (`fig1_embeddings`)         | `fig1_embeddings.pdf` |
| 2 pipeline     | hand-made schematic (PNG)                | `fig2_pipeline.png`   |
| 3 islands      | **scripted** (`fig3_islands`): A schematic + B real SNHG12 SW-matrix | `fig3_islands.pdf` |
| 4 MMD          | **scripted** (`fig4_mmd`)                | `fig4_mmd.pdf`        |
| 5 case studies | UCSC screenshots, hand-arranged          | `fig5_cases.pdf`      |
| 6 cores        | **scripted** (`fig6_cores`)              | `fig6_cores.pdf`      |

Figs 2 (pipeline schematic) and 5 (genome-browser screenshots) are hand-made
assets; Figs 1/3/4/6 are matplotlib. Figs 4/6 are wired to `--results-dir` and
still stubbed pending the wider run.

Scripted figures are composed **entirely in matplotlib** — panel letters, sizing
to the paper text width, embedded (editable) fonts — so no hand-composition step
is needed. Shared style lives in `analysis/figstyle.py`; each figure is one
builder in `analysis/make_figures.py`.

```bash
# from repo root, using the project venv:
.venv/bin/python analysis/make_figures.py --outdir paper/figures
.venv/bin/python analysis/make_figures.py --outdir paper/figures --only fig4_mmd
# data-dependent panels read from a results dir (plug in the RiNALMo run):
.venv/bin/python analysis/make_figures.py --outdir paper/figures --results-dir ../preprint_results
```
(Or `make figures PYTHON=../.venv/bin/python` from `paper/`.)

**Figure 1 is two-step** (embeddings are expensive; plotting stays torch-free):
```bash
# 1) heavy: RiNALMo embeddings -> analysis/data/fig1_embeddings.npz  (run once)
.venv/bin/python analysis/compute_fig1_embeddings.py
# 2) light: compose the panels from the cached npz
.venv/bin/python analysis/make_figures.py --only fig1_embeddings --outdir paper/figures
```
Panels A/B (per-token tRNA/miRNA; mean-pooled signal-vs-background) need only a
handful of model RNAs + RiNALMo — no pipeline run. Panels C/D still stubbed.
Committing `analysis/data/fig1_embeddings.npz` lets `make_figures` rebuild Fig 1
without torch/the 2.4 GB model.

Notes:
- Scripted builders emit **PDF** (vector) + a PNG preview. The `.tex` currently
  `\includegraphics` the `.png` placeholders (the old RNA-FM figures); switch each
  to `.pdf` once the scripted version is wired to real data.
- Iterate in `analysis/notebooks/`, but keep the *final* composition in
  `make_figures.py` so `make figures` reproduces every scripted panel.
- Panels currently draw labelled TODO stubs; each names its data source (old
  notebook / raw plot) so wiring tomorrow's results is mechanical.

## Keeping the bibliography in sync with Zotero

The bibliography lives in **`CURIA_Preprint.bib`** (exported from Zotero via
Better BibTeX) and `main.tex` reads it with `\bibliography{CURIA_Preprint}`. It is
a generated file — treat Zotero as the source of truth and let Better BibTeX keep
the `.bib` continuously in sync (no manual re-export).

Your intended workflow — *"export everything from Zotero, then add dynamically as
things come up"* — is exactly right, and Better BibTeX supports it directly:

1. Install Better BibTeX for Zotero:
   <https://retorque.re/zotero-better-bibtex/installation/>.
2. Set stable citation keys: Zotero → Settings → Better BibTeX → Citation keys.
   The current export uses the default `[auth][title][year]` style (e.g.
   `kirilenkoIntegratingGeneAnnotation2023`). Keep whatever format you pick — if
   you change it later, existing `\cite{...}` keys in the `.tex` must be updated.
3. Right-click your library (or a dedicated collection) → **Export Library…** /
   **Export Collection…**
   - Format: **Better BibTeX**,
   - tick **Keep updated**.
   Point it at `paper/CURIA_Preprint.bib` (overwrite the existing file).
4. From now on, adding or editing an item in Zotero rewrites
   `paper/CURIA_Preprint.bib` automatically. Just `make` again.

Cite in the text with natbib: `\citep{key}` (parenthetical) or `\citet{key}`
(textual); multiple keys are comma-separated, `\citep{a,b,c}`.

### Citation style (numeric, arXiv-like)

- `main.tex` loads `\usepackage[numbers,sort&compress]{natbib}`, so citations
  render as numbers with range compression: `\citep{a,b,c,f}` → `[1-3,6]`.
- The reference list uses **`unsrtnatetal.bst`** (committed in `paper/`): it is
  `unsrtnat` (numbered by order of appearance) patched to truncate long author
  lists to *"First, Second, Third et al."*. The cutoff is one line at the top of
  the `.bst`: `FUNCTION {max.num.names} { #3 }` — change `#3` to `#6` (etc.) to
  show more authors. A copy also lives in `~/Library/texmf/...`; the in-repo copy
  is what makes Overleaf / fresh clones build.

Notes:
- Exporting the **whole library** is fine — only entries actually `\cite`d appear
  in the compiled bibliography, so unused entries cost nothing. A dedicated
  collection just makes the file smaller and the scope tighter; optional.
- There is currently a duplicate RiNALMo entry
  (`penicRiNALMoGeneralpurposeRNA2025` and `...2025a`); the paper uses the non-`a`
  key. Merge the duplicate in Zotero when convenient.
- Commit `CURIA_Preprint.bib` so co-authors / Overleaf build without Zotero.
