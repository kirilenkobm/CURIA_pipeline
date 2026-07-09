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
```

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
| 1 embeddings   | **scripted** (`fig1_embeddings`) | `fig1_embeddings.pdf` |
| 2 pipeline     | schematic (Affinity/draw.io)     | `fig2_pipeline.pdf`   |
| 3 islands      | A = schematic, **B scripted** (`fig3b_dotplot`) | `fig3_islands.pdf` |
| 4 MMD          | **scripted** (`fig4_mmd`)        | `fig4_mmd.pdf`        |
| 5 case studies | UCSC screenshots, hand-arranged  | `fig5_cases.pdf`      |
| 6 cores        | **scripted** (`fig6_cores`)      | `fig6_cores.pdf`      |

Scripted figures are composed **entirely in matplotlib** — panel letters, sizing
to the paper text width, embedded (editable) fonts — so no hand-composition step
is needed. Shared style lives in `analysis/figstyle.py`; each figure is one
builder in `analysis/make_figures.py`.

```bash
# from repo root, using the project venv:
.venv/bin/python analysis/make_figures.py --outdir paper/figures
.venv/bin/python analysis/make_figures.py --outdir paper/figures --only fig4_mmd
# data-dependent panels read from a results dir (plug in the RiNALMo run):
.venv/bin/python analysis/make_figures.py --outdir paper/figures --results-dir ../rinalmo_version_outputs
```
(Or `make figures PYTHON=../.venv/bin/python` from `paper/`.)

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
