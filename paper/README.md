# Paper

LaTeX source for the CURIA preprint.

```
paper/
  main.tex              # top-level: pulls in sections, refs, bib style
  refs.bib              # bibliography (auto-exported from Zotero, see below)
  Makefile              # build commands
  sections/*.tex        # 00_abstract .. 04_discussion
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

## Keeping refs.bib in sync with Zotero

`refs.bib` is a generated file. The clean way to "update dynamically" is the
**Better BibTeX** Zotero add-on, which keeps the `.bib` file continuously in
sync with your library — no manual re-export.

1. Install Better BibTeX for Zotero:
   <https://retorque.re/zotero-better-bibtex/installation/>.
2. Set stable citation keys: Zotero → Settings → Better BibTeX → Citation keys,
   e.g. `[auth:lower][year]` → `kirilenko2023`. (This repo currently uses keys
   like `kirilenko2023toga`; pick a format and keep it.)
3. Right-click the collection for this paper → **Export Collection…**
   - Format: **Better BibLaTeX** (or **Better BibTeX** for classic bibtex),
   - tick **Keep updated**.
   Point it at this file: `paper/refs.bib`.
4. From now on, whenever you add/edit an item in that Zotero collection, Better
   BibTeX rewrites `paper/refs.bib` automatically. Just `make` again.

Cite in the text with natbib: `\citep{kirilenko2023toga}` (parenthetical) or
`\citet{...}` (textual).

Notes:
- Keep one Zotero **collection** dedicated to this paper so the auto-export scope
  is exactly the papers you cite.
- The "extra / non-Zotero entries" marker at the bottom of `refs.bib` is for
  one-off entries you don't want in Zotero. Note that a full re-export can
  overwrite the file, so prefer putting everything in Zotero.
- Commit `refs.bib` so co-authors/Overleaf build without Zotero installed.
