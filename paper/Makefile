# Build the paper. Requires a LaTeX toolchain (latexmk + pdflatex) and bibtex.
#
#   make          -> build main.pdf
#   make assets   -> regenerate figures/*.pdf and tables/*.tex from analysis/
#   make all      -> assets + main.pdf
#   make watch    -> rebuild on save (latexmk -pvc)
#   make clean    -> remove LaTeX aux files
#   make distclean-> also remove main.pdf

MAIN    := main
PYTHON  := python
ANALYSIS:= ../analysis

.PHONY: all pdf assets figures tables cores watch clean distclean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex CURIA_Preprint.bib sections/*.tex tables/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

all: assets pdf

assets: figures tables

figures:
	$(PYTHON) $(ANALYSIS)/make_figures.py --outdir figures

tables:
	$(PYTHON) $(ANALYSIS)/make_tables.py --outdir tables

# Supplementary conserved-cores TSV (large; linked from the paper, not inlined).
# Writes lncRNAs_with_conserved_cores.tsv (in paper/) from the run panel.
cores:
	$(PYTHON) $(ANALYSIS)/make_conserved_cores.py --results-dir ../rinalmo_version_outputs --out lncRNAs_with_conserved_cores.tsv

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl

distclean: clean
	rm -f $(MAIN).pdf
