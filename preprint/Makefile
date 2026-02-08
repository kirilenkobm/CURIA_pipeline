PDF=main.pdf
TEX=main.tex

all: $(PDF)

$(PDF): $(TEX) references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	latexmk -C
