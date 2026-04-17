all :
	cd rapport ;pdflatex rapport.tex && bibtex rapport && pdflatex rapport.tex && pdflatex rapport.tex && evince rapport.pdf
clean:
	cd rapport; rm -f *.aux *.log *.out *.toc *.pdf *.bbl *.blg
.PHONY: all clean
