pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
Copy-Item build\main.pdf AI_Anxiety_Map.pdf -Force