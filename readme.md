# Vizualizace Fraktálů

### Projekt do předmětu VVP 
### Kateřina Štefánková, 15.4.2023

## Popis

Projekt se zabývá interaktivní vizualizací Mandelbrotovy a Juliovy množiny.
Teoretický popis problému lze nalézt např. na Wikipedii: https://en.wikipedia.org/wiki/Mandelbrot_set, https://en.wikipedia.org/wiki/Julia_set.

#### Repozitář obsahuje soubory:
- *projectLibrary.py* - knihovna obsahující třídy *Mandelbrot* a *Julia*, pomocí kterých lze vykreslit interaktivní graf těchto množin
- *examples.ipybnb* - soubor obsahující ukázky použití knihovny *projectLibrary*
- *projectAssignment.md* - soubor se zadáním projektu
- *readme.md* - soubor readme s informacemi k repozitáři

## Použití

Ukázkové použití je uvedeno v souboru *examples.ipybnb*
#### Doporučeno:
- import projectLibrary
- vytvořit objekt třídy *Mandelbrot* nebo *Julia* s odpovídajícími parametry
- zavolat funkci objektu *interactive_plot()*

Lze také vypsat divergenční matici množiny (funkce *compute()*) nebo vykreslit neinteraktivní graf (funkce *plot_set()*).
