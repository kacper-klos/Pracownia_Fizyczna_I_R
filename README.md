# Pracownia Fizyczna I R

To repozytorium zawiera raporty sporządzone w trakcie realizacji przedmiotu [Pracownia Fizyczna I R](https://usosweb.uw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1100-1Ind25) na Uniwersytecie Warszawskim.

## Struktura repozytorium

```
.
├── pelne/          # pierwsze pięć raportów, wszystkie ocenione na 5
└── skrocone/       # pozostałe raporty w wersji skróconej
     └── <nazwa_eksperymentu>/
         ├── raport.tex     # źródło LaTeX
         ├── raport.pdf     # wynikowy PDF
         ├── analiza/       # skrypty Python + dane pomiarowe
         └── images/        # wykresy i zdjęcia używane w LaTeX
```

> **Uwaga**  
> Zarówno raporty pełne, jak i skrócone mają identyczną strukturę katalogów.

## Wymagania

- Python ≥ 3.10  
- biblioteki: `numpy`, `pandas`, `matplotlib`, `scipy` (oraz inne w razie potrzeby)  
- menedżer zależności **Poetry** (zalecany)

## Uruchamianie skryptów

1. Przejdź do katalogu wybranego doświadczenia.
2. Uruchom skrypt, np.:

```bash
python analiza/analiza.py
# lub
poetry run python analiza/analiza.py
```

> **Ważne**  
> • Skrypt zapisuje wykresy w podkatalogu `images/`. Jeśli uruchomisz go z innej lokalizacji, mogą wystąpić błędy ścieżek.  
> • W niektórych przypadkach przed uruchomieniem trzeba odkomentować kilka ostatnich linii kodu (zależnie od eksperymentu).

## Licencja i etyka

Materiały zawarte w repozytorium służą wyłącznie jako przykład poprawnie przygotowanych raportów.  
Podczas realizacji przedmiotu **nie wolno** kopiować danych ani kodu z tego repozytorium.
