
# PlotService

`PlotService` ist eine Python-Klasse für erweiterte Datenvisualisierungen mit Plotly. Sie bietet Methoden zum Erstellen von Liniendiagrammen, Balkendiagrammen, Histogrammen, Boxplots, Autokorrelationen (ACF und PACF) und einer Streudiagrammmatrix. Diese Visualisierungen sind besonders nützlich für die Analyse von Zeitreihen- und statistischen Daten.

## Installation

Um das Projekt lokal zu verwenden, stellen Sie sicher, dass Python und die folgenden Bibliotheken installiert sind:

```bash
pip install numpy pandas plotly statsmodels
```

## Funktionen

Die `PlotService`-Klasse umfasst mehrere Methoden zur Datenvisualisierung, darunter:

1. **Line Chart** (`plot_line_chart`): Erstellt ein Liniendiagramm mit mehreren Spalten als Y-Werte.
2. **Bar Chart** (`plot_bar_chart`): Visualisiert die Daten als Balkendiagramm.
3. **Histogram** (`plot_histogram`): Erstellt ein Histogramm, um die Verteilung von Daten zu visualisieren.
4. **Boxplot** (`plot_boxplot`): Erstellt Boxplots für ausgewählte Spalten zur Verteilung von Werten.
5. **ACF und PACF** (`plot_acf_pacf`): Berechnet und visualisiert die Autokorrelation (ACF) und partielle Autokorrelation (PACF) einer Zeitreihe.
6. **Scatter Plot Matrix** (`plot_scatter_matrix`): Zeigt die Korrelation zwischen allen Spalten als Streudiagrammmatrix an.

## Verwendung

Hier sind einige Beispiele für die Verwendung der `PlotService`-Klasse:

```python
import numpy as np
import pandas as pd
from plot_service import PlotService  # Die PlotService-Klasse importieren

np.random.seed(42)
num_entries = 1000
data = {
    'A': np.random.normal(loc=0, scale=1, size=num_entries),
    'B': np.random.normal(loc=5, scale=2, size=num_entries),
    'C': np.random.exponential(scale=1, size=num_entries),
    'D': np.random.uniform(low=0, high=10, size=num_entries),
    'E': np.random.lognormal(mean=1, sigma=0.5, size=num_entries),
    'F': np.random.binomial(n=100, p=0.3, size=num_entries)
}

df = pd.DataFrame(data)

# Instanz der PlotService-Klasse erstellen
ps = PlotService()

# Liniendiagramm für die Spalten 'A' und 'B'
ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot")

# Balkendiagramm für die ersten 10 Werte der Spalten 'A' und 'C'
ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot")

# Histogramm für die Spalten 'A' und 'B'
ps.plot_histogram(df, columns=['A', 'B'], title="Histogram")

# Boxplot für die Spalten 'A' und 'C'
ps.plot_boxplot(df, columns=['A', 'C'], title="Boxplot")

# PACF (partielle Autokorrelation) für die Spalte 'A'
ps.plot_acf_pacf(df['A'], b_pacf=True)

# ACF (Autokorrelation) für die Spalte 'A'
ps.plot_acf_pacf(df['A'])

# Streudiagrammmatrix für alle Spalten
ps.plot_scatter_matrix(df, title="Scatter Plot Matrix")
```

## Methodenbeschreibung

- **plot_line_chart**: Visualisiert die Daten in einem Liniendiagramm.
- **plot_bar_chart**: Erstellt ein Balkendiagramm, das die Datenreihen für die ausgewählten Spalten zeigt.
- **plot_histogram**: Zeigt die Häufigkeitsverteilung der ausgewählten Spalten.
- **plot_boxplot**: Erstellt Boxplots, die die Quartilsverteilung der Werte zeigen.
- **plot_acf_pacf**: Berechnet die Autokorrelation (ACF) oder partielle Autokorrelation (PACF) für eine Zeitreihe.
- **plot_scatter_matrix**: Visualisiert die Korrelationen zwischen mehreren Variablen in einer Streudiagrammmatrix.

## Abhängigkeiten

- `numpy`: Für numerische Operationen und Array-Verarbeitung
- `pandas`: Für die Arbeit mit Datenrahmen
- `plotly`: Für die interaktive Visualisierung
- `statsmodels`: Für ACF- und PACF-Berechnungen

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).
