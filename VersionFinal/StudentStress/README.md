# Proyecto: Clasificacion de niveles de ansiedad en estudiantes

Este proyecto implementa un flujo simple de reconocimiento de patrones para clasificar niveles de ansiedad a partir del dataset `StressLevelDataset.csv`.

Aunque el dataset original se enfoca en monitoreo de estres estudiantil, en esta actividad se usa la columna `anxiety_level` como variable objetivo. La variable original se conserva y se crea una nueva clase multiclase llamada `anxiety_class`.

## Estructura

```text
Proyecto/
│
├── data/
│   └── StressLevelDataset.csv
│
├── modelos/
│   ├── __init__.py
│   ├── bayes.py
│   ├── arbol.py
│   └── common.py
│
├── resultados/
│   ├── eda/
│   ├── preprocesamiento/
│   ├── seleccion_caracteristicas/
│   └── clasificacion/
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── training.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Fases implementadas

1. Analisis exploratorio de datos.
2. Preprocesamiento.
3. Seleccion de caracteristicas.
4. Entrenamiento de modelos.
5. Evaluacion multiclase.

## Modelos usados

Se entrenan seis variantes principales:

- Bayes ingenuo con todas las caracteristicas.
- Bayes ingenuo con mejores caracteristicas por Informacion Mutua.
- Bayes ingenuo con mejores caracteristicas por Chi-cuadrado.
- Arbol de decision con todas las caracteristicas.
- Arbol de decision con mejores caracteristicas por Informacion Mutua.
- Arbol de decision con mejores caracteristicas por Chi-cuadrado.

## Metricas de evaluacion

Para cada modelo se calculan:

- Exactitud o tasa de reconocimiento.
- Tasa de error.
- Precision por clase.
- Recall por clase.
- F-score por clase.
- F1 macro.
- F1 ponderado.
- Matriz de confusion.

## Nota metodologica importante

La seleccion de caracteristicas se calcula solo con `X_train` y `y_train`. Esto evita fuga de informacion hacia el conjunto de prueba.

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Validar estructura:

```bash
python main.py --check
```

Ejecutar pipeline completo:

```bash
python main.py --top-k 10
```

Ejecutar eliminando `stress_level` como predictor:

```bash
python main.py --top-k 10 --drop-stress-level
```

Cambiar variante de Bayes:

```bash
python main.py --bayes-type categorical
```

## Resultados principales

Los resultados se guardan en:

```text
resultados/
```

La tabla comparativa final aparece en:

```text
resultados/clasificacion/resumen_modelos.csv
```

Cada modelo tiene su propia carpeta con:

- `metrics.json`
- `classification_report.txt`
- `classification_report.csv`
- `metricas_por_clase.csv`
- `predicciones_test.csv`
- `matriz_confusion.png`
- `metricas_por_clase.png`


## Nuevas visualizaciones agregadas

Para cada experimento de clasificacion se generan salidas adicionales:

- `regiones_decision_2d.png`: vista 2D de clases, regiones y errores de clasificacion.
- `arbol_nodos_hojas.png`: solo para arboles, muestra nodos y hojas.
- `campana_gaussiana_bayes.png`: solo para Bayes gaussiano, muestra una campana de Gauss asociada a la primera caracteristica del experimento.


## Actualizacion de requisitos

El proyecto ahora incluye:

- Reporte TXT de EDA en `resultados/eda/reporte_eda.txt`.
- Discretizacion en tres niveles de `anxiety_level`, `self_esteem` y `depression`.
- Seleccion de caracteristicas por:
  - Informacion Mutua.
  - Chi-cuadrado.
  - Importancia de Arbol de Decision.
- Variantes de seleccion por cantidad de caracteristicas: `n=5`, `n=10` y `n=15` por defecto.
- Resultados de clasificacion organizados por subcarpetas:
  - `resultados/clasificacion/n_5/`
  - `resultados/clasificacion/n_10/`
  - `resultados/clasificacion/n_15/`

### Ejecucion recomendada

```bash
python main.py --top-k-list 5,10,15
```

Para ejecutar un solo valor:

```bash
python main.py --top-k 10
```
