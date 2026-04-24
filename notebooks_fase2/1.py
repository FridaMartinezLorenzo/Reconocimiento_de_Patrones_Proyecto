"""
Analisis del dataset StressLevelDataset.csv
Actividad: Preprocesamiento y seleccion de caracteristicas

En este dataset las columnas no son continuas: son variables ordinales/categoricas
codificadas como enteros. Por eso el flujo evita discretizar con cuantiles y, en
su lugar, conserva los codigos originales para analizarlos como variables discretas.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_selection import mutual_info_classif

# ------------------------------------------------------------
# 1 CARGAR DATASET
# ------------------------------------------------------------

print("\nCargando dataset...\n")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "StressLevelDataset.csv"

df = pd.read_csv(DATASET_PATH)

TARGET_COL = "stress_level"

print("Dimensiones del dataset:", df.shape)
print("\nColumnas:")
print(df.columns)

print("\nPrimeras filas:")
print(df.head())

# ------------------------------------------------------------
# 1.1 PERFIL DE VARIABLES
# ------------------------------------------------------------

print("\nNormalizando tipos a enteros y perfilando cardinalidad...")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

summary_rows = []

for col in df.columns:
    summary_rows.append(
        {
            "columna": col,
            "tipo": "target ordinal" if col == TARGET_COL else "predictor ordinal/categorico",
            "unicos": int(df[col].nunique()),
            "minimo": int(df[col].min()),
            "maximo": int(df[col].max()),
        }
    )

schema_df = pd.DataFrame(summary_rows)

print("\nEsquema de variables:")
print(schema_df)

schema_df.to_csv(BASE_DIR / "esquema_variables.csv", index=False)

# ------------------------------------------------------------
# 2 REVISION BASICA DEL DATASET
# ------------------------------------------------------------

print("\nValores faltantes:")
print(df.isnull().sum())

print("\nEstadisticas descriptivas:")
print(df.describe())

# Guardar dataset normalizado sin alterar el significado ordinal
df.to_csv(BASE_DIR / "dataset_normalizado.csv", index=False)

# ------------------------------------------------------------
# 3 DEFINIR VARIABLES PARA CLASIFICACION
# ------------------------------------------------------------

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ------------------------------------------------------------
# 4 CALCULAR INFORMACION MUTUA
# ------------------------------------------------------------

print("\nCalculando Informacion Mutua...")

mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)

mi_scores = pd.Series(mi, index=X.columns)

mi_scores = mi_scores.sort_values(ascending=False)

print("\nRanking de caracteristicas:")
print(mi_scores)

# guardar ranking
mi_scores.to_csv(BASE_DIR / "ranking_informacion_mutua.csv")

# ------------------------------------------------------------
# 5 GRAFICA DE IMPORTANCIA DE CARACTERISTICAS
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.barplot(
    x=mi_scores.values,
    y=mi_scores.index,
    color="#2a9d8f"
)

plt.title("Ranking de caracteristicas por Informacion Mutua")
plt.xlabel("Informacion Mutua")
plt.ylabel("Caracteristica")

plt.tight_layout()

plt.savefig(BASE_DIR / "ranking_informacion_mutua.png")
plt.close()

# ------------------------------------------------------------
# 6 HEATMAP DE CORRELACION
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

# Spearman es mas apropiado para variables ordinales codificadas como enteros.
corr = df.corr(method="spearman")

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Matriz de correlacion")

plt.tight_layout()

plt.savefig(BASE_DIR / "heatmap_correlacion.png")
plt.close()

# ------------------------------------------------------------
# 7 DISTRIBUCION DE VARIABLES CLAVE
# ------------------------------------------------------------

features_to_plot = ["anxiety_level", "self_esteem", TARGET_COL]

for feature in features_to_plot:

    plt.figure(figsize=(6,4))
    
    sns.countplot(
        x=df[feature],
        color="#457b9d"
    )
    
    plt.title(f"Distribucion de niveles de {feature}")
    plt.xlabel("Nivel")
    plt.ylabel("Frecuencia")
    
    plt.tight_layout()
    
    plt.savefig(BASE_DIR / f"distribucion_{feature}.png")
    plt.close()

# ------------------------------------------------------------
# 8 CONCLUSION AUTOMATICA
# ------------------------------------------------------------

print("\nCaracteristicas mas relevantes para clasificar depresion:\n")

print(mi_scores.head(5))

print("\nAnalisis terminado.")
print("Archivos generados:")
print("- esquema_variables.csv")
print("- dataset_normalizado.csv")
print("- ranking_informacion_mutua.csv")
print("- ranking_informacion_mutua.png")
print("- heatmap_correlacion.png")