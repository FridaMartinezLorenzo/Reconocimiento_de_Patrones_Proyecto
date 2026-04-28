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
from sklearn.feature_selection import chi2

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

schema_df.to_csv(BASE_DIR /"resultados_1"/ "esquema_variables.csv", index=False)

# ------------------------------------------------------------
# 2 REVISION BASICA DEL DATASET
# ------------------------------------------------------------

print("\nValores faltantes:")
print(df.isnull().sum())

print("\nEstadisticas descriptivas:")
print(df.describe())

# Guardar dataset normalizado sin alterar el significado ordinal
df.to_csv(BASE_DIR /"resultados_1"/ "dataset_normalizado.csv", index=False)

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
mi_scores.to_csv(BASE_DIR /"resultados_1"/ "ranking_informacion_mutua.csv")

# ------------------------------------------------------------
# 4.1 CALCULAR CHI-CUADRADO
# ------------------------------------------------------------

print("\nCalculando Chi-cuadrado...")

chi2_scores, chi2_pvalues = chi2(X, y)

chi2_series = pd.Series(chi2_scores, index=X.columns).sort_values(ascending=False)
chi2_pvalues_series = pd.Series(chi2_pvalues, index=X.columns)

print("\nRanking de caracteristicas (Chi-cuadrado):")
print(chi2_series)

chi2_df = pd.DataFrame(
    {
        "feature": chi2_series.index,
        "chi2_score": chi2_series.values,
        "p_value": chi2_pvalues_series.loc[chi2_series.index].values,
    }
)

chi2_df.to_csv(BASE_DIR /"resultados_1"/ "ranking_chi2.csv", index=False)

print("\nTop 10 caracteristicas por Chi-cuadrado (score + p-value):")
print(chi2_df.head(10).to_string(index=False))

# Grafica exclusiva de Chi-cuadrado (Top 10)
chi2_top10 = chi2_series.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=chi2_top10.values,
    y=chi2_top10.index,
    color="#f4a261",
)

plt.title("Top 10 caracteristicas por Chi-cuadrado")
plt.xlabel("Chi-cuadrado")
plt.ylabel("Caracteristica")
plt.tight_layout()
plt.savefig(BASE_DIR /"resultados_1"/ "ranking_chi2_top10.png")
plt.close()

# Comparacion de metodos para facilitar interpretacion
comparison_df = pd.DataFrame(
    {
        "feature": X.columns,
        "mutual_info": mi_scores.reindex(X.columns).values,
        "chi2_score": chi2_series.reindex(X.columns).values,
        "chi2_p_value": chi2_pvalues_series.reindex(X.columns).values,
    }
)

comparison_df["rank_mutual_info"] = comparison_df["mutual_info"].rank(
    ascending=False,
    method="min",
)
comparison_df["rank_chi2"] = comparison_df["chi2_score"].rank(
    ascending=False,
    method="min",
)

comparison_df = comparison_df.sort_values("rank_mutual_info")
comparison_df.to_csv(BASE_DIR /"resultados_1"/ "comparacion_metodos_features.csv", index=False)

# ------------------------------------------------------------
# 4.2 COMPARACION GRAFICA ENTRE METODOS
# ------------------------------------------------------------

print("\nGenerando comparacion grafica entre metodos...")

def minmax_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


comparison_plot_df = comparison_df.copy()
comparison_plot_df["mutual_info_norm"] = minmax_normalize(comparison_plot_df["mutual_info"])
comparison_plot_df["chi2_norm"] = minmax_normalize(comparison_plot_df["chi2_score"])
comparison_plot_df["rank_promedio"] = (
    comparison_plot_df["rank_mutual_info"] + comparison_plot_df["rank_chi2"]
) / 2

top_features_plot = comparison_plot_df.nsmallest(10, "rank_promedio").copy()
top_features_plot = top_features_plot.sort_values("rank_promedio", ascending=True)

plot_long = top_features_plot.melt(
    id_vars="feature",
    value_vars=["mutual_info_norm", "chi2_norm"],
    var_name="metodo",
    value_name="score_normalizado",
)

plot_long["metodo"] = plot_long["metodo"].replace(
    {
        "mutual_info_norm": "Informacion Mutua",
        "chi2_norm": "Chi-cuadrado",
    }
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=plot_long,
    y="feature",
    x="score_normalizado",
    hue="metodo",
)

plt.title("Comparacion de metodos (Top 10 por ranking promedio)")
plt.xlabel("Score normalizado")
plt.ylabel("Caracteristica")
plt.legend(title="Metodo")
plt.tight_layout()
plt.savefig(BASE_DIR /"resultados_1"/ "comparacion_metodos_top10.png")
plt.close()

# ------------------------------------------------------------
# 4.3 GRAFICAS ADICIONALES DE INTERPRETACION
# ------------------------------------------------------------

print("\nGenerando graficas adicionales de comparacion...")

# 1) Dispersion de rankings MI vs Chi2
rank_scatter_df = comparison_df.copy()
rank_scatter_df["rank_diff_abs"] = (
    rank_scatter_df["rank_mutual_info"] - rank_scatter_df["rank_chi2"]
).abs()

plt.figure(figsize=(8, 8))

sns.scatterplot(
    data=rank_scatter_df,
    x="rank_mutual_info",
    y="rank_chi2",
    s=80,
    color="#1d3557",
)

min_rank = 1
max_rank = int(max(rank_scatter_df["rank_mutual_info"].max(), rank_scatter_df["rank_chi2"].max()))
plt.plot([min_rank, max_rank], [min_rank, max_rank], linestyle="--", color="#e63946", linewidth=1.5)

# Etiquetar solo las variables con mayor desacuerdo para mantener legibilidad
top_disagreement = rank_scatter_df.nlargest(6, "rank_diff_abs")
for _, row in top_disagreement.iterrows():
    plt.text(
        row["rank_mutual_info"] + 0.15,
        row["rank_chi2"] + 0.15,
        row["feature"],
        fontsize=8,
    )

plt.title("Comparacion de rankings: Informacion Mutua vs Chi-cuadrado")
plt.xlabel("Rank (Informacion Mutua)")
plt.ylabel("Rank (Chi-cuadrado)")
plt.tight_layout()
plt.savefig(BASE_DIR /"resultados_1"/ "comparacion_ranks_scatter.png")
plt.close()

# 2) Variables con mayor diferencia de ranking entre metodos
rank_gap_df = rank_scatter_df.sort_values("rank_diff_abs", ascending=False).head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=rank_gap_df,
    y="feature",
    x="rank_diff_abs",
    color="#ffb703",
)

plt.title("Top 10 diferencias de ranking entre metodos")
plt.xlabel("Diferencia absoluta de rank")
plt.ylabel("Caracteristica")
plt.tight_layout()
plt.savefig(BASE_DIR /"resultados_1"/ "diferencia_ranks_top10.png")
plt.close()

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

plt.savefig(BASE_DIR /"resultados_1"/ "ranking_informacion_mutua.png")
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

plt.savefig(BASE_DIR /"resultados_1"/ "heatmap_correlacion.png")
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
    
    plt.savefig(BASE_DIR /"resultados_1"/ f"distribucion_{feature}.png")
    plt.close()

# ------------------------------------------------------------
# 8 CONCLUSION AUTOMATICA
# ------------------------------------------------------------

print("\nCaracteristicas mas relevantes para clasificar stress_level:\n")

print(mi_scores.head(5))

print("\nTop 5 por Chi-cuadrado:\n")
print(chi2_series.head(5))

print("\nAnalisis terminado.")
print("Archivos generados:")
print("- esquema_variables.csv")
print("- dataset_normalizado.csv")
print("- ranking_informacion_mutua.csv")
print("- ranking_chi2.csv")
print("- ranking_chi2_top10.png")
print("- comparacion_metodos_features.csv")
print("- comparacion_metodos_top10.png")
print("- comparacion_ranks_scatter.png")
print("- diferencia_ranks_top10.png")
print("- ranking_informacion_mutua.png")
print("- heatmap_correlacion.png")