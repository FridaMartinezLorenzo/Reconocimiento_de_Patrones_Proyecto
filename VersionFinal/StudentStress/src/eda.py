"""Modulo de Analisis Exploratorio de Datos (EDA).

Genera tablas y graficas basicas para conocer el dataset antes del
preprocesamiento y el entrenamiento de modelos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def save_variable_schema(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Guarda una tabla con tipo, cardinalidad, minimo y maximo por columna."""
    rows = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "columna": col,
                "tipo_pandas": str(df[col].dtype),
                "valores_unicos": int(df[col].nunique(dropna=True)),
                "valores_faltantes": int(df[col].isna().sum()),
                "minimo": numeric.min() if numeric.notna().any() else None,
                "maximo": numeric.max() if numeric.notna().any() else None,
            }
        )

    schema = pd.DataFrame(rows)
    schema.to_csv(output_dir / "esquema_variables.csv", index=False, encoding="utf-8")
    return schema


def save_basic_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Guarda valores faltantes y estadisticas descriptivas."""
    missing = df.isnull().sum().reset_index()
    missing.columns = ["columna", "valores_faltantes"]
    missing.to_csv(output_dir / "valores_faltantes.csv", index=False, encoding="utf-8")

    df.describe(include="all").transpose().to_csv(
        output_dir / "resumen_estadistico.csv", encoding="utf-8"
    )


def plot_count_distribution(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """Genera una grafica de barras para una columna discreta."""
    if column not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    order = sorted(df[column].dropna().unique())
    sns.countplot(data=df, x=column, order=order, color="#457b9d")
    plt.title(title or f"Distribucion de {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_dir / (filename or f"distribucion_{column}.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera un mapa de calor de correlacion de Spearman."""
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="spearman")
    corr.to_csv(output_dir / "correlacion_spearman.csv", encoding="utf-8")

    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Matriz de correlacion de Spearman")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_correlacion_spearman.png", dpi=150)
    plt.close()


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Ejecuta el EDA completo y guarda los resultados."""
    ensure_dir(output_dir)

    save_variable_schema(df, output_dir)
    save_basic_tables(df, output_dir)

    plot_count_distribution(
        df,
        "anxiety_level",
        output_dir,
        title="Distribucion original de anxiety_level",
        filename="distribucion_anxiety_original.png",
    )
    plot_count_distribution(df, "stress_level", output_dir)
    plot_count_distribution(df, "depression", output_dir)
    plot_correlation_heatmap(df, output_dir)
