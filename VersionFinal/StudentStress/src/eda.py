"""Modulo de Analisis Exploratorio de Datos (EDA).

Genera tablas, graficas y un reporte TXT completo para documentar la
comprension inicial del dataset antes del preprocesamiento y modelado.
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
                "rango": (numeric.max() - numeric.min()) if numeric.notna().any() else None,
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


def _format_section(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title) + "\n"


def _detect_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta posibles outliers por IQR en variables numericas."""
    rows = []
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        if outliers > 0:
            rows.append(
                {
                    "variable": col,
                    "outliers": int(outliers),
                    "limite_inferior": float(lower),
                    "limite_superior": float(upper),
                }
            )
    return pd.DataFrame(rows)


def generate_txt_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera un reporte TXT del EDA, similar al formato solicitado."""
    ensure_dir(output_dir)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    describe_df = numeric_df.describe().round(2)
    range_df = pd.DataFrame(
        {
            "Variable": numeric_df.columns,
            "Minimo": numeric_df.min().values,
            "Maximo": numeric_df.max().values,
            "Rango": (numeric_df.max() - numeric_df.min()).values,
        }
    )
    unique_df = pd.DataFrame(
        {"Variable": df.columns, "Valores_unicos": [df[col].nunique(dropna=True) for col in df.columns]}
    )
    outlier_df = _detect_iqr_outliers(df)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("ANALISIS EXPLORATORIO DEL DATASET: STUDENT STRESS MONITORING")
    lines.append("=" * 80)

    lines.append(_format_section("1. FORMA DEL DATASET (filas, columnas):"))
    lines.append(f"   {df.shape[0]} filas, {df.shape[1]} columnas")

    lines.append(_format_section("2. NOMBRES DE LAS VARIABLES:"))
    for idx, col in enumerate(df.columns, start=1):
        lines.append(f"   {idx:2d}. {col}")

    lines.append(_format_section("3. PRIMERAS 10 FILAS:"))
    lines.append(df.head(10).to_string())

    lines.append(_format_section("4. ULTIMAS 10 FILAS:"))
    lines.append(df.tail(10).to_string())

    lines.append(_format_section("5. TIPOS DE DATOS:"))
    lines.append(df.dtypes.to_string())

    lines.append(_format_section("6. VALORES NULOS POR COLUMNA:"))
    missing = df.isnull().sum()
    if int(missing.sum()) == 0:
        lines.append("   No hay valores nulos en el dataset")
    else:
        lines.append(missing.to_string())

    lines.append(_format_section("7. ESTADISTICAS DESCRIPTIVAS (variables numericas):"))
    lines.append(describe_df.to_string())

    lines.append(_format_section("8. RANGO DE VALORES POR VARIABLE:"))
    lines.append(range_df.to_string(index=False))

    if "mental_health_history" in df.columns:
        lines.append(_format_section("9. DISTRIBUCION DE VARIABLE BINARIA (mental_health_history):"))
        lines.append(df["mental_health_history"].value_counts().sort_index().to_string())
        lines.append("   0 = no tiene historial, 1 = si tiene historial")

    if "stress_level" in df.columns:
        lines.append(_format_section("10. DISTRIBUCION DE LA VARIABLE stress_level:"))
        stress_counts = df["stress_level"].value_counts().sort_index()
        stress_percent = (stress_counts / len(df) * 100).round(1)
        stress_table = pd.DataFrame({"Frecuencia": stress_counts, "Porcentaje": stress_percent})
        lines.append(stress_table.to_string())

    lines.append(_format_section("11. NUMERO DE VALORES UNICOS POR VARIABLE:"))
    lines.append(unique_df.to_string(index=False))

    lines.append(_format_section("12. RESUMEN DE VARIABLES CLAVE (media y desviacion estandar):"))
    key_vars = [
        "anxiety_level",
        "self_esteem",
        "depression",
        "sleep_quality",
        "academic_performance",
        "study_load",
        "social_support",
        "stress_level",
    ]
    key_vars = [col for col in key_vars if col in numeric_df.columns]
    key_summary = pd.DataFrame(
        {
            "Variable": key_vars,
            "Media": [numeric_df[col].mean() for col in key_vars],
            "Desv_Estandar": [numeric_df[col].std() for col in key_vars],
        }
    ).round(2)
    lines.append(key_summary.to_string(index=False))

    lines.append(_format_section("13. PERCENTILES CLAVE PARA VARIABLES PRINCIPALES:"))
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    for col in key_vars:
        lines.append(f"\n   {col}:")
        pct = numeric_df[col].quantile(percentiles).round(2)
        for pct_idx, value in pct.items():
            lines.append(f"      Percentil {int(pct_idx * 100):2d}: {value:.2f}")

    if "stress_level" in numeric_df.columns:
        lines.append(_format_section("14. CORRELACION DE SPEARMAN CON stress_level:"))
        corr_stress = numeric_df.corr(method="spearman")["stress_level"].drop("stress_level")
        corr_stress = corr_stress.sort_values(key=lambda series: series.abs(), ascending=False).round(3)
        lines.append(corr_stress.to_string())

    lines.append(_format_section("15. DETECCION DE POSIBLES VALORES ATIPICOS (IQR):"))
    if outlier_df.empty:
        lines.append("   No se detectaron posibles valores atipicos por IQR.")
    else:
        lines.append(outlier_df.round(2).to_string(index=False))

    lines.append("\n" + "=" * 80)
    lines.append("ANALISIS COMPLETADO")
    lines.append("=" * 80)

    (output_dir / "reporte_eda.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Ejecuta el EDA completo y guarda los resultados."""
    ensure_dir(output_dir)

    save_variable_schema(df, output_dir)
    save_basic_tables(df, output_dir)
    generate_txt_report(df, output_dir)

    plot_count_distribution(
        df,
        "anxiety_level",
        output_dir,
        title="Distribucion original de anxiety_level",
        filename="distribucion_anxiety_original.png",
    )
    plot_count_distribution(df, "stress_level", output_dir)
    plot_count_distribution(df, "depression", output_dir)
    plot_count_distribution(df, "self_esteem", output_dir)
    plot_correlation_heatmap(df, output_dir)
