"""Graficas y resumen global de evaluacion multiclase."""

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


def save_models_summary(results_summary: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    """Guarda la tabla comparativa general de modelos."""
    ensure_dir(output_dir)
    summary_df = pd.DataFrame(results_summary).sort_values(
        "accuracy_tasa_reconocimiento", ascending=False
    )
    summary_df.to_csv(output_dir / "resumen_modelos.csv", index=False, encoding="utf-8")
    return summary_df


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Guarda una grafica de barras para comparar una metrica entre modelos."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x=metric_col, y="model", color="#457b9d")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Modelo")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_general_comparisons(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Genera graficas generales para la fase de evaluacion."""
    ensure_dir(output_dir)
    plot_metric_comparison(
        summary_df,
        "accuracy_tasa_reconocimiento",
        output_dir / "comparacion_accuracy_tasa_reconocimiento.png",
        "Comparacion de exactitud / tasa de reconocimiento",
        "Exactitud",
    )
    plot_metric_comparison(
        summary_df,
        "error_rate_tasa_error",
        output_dir / "comparacion_tasa_error.png",
        "Comparacion de tasa de error",
        "Tasa de error",
    )
    plot_metric_comparison(
        summary_df,
        "f1_macro",
        output_dir / "comparacion_f1_macro.png",
        "Comparacion de F1 macro",
        "F1 macro",
    )
    plot_metric_comparison(
        summary_df,
        "f1_weighted",
        output_dir / "comparacion_f1_weighted.png",
        "Comparacion de F1 ponderado",
        "F1 ponderado",
    )


def plot_metrics_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Grafica tipo heatmap con metricas principales por modelo."""
    ensure_dir(output_dir)
    metric_cols = [
        "accuracy_tasa_reconocimiento",
        "error_rate_tasa_error",
        "f1_macro",
        "f1_weighted",
    ]
    heatmap_df = summary_df.set_index("model")[metric_cols]
    plt.figure(figsize=(9, max(4, len(summary_df) * 0.55)))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Resumen visual de metricas por modelo")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_metricas_modelos.png", dpi=150)
    plt.close()


def run_evaluation_summary(results_summary: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    """Ejecuta el resumen final de evaluacion."""
    summary_df = save_models_summary(results_summary, output_dir)
    plot_general_comparisons(summary_df, output_dir)
    plot_metrics_heatmap(summary_df, output_dir)
    return summary_df
