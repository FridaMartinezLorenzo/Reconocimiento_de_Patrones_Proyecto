"""Modulo de seleccion de caracteristicas.

Calcula ranking por Informacion Mutua y Chi-cuadrado usando solamente los datos
de entrenamiento para evitar fuga de informacion.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import chi2, mutual_info_classif


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def select_by_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Calcula ranking de caracteristicas por Informacion Mutua."""
    scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=True,
        random_state=random_state,
    )
    ranking = (
        pd.DataFrame({"feature": X_train.columns, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )
    selected_features = ranking.head(top_k)["feature"].tolist()
    return ranking, selected_features


def select_by_chi2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Calcula ranking de caracteristicas por Chi-cuadrado."""
    X_non_negative = X_train.copy()

    # Chi-cuadrado requiere valores no negativos. El dataset original ya cumple,
    # pero esta revision evita errores si se agrega otra fuente de datos.
    for col in X_non_negative.columns:
        min_value = X_non_negative[col].min()
        if min_value < 0:
            X_non_negative[col] = X_non_negative[col] - min_value

    scores, p_values = chi2(X_non_negative, y_train)
    ranking = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "chi2_score": scores,
                "p_value": p_values,
            }
        )
        .sort_values("chi2_score", ascending=False)
        .reset_index(drop=True)
    )
    selected_features = ranking.head(top_k)["feature"].tolist()
    return ranking, selected_features


def plot_ranking(
    ranking: pd.DataFrame,
    score_col: str,
    output_path: Path,
    title: str,
    top_k: int = 10,
) -> None:
    """Grafica el top de caracteristicas de un ranking."""
    top = ranking.head(top_k)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x=score_col, y="feature", color="#2a9d8f")
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Caracteristica")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_selected_features(features: list[str], output_path: Path) -> None:
    """Guarda una lista de caracteristicas en CSV y JSON."""
    pd.DataFrame({"feature": features}).to_csv(
        output_path.with_suffix(".csv"), index=False, encoding="utf-8"
    )
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(features, file, indent=2, ensure_ascii=False)


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    top_k: int = 10,
) -> dict[str, object]:
    """Ejecuta seleccion por Informacion Mutua y Chi-cuadrado."""
    ensure_dir(output_dir)

    mi_ranking, mi_features = select_by_mutual_info(X_train, y_train, top_k=top_k)
    chi2_ranking, chi2_features = select_by_chi2(X_train, y_train, top_k=top_k)

    mi_ranking.to_csv(output_dir / "ranking_informacion_mutua.csv", index=False, encoding="utf-8")
    chi2_ranking.to_csv(output_dir / "ranking_chi2.csv", index=False, encoding="utf-8")

    save_selected_features(mi_features, output_dir / "features_mi_top10")
    save_selected_features(chi2_features, output_dir / "features_chi2_top10")

    plot_ranking(
        mi_ranking,
        "mi_score",
        output_dir / "ranking_informacion_mutua_top10.png",
        "Top 10 caracteristicas por Informacion Mutua",
        top_k=top_k,
    )
    plot_ranking(
        chi2_ranking,
        "chi2_score",
        output_dir / "ranking_chi2_top10.png",
        "Top 10 caracteristicas por Chi-cuadrado",
        top_k=top_k,
    )

    return {
        "mi_ranking": mi_ranking,
        "mi_features": mi_features,
        "chi2_ranking": chi2_ranking,
        "chi2_features": chi2_features,
    }
