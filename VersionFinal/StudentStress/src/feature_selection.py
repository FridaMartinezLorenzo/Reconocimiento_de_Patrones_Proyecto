"""Modulo de seleccion de caracteristicas.

Calcula rankings por Informacion Mutua, Chi-cuadrado e importancia de Arbol de
Decision usando solamente los datos de entrenamiento para evitar fuga de
informacion.
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
from sklearn.tree import DecisionTreeClassifier


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def select_by_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula ranking de caracteristicas por Informacion Mutua."""
    scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=True,
        random_state=random_state,
    )
    return (
        pd.DataFrame({"feature": X_train.columns, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )


def select_by_chi2(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Calcula ranking de caracteristicas por Chi-cuadrado."""
    X_non_negative = X_train.copy()
    for col in X_non_negative.columns:
        min_value = X_non_negative[col].min()
        if min_value < 0:
            X_non_negative[col] = X_non_negative[col] - min_value

    scores, p_values = chi2(X_non_negative, y_train)
    return (
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


def select_by_tree_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula ranking por importancia de un Arbol de Decision."""
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=8,
        random_state=random_state,
    )
    tree.fit(X_train, y_train)
    return (
        pd.DataFrame({"feature": X_train.columns, "tree_importance": tree.feature_importances_})
        .sort_values("tree_importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_ranking(
    ranking: pd.DataFrame,
    score_col: str,
    output_path: Path,
    title: str,
    top_k: int = 10,
    color: str = "#2a9d8f",
) -> None:
    """Grafica el top de caracteristicas de un ranking."""
    top = ranking.head(top_k)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x=score_col, y="feature", color=color)
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Caracteristica")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_selected_features(features: list[str], output_path: Path) -> None:
    """Guarda una lista de caracteristicas en CSV y JSON."""
    ensure_dir(output_path.parent)
    pd.DataFrame({"feature": features}).to_csv(
        output_path.with_suffix(".csv"), index=False, encoding="utf-8"
    )
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(features, file, indent=2, ensure_ascii=False)


def _normalize_k_values(k_values: list[int], n_features: int) -> list[int]:
    """Limpia valores de k para que no excedan el numero de caracteristicas."""
    cleaned = sorted({int(k) for k in k_values if int(k) > 0})
    return [min(k, n_features) for k in cleaned]


def _features_from_ranking(ranking: pd.DataFrame, k: int) -> list[str]:
    return ranking.head(k)["feature"].tolist()


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    k_values: list[int] | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    """Ejecuta seleccion por MI, Chi-cuadrado y Arbol para varios valores de k."""
    ensure_dir(output_dir)
    if k_values is None:
        k_values = [5, 10, 15]
    k_values = _normalize_k_values(k_values, n_features=X_train.shape[1])

    mi_ranking = select_by_mutual_info(X_train, y_train, random_state=random_state)
    chi2_ranking = select_by_chi2(X_train, y_train)
    tree_ranking = select_by_tree_importance(X_train, y_train, random_state=random_state)

    mi_ranking.to_csv(output_dir / "ranking_informacion_mutua.csv", index=False, encoding="utf-8")
    chi2_ranking.to_csv(output_dir / "ranking_chi2.csv", index=False, encoding="utf-8")
    tree_ranking.to_csv(output_dir / "ranking_arboles.csv", index=False, encoding="utf-8")

    selected_by_k: dict[int, dict[str, list[str]]] = {}
    for k in k_values:
        k_dir = output_dir / f"n_{k}"
        ensure_dir(k_dir)

        mi_features = _features_from_ranking(mi_ranking, k)
        chi2_features = _features_from_ranking(chi2_ranking, k)
        tree_features = _features_from_ranking(tree_ranking, k)

        selected_by_k[k] = {
            "mi_features": mi_features,
            "chi2_features": chi2_features,
            "arboles_features": tree_features,
        }

        save_selected_features(mi_features, k_dir / f"features_mi_top{k}")
        save_selected_features(chi2_features, k_dir / f"features_chi2_top{k}")
        save_selected_features(tree_features, k_dir / f"features_arboles_top{k}")

        plot_ranking(
            mi_ranking,
            "mi_score",
            k_dir / f"ranking_informacion_mutua_top{k}.png",
            f"Top {k} caracteristicas por Informacion Mutua",
            top_k=k,
            color="#2a9d8f",
        )
        plot_ranking(
            chi2_ranking,
            "chi2_score",
            k_dir / f"ranking_chi2_top{k}.png",
            f"Top {k} caracteristicas por Chi-cuadrado",
            top_k=k,
            color="#f4a261",
        )
        plot_ranking(
            tree_ranking,
            "tree_importance",
            k_dir / f"ranking_arboles_top{k}.png",
            f"Top {k} caracteristicas por importancia de Arbol",
            top_k=k,
            color="#457b9d",
        )

    # Archivos top10 en la raiz para cumplir el formato solicitado y mantener compatibilidad.
    reference_k = 10 if 10 in selected_by_k else k_values[0]
    ref_features = selected_by_k[reference_k]
    save_selected_features(ref_features["mi_features"], output_dir / "features_mi_top10")
    save_selected_features(ref_features["chi2_features"], output_dir / "features_chi2_top10")
    save_selected_features(ref_features["arboles_features"], output_dir / "features_arboles_top10")
    plot_ranking(
        mi_ranking,
        "mi_score",
        output_dir / "ranking_informacion_mutua_top10.png",
        "Top 10 caracteristicas por Informacion Mutua",
        top_k=reference_k,
        color="#2a9d8f",
    )
    plot_ranking(
        chi2_ranking,
        "chi2_score",
        output_dir / "ranking_chi2_top10.png",
        "Top 10 caracteristicas por Chi-cuadrado",
        top_k=reference_k,
        color="#f4a261",
    )
    plot_ranking(
        tree_ranking,
        "tree_importance",
        output_dir / "ranking_arboles_top10.png",
        "Top 10 caracteristicas por importancia de Arbol",
        top_k=reference_k,
        color="#457b9d",
    )

    with open(output_dir / "features_por_n.json", "w", encoding="utf-8") as file:
        json.dump({str(k): value for k, value in selected_by_k.items()}, file, indent=2, ensure_ascii=False)

    return {
        "mi_ranking": mi_ranking,
        "chi2_ranking": chi2_ranking,
        "arboles_ranking": tree_ranking,
        "k_values": k_values,
        "selected_by_k": selected_by_k,
        "mi_features": ref_features["mi_features"],
        "chi2_features": ref_features["chi2_features"],
        "arboles_features": ref_features["arboles_features"],
    }
