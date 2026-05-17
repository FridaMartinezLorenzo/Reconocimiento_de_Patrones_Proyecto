"""Modelos de arbol de decision para la clasificacion de ansiedad."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .common import evaluate_and_save_model


def _get_tree_model(random_state: int = 42) -> DecisionTreeClassifier:
    """Devuelve un arbol de decision simple y controlado.

    Se usa max_depth y min_samples_leaf para reducir sobreajuste sin agregar
    busquedas de hiperparametros complejas.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=8,
        random_state=random_state,
    )


def run_tree_models(
    X_train_all: pd.DataFrame,
    X_test_all: pd.DataFrame,
    X_train_mi: pd.DataFrame,
    X_test_mi: pd.DataFrame,
    X_train_chi2: pd.DataFrame,
    X_test_chi2: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_models_dir: Path,
    random_state: int = 42,
) -> list[dict[str, object]]:
    """Ejecuta arbol con todas, top MI y top Chi-cuadrado."""
    results = []
    tree_dir = base_models_dir / "arbol"

    experiments = [
        ("Arbol (todas)", X_train_all, X_test_all, tree_dir / "todas_las_caracteristicas"),
        ("Arbol (top MI)", X_train_mi, X_test_mi, tree_dir / "top_mi"),
        ("Arbol (top Chi2)", X_train_chi2, X_test_chi2, tree_dir / "top_chi2"),
    ]

    for model_name, X_train_exp, X_test_exp, output_dir in experiments:
        results.append(
            evaluate_and_save_model(
                model_name=model_name,
                model=_get_tree_model(random_state=random_state),
                X_train_m=X_train_exp,
                X_test_m=X_test_exp,
                y_train_m=y_train,
                y_test_m=y_test,
                output_dir=output_dir,
            )
        )

    return results
