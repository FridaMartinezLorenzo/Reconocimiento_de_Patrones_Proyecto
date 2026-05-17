"""Modelos de Bayes ingenuo para la clasificacion de ansiedad."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB

from .common import evaluate_and_save_model


def _get_bayes_model(model_type: str):
    """Devuelve el tipo de Bayes solicitado."""
    model_type = model_type.lower()
    if model_type == "gaussian":
        return GaussianNB()
    if model_type == "multinomial":
        return MultinomialNB()
    if model_type == "categorical":
        return CategoricalNB()
    raise ValueError("model_type debe ser: gaussian, multinomial o categorical")


def run_bayes_models(
    X_train_all: pd.DataFrame,
    X_test_all: pd.DataFrame,
    X_train_mi: pd.DataFrame,
    X_test_mi: pd.DataFrame,
    X_train_chi2: pd.DataFrame,
    X_test_chi2: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_models_dir: Path,
    model_type: str = "gaussian",
) -> list[dict[str, object]]:
    """Ejecuta Bayes con todas, top MI y top Chi-cuadrado."""
    results = []
    bayes_dir = base_models_dir / "bayes"

    experiments = [
        ("Bayes (todas)", X_train_all, X_test_all, bayes_dir / "todas_las_caracteristicas"),
        ("Bayes (top MI)", X_train_mi, X_test_mi, bayes_dir / "top_mi"),
        ("Bayes (top Chi2)", X_train_chi2, X_test_chi2, bayes_dir / "top_chi2"),
    ]

    for model_name, X_train_exp, X_test_exp, output_dir in experiments:
        results.append(
            evaluate_and_save_model(
                model_name=model_name,
                model=_get_bayes_model(model_type),
                X_train_m=X_train_exp,
                X_test_m=X_test_exp,
                y_train_m=y_train,
                y_test_m=y_test,
                output_dir=output_dir,
            )
        )

    return results
