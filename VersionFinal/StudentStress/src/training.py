"""Entrenamiento general de los modelos solicitados."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modelos.arbol import run_tree_models
from modelos.bayes import run_bayes_models
from src.evaluation import run_evaluation_summary


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def _validate_selected_features(features: list[str], X_train: pd.DataFrame, selection_name: str) -> None:
    """Verifica que las caracteristicas seleccionadas existan en X_train."""
    missing = [feature for feature in features if feature not in X_train.columns]
    if missing:
        raise KeyError(f"Caracteristicas faltantes en {selection_name}: {missing}")


def run_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    mi_features: list[str],
    chi2_features: list[str],
    output_dir: Path,
    bayes_type: str = "gaussian",
) -> pd.DataFrame:
    """Entrena y evalua Bayes y Arbol con todas, top MI y top Chi-cuadrado."""
    ensure_dir(output_dir)
    _validate_selected_features(mi_features, X_train, "Informacion Mutua")
    _validate_selected_features(chi2_features, X_train, "Chi-cuadrado")

    X_train_mi = X_train[mi_features]
    X_test_mi = X_test[mi_features]
    X_train_chi2 = X_train[chi2_features]
    X_test_chi2 = X_test[chi2_features]

    results_summary: list[dict[str, object]] = []

    results_summary.extend(
        run_bayes_models(
            X_train_all=X_train,
            X_test_all=X_test,
            X_train_mi=X_train_mi,
            X_test_mi=X_test_mi,
            X_train_chi2=X_train_chi2,
            X_test_chi2=X_test_chi2,
            y_train=y_train,
            y_test=y_test,
            base_models_dir=output_dir,
            model_type=bayes_type,
        )
    )

    results_summary.extend(
        run_tree_models(
            X_train_all=X_train,
            X_test_all=X_test,
            X_train_mi=X_train_mi,
            X_test_mi=X_test_mi,
            X_train_chi2=X_train_chi2,
            X_test_chi2=X_test_chi2,
            y_train=y_train,
            y_test=y_test,
            base_models_dir=output_dir,
        )
    )

    with open(output_dir / "resultados_modelos_raw.json", "w", encoding="utf-8") as file:
        json.dump(results_summary, file, indent=2, ensure_ascii=False)

    summary_df = run_evaluation_summary(results_summary, output_dir)
    return summary_df
