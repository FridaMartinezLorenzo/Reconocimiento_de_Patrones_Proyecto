"""Entrenamiento general de los modelos solicitados.

Entrena Bayes ingenuo y Arboles de Decision usando todas las caracteristicas y
subconjuntos seleccionados por Informacion Mutua, Chi-cuadrado e importancia de
Arbol para varios valores de n.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from modelos.common import evaluate_and_save_model
from src.evaluation import run_evaluation_summary


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def _validate_selected_features(features: list[str], X_train: pd.DataFrame, selection_name: str) -> None:
    """Verifica que las caracteristicas seleccionadas existan en X_train."""
    missing = [feature for feature in features if feature not in X_train.columns]
    if missing:
        raise KeyError(f"Caracteristicas faltantes en {selection_name}: {missing}")


def _get_bayes_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "gaussian":
        return GaussianNB()
    if model_type == "multinomial":
        return MultinomialNB()
    if model_type == "categorical":
        return CategoricalNB()
    raise ValueError("model_type debe ser: gaussian, multinomial o categorical")


def _get_tree_model(random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=8,
        random_state=random_state,
    )


def _build_feature_sets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: dict[str, list[str]],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Construye los subconjuntos de caracteristicas para un valor de n."""
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame, int]] = {
        "todas_las_caracteristicas": (X_train, X_test, int(X_train.shape[1])),
    }

    selectors = {
        "top_mi": selected_features["mi_features"],
        "top_chi2": selected_features["chi2_features"],
        "top_arboles": selected_features["arboles_features"],
    }
    for selector_name, features in selectors.items():
        _validate_selected_features(features, X_train, selector_name)
        feature_sets[selector_name] = (X_train[features], X_test[features], len(features))

    return feature_sets


def _run_models_for_k(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_features: dict[str, list[str]],
    output_dir: Path,
    k: int,
    bayes_type: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Entrena todos los modelos para una cantidad n de caracteristicas."""
    ensure_dir(output_dir)
    results_summary: list[dict[str, object]] = []
    feature_sets = _build_feature_sets(X_train, X_test, selected_features)

    for subset_name, (X_train_exp, X_test_exp, n_features_used) in feature_sets.items():
        bayes_name = f"Bayes ({subset_name}, n={k})" if subset_name != "todas_las_caracteristicas" else "Bayes (todas)"
        bayes_result = evaluate_and_save_model(
            model_name=bayes_name,
            model=_get_bayes_model(bayes_type),
            X_train_m=X_train_exp,
            X_test_m=X_test_exp,
            y_train_m=y_train,
            y_test_m=y_test,
            output_dir=output_dir / "bayes" / subset_name,
        )
        bayes_result.update({"selector": subset_name, "n_selection": k, "n_features_used": n_features_used})
        results_summary.append(bayes_result)

        tree_name = f"Arbol ({subset_name}, n={k})" if subset_name != "todas_las_caracteristicas" else "Arbol (todas)"
        tree_result = evaluate_and_save_model(
            model_name=tree_name,
            model=_get_tree_model(random_state=random_state),
            X_train_m=X_train_exp,
            X_test_m=X_test_exp,
            y_train_m=y_train,
            y_test_m=y_test,
            output_dir=output_dir / "arbol" / subset_name,
        )
        tree_result.update({"selector": subset_name, "n_selection": k, "n_features_used": n_features_used})
        results_summary.append(tree_result)

    with open(output_dir / "resultados_modelos_raw.json", "w", encoding="utf-8") as file:
        json.dump(results_summary, file, indent=2, ensure_ascii=False)

    return run_evaluation_summary(results_summary, output_dir)


def run_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_by_k: dict[int, dict[str, list[str]]],
    output_dir: Path,
    bayes_type: str = "gaussian",
    random_state: int = 42,
) -> pd.DataFrame:
    """Entrena y evalua modelos para cada valor de n seleccionado."""
    ensure_dir(output_dir)
    all_summaries: list[pd.DataFrame] = []

    for k, selected_features in selected_by_k.items():
        k_dir = output_dir / f"n_{k}"
        summary_k = _run_models_for_k(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            selected_features=selected_features,
            output_dir=k_dir,
            k=k,
            bayes_type=bayes_type,
            random_state=random_state,
        )
        summary_k["n_selection"] = k
        all_summaries.append(summary_k)

    global_summary = pd.concat(all_summaries, ignore_index=True).sort_values(
        "accuracy_tasa_reconocimiento", ascending=False
    )
    global_summary.to_csv(output_dir / "resumen_modelos_global.csv", index=False, encoding="utf-8")
    run_evaluation_summary(global_summary.to_dict(orient="records"), output_dir)
    return global_summary
