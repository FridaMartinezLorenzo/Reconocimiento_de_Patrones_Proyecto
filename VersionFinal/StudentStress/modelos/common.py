"""Funciones comunes para entrenamiento y evaluacion de modelos.

Este modulo concentra la evaluacion multiclase clasica solicitada en la actividad:
exactitud, tasa de error, precision, recall y F-score por clase.

Ademas, genera visualizaciones complementarias:
- Matriz de confusion.
- Metricas por clase.
- Regiones de decision 2D con aciertos y errores de clasificacion.
- Representacion visual del arbol (nodos y hojas).
- Curva tipo campana de Gauss para Bayes ingenuo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree


sns.set_style("whitegrid")


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(cm, labels: list[int], model_name: str, output_path: Path) -> None:
    """Guarda la matriz de confusion de un modelo."""
    fig_width = max(6, len(labels) * 1.2)
    fig_height = max(5, len(labels) * 1.0)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Matriz de confusion - {model_name}")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_class_metrics(class_metrics: pd.DataFrame, model_name: str, output_path: Path) -> None:
    """Guarda grafica con precision, recall y F-score por clase."""
    plot_df = class_metrics.melt(
        id_vars="clase",
        value_vars=["precision", "recall", "f1_score"],
        var_name="metrica",
        value_name="valor",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="clase", y="valor", hue="metrica")
    plt.title(f"Metricas por clase - {model_name}")
    plt.xlabel("Clase de ansiedad")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_decision_regions(
    model,
    X_train_m: pd.DataFrame,
    X_test_m: pd.DataFrame,
    y_train_m: pd.Series,
    y_test_m: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Genera una representacion 2D de clases, regiones y errores de clasificacion.

    Para mantener el codigo simple y facil de leer, la grafica se construye con las
    dos primeras caracteristicas del subconjunto utilizado por el modelo. Se ajusta
    un clon del mismo clasificador en 2D solo con fines visuales.
    """
    if X_train_m.shape[1] < 2 or X_test_m.shape[1] < 2:
        return

    feature_x, feature_y = X_train_m.columns[:2]
    X_train_2d = X_train_m[[feature_x, feature_y]]
    X_test_2d = X_test_m[[feature_x, feature_y]]

    vis_model = clone(model)
    vis_model.fit(X_train_2d, y_train_m)

    x_min = min(X_train_2d[feature_x].min(), X_test_2d[feature_x].min()) - 1
    x_max = max(X_train_2d[feature_x].max(), X_test_2d[feature_x].max()) + 1
    y_min = min(X_train_2d[feature_y].min(), X_test_2d[feature_y].min()) - 1
    y_max = max(X_train_2d[feature_y].max(), X_test_2d[feature_y].max()) + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )

    grid = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})
    zz = vis_model.predict(grid).reshape(xx.shape)

    labels = sorted(pd.Series(y_test_m).unique().tolist())
    cmap_bg = plt.cm.RdYlGn
    cmap_pts = plt.cm.Set1

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.32, cmap=cmap_bg)
    plt.contour(xx, yy, zz, colors="gray", linewidths=0.8, alpha=0.6)

    y_pred_2d = vis_model.predict(X_test_2d)
    colors = [cmap_pts(i / max(1, len(labels) - 1)) for i in range(len(labels))]

    for idx, label in enumerate(labels):
        correct_mask = (y_test_m.to_numpy() == label) & (y_pred_2d == label)
        error_mask = (y_test_m.to_numpy() == label) & (y_pred_2d != label)

        if correct_mask.any():
            plt.scatter(
                X_test_2d.loc[correct_mask, feature_x],
                X_test_2d.loc[correct_mask, feature_y],
                c=[colors[idx]],
                marker="o",
                edgecolor="black",
                s=55,
                label=f"Clase {label} correcta",
            )
        if error_mask.any():
            plt.scatter(
                X_test_2d.loc[error_mask, feature_x],
                X_test_2d.loc[error_mask, feature_y],
                c=[colors[idx]],
                marker="X",
                edgecolor="black",
                s=90,
                label=f"Clase {label} error",
            )

    plt.title(f"Regiones de decision 2D - {model_name}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_tree_structure(
    model,
    feature_names: list[str],
    class_names: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Guarda la representacion visual del arbol con nodos y hojas."""
    if not isinstance(model, DecisionTreeClassifier):
        return

    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
    )
    plt.title(f"Representacion del arbol - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def _normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Calcula la densidad normal sin depender de scipy."""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _plot_gaussian_bell_for_bayes(
    model,
    X_train_m: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    """Genera una grafica estilo campana de Gauss para Bayes ingenuo.

    La grafica usa la primera caracteristica del subconjunto de entrenamiento,
    se estandariza a z-score y se compara contra la distribucion normal estandar.
    """
    if X_train_m.shape[1] < 1:
        return

    # La representacion es especialmente adecuada para GaussianNB, pero puede
    # mostrarse tambien como referencia visual del supuesto gaussiano.
    if not isinstance(model, GaussianNB):
        return

    feature_name = X_train_m.columns[0]
    values = X_train_m[feature_name].astype(float).to_numpy()

    sigma = float(np.std(values))
    if sigma == 0:
        sigma = 1.0
    mu = float(np.mean(values))
    z_values = (values - mu) / sigma

    x = np.linspace(-3.5, 3.5, 600)
    y = _normal_pdf(x, mu=0.0, sigma=1.0)

    plt.figure(figsize=(10, 6))
    plt.hist(z_values, bins=8, density=True, alpha=0.35, edgecolor="white")
    plt.plot(x, y, linewidth=2)

    intervals = [(-1, 1, "34%", 0.24), (-2, -1, "13.5%", 0.10), (1, 2, "13.5%", 0.10),
                 (-3, -2, "2.35%", 0.03), (2, 3, "2.35%", 0.03), (-3.3, -3.0, "0.15%", 0.008), (3.0, 3.3, "0.15%", 0.008)]

    for start, end, label, y_text in intervals:
        xs = np.linspace(start, end, 100)
        plt.fill_between(xs, _normal_pdf(xs), alpha=0.15)
        plt.text((start + end) / 2, y_text, label, ha="center", fontsize=9)

    for tick in range(-3, 4):
        plt.axvline(tick, linestyle="--", linewidth=0.8, alpha=0.5)

    plt.text(0, -0.015, "μ", ha="center", fontsize=12)
    plt.text(-1, -0.015, "-1σ", ha="center", fontsize=10)
    plt.text(1, -0.015, "+1σ", ha="center", fontsize=10)

    plt.title(f"Campana de Gauss - {model_name}\nCaracteristica: {feature_name}")
    plt.xlabel("Valor estandarizado (z-score)")
    plt.ylabel("Densidad")
    plt.xlim(-3.5, 3.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_and_save_model(
    model_name: str,
    model,
    X_train_m: pd.DataFrame,
    X_test_m: pd.DataFrame,
    y_train_m: pd.Series,
    y_test_m: pd.Series,
    output_dir: Path,
) -> dict[str, object]:
    """Entrena, evalua y guarda resultados de un modelo multiclase.

    Devuelve un resumen compacto para la tabla final de comparacion.
    """
    ensure_dir(output_dir)

    model.fit(X_train_m, y_train_m)
    y_pred = model.predict(X_test_m)

    labels = sorted(pd.Series(y_test_m).unique().tolist())
    accuracy = accuracy_score(y_test_m, y_pred)
    error_rate = 1.0 - accuracy
    f1_macro = f1_score(y_test_m, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test_m, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_m,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    class_metrics = pd.DataFrame(
        {
            "clase": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )

    cm = confusion_matrix(y_test_m, y_pred, labels=labels)
    report_dict = classification_report(
        y_test_m,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test_m,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "model": model_name,
        "n_features": int(X_train_m.shape[1]),
        "features": list(X_train_m.columns),
        "train_size": 0.70,
        "test_size": 0.30,
        "accuracy_tasa_reconocimiento": float(accuracy),
        "error_rate_tasa_error": float(error_rate),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classes": [int(label) for label in labels],
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / "classification_report.csv", encoding="utf-8"
    )
    class_metrics.to_csv(output_dir / "metricas_por_clase.csv", index=False, encoding="utf-8")

    pd.DataFrame({"y_true": y_test_m, "y_pred": y_pred}).to_csv(
        output_dir / "predicciones_test.csv", index=False, encoding="utf-8"
    )

    _plot_confusion_matrix(cm, labels, model_name, output_dir / "matriz_confusion.png")
    _plot_class_metrics(class_metrics, model_name, output_dir / "metricas_por_clase.png")

    # Las visualizaciones interpretables mas pesadas se generan solo para la
    # variante representativa n_10/top_mi. Esto mantiene el pipeline rapido y,
    # al mismo tiempo, conserva las imagenes necesarias para la exposicion.
    parts = set(output_dir.parts)
    generate_interpretability_plots = ("n_10" in parts and output_dir.name == "top_mi") or ("n_10" not in parts)
    if generate_interpretability_plots:
        _plot_decision_regions(
            model,
            X_train_m,
            X_test_m,
            y_train_m,
            y_test_m,
            y_pred,
            model_name,
            output_dir / "regiones_decision_2d.png",
        )
        _plot_tree_structure(
            model,
            feature_names=list(X_train_m.columns),
            class_names=[str(label) for label in labels],
            model_name=model_name,
            output_path=output_dir / "arbol_nodos_hojas.png",
        )
        _plot_gaussian_bell_for_bayes(
            model,
            X_train_m,
            model_name,
            output_dir / "campana_gaussiana_bayes.png",
        )

    return {
        "model": model_name,
        "accuracy_tasa_reconocimiento": float(accuracy),
        "error_rate_tasa_error": float(error_rate),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n_features": int(X_train_m.shape[1]),
    }
