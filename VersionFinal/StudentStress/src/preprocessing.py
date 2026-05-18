"""Modulo de preprocesamiento.

Incluye carga de datos, normalizacion de tipos, discretizacion de variables clave
y separacion entrenamiento/prueba 70/30.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> None:
    """Crea una carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Carga el archivo CSV del dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {dataset_path}")
    return pd.read_csv(dataset_path)


def normalize_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte todas las columnas a enteros."""
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="raise").astype(int)
    return df_norm


def _discretize_three_levels(series: pd.Series) -> tuple[pd.Series, list[float]]:
    """Discretiza una serie en tres niveles por cuantiles: 0, 1 y 2."""
    qcut_result = pd.qcut(
        series,
        q=3,
        labels=[0, 1, 2],
        retbins=True,
        duplicates="drop",
    )
    return qcut_result[0].astype(int), [float(value) for value in qcut_result[1]]


def discretize_key_variables(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str = "anxiety_level",
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Discretiza anxiety_level, self_esteem y depression en tres niveles.

    - anxiety_level se usa para crear la variable objetivo anxiety_class.
    - self_esteem y depression se reemplazan por su version discretizada como
      variables predictoras, conservando sus copias originales.
    """
    ensure_dir(output_dir)
    df_processed = df.copy()
    cut_points: dict[str, list[float]] = {}

    required_cols = [target_col, "self_esteem", "depression"]
    for col in required_cols:
        if col not in df_processed.columns:
            raise KeyError(f"No existe la columna requerida para discretizar: {col}")

    df_processed["anxiety_level_original"] = df_processed[target_col]
    anxiety_class, anxiety_cuts = _discretize_three_levels(df_processed[target_col])
    df_processed["anxiety_class"] = anxiety_class
    cut_points["anxiety_level"] = anxiety_cuts

    labels_map = {0: "baja", 1: "media", 2: "alta"}
    df_processed["anxiety_class_label"] = df_processed["anxiety_class"].map(labels_map)

    for col in ["self_esteem", "depression"]:
        original_col = f"{col}_original"
        df_processed[original_col] = df_processed[col]
        discretized, cuts = _discretize_three_levels(df_processed[col])
        df_processed[col] = discretized
        cut_points[col] = cuts

    cut_rows = []
    for variable, cuts in cut_points.items():
        for idx, cut in enumerate(cuts):
            cut_rows.append({"variable": variable, "cut_index": idx, "cut_point": cut})
    pd.DataFrame(cut_rows).to_csv(
        output_dir / "cortes_discretizacion_variables.csv",
        index=False,
        encoding="utf-8",
    )

    # Mantener compatibilidad con el archivo anterior de anxiety_level.
    pd.DataFrame({"cut_point": cut_points["anxiety_level"]}).to_csv(
        output_dir / "cortes_discretizacion_clase.csv",
        index=False,
        encoding="utf-8",
    )

    class_distribution = (
        df_processed["anxiety_class"]
        .value_counts()
        .sort_index()
        .rename_axis("clase")
        .reset_index(name="frecuencia")
    )
    class_distribution.to_csv(
        output_dir / "distribucion_anxiety_class.csv", index=False, encoding="utf-8"
    )

    return df_processed, cut_points


def build_features_and_target(
    df: pd.DataFrame,
    target_col: str = "anxiety_class",
    drop_stress_level: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y objetivo."""
    columns_to_drop = [
        target_col,
        "anxiety_class_label",
        "anxiety_level_original",
        "anxiety_level",
        "self_esteem_original",
        "depression_original",
    ]
    if drop_stress_level and "stress_level" in df.columns:
        columns_to_drop.append("stress_level")

    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en 70% entrenamiento y 30% prueba con estratificacion."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_data(
    df_processed: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Guarda datasets procesados y particiones."""
    ensure_dir(output_dir)
    df_processed.to_csv(output_dir / "dataset_preprocesado.csv", index=False, encoding="utf-8")
    X_train.to_csv(output_dir / "X_train.csv", index=False, encoding="utf-8")
    X_test.to_csv(output_dir / "X_test.csv", index=False, encoding="utf-8")
    y_train.to_frame("anxiety_class").to_csv(
        output_dir / "y_train.csv", index=False, encoding="utf-8"
    )
    y_test.to_frame("anxiety_class").to_csv(
        output_dir / "y_test.csv", index=False, encoding="utf-8"
    )


def plot_train_test_distribution(y_train: pd.Series, y_test: pd.Series, output_dir: Path) -> None:
    """Grafica la distribucion de clases en entrenamiento y prueba."""
    ensure_dir(output_dir)
    plot_df = pd.concat(
        [
            pd.DataFrame({"clase": y_train, "particion": "Entrenamiento"}),
            pd.DataFrame({"clase": y_test, "particion": "Prueba"}),
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x="clase", hue="particion")
    plt.title("Distribucion de clases en entrenamiento y prueba")
    plt.xlabel("Clase de ansiedad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_dir / "distribucion_clases_train_test.png", dpi=150)
    plt.close()


def plot_discretized_variables(df_processed: pd.DataFrame, output_dir: Path) -> None:
    """Grafica las variables discretizadas en tres niveles."""
    plot_cols = ["anxiety_class", "self_esteem", "depression"]
    plot_df = df_processed[plot_cols].melt(var_name="variable", value_name="nivel")
    plt.figure(figsize=(9, 5))
    sns.countplot(data=plot_df, x="nivel", hue="variable")
    plt.title("Variables discretizadas en 3 niveles")
    plt.xlabel("Nivel discretizado")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_dir / "distribucion_variables_discretizadas.png", dpi=150)
    plt.close()


def run_preprocessing(
    dataset_path: Path,
    output_dir: Path,
    target_col: str = "anxiety_level",
    drop_stress_level: bool = False,
) -> dict[str, object]:
    """Ejecuta el preprocesamiento completo."""
    ensure_dir(output_dir)

    df_raw = load_dataset(dataset_path)
    df_norm = normalize_numeric_types(df_raw)
    df_processed, cut_points = discretize_key_variables(df_norm, output_dir, target_col=target_col)
    X, y = build_features_and_target(
        df_processed,
        target_col="anxiety_class",
        drop_stress_level=drop_stress_level,
    )
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    save_processed_data(df_processed, X_train, X_test, y_train, y_test, output_dir)
    plot_train_test_distribution(y_train, y_test, output_dir)
    plot_discretized_variables(df_processed, output_dir)

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cut_points": cut_points,
    }
