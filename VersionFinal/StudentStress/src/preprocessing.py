"""Modulo de preprocesamiento.

Incluye carga de datos, normalizacion de tipos, creacion de la clase de ansiedad
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
    """Convierte todas las columnas a enteros.

    El dataset contiene variables ordinales/categoricas codificadas como numeros.
    """
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="raise").astype(int)
    return df_norm


def create_anxiety_class(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    n_classes: int = 3,
) -> tuple[pd.DataFrame, list[float]]:
    """Crea la variable objetivo multiclase anxiety_class sin borrar la original."""
    if target_col not in df.columns:
        raise KeyError(f"No existe la columna objetivo: {target_col}")

    ensure_dir(output_dir)
    df_processed = df.copy()
    df_processed["anxiety_level_original"] = df_processed[target_col]

    qcut_result = pd.qcut(
        df_processed[target_col],
        q=n_classes,
        labels=list(range(n_classes)),
        retbins=True,
        duplicates="drop",
    )

    df_processed["anxiety_class"] = qcut_result[0].astype(int)
    cut_points = [float(value) for value in qcut_result[1]]

    labels_map = {0: "baja", 1: "media", 2: "alta"}
    df_processed["anxiety_class_label"] = df_processed["anxiety_class"].map(labels_map)

    pd.DataFrame({"cut_point": cut_points}).to_csv(
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
    """Separa variables predictoras y objetivo.

    Por defecto se conserva stress_level para respetar la actividad original.
    Puede eliminarse con drop_stress_level=True para un experimento adicional.
    """
    columns_to_drop = [target_col, "anxiety_class_label", "anxiety_level_original", "anxiety_level"]
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


def plot_train_test_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
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
    df_processed, cut_points = create_anxiety_class(df_norm, target_col, output_dir)
    X, y = build_features_and_target(
        df_processed,
        target_col="anxiety_class",
        drop_stress_level=drop_stress_level,
    )
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    save_processed_data(df_processed, X_train, X_test, y_train, y_test, output_dir)
    plot_train_test_distribution(y_train, y_test, output_dir)

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cut_points": cut_points,
    }
