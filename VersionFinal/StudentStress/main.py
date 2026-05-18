"""Punto de entrada del proyecto de clasificacion de niveles de ansiedad.

Flujo integrado:
1. Analisis exploratorio de datos.
2. Preprocesamiento.
3. Seleccion de caracteristicas.
4. Entrenamiento de modelos.
5. Evaluacion multiclase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda import run_eda
from src.feature_selection import run_feature_selection
from src.preprocessing import load_dataset, run_preprocessing
from src.training import run_training


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "resultados"
DATASET_PATH = DATA_DIR / "StressLevelDataset.csv"

EDA_DIR = RESULTS_DIR / "eda"
PREPROCESSING_DIR = RESULTS_DIR / "preprocesamiento"
FEATURE_SELECTION_DIR = RESULTS_DIR / "seleccion_caracteristicas"
CLASSIFICATION_DIR = RESULTS_DIR / "clasificacion"

REQUIRED_DIRECTORIES = [
    DATA_DIR,
    PROJECT_ROOT / "modelos",
    PROJECT_ROOT / "src",
    EDA_DIR,
    PREPROCESSING_DIR,
    FEATURE_SELECTION_DIR,
    CLASSIFICATION_DIR,
]

REQUIRED_FILES = [
    DATASET_PATH,
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "modelos" / "__init__.py",
    PROJECT_ROOT / "modelos" / "bayes.py",
    PROJECT_ROOT / "modelos" / "arbol.py",
    PROJECT_ROOT / "modelos" / "common.py",
    PROJECT_ROOT / "src" / "eda.py",
    PROJECT_ROOT / "src" / "preprocessing.py",
    PROJECT_ROOT / "src" / "feature_selection.py",
    PROJECT_ROOT / "src" / "training.py",
    PROJECT_ROOT / "src" / "evaluation.py",
]


def parse_k_values(raw_values: str) -> list[int]:
    """Convierte una cadena como '5,10,15' en lista de enteros."""
    values = []
    for item in raw_values.split(','):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Debe indicarse al menos un valor de n para seleccion de caracteristicas.")
    return sorted(set(values))


def create_result_directories() -> None:
    """Crea las carpetas necesarias si no existen."""
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def validate_project_structure() -> bool:
    """Valida que los archivos y carpetas base existan."""
    create_result_directories()

    missing_dirs = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_DIRECTORIES if not path.exists()]
    missing_files = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_FILES if not path.exists()]

    if missing_dirs or missing_files:
        print("[ERROR] La estructura base no esta completa.")
        if missing_dirs:
            print("\nCarpetas faltantes:")
            for item in missing_dirs:
                print(f"  - {item}")
        if missing_files:
            print("\nArchivos faltantes:")
            for item in missing_files:
                print(f"  - {item}")
        return False

    print("[OK] Estructura base validada correctamente.")
    print(f"[OK] Dataset encontrado: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    return True


def run_full_pipeline(
    k_values: list[int],
    drop_stress_level: bool = False,
    bayes_type: str = "gaussian",
) -> None:
    """Ejecuta todas las fases del proyecto."""
    if not validate_project_structure():
        raise SystemExit(1)

    print("\n[1/5] Ejecutando Analisis Exploratorio de Datos...")
    df_raw = load_dataset(DATASET_PATH)
    run_eda(df_raw, EDA_DIR)
    print(f"[OK] Resultados EDA guardados en: {EDA_DIR.relative_to(PROJECT_ROOT)}")
    print("[OK] Reporte TXT generado en: resultados/eda/reporte_eda.txt")

    print("\n[2/5] Ejecutando preprocesamiento...")
    preprocessing_result = run_preprocessing(
        dataset_path=DATASET_PATH,
        output_dir=PREPROCESSING_DIR,
        target_col="anxiety_level",
        drop_stress_level=drop_stress_level,
    )
    print(f"[OK] Resultados de preprocesamiento guardados en: {PREPROCESSING_DIR.relative_to(PROJECT_ROOT)}")
    print("[INFO] Se discretizaron anxiety_level, self_esteem y depression en 3 niveles.")

    print("\n[3/5] Ejecutando seleccion de caracteristicas...")
    selection_result = run_feature_selection(
        X_train=preprocessing_result["X_train"],
        y_train=preprocessing_result["y_train"],
        output_dir=FEATURE_SELECTION_DIR,
        k_values=k_values,
    )
    print(f"[OK] Resultados de seleccion guardados en: {FEATURE_SELECTION_DIR.relative_to(PROJECT_ROOT)}")
    print("[INFO] Metodos usados: Informacion Mutua, Chi-cuadrado e importancia de Arbol.")
    print("[INFO] La seleccion de caracteristicas se calculo solo con entrenamiento.")

    print("\n[4/5] Entrenando modelos Bayes y Arbol por cada n...")
    summary_df = run_training(
        X_train=preprocessing_result["X_train"],
        X_test=preprocessing_result["X_test"],
        y_train=preprocessing_result["y_train"],
        y_test=preprocessing_result["y_test"],
        selected_by_k=selection_result["selected_by_k"],
        output_dir=CLASSIFICATION_DIR,
        bayes_type=bayes_type,
    )
    print(f"[OK] Resultados de clasificacion guardados en: {CLASSIFICATION_DIR.relative_to(PROJECT_ROOT)}")

    print("\n[5/5] Resumen final de mejores modelos:")
    print(
        summary_df[
            [
                "model",
                "accuracy_tasa_reconocimiento",
                "error_rate_tasa_error",
                "f1_macro",
                "f1_weighted",
                "n_features",
                "n_selection",
            ]
        ].head(12).to_string(index=False)
    )

    execution_summary = {
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "target_original": "anxiety_level",
        "target_modelado": "anxiety_class",
        "variables_discretizadas": ["anxiety_level", "self_esteem", "depression"],
        "k_values": selection_result["k_values"],
        "drop_stress_level": drop_stress_level,
        "bayes_type": bayes_type,
        "n_train": int(len(preprocessing_result["y_train"])),
        "n_test": int(len(preprocessing_result["y_test"])),
        "n_features_train": int(preprocessing_result["X_train"].shape[1]),
        "mejor_modelo": summary_df.iloc[0].to_dict(),
        "nota_metodologica": "La seleccion de caracteristicas se calculo solo con X_train y y_train para evitar fuga de informacion.",
    }

    with open(RESULTS_DIR / "resumen_ejecucion_completa.json", "w", encoding="utf-8") as file:
        json.dump(execution_summary, file, indent=2, ensure_ascii=False)

    print("\n[OK] Pipeline completo ejecutado correctamente.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proyecto de clasificacion multiclase de niveles de ansiedad."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Valida la estructura del proyecto sin ejecutar el pipeline.",
    )
    parser.add_argument(
        "--top-k-list",
        default="5,10,15",
        help="Valores de n para seleccion de caracteristicas. Ejemplo: 5,10,15",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Compatibilidad: ejecuta un solo valor de n. Si se usa, reemplaza --top-k-list.",
    )
    parser.add_argument(
        "--drop-stress-level",
        action="store_true",
        help="Elimina stress_level de los predictores para un experimento adicional.",
    )
    parser.add_argument(
        "--bayes-type",
        choices=["gaussian", "multinomial", "categorical"],
        default="gaussian",
        help="Tipo de Bayes ingenuo a usar. Valor por defecto: gaussian.",
    )
    args = parser.parse_args()

    if args.check:
        ok = validate_project_structure()
        raise SystemExit(0 if ok else 1)

    k_values = [args.top_k] if args.top_k is not None else parse_k_values(args.top_k_list)

    run_full_pipeline(
        k_values=k_values,
        drop_stress_level=args.drop_stress_level,
        bayes_type=args.bayes_type,
    )


if __name__ == "__main__":
    main()
