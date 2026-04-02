"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Compara el nuevo modelo entrenado contra el modelo actualmente en producción.
Si el nuevo modelo es mejor, lo promueve automáticamente.

Uso:
    python src/evaluate.py --model RandomForest --year 2024
    python src/evaluate.py --model RandomForest --year 2024 --force
"""

import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME      = os.getenv("MODEL_NAME", "RiesgoMetabolico")
MIN_F1_MACRO    = float(os.getenv("MIN_F1_MACRO", "0.65"))
MAX_F1_DROP     = float(os.getenv("MAX_F1_DROP", "0.05"))


def obtener_metricas_produccion(nombre_modelo: str) -> dict | None:
    """Obtiene las métricas del modelo actualmente en producción."""
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    nombre_completo = f"{MODEL_NAME}_{nombre_modelo}"

    try:
        versiones = client.get_latest_versions(nombre_completo, stages=["Production"])
        if not versiones:
            logger.warning("No hay modelo en producción todavía.")
            return None

        version = versiones[0]
        run = client.get_run(version.run_id)
        metricas = run.data.metrics
        logger.info(f"Modelo en producción: versión {version.version}")
        logger.info(f"  F1-Macro: {metricas.get('f1_macro', 0):.4f}")
        return metricas

    except Exception as e:
        logger.warning(f"No se pudo obtener modelo en producción: {e}")
        return None


def obtener_metricas_nuevo(nombre_modelo: str, year: str) -> tuple[dict, str]:
    """Obtiene las métricas del modelo recién entrenado."""
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    nombre_completo = f"{MODEL_NAME}_{nombre_modelo}"

    versiones = client.get_latest_versions(nombre_completo, stages=["None", "Staging"])
    if not versiones:
        raise ValueError(f"No se encontró modelo {nombre_completo} en Staging/None")

    # Tomar la versión más reciente del año indicado
    for v in sorted(versiones, key=lambda x: -int(x.version)):
        run = client.get_run(v.run_id)
        if run.data.tags.get("year") == year:
            metricas = run.data.metrics
            logger.info(f"Nuevo modelo: versión {v.version} (año {year})")
            logger.info(f"  F1-Macro: {metricas.get('f1_macro', 0):.4f}")
            return metricas, v.version

    raise ValueError(f"No se encontró modelo del año {year}")


def decidir_promocion(metricas_nuevo: dict, metricas_prod: dict | None,
                      forzar: bool = False) -> tuple[bool, str]:
    """
    Decide si el nuevo modelo debe pasar a producción.

    Reglas:
    1. F1-Macro mínimo absoluto: >= MIN_F1_MACRO
    2. No caer más de MAX_F1_DROP respecto al modelo en producción
    3. Si no hay modelo en producción, promover automáticamente
    """
    f1_nuevo = metricas_nuevo.get("f1_macro", 0)

    # Regla 1: mínimo absoluto
    if f1_nuevo < MIN_F1_MACRO:
        return False, f"F1-Macro {f1_nuevo:.4f} < mínimo {MIN_F1_MACRO}"

    # Sin modelo en producción → promover siempre
    if metricas_prod is None:
        return True, "Primera versión en producción"

    f1_prod = metricas_prod.get("f1_macro", 0)

    # Regla 2: no caer más del umbral
    if f1_nuevo < f1_prod - MAX_F1_DROP:
        return False, (
            f"F1-Macro cayó {f1_prod - f1_nuevo:.4f} "
            f"(umbral máximo: {MAX_F1_DROP})"
        )

    if f1_nuevo >= f1_prod:
        return True, f"Mejora: {f1_nuevo:.4f} > {f1_prod:.4f}"

    return True, f"Dentro del umbral permitido: {f1_nuevo:.4f} vs {f1_prod:.4f}"


def promover(nombre_modelo: str, version: str) -> None:
    """Mueve el modelo a Production en MLflow."""
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    nombre_completo = f"{MODEL_NAME}_{nombre_modelo}"

    # Archivar la versión anterior en producción
    versiones_prod = client.get_latest_versions(nombre_completo, stages=["Production"])
    for v in versiones_prod:
        client.transition_model_version_stage(
            name=nombre_completo,
            version=v.version,
            stage="Archived",
        )
        logger.info(f"Versión {v.version} archivada")

    # Promover la nueva versión
    client.transition_model_version_stage(
        name=nombre_completo,
        version=version,
        stage="Production",
    )
    logger.success(f"Versión {version} promovida a Production")


def evaluar_y_promover(nombre_modelo: str, year: str, forzar: bool = False) -> bool:
    """Pipeline completo de evaluación y promoción."""
    mlflow.set_tracking_uri(MLFLOW_URI)

    logger.info(f"\n{'='*55}")
    logger.info(f"EVALUACIÓN: {nombre_modelo} — Año {year}")
    logger.info(f"{'='*55}")

    # Obtener métricas
    metricas_prod = obtener_metricas_produccion(nombre_modelo)
    metricas_nuevo, version = obtener_metricas_nuevo(nombre_modelo, year)

    # Comparar
    logger.info("\nComparación:")
    campos = ["f1_macro", "f1_clase1", "auc_roc_macro", "accuracy"]
    for campo in campos:
        nuevo = metricas_nuevo.get(campo, 0)
        prod  = metricas_prod.get(campo, 0) if metricas_prod else None
        if prod is not None:
            diff  = nuevo - prod
            signo = "↑" if diff > 0 else "↓"
            logger.info(f"  {campo:<18}: {nuevo:.4f} vs {prod:.4f} {signo}{abs(diff):.4f}")
        else:
            logger.info(f"  {campo:<18}: {nuevo:.4f} (sin baseline)")

    # Decisión
    promover_flag, razon = decidir_promocion(metricas_nuevo, metricas_prod, forzar)

    logger.info(f"\nDecisión: {'PROMOVER' if promover_flag else 'MANTENER ACTUAL'}")
    logger.info(f"  Razón: {razon}")

    if promover_flag or forzar:
        promover(nombre_modelo, version)
        return True
    else:
        logger.warning("Modelo actual se mantiene en producción")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación y promoción de modelos")
    parser.add_argument("--model", required=True,
                        choices=["DecisionTree", "RandomForest", "GradientBoosting",
                                 "MLP", "NaiveBayes"],
                        help="Nombre del modelo a evaluar")
    parser.add_argument("--year",  required=True, help="Año del modelo nuevo (ej: 2024)")
    parser.add_argument("--force", action="store_true",
                        help="Forzar promoción aunque no mejore")
    args = parser.parse_args()

    evaluar_y_promover(args.model, args.year, args.force)
