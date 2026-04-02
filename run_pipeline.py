#!/usr/bin/env python3
"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Pipeline anual completo: preprocesar → entrenar → evaluar → promover.
Ejecutar una vez al año cuando llega la nueva encuesta EDSA.

Uso:
    python run_pipeline.py --input data/edsa_2024.csv --year 2024
    python run_pipeline.py --input data/edsa_2024.csv --year 2024 --model RandomForest
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Configurar logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day")

Path("logs").mkdir(exist_ok=True)


def ejecutar_pipeline(ruta_input: str, year: str, modelo: str = "RandomForest"):
    """Ejecuta el pipeline MLOps completo para un nuevo año."""

    logger.info(f"\n{'█'*55}")
    logger.info(f"  PIPELINE MLOPS BOLIVIA — AÑO {year}")
    logger.info(f"{'█'*55}\n")

    # ── PASO 1: Preprocesamiento ──────────────────────────────────────────────
    logger.info("PASO 1/3 — Preprocesamiento")
    ruta_procesado = f"data/processed/edsa_{year}_procesado.csv"

    from src.preprocess import preprocesar
    df = preprocesar(ruta_input, ruta_procesado)
    logger.success(f"Dataset procesado: {len(df):,} registros → {ruta_procesado}")

    # ── PASO 2: Entrenamiento ─────────────────────────────────────────────────
    logger.info("\nPASO 2/3 — Entrenamiento de 5 modelos")
    from src.train import entrenar_todos
    resultados, mejor_modelo = entrenar_todos(ruta_procesado, year)
    logger.success(f"Mejor modelo del año {year}: {mejor_modelo}")

    # ── PASO 3: Evaluación y promoción ────────────────────────────────────────
    logger.info(f"\nPASO 3/3 — Evaluación y promoción del modelo: {modelo}")
    from src.evaluate import evaluar_y_promover
    promovido = evaluar_y_promover(modelo, year)

    # ── Monitoreo: actualizar baseline ────────────────────────────────────────
    logger.info("\nActualizando baseline de monitoreo")
    import pandas as pd
    from monitoring.monitor import actualizar_baseline
    df_clean = pd.read_csv(ruta_procesado)
    actualizar_baseline(df_clean)

    # ── Resumen final ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"PIPELINE COMPLETADO — AÑO {year}")
    logger.info(f"{'='*55}")
    logger.info(f"  Dataset:       {len(df):,} registros")
    logger.info(f"  Mejor modelo:  {mejor_modelo}")
    logger.info(f"  F1-Macro:      {resultados[mejor_modelo]['f1_macro']:.4f}")
    logger.info(f"  Promovido:     {'Sí' if promovido else 'No (se mantiene versión anterior)'}")
    logger.info(f"  Próximo paso:  {'Recargar API con /modelo/recargar' if promovido else 'Sin cambios en API'}")

    return promovido


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps anual Bolivia")
    parser.add_argument("--input",  required=True, help="CSV crudo de la nueva EDSA")
    parser.add_argument("--year",   required=True, help="Año de la encuesta (ej: 2024)")
    parser.add_argument("--model",  default="RandomForest",
                        choices=["DecisionTree", "RandomForest", "GradientBoosting",
                                 "MLP", "NaiveBayes"],
                        help="Modelo a evaluar para producción")
    args = parser.parse_args()

    ejecutar_pipeline(args.input, args.year, args.model)
