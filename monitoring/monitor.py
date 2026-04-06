"""
monitoring/monitor.py
─────────────────────────────────────────────────────────────────────────────
Monitoreo de drift en producción.
Compara la distribución actual de predicciones con el baseline de entrenamiento.

Uso:
    python monitoring/monitor.py --data data/processed/edsa_2024_procesado.csv
    python monitoring/monitor.py --predictions logs/predicciones.csv
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DRIFT_IMC_THRESHOLD = float(os.getenv("DRIFT_IMC_THRESHOLD", "0.10"))
DRIFT_CLASS2_THRESHOLD = float(os.getenv("DRIFT_CLASS2_THRESHOLD", "0.15"))
TARGET = "RiesgoMetabolicoClase"

# Estadísticas baseline del dataset de entrenamiento EDSA 2023
# (actualizar cada año con las del nuevo dataset)
BASELINE = {
    "IMC_mean": 25.64,
    "IMC_std": 6.48,
    "Peso_mean": 62.8,
    "proporcion_clase0": 0.482,
    "proporcion_clase1": 0.187,
    "proporcion_clase2": 0.332,
}


def to_python_type(obj):
    """Convierte tipos NumPy/Pandas a tipos nativos serializables por JSON."""
    if isinstance(obj, dict):
        return {str(k): to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_type(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj):
        return None
    return obj


def detectar_drift_imc(df: pd.DataFrame) -> dict:
    """Detecta cambios en la distribución del IMC."""
    imc_actual = df["IMC"].dropna()
    mean_actual = float(imc_actual.mean())
    mean_base = float(BASELINE["IMC_mean"])
    cambio_pct = abs(mean_actual - mean_base) / mean_base

    alerta = bool(cambio_pct > DRIFT_IMC_THRESHOLD)

    return {
        "variable": "IMC",
        "mean_baseline": round(mean_base, 4),
        "mean_actual": round(mean_actual, 4),
        "cambio_pct": round(cambio_pct * 100, 2),
        "umbral_pct": float(DRIFT_IMC_THRESHOLD * 100),
        "alerta": alerta,
        "mensaje": f"ALERTA: IMC media cambió {cambio_pct*100:.1f}%" if alerta else "OK",
    }


def detectar_drift_clases(df: pd.DataFrame, col_pred: str = TARGET) -> dict:
    """Detecta cambios en la proporción de cada clase predicha."""
    vc = df[col_pred].value_counts(normalize=True)
    prop_clase2_actual = float(vc.get(2, 0))
    prop_clase2_base = float(BASELINE["proporcion_clase2"])
    cambio = abs(prop_clase2_actual - prop_clase2_base)

    alerta = bool(cambio > DRIFT_CLASS2_THRESHOLD)

    return {
        "variable": "RiesgoMetabolicoClase",
        "prop_clase2_baseline": round(prop_clase2_base, 4),
        "prop_clase2_actual": round(prop_clase2_actual, 4),
        "cambio_absoluto": round(cambio, 4),
        "umbral": float(DRIFT_CLASS2_THRESHOLD),
        "alerta": alerta,
        "distribucion_actual": {
            str(k): round(float(v), 4) for k, v in vc.items()
        },
        "mensaje": f"ALERTA: Clase 2 cambió {cambio:.3f} puntos" if alerta else "OK",
    }


def generar_reporte(df: pd.DataFrame, ruta_salida: str = None) -> dict:
    """Genera el reporte completo de monitoreo."""
    timestamp = datetime.now().isoformat()

    reporte = {
        "timestamp": timestamp,
        "n_registros": int(len(df)),
        "drift_imc": detectar_drift_imc(df),
        "drift_clases": detectar_drift_clases(df),
        "estadisticas": {
            "IMC_mean": round(float(df["IMC"].mean()), 4),
            "IMC_std": round(float(df["IMC"].std()), 4),
            "Peso_mean": round(float(df["Peso"].mean()), 4),
            "Talla_mean": round(float(df["Talla"].mean()), 4),
        },
    }

    alertas = [
        bool(v["alerta"])
        for k, v in reporte.items()
        if isinstance(v, dict) and "alerta" in v
    ]
    reporte["hay_alertas"] = bool(any(alertas))
    reporte["n_alertas"] = int(sum(alertas))

    logger.info(f"\n{'='*55}")
    logger.info(f"REPORTE DE MONITOREO — {timestamp}")
    logger.info(f"{'='*55}")
    logger.info(f"Registros analizados: {len(df):,}")

    for nombre, resultado in [
        ("Drift IMC", reporte["drift_imc"]),
        ("Drift Clases", reporte["drift_clases"]),
    ]:
        estado = "⚠ ALERTA" if resultado["alerta"] else "✓ OK"
        logger.info(f"  {nombre}: {estado} — {resultado['mensaje']}")

    if reporte["hay_alertas"]:
        logger.warning(f"\n{'!'*30}")
        logger.warning(f"Se detectaron {reporte['n_alertas']} alerta(s).")
        logger.warning("Considera reentrenar con datos más recientes.")
        logger.warning(f"{'!'*30}")
    else:
        logger.success("Sin alertas de drift detectadas.")

    reporte = to_python_type(reporte)

    if ruta_salida:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        logger.info(f"Reporte guardado: {ruta_salida}")

    return reporte


def actualizar_baseline(df: pd.DataFrame) -> None:
    """Actualiza el baseline con los datos del nuevo año."""
    global BASELINE
    vc = df[TARGET].value_counts(normalize=True)
    BASELINE = {
        "IMC_mean": round(float(df["IMC"].mean()), 4),
        "IMC_std": round(float(df["IMC"].std()), 4),
        "Peso_mean": round(float(df["Peso"].mean()), 4),
        "proporcion_clase0": round(float(vc.get(0, 0)), 4),
        "proporcion_clase1": round(float(vc.get(1, 0)), 4),
        "proporcion_clase2": round(float(vc.get(2, 0)), 4),
    }

    Path("monitoring").mkdir(parents=True, exist_ok=True)
    with open("monitoring/baseline.json", "w", encoding="utf-8") as f:
        json.dump(to_python_type(BASELINE), f, indent=2, ensure_ascii=False)
    logger.success("Baseline actualizado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoreo de drift")
    parser.add_argument("--data", required=True, help="CSV con datos actuales")
    parser.add_argument("--output", default=None, help="Ruta para guardar el reporte JSON")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Actualizar el baseline con estos datos",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.update_baseline:
        actualizar_baseline(df)

    ruta_salida = args.output or f"monitoring/reporte_{datetime.now().strftime('%Y%m%d')}.json"
    generar_reporte(df, ruta_salida)