"""
src/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Preprocesamiento del dataset EDSA Bolivia.
Transforma el CSV crudo en el dataset listo para entrenamiento.

Uso:
    python src/preprocess.py --input data/edsa_2024.csv
    python src/preprocess.py --input data/edsa_2024.csv --output data/processed/2024.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ── Constantes ────────────────────────────────────────────────────────────────
COLS_FEATURES = [
    "IMC", "Peso", "Talla", "Anemia", "AreaUrbana",
    "AltitudAlta", "Embarazada", "Departamento", "ZonaGeografica",
]
TARGET = "RiesgoMetabolicoClase"

NOMBRES_DEPTO = {
    1: "Chuquisaca", 2: "La Paz",    3: "Cochabamba",
    4: "Oruro",      5: "Potosí",    6: "Tarija",
    7: "Santa Cruz", 8: "Beni",      9: "Pando",
}


def cargar_raw(ruta: str) -> pd.DataFrame:
    """Carga el CSV crudo de la EDSA y filtra mujeres adultas."""
    logger.info(f"Cargando dataset: {ruta}")
    df = pd.read_csv(ruta)
    logger.info(f"  Filas originales: {len(df):,}")

    # Filtrar subpoblación: mujeres adultas (tienen imc_m calculado)
    df = df[df["imc_m"].notna()].copy()
    logger.info(f"  Mujeres adultas: {len(df):,}")
    return df


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las variables del modelo desde el dataset crudo.
    El mismo pipeline exacto que en el notebook de preprocesamiento.
    """
    out = pd.DataFrame()

    # ── Variables continuas ───────────────────────────────────────────────────
    out["IMC"]   = df["imc_m"].round(4)
    out["Peso"]  = df["hs05_0095"].round(1)
    out["Talla"] = df["hs05_0096"].round(1)

    # ── Variables binarias ────────────────────────────────────────────────────
    # Anemia: tip_anemia_m >= 2 (leve, moderada, severa)
    # Imputación: 0 donde no hay medición de hemoglobina
    out["Anemia"] = 0
    mask_anemia = df["tip_anemia_m"].notna()
    out.loc[mask_anemia, "Anemia"] = (
        df.loc[mask_anemia, "tip_anemia_m"] >= 2
    ).astype(int)

    # Área urbana: area == 1
    out["AreaUrbana"] = (df["area"] == 1).astype(int)

    # Altitud alta: >= 2500 msnm (altiplano boliviano)
    out["AltitudAlta"] = (df["altitud"] >= 2500).astype(int)

    # Embarazada: hs06_0121 == 1 (imputar 0 donde no aplica)
    out["Embarazada"] = 0
    mask_emb = df["hs06_0121"].notna()
    out.loc[mask_emb, "Embarazada"] = (
        df.loc[mask_emb, "hs06_0121"] == 1
    ).astype(int)

    # ── Variables ordinales ───────────────────────────────────────────────────
    # Departamento: estrato // 100
    out["Departamento"] = (df["estrato"] // 100).astype(int)

    # Zona geográfica: 1=Llanos, 2=Valles, 3=Altiplano
    out["ZonaGeografica"] = pd.cut(
        df["altitud"],
        bins=[-1, 500, 2500, 9999],
        labels=[1, 2, 3],
    ).astype(int)

    # ── TARGET: Riesgo metabólico por cintura ─────────────────────────────────
    # Estándar OMS — WHO Technical Report Series 894 (2000)
    # 0 = sin riesgo (≤ 80 cm)
    # 1 = riesgo leve (81–88 cm)
    # 2 = riesgo alto (> 88 cm)
    out[TARGET] = pd.cut(
        df["hs05_0097"],
        bins=[-1, 80, 88, 999],
        labels=[0, 1, 2],
    ).astype(int)

    return out


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y NaN del dataset final."""
    n_antes = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    n_dup = n_antes - len(df)
    logger.info(f"  Duplicados/NaN eliminados: {n_dup:,}")
    logger.info(f"  Registros finales: {len(df):,}")
    return df


def reportar(df: pd.DataFrame) -> None:
    """Imprime un resumen del dataset procesado."""
    vc = df[TARGET].value_counts().sort_index()
    logger.info("  Distribución del target:")
    etiquetas = {0: "Sin riesgo (≤80cm)", 1: "Riesgo leve (81-88cm)", 2: "Riesgo alto (>88cm)"}
    for cls, label in etiquetas.items():
        n = vc.get(cls, 0)
        logger.info(f"    Clase {cls} — {label}: {n:,} ({n/len(df)*100:.1f}%)")
    logger.info(f"  NaN totales: {df.isna().sum().sum()}")


def preprocesar(ruta_entrada: str, ruta_salida: str = None) -> pd.DataFrame:
    """Pipeline completo: carga → construye variables → limpia → guarda."""
    df_raw   = cargar_raw(ruta_entrada)
    df_vars  = construir_variables(df_raw)
    df_clean = limpiar(df_vars)
    reportar(df_clean)

    if ruta_salida:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(ruta_salida, index=False)
        logger.success(f"Dataset guardado: {ruta_salida}")

    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento EDSA Bolivia")
    parser.add_argument("--input",  required=True, help="Ruta al CSV crudo de la EDSA")
    parser.add_argument("--output", default=None,  help="Ruta de salida (opcional)")
    args = parser.parse_args()

    if args.output is None:
        # Inferir nombre de salida desde el año del archivo
        año = Path(args.input).stem.split("_")[-1]
        args.output = f"data/processed/edsa_{año}_procesado.csv"

    preprocesar(args.input, args.output)
