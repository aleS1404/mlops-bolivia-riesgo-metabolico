"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
API REST con FastAPI para servir el modelo de predicción de riesgo metabólico.
El modelo se carga automáticamente desde MLflow (versión en Production).

Endpoints:
    GET  /               → info del servicio
    GET  /salud          → health check
    POST /predecir       → predicción individual
    POST /predecir/lote  → predicción en lote (varios registros)
    GET  /modelo/info    → información del modelo activo
    GET  /metricas       → métricas del modelo en producción
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME    = os.getenv("MODEL_NAME", "RiesgoMetabolico")
MODEL_TYPE    = os.getenv("MODEL_TYPE", "RandomForest")   # modelo activo por defecto
API_VERSION   = os.getenv("API_VERSION", "v1")

ETIQUETAS = {
    0: "Sin riesgo (cintura ≤ 80 cm)",
    1: "Riesgo leve (cintura 81-88 cm)",
    2: "Riesgo alto (cintura > 88 cm)",
}
ADVERTENCIA = (
    "Este modelo es una herramienta de tamizaje investigativo basada en "
    "datos poblacionales bolivianos (EDSA 2023). NO reemplaza diagnóstico "
    "clínico ni evaluación médica individual. Uso exclusivo para "
    "investigación en salud pública."
)

COLS_FEATURES = [
    "IMC", "Peso", "Talla", "Anemia", "AreaUrbana",
    "AltitudAlta", "Embarazada", "Departamento", "ZonaGeografica",
]

# Estado global del modelo
estado = {
    "modelo": None,
    "scaler": None,
    "version": None,
    "cargado_en": None,
    "tipo": MODEL_TYPE,
}


# ── Carga del modelo ──────────────────────────────────────────────────────────
def cargar_modelo():
    """Carga el modelo en Production desde MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    nombre_completo = f"{MODEL_NAME}_{MODEL_TYPE}"

    try:
        uri = f"models:/{nombre_completo}/Production"
        modelo = mlflow.sklearn.load_model(uri)

        # Obtener versión
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        versiones = client.get_latest_versions(nombre_completo, stages=["Production"])
        version = versiones[0].version if versiones else "unknown"

        estado["modelo"]     = modelo
        estado["version"]    = version
        estado["cargado_en"] = datetime.now().isoformat()

        # Cargar scaler (buscar el más reciente disponible)
        scalers = sorted(Path("models").glob("scaler_*.pkl"), reverse=True)
        if scalers:
            with open(scalers[0], "rb") as f:
                estado["scaler"] = pickle.load(f)
            logger.info(f"Scaler cargado: {scalers[0]}")

        logger.success(f"Modelo cargado: {nombre_completo} v{version}")

    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        logger.warning("Iniciando sin modelo — usa /modelo/recargar para cargar uno")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la aplicación."""
    cargar_modelo()
    yield
    logger.info("Cerrando API")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Riesgo Metabólico Bolivia",
    description=(
        "Predicción del riesgo metabólico en mujeres bolivianas en edad reproductiva "
        "basada en el dataset EDSA 2023 del INE Bolivia. "
        "Modelo de Machine Learning — uso exclusivo para tamizaje investigativo."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas de entrada/salida ─────────────────────────────────────────────────
class DatosPaciente(BaseModel):
    """Variables predictoras de una mujer adulta."""
    IMC          : float = Field(..., ge=10, le=80,  description="Índice de masa corporal (kg/m²)")
    Peso         : float = Field(..., ge=25, le=200, description="Peso en kg")
    Talla        : float = Field(..., ge=100, le=200,description="Talla en cm")
    Anemia       : int   = Field(..., ge=0, le=1,    description="0=No, 1=Sí (hemoglobina baja)")
    AreaUrbana   : int   = Field(..., ge=0, le=1,    description="0=Rural, 1=Urbana")
    AltitudAlta  : int   = Field(..., ge=0, le=1,    description="0=Menos de 2500m, 1=Altiplano")
    Embarazada   : int   = Field(..., ge=0, le=1,    description="0=No, 1=Sí")
    Departamento : int   = Field(..., ge=1, le=9,    description="1=Chuquisaca … 9=Pando")
    ZonaGeografica: int  = Field(..., ge=1, le=3,    description="1=Llanos, 2=Valles, 3=Altiplano")

    @field_validator("IMC")
    @classmethod
    def validar_imc(cls, v, info):
        """Verificar coherencia IMC con Peso/Talla si están disponibles."""
        return round(v, 4)

    model_config = {
        "json_schema_extra": {
            "example": {
                "IMC": 28.5, "Peso": 65.0, "Talla": 155.0,
                "Anemia": 0, "AreaUrbana": 1, "AltitudAlta": 1,
                "Embarazada": 0, "Departamento": 2, "ZonaGeografica": 3,
            }
        }
    }


class ResultadoPrediccion(BaseModel):
    clase         : int
    etiqueta      : str
    probabilidades: dict
    modelo_version: str
    advertencia   : str
    timestamp     : str


class LoteEntrada(BaseModel):
    registros: list[DatosPaciente] = Field(..., max_length=1000)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def raiz():
    return {
        "servicio"   : "API Riesgo Metabólico Bolivia",
        "version"    : API_VERSION,
        "descripcion": "Tamizaje de riesgo metabólico en mujeres bolivianas",
        "endpoints"  : ["/predecir", "/predecir/lote", "/modelo/info", "/salud"],
        "docs"       : "/docs",
    }


@app.get("/salud", tags=["Info"])
async def health_check():
    return {
        "estado"        : "activo" if estado["modelo"] is not None else "sin_modelo",
        "modelo_cargado": estado["modelo"] is not None,
        "version"       : estado["version"],
        "timestamp"     : datetime.now().isoformat(),
    }


@app.get("/modelo/info", tags=["Modelo"])
async def info_modelo():
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {
        "tipo"       : estado["tipo"],
        "version"    : estado["version"],
        "cargado_en" : estado["cargado_en"],
        "features"   : COLS_FEATURES,
        "clases"     : ETIQUETAS,
        "dataset"    : "EDSA 2023 — INE Bolivia",
        "poblacion"  : "Mujeres adultas en edad reproductiva",
        "advertencia": ADVERTENCIA,
    }


@app.post("/modelo/recargar", tags=["Modelo"])
async def recargar_modelo():
    """Recarga el modelo desde MLflow sin reiniciar la API."""
    cargar_modelo()
    return {"mensaje": "Modelo recargado", "version": estado["version"]}


@app.post("/predecir", response_model=ResultadoPrediccion, tags=["Predicción"])
async def predecir(datos: DatosPaciente):
    """Predicción de riesgo metabólico para una mujer."""
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Preparar features
    df = pd.DataFrame([datos.model_dump()])[COLS_FEATURES]

    # Aplicar scaler si está disponible
    if estado["scaler"] is not None:
        df_sc = pd.DataFrame(
            estado["scaler"].transform(df), columns=COLS_FEATURES
        )
    else:
        df_sc = df

    # Predicción
    try:
        clase = int(estado["modelo"].predict(df_sc)[0])
        prob  = estado["modelo"].predict_proba(df_sc)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    return ResultadoPrediccion(
        clase=clase,
        etiqueta=ETIQUETAS[clase],
        probabilidades={
            "sin_riesgo" : round(prob[0], 4),
            "riesgo_leve": round(prob[1], 4),
            "riesgo_alto": round(prob[2], 4),
        },
        modelo_version=f"v{estado['version']}_{estado['tipo']}",
        advertencia=ADVERTENCIA,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predecir/lote", tags=["Predicción"])
async def predecir_lote(lote: LoteEntrada):
    """Predicción en lote para hasta 1000 registros."""
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    df = pd.DataFrame([r.model_dump() for r in lote.registros])[COLS_FEATURES]

    if estado["scaler"] is not None:
        df_sc = pd.DataFrame(
            estado["scaler"].transform(df), columns=COLS_FEATURES
        )
    else:
        df_sc = df

    clases = estado["modelo"].predict(df_sc).tolist()
    probs  = estado["modelo"].predict_proba(df_sc).tolist()

    resultados = []
    for i, (c, p) in enumerate(zip(clases, probs)):
        resultados.append({
            "indice"        : i,
            "clase"         : int(c),
            "etiqueta"      : ETIQUETAS[int(c)],
            "probabilidades": {
                "sin_riesgo" : round(p[0], 4),
                "riesgo_leve": round(p[1], 4),
                "riesgo_alto": round(p[2], 4),
            },
        })

    return {
        "total"          : len(resultados),
        "modelo_version" : f"v{estado['version']}_{estado['tipo']}",
        "resultados"     : resultados,
        "advertencia"    : ADVERTENCIA,
        "timestamp"      : datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
