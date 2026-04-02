"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Entrenamiento de los 5 modelos con registro automático en MLflow.
Cada ejecución queda registrada con sus métricas, parámetros e hiperparámetros.

Uso:
    python src/train.py --data data/processed/edsa_2024_procesado.csv
    python src/train.py --data data/processed/edsa_2024_procesado.csv --year 2024
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "riesgo_metabolico_bolivia")
MODEL_NAME      = os.getenv("MODEL_NAME", "RiesgoMetabolico")
TARGET          = "RiesgoMetabolicoClase"
COLS_FEATURES   = [
    "IMC", "Peso", "Talla", "Anemia", "AreaUrbana",
    "AltitudAlta", "Embarazada", "Departamento", "ZonaGeografica",
]
LABELS = ["Sin Riesgo", "Riesgo Leve", "Riesgo Alto"]
RANDOM_STATE = 42


# ── Utilidades ────────────────────────────────────────────────────────────────
def calcular_metricas(y_true, y_pred, y_prob) -> dict:
    """Calcula todas las métricas de evaluación multiclase."""
    return {
        "accuracy"     : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro"     : round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted"  : round(f1_score(y_true, y_pred, average="weighted"), 4),
        "auc_roc_macro": round(roc_auc_score(y_true, y_prob, multi_class="ovr"), 4),
        "f1_clase0"    : round(f1_score(y_true, y_pred, average=None)[0], 4),
        "f1_clase1"    : round(f1_score(y_true, y_pred, average=None)[1], 4),
        "f1_clase2"    : round(f1_score(y_true, y_pred, average=None)[2], 4),
    }


def cargar_datos(ruta: str):
    """Carga el dataset procesado y lo divide en train/val/test."""
    df = pd.read_csv(ruta)
    logger.info(f"Dataset cargado: {len(df):,} registros")

    X = df[COLS_FEATURES]
    y = df[TARGET]

    # División estratificada 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Normalización
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=COLS_FEATURES)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=COLS_FEATURES)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=COLS_FEATURES)

    # Pesos de clase
    pesos = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
    dict_pesos = dict(zip([0, 1, 2], pesos))

    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return (
        X_train, X_val, X_test,
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        scaler, dict_pesos,
    )


# ── Modelos ───────────────────────────────────────────────────────────────────
def entrenar_decision_tree(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test):
    """Modelo 1: Árbol de Clasificación con búsqueda de profundidad óptima."""
    logger.info("Entrenando: Árbol de Decisión")

    # Búsqueda de max_depth
    mejor_f1, mejor_depth = 0, 3
    for d in [3, 4, 5, 6, 7, 8]:
        dt = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced",
            min_samples_leaf=20, random_state=RANDOM_STATE
        )
        dt.fit(X_train_sc, y_train)
        f1v = f1_score(y_val, dt.predict(X_val_sc), average="macro")
        if f1v > mejor_f1:
            mejor_f1, mejor_depth = f1v, d

    # Modelo final
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=mejor_depth,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train_sc, y_train)

    params = {
        "max_depth": mejor_depth,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
    }
    return modelo, params, X_train_sc, X_test_sc


def entrenar_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    """Modelo 2: Random Forest con búsqueda de n_estimators y max_depth."""
    logger.info("Entrenando: Random Forest")

    # Búsqueda n_estimators
    mejor_f1, mejor_n = 0, 100
    for n in [50, 100, 200, 300]:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        f1v = f1_score(y_val, rf.predict(X_val), average="macro")
        if f1v > mejor_f1:
            mejor_f1, mejor_n = f1v, n

    # Búsqueda max_depth
    mejor_f1, mejor_depth = 0, 10
    for d in [8, 10, 12, 15]:
        rf = RandomForestClassifier(
            n_estimators=mejor_n, max_depth=d,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        f1v = f1_score(y_val, rf.predict(X_val), average="macro")
        if f1v > mejor_f1:
            mejor_f1, mejor_depth = f1v, d

    modelo = RandomForestClassifier(
        n_estimators=mejor_n,
        max_depth=mejor_depth,
        max_features="sqrt",
        min_samples_split=10,
        min_samples_leaf=5,
        bootstrap=True,
        class_weight="balanced_subsample",
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    params = {
        "n_estimators": mejor_n,
        "max_depth": mejor_depth,
        "class_weight": "balanced_subsample",
    }
    return modelo, params, X_train, X_test


def entrenar_gradient_boosting(X_train, X_val, X_test, y_train, y_val, y_test):
    """Modelo 3: Gradient Boosting con búsqueda de learning_rate."""
    logger.info("Entrenando: Gradient Boosting")

    mejor_f1, mejor_lr = 0, 0.05
    for lr in [0.30, 0.10, 0.05, 0.02]:
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=lr,
            max_depth=4, subsample=0.8, random_state=RANDOM_STATE,
        )
        gb.fit(X_train, y_train)
        f1v = f1_score(y_val, gb.predict(X_val), average="macro")
        if f1v > mejor_f1:
            mejor_f1, mejor_lr = f1v, lr

    modelo = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=mejor_lr,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)

    params = {
        "n_estimators": 200,
        "learning_rate": mejor_lr,
        "max_depth": 4,
        "subsample": 0.8,
    }
    return modelo, params, X_train, X_test


def entrenar_mlp(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, dict_pesos):
    """Modelo 4: Red Neuronal MLP con Keras."""
    logger.info("Entrenando: Red Neuronal MLP")

    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    y_train_ohe = to_categorical(y_train.values, num_classes=3)
    y_val_ohe   = to_categorical(y_val.values,   num_classes=3)

    inp = keras.Input(shape=(9,))
    x   = layers.BatchNormalization()(inp)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax")(x)

    modelo = keras.Model(inputs=inp, outputs=out)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
    ]

    modelo.fit(
        X_train_sc.values, y_train_ohe,
        validation_data=(X_val_sc.values, y_val_ohe),
        epochs=100, batch_size=512,
        class_weight=dict_pesos,
        callbacks=callbacks, verbose=0,
    )

    params = {
        "arquitectura": "128-64-32",
        "optimizer": "adam",
        "batch_size": 512,
        "dropout": "0.4-0.3-0.2",
    }
    return modelo, params, X_train_sc, X_test_sc


def entrenar_naive_bayes(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test):
    """Modelo 5: Gaussian Naive Bayes."""
    logger.info("Entrenando: Naive Bayes")

    mejor_f1, mejor_vs = 0, 1e-9
    for vs in [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]:
        nb = GaussianNB(var_smoothing=vs)
        nb.fit(X_train_sc, y_train)
        f1v = f1_score(y_val, nb.predict(X_val_sc), average="macro")
        if f1v > mejor_f1:
            mejor_f1, mejor_vs = f1v, vs

    modelo = GaussianNB(var_smoothing=mejor_vs)
    modelo.fit(X_train_sc, y_train)

    params = {"var_smoothing": mejor_vs}
    return modelo, params, X_train_sc, X_test_sc


# ── Pipeline principal ────────────────────────────────────────────────────────
def entrenar_todos(ruta_datos: str, year: str = None):
    """Entrena los 5 modelos y registra todo en MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if year is None:
        year = datetime.now().strftime("%Y")

    # Cargar datos
    (X_train, X_val, X_test,
     X_train_sc, X_val_sc, X_test_sc,
     y_train, y_val, y_test,
     scaler, dict_pesos) = cargar_datos(ruta_datos)

    resultados = {}

    # Definir los 5 modelos a entrenar
    definiciones = [
        ("DecisionTree",     entrenar_decision_tree,
         (X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test)),
        ("RandomForest",     entrenar_random_forest,
         (X_train, X_val, X_test, y_train, y_val, y_test)),
        ("GradientBoosting", entrenar_gradient_boosting,
         (X_train, X_val, X_test, y_train, y_val, y_test)),
        ("MLP",              entrenar_mlp,
         (X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, dict_pesos)),
        ("NaiveBayes",       entrenar_naive_bayes,
         (X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test)),
    ]

    for nombre, func, args in definiciones:
        with mlflow.start_run(run_name=f"{nombre}_{year}"):
            logger.info(f"\n{'─'*50}")

            # Entrenar
            modelo, params, X_tr, X_te = func(*args)

            # Predicciones
            if nombre == "MLP":
                y_prob = modelo.predict(X_te.values, verbose=0)
                y_pred = np.argmax(y_prob, axis=1)
            else:
                y_pred = modelo.predict(X_te)
                y_prob = modelo.predict_proba(X_te)

            # Métricas
            metricas = calcular_metricas(y_test, y_pred, y_prob)
            resultados[nombre] = metricas

            # Registrar en MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metricas)
            mlflow.set_tag("year", year)
            mlflow.set_tag("model_type", nombre)
            mlflow.set_tag("dataset", ruta_datos)

            # Guardar scaler junto al modelo
            mlflow.log_dict(
                {"features": COLS_FEATURES, "target": TARGET},
                "schema.json"
            )

            # Guardar el modelo
            if nombre == "MLP":
                modelo.save(f"models/{nombre}_{year}.keras")
                mlflow.log_artifact(f"models/{nombre}_{year}.keras")
            else:
                mlflow.sklearn.log_model(
                    modelo,
                    artifact_path="model",
                    registered_model_name=f"{MODEL_NAME}_{nombre}",
                )

            # Imprimir resultado
            logger.success(
                f"{nombre}: acc={metricas['accuracy']:.4f} "
                f"f1={metricas['f1_macro']:.4f} "
                f"auc={metricas['auc_roc_macro']:.4f}"
            )

    # Guardar scaler para uso en la API
    Path("models").mkdir(exist_ok=True)
    with open(f"models/scaler_{year}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler guardado: models/scaler_{year}.pkl")

    # Resumen final
    logger.info(f"\n{'='*55}")
    logger.info("RESUMEN COMPARATIVO")
    logger.info(f"{'Modelo':<20} {'F1-Macro':>10} {'AUC-ROC':>10}")
    logger.info(f"{'─'*42}")
    mejor = max(resultados, key=lambda k: resultados[k]["f1_macro"])
    for nombre, m in sorted(resultados.items(), key=lambda x: -x[1]["f1_macro"]):
        marca = " ← MEJOR" if nombre == mejor else ""
        logger.info(f"{nombre:<20} {m['f1_macro']:>10.4f} {m['auc_roc_macro']:>10.4f}{marca}")

    return resultados, mejor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento MLOps Bolivia")
    parser.add_argument("--data", required=True, help="Ruta al CSV procesado")
    parser.add_argument("--year", default=None,  help="Año de la encuesta (ej: 2024)")
    args = parser.parse_args()

    entrenar_todos(args.data, args.year)
