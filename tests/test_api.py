"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
Tests de la API REST. Verifica que todos los endpoints responden correctamente.

Uso:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short
"""

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, ".")
from api.main import app

client = TestClient(app)

# Datos de ejemplo válidos
EJEMPLO_VALIDO = {
    "IMC": 28.5, "Peso": 65.0, "Talla": 155.0,
    "Anemia": 0, "AreaUrbana": 1, "AltitudAlta": 1,
    "Embarazada": 0, "Departamento": 2, "ZonaGeografica": 3,
}

EJEMPLO_BAJO_RIESGO = {
    "IMC": 20.0, "Peso": 50.0, "Talla": 158.0,
    "Anemia": 0, "AreaUrbana": 1, "AltitudAlta": 0,
    "Embarazada": 0, "Departamento": 7, "ZonaGeografica": 1,
}


class TestEndpointsBasicos:
    def test_raiz_responde(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "servicio" in r.json()

    def test_health_check(self):
        r = client.get("/salud")
        assert r.status_code == 200
        assert "estado" in r.json()

    def test_info_modelo(self):
        r = client.get("/modelo/info")
        # 200 si hay modelo, 503 si no hay modelo cargado
        assert r.status_code in [200, 503]


class TestPrediccion:
    def test_prediccion_individual_estructura(self):
        r = client.post("/predecir", json=EJEMPLO_VALIDO)
        if r.status_code == 503:
            pytest.skip("Modelo no disponible en entorno de test")
        assert r.status_code == 200
        data = r.json()
        assert "clase" in data
        assert "etiqueta" in data
        assert "probabilidades" in data
        assert "advertencia" in data

    def test_prediccion_clase_valida(self):
        r = client.post("/predecir", json=EJEMPLO_VALIDO)
        if r.status_code == 503:
            pytest.skip("Modelo no disponible")
        assert r.json()["clase"] in [0, 1, 2]

    def test_probabilidades_suman_uno(self):
        r = client.post("/predecir", json=EJEMPLO_VALIDO)
        if r.status_code == 503:
            pytest.skip("Modelo no disponible")
        prob = r.json()["probabilidades"]
        suma = prob["sin_riesgo"] + prob["riesgo_leve"] + prob["riesgo_alto"]
        assert abs(suma - 1.0) < 0.01

    def test_advertencia_presente(self):
        r = client.post("/predecir", json=EJEMPLO_VALIDO)
        if r.status_code == 503:
            pytest.skip("Modelo no disponible")
        assert len(r.json()["advertencia"]) > 20

    def test_prediccion_lote(self):
        lote = {"registros": [EJEMPLO_VALIDO, EJEMPLO_BAJO_RIESGO]}
        r = client.post("/predecir/lote", json=lote)
        if r.status_code == 503:
            pytest.skip("Modelo no disponible")
        assert r.status_code == 200
        assert r.json()["total"] == 2
        assert len(r.json()["resultados"]) == 2


class TestValidaciones:
    def test_imc_fuera_de_rango(self):
        datos = {**EJEMPLO_VALIDO, "IMC": 500}
        r = client.post("/predecir", json=datos)
        assert r.status_code == 422   # Validation error

    def test_departamento_invalido(self):
        datos = {**EJEMPLO_VALIDO, "Departamento": 10}
        r = client.post("/predecir", json=datos)
        assert r.status_code == 422

    def test_campo_faltante(self):
        datos = {k: v for k, v in EJEMPLO_VALIDO.items() if k != "IMC"}
        r = client.post("/predecir", json=datos)
        assert r.status_code == 422

    def test_valor_binario_invalido(self):
        datos = {**EJEMPLO_VALIDO, "Anemia": 5}
        r = client.post("/predecir", json=datos)
        assert r.status_code == 422
