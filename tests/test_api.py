"""Testes da FastAPI."""
import os
import sys
sys.path.insert(0, ".")

import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

# Garante que o working directory é a raiz do projeto
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app, raise_server_exceptions=False)

MODEL_EXISTS = os.path.exists("models/champion_v3.joblib")
DATA_EXISTS = os.path.exists("data/processed/features.csv")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


@pytest.mark.skipif(
    not (MODEL_EXISTS and DATA_EXISTS),
    reason="modelo ou dados não disponíveis no ambiente CI"
)

def test_predict_valid():
    response = client.post(
        "/predict",
        json={"transaction_id": "TXN_009930"},
        timeout=120,
    )
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "prediction" in data
    assert data["prediction"] in ["FRAUDE", "LEGITIMA"]


@pytest.mark.skipif(
    not (MODEL_EXISTS and DATA_EXISTS),
    reason="modelo ou dados não disponíveis no ambiente CI"
)
def test_predict_not_found():
    response = client.post(
        "/predict",
        json={"transaction_id": "TXN_999999"},
        timeout=120,
    )
    assert response.status_code == 404