"""Testes da FastAPI."""
import sys
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_predict_valid():
    response = client.post(
        "/predict",
        json={"transaction_id": "TXN_009930"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "prediction" in data
    assert data["prediction"] in ["FRAUDE", "LEGITIMA"]


def test_predict_not_found():
    response = client.post(
        "/predict",
        json={"transaction_id": "TXN_999999"}
    )
    assert response.status_code == 404