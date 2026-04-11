"""
SHAP — explicabilidade do modelo XGBoost champion.
Calcula importância das features por transação.
"""
import logging
import os

import mlflow
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

FEATURE_NAMES = [
    "amount",
    "distance_from_home",
    "velocity_1h",
    "velocity_24h",
    "avg_amount_30d",
    "amount_ratio",
    "is_night",
    "high_velocity",
    "merchant_category_encoded",
]


def load_champion() -> object:
    """Carrega o modelo champion do Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.sklearn.load_model("models:/fraud-detector-champion/1")
    logger.info("Champion carregado do Model Registry.")
    return model


def explain_prediction(
    transaction: dict,
    model: object,
) -> dict:
    """
    Explica a predição para uma transação via SHAP.

    Args:
        transaction: dicionário com features da transação.
        model: modelo XGBoost carregado.

    Returns:
        Dicionário com score, top features e interpretação.
    """
    df = pd.DataFrame([transaction])[FEATURE_NAMES]

    # score de fraude
    fraud_score = float(model.predict_proba(df)[0][1])
    prediction = "FRAUDE" if fraud_score >= 0.5 else "LEGITIMA"

    # calcula SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # para XGBoost binário, shap_values pode ser lista ou array
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    # top 3 features por importância absoluta
    feature_importance = sorted(
        zip(FEATURE_NAMES, values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    top_features = [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "aumenta risco" if val > 0 else "reduz risco",
        }
        for name, val in feature_importance
    ]

    result = {
        "fraud_score": round(fraud_score, 4),
        "prediction": prediction,
        "top_features": top_features,
        "interpretation": _generate_interpretation(fraud_score, top_features),
    }

    return result


def _generate_interpretation(score: float, top_features: list) -> str:
    """Gera interpretação em linguagem natural."""
    level = "ALTO" if score >= 0.7 else "MÉDIO" if score >= 0.4 else "BAIXO"
    top = top_features[0]["feature"] if top_features else "N/A"
    direction = top_features[0]["direction"] if top_features else ""

    return (
        f"Risco {level} de fraude (score: {score:.2f}). "
        f"Principal fator: {top} ({direction})."
    )


if __name__ == "__main__":
    model = load_champion()

    # exemplo de transação suspeita
    transacao_suspeita = {
        "amount": 3200.0,
        "distance_from_home": 120.0,
        "velocity_1h": 5,
        "velocity_24h": 12,
        "avg_amount_30d": 150.0,
        "amount_ratio": 21.3,
        "is_night": 1,
        "high_velocity": 1,
        "merchant_category_encoded": 5,
    }

    # exemplo de transação normal
    transacao_normal = {
        "amount": 85.0,
        "distance_from_home": 2.0,
        "velocity_1h": 1,
        "velocity_24h": 3,
        "avg_amount_30d": 90.0,
        "amount_ratio": 0.94,
        "is_night": 0,
        "high_velocity": 0,
        "merchant_category_encoded": 0,
    }

    print("\n=== Transação Suspeita ===")
    result = explain_prediction(transacao_suspeita, model)
    print(f"Score:       {result['fraud_score']}")
    print(f"Predição:    {result['prediction']}")
    print(f"Interpretação: {result['interpretation']}")
    print("Top features:")
    for f in result["top_features"]:
        print(f"  {f['feature']:30s} SHAP: {f['shap_value']:+.4f} ({f['direction']})")

    print("\n=== Transação Normal ===")
    result2 = explain_prediction(transacao_normal, model)
    print(f"Score:       {result2['fraud_score']}")
    print(f"Predição:    {result2['prediction']}")
    print(f"Interpretação: {result2['interpretation']}")
    print("Top features:")
    for f in result2["top_features"]:
        print(f"  {f['feature']:30s} SHAP: {f['shap_value']:+.4f} ({f['direction']})")
