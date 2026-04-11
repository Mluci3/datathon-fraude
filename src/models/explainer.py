"""
SHAP — explicabilidade do modelo XGBoost champion.
Retorna transaction_id e customer_id para ação imediata do analista.
"""
import logging
import os

import mlflow
import pandas as pd
import shap
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

FEATURE_COLS = [
    "amount", "distance_from_home", "velocity_1h", "velocity_24h",
    "avg_amount_30d", "account_balance", "amount_ratio", "is_night",
    "high_velocity", "is_online", "is_credit", "merchant_category_encoded",
]


def load_champion() -> object:
    """Carrega o modelo champion do Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.sklearn.load_model("models:/fraud-detector-champion/2")
    logger.info("Champion v2 carregado do Model Registry.")
    return model


def explain_prediction(transaction: dict, model: object) -> dict:
    """
    Explica a predição para uma transação via SHAP.

    Args:
        transaction: dicionário com TODOS os campos da transação
                     incluindo transaction_id e customer_id.
        model: modelo XGBoost carregado.

    Returns:
        Dicionário com identificadores, score, top features e ação sugerida.
    """
    # extrai identificadores para retornar ao analista
    transaction_id = transaction.get("transaction_id", "N/A")
    customer_id = transaction.get("customer_id", "N/A")

    # usa apenas features do modelo
    df = pd.DataFrame([transaction])[FEATURE_COLS]

    # score de fraude
    fraud_score = float(model.predict_proba(df)[0][1])
    prediction = "FRAUDE" if fraud_score >= 0.5 else "LEGITIMA"

    # calcula SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    # top 3 features por importância absoluta
    feature_importance = sorted(
        zip(FEATURE_COLS, values),
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

    # ação sugerida baseada no score
    if fraud_score >= 0.8:
        action = "BLOQUEAR — risco crítico, acionar time de fraude imediatamente"
    elif fraud_score >= 0.5:
        action = "REVISAR — risco moderado, solicitar confirmação ao cliente"
    else:
        action = "APROVAR — transação dentro do padrão normal"

    result = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "fraud_score": round(fraud_score, 4),
        "prediction": prediction,
        "action": action,
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

    transacao_suspeita = {
        "transaction_id": "TXN_009930",
        "customer_id": "CUST_0139",
        "amount": 3200.0,
        "distance_from_home": 120.0,
        "velocity_1h": 5,
        "velocity_24h": 12,
        "avg_amount_30d": 150.0,
        "account_balance": 887.44,
        "amount_ratio": 21.3,
        "is_night": 1,
        "high_velocity": 1,
        "is_online": 1,
        "is_credit": 1,
        "merchant_category_encoded": 5,
    }

    transacao_normal = {
        "transaction_id": "TXN_001234",
        "customer_id": "CUST_0456",
        "amount": 85.0,
        "distance_from_home": 2.0,
        "velocity_1h": 1,
        "velocity_24h": 3,
        "avg_amount_30d": 90.0,
        "account_balance": 5200.0,
        "amount_ratio": 0.94,
        "is_night": 0,
        "high_velocity": 0,
        "is_online": 0,
        "is_credit": 0,
        "merchant_category_encoded": 0,
    }

    for label, txn in [("Suspeita", transacao_suspeita), ("Normal", transacao_normal)]:
        print(f"\n=== Transação {label} ===")
        result = explain_prediction(txn, model)
        print(f"Transaction ID: {result['transaction_id']}")
        print(f"Customer ID:    {result['customer_id']}")
        print(f"Score:          {result['fraud_score']}")
        print(f"Predição:       {result['prediction']}")
        print(f"Ação sugerida:  {result['action']}")
        print(f"Interpretação:  {result['interpretation']}")
        print("Top features:")
        for f in result["top_features"]:
            print(f"  {f['feature']:30s} SHAP: {f['shap_value']:+.4f} ({f['direction']})")