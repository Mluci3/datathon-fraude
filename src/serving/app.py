"""
FastAPI — endpoint do ML Copilot para prevenção a fraude.
Expõe o agente ReAct e predição direta via HTTP.

v2: endpoint /chat retorna 'contexts' com os outputs reais das tools
    capturados via ContextCollectorCallback — necessário para RAGAS faithfulness.
"""
import logging
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Copilot — Prevenção a Fraude",
    description="Agente ReAct para análise de fraude financeira com MLflow, SHAP e RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        import sys
        sys.path.insert(0, ".")
        from src.agent.react_agent import create_agent
        _agent = create_agent()
        logger.info("Agente inicializado.")
    return _agent


# ─────────────────────────────────────────────
# CALLBACK: captura outputs das tools para RAGAS
# ─────────────────────────────────────────────

class ContextCollectorCallback(BaseCallbackHandler):
    """
    Intercepta os outputs das tools durante a execução do agente
    e os coleta como 'contextos recuperados'.
    Necessário para avaliação correta de faithfulness no RAGAS.
    """

    def __init__(self):
        self.contexts: list[str] = []

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        if isinstance(output, str) and output.strip():
            self.contexts.append(output.strip())


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    contexts: list[str]   # outputs reais das tools — usado pelo RAGAS
    timestamp: str


class PredictRequest(BaseModel):
    transaction_id: str


class PredictResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    prediction: str
    action: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    """Verifica se a API está funcionando."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agente"])
def chat(request: ChatRequest):
    """
    Endpoint principal do agente ReAct.
    Aceita perguntas em linguagem natural sobre fraude,
    modelos e transações.

    Retorna 'contexts' com os outputs reais das tools invocadas
    durante a execução — usado pelo ragas_eval.py para avaliação
    correta de faithfulness.
    """
    try:
        logger.info("Chat request: %s", request.message[:100])
        agent = get_agent()

        # Callback coleta outputs das tools durante a execução
        collector = ContextCollectorCallback()

        result = agent.invoke(
            {"input": request.message},
            config={"callbacks": [collector]},
        )

        logger.info(
            "Contexts coletados: %d chunks | resposta: %d chars",
            len(collector.contexts),
            len(result["output"]),
        )

        return ChatResponse(
            response=result["output"],
            contexts=collector.contexts,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error("Erro no chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Modelo"])
def predict(request: PredictRequest):
    """
    Predição direta de fraude para uma transação.
    Não usa o agente — chama o modelo diretamente.
    """
    import json
    import subprocess
    import sys

    FEATURE_COLS = [
        "amount", "distance_from_home", "velocity_1h", "velocity_24h",
        "avg_amount_30d", "account_balance", "is_new_device",
        "time_since_last_txn_min", "failed_txns_last_24h", "ip_risk_score",
        "amount_ratio", "is_night", "high_velocity", "is_online",
        "is_credit", "is_urgent", "merchant_category_encoded",
    ]

    script = f"""
import joblib, pandas as pd, json, sys
model = joblib.load('models/champion_v3.joblib')
FEATURE_COLS = {FEATURE_COLS}
df = pd.read_csv('data/processed/features.csv')
row = df[df['transaction_id'] == '{request.transaction_id}']
if row.empty:
    print(json.dumps({{"error": "not_found"}}))
    sys.exit(0)
X = row.iloc[0][FEATURE_COLS]
score = float(model.predict_proba([X])[0][1])
print(json.dumps({{"score": round(score, 4)}}))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60
        )
        data = json.loads(result.stdout.strip())

        if "error" in data:
            raise HTTPException(
                status_code=404,
                detail=f"Transação {request.transaction_id} não encontrada."
            )

        score = data["score"]
        prediction = "FRAUDE" if score >= 0.5 else "LEGITIMA"
        action = (
            "BLOQUEAR" if score >= 0.8
            else "REVISAR" if score >= 0.5
            else "APROVAR"
        )

        return PredictResponse(
            transaction_id=request.transaction_id,
            fraud_score=score,
            prediction=prediction,
            action=action,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Modelo"])
def list_models():
    """Lista modelos registrados no MLflow Model Registry."""
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = mlflow.tracking.MlflowClient()

    models = []
    for name in ["fraud-detector-champion", "fraud-detector-challenger"]:
        versions = client.search_model_versions(f"name='{name}'")
        for v in versions:
            models.append({
                "name": name,
                "version": v.version,
                "stage": v.tags.get("stage", "N/A"),
                "algorithm": v.tags.get("algorithm", "N/A"),
                "auc": v.tags.get("auc", "N/A"),
                "recall": v.tags.get("recall", "N/A"),
                "f1": v.tags.get("f1", "N/A"),
            })

    return {"models": models}