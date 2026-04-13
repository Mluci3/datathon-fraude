"""
Tools do agente ReAct para o ML Copilot de fraude.
3 tools: explain_prediction, query_model_registry, query_transactions.
"""
import logging
import os

import mlflow
from dotenv import load_dotenv
from langchain.tools import Tool

from src.agent.rag_pipeline import (
    get_chroma_client,
    get_embedder,
    search_fraud_rules,
    search_similar_transactions,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

FEATURE_COLS = [
    "amount", "distance_from_home", "velocity_1h", "velocity_24h",
    "avg_amount_30d", "account_balance", "is_new_device",
    "time_since_last_txn_min", "failed_txns_last_24h", "ip_risk_score",
    "amount_ratio", "is_night", "high_velocity", "is_online",
    "is_credit", "is_urgent", "merchant_category_encoded",
]

_chroma_client = None
_embedder = None





def _get_rag():
    global _chroma_client, _embedder
    if _chroma_client is None:
        _chroma_client = get_chroma_client()
        _embedder = get_embedder()
    return _chroma_client, _embedder


# ─────────────────────────────────────────────
# TOOL 1 — explain_prediction_tool
# ─────────────────────────────────────────────

def _explain_prediction_fn(input_str: str) -> str:
    """Explica predição rodando em subprocess separado — evita segfault no M1."""
    import json
    import subprocess
    import sys
    try:
        transaction_id = input_str.strip()

        script = f"""
import joblib, pandas as pd, json, sys
sys.path.insert(0, '.')

model = joblib.load('models/champion_v3.joblib')
FEATURE_COLS = {FEATURE_COLS}

df_raw = pd.read_csv('data/raw/transactions.csv')
df_feat = pd.read_csv('data/processed/features.csv')

raw_row = df_raw[df_raw['transaction_id'] == '{transaction_id}']
feat_row = df_feat[df_feat['transaction_id'] == '{transaction_id}']

if raw_row.empty:
    print(json.dumps({{"error": "not found"}}))
    sys.exit(0)

raw = raw_row.iloc[0].to_dict()
feat = feat_row.iloc[0].to_dict()

X = pd.DataFrame([feat])[FEATURE_COLS]
fraud_score = float(model.predict_proba(X)[0][1])
importances = model.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances.tolist()), key=lambda x: x[1], reverse=True)[:4]

result = {{
    "transaction_id": "{transaction_id}",
    "customer_id": raw.get("customer_id", "N/A"),
    "fraud_score": round(fraud_score, 4),
    "prediction": "FRAUDE" if fraud_score >= 0.5 else "LEGITIMA",
    "action": "BLOQUEAR" if fraud_score >= 0.8 else ("REVISAR" if fraud_score >= 0.5 else "APROVAR"),
    "top_features": [dict(feature=n, importance=round(v,4), value=feat.get(n,"N/A")) for n,v in feat_imp],
    "amount": raw.get("amount", 0),
    "channel": raw.get("channel", "N/A"),
    "city": raw.get("city", "N/A"),
    "timestamp": raw.get("timestamp", "N/A"),
    "is_new_device": raw.get("is_new_device", 0),
    "ip_risk_score": raw.get("ip_risk_score", 0),
    "failed_txns_last_24h": raw.get("failed_txns_last_24h", 0),
    "distance_from_home": raw.get("distance_from_home", 0),
}}
print(json.dumps(result))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return f"Erro ao analisar: {result.stderr[:200]}"

        data = json.loads(result.stdout.strip())

        if "error" in data:
            return f"Transação {transaction_id} não encontrada."

        output = (
            f"ANÁLISE DA TRANSAÇÃO {data['transaction_id']}\n"
            f"Cliente: {data['customer_id']}\n"
            f"Score de fraude: {data['fraud_score']}\n"
            f"Predição: {data['prediction']}\n"
            f"Ação sugerida: {data['action']}\n"
            f"\nPrincipais fatores:\n"
        )
        for f in data["top_features"]:
            output += f"  - {f['feature']}: importância {f['importance']} | valor: {f['value']}\n"

        output += (
            f"\nDetalhes:\n"
            f"  Valor: R${data['amount']:.2f}\n"
            f"  Canal: {data['channel']}\n"
            f"  Cidade: {data['city']}\n"
            f"  Horário: {data['timestamp']}\n"
            f"  Device novo: {'Sim' if data['is_new_device'] == 1 else 'Não'}\n"
            f"  IP risk score: {data['ip_risk_score']:.2f}\n"
            f"  Tentativas falhas 24h: {data['failed_txns_last_24h']}\n"
            f"  Dist. de casa: {data['distance_from_home']:.1f}km\n"
        )
        return output

    except Exception as e:
        logger.error("Erro em explain_prediction_fn: %s", e)
        return f"Erro ao analisar transação: {str(e)}"


# ─────────────────────────────────────────────
# TOOL 2 — query_model_registry_tool
# ─────────────────────────────────────────────

def _query_model_registry_fn(input_str: str) -> str:
    """Consulta o MLflow Model Registry."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        model_name = input_str.strip() or "fraud-detector-champion"
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            return f"Modelo '{model_name}' não encontrado no registry."

        output = f"MODELO: {model_name}\n\n"
        for v in versions:
            tags = v.tags
            output += (
                f"Versão {v.version}:\n"
                f"  Stage: {tags.get('stage', 'N/A')}\n"
                f"  Algoritmo: {tags.get('algorithm', 'N/A')}\n"
                f"  Dataset: {tags.get('dataset', 'N/A')}\n"
                f"  AUC: {tags.get('auc', 'N/A')}\n"
                f"  Recall: {tags.get('recall', 'N/A')}\n"
                f"  F1: {tags.get('f1', 'N/A')}\n"
                f"  Decisão: {tags.get('decision', 'N/A')}\n\n"
            )
        return output

    except Exception as e:
        logger.error("Erro em query_model_registry_fn: %s", e)
        return f"Erro ao consultar registry: {str(e)}"


# ─────────────────────────────────────────────
# TOOL 3 — query_transactions_tool
# ─────────────────────────────────────────────

def _query_transactions_fn(input_str: str) -> str:
    """Busca transações similares no histórico via RAG."""
    try:
        client, embedder = _get_rag()
        query = input_str.strip()

        similar = search_similar_transactions(
            query, client, embedder, n_results=5
        )
        rules = search_fraud_rules(query, client, embedder, n_results=2)

        output = f"CASOS SIMILARES para: '{query}'\n\n"

        frauds = [s for s in similar if s["status"] == "FRAUDE"]
        legit = [s for s in similar if s["status"] == "LEGITIMA"]

        if frauds:
            output += f"Fraudes confirmadas ({len(frauds)}):\n"
            for s in frauds:
                output += (
                    f"  {s['transaction_id']} | {s['customer_id']} | "
                    f"R${s['amount']:.2f} | "
                    f"similaridade: {s['similarity']}\n"
                )

        if legit:
            output += f"\nTransações legítimas similares ({len(legit)}):\n"
            for s in legit:
                output += (
                    f"  {s['transaction_id']} | {s['customer_id']} | "
                    f"R${s['amount']:.2f} | "
                    f"similaridade: {s['similarity']}\n"
                )

        if rules:
            output += "\nRegras de fraude relevantes:\n"
            for r in rules:
                output += f"  [{r['relevance']:.2f}] {r['rule'][:120]}...\n"

        return output

    except Exception as e:
        logger.error("Erro em query_transactions_fn: %s", e)
        return f"Erro ao buscar transações: {str(e)}"


# ─────────────────────────────────────────────
# INSTÂNCIAS DAS TOOLS
# ─────────────────────────────────────────────

explain_prediction_tool = Tool(
    name="explain_prediction_tool",
    func=_explain_prediction_fn,
    description=(
        "Explica a predição de fraude para uma transação específica. "
        "Use quando perguntarem sobre uma transação específica, score de fraude, "
        "por que foi bloqueada ou aprovada. "
        "Input: transaction_id (ex: TXN_009930)"
    ),
)

query_model_registry_tool = Tool(
    name="query_model_registry_tool",
    func=_query_model_registry_fn,
    description=(
        "Consulta o MLflow Model Registry para informações sobre modelos em produção. "
        "Use quando perguntarem sobre métricas do modelo, qual versão está em produção, "
        "AUC, recall, ou histórico de versões. "
        "Input: apenas o nome exato do modelo, sem texto adicional. "
        "Exemplos de input válido: fraud-detector-champion | fraud-detector-challenger"
    ),
)

query_transactions_tool = Tool(
    name="query_transactions_tool",
    func=_query_transactions_fn,
    description=(
        "Busca transações similares no histórico usando RAG semântico. "
        "Use quando o analista quiser ver casos parecidos ou precedentes históricos. "
        "Input: descrição do padrão ou características da transação"
    ),
)

TOOLS = [
    explain_prediction_tool,
    query_model_registry_tool,
    query_transactions_tool,
]