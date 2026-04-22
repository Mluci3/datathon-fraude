"""
Avaliação RAGAS do pipeline RAG — 4 métricas obrigatórias.
LLM juiz: Gemini 2.5 Flash (Google)
Embeddings: Google Generative AI Embeddings (gemini-embedding-001)

Correções v2:
- Embedding corrigido para models/gemini-embedding-001 (era embedding-001 descontinuado)
- Contexto capturado dinamicamente via API ao invés de contexto estático do golden set
  (contexto estático curto causava faithfulness baixo — agente respondia além do contexto)
"""
import json
import logging
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/chat"
GOLDEN_SET_PATH = "data/golden_set/golden_set.json"

# ─────────────────────────────────────────────
# CORREÇÃO 1: endpoint que retorna resposta + contexto real
# ─────────────────────────────────────────────

def query_agent_with_context(question: str, retries: int = 2) -> tuple[str, list[str]]:
    """
    Envia pergunta para o agente e retorna (resposta, contextos_recuperados).
    
    O endpoint /chat deve retornar também os contextos recuperados pelas tools
    para que o RAGAS avalie faithfulness corretamente.
    
    Se a API não retornar 'contexts', usa o fallback do golden set.
    """
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json={"message": question}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                ans = data.get("response", "")
                
                if "stopped due to iteration" in ans or not ans:
                    logger.warning("Tentativa %d: iteration limit ou vazio", attempt + 1)
                    time.sleep(10)
                    continue
                
                # Pega contextos dinâmicos se a API retornar, senão retorna lista vazia
                # (fallback para contexto estático do golden set será feito no loop principal)
                contexts = data.get("contexts", [])
                return ans, contexts
                
        except Exception as e:
            logger.warning("Tentativa %d erro: %s", attempt + 1, e)
        time.sleep(10)
    
    return "", []


def evaluate_rag(golden_set_path: str = GOLDEN_SET_PATH) -> dict:
    """Avalia o pipeline RAG usando Gemini como juiz."""
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, AnswerCorrectness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    samples = []
    skipped = 0

    for i, item in enumerate(golden_set):
        question = item["question"]
        logger.info("Pergunta %d/%d: %s", i + 1, len(golden_set), question[:60])

        answer, dynamic_contexts = query_agent_with_context(question)

        if not answer:
            logger.warning("⚠️ Pergunta %d ignorada (vazia após retries)", i + 1)
            skipped += 1
            continue

        # ─────────────────────────────────────────────
        # CORREÇÃO 2: contexto dinâmico > contexto estático
        # Se a API retornou contextos reais das tools, usa eles.
        # Senão, usa o contexto estático do golden set como fallback.
        # Isso corrige o faithfulness baixo causado por contexto estático curto.
        # ─────────────────────────────────────────────
        if dynamic_contexts:
            contexts = dynamic_contexts
            logger.info("✅ Contexto dinâmico: %d chunks", len(contexts))
        else:
            contexts = item["contexts"]
            logger.info("⚠️ Usando contexto estático do golden set (API não retornou contexts)")

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=item["ground_truth"],
        ))
        logger.info("✅ %d chars recebidos", len(answer))

        # Pausa entre queries para evitar rate limit do Gemini
        if i < len(golden_set) - 1:
            time.sleep(4)

    logger.info("%d pares válidos | %d ignorados", len(samples), skipped)

    # ─────────────────────────────────────────────
    # CORREÇÃO 3: embedding correto (gemini-embedding-001, não embedding-001)
    # embedding-001 está descontinuado e retornava vetores inválidos
    # causando NaN em answer_relevancy e answer_correctness
    # ─────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",   # ← CORRIGIDO
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    logger.info("Iniciando avaliação RAGAS (Juiz Gemini 2.5 Flash)...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    logger.info("Avaliação concluída. Processando resultados...")

    df_results = result.to_pandas()
    df_results.to_csv("evaluation/ragas_detailed_results.csv", index=False)

    # Log por amostra para diagnóstico
    logger.info("\n=== Detalhamento por amostra ===")
    for _, row in df_results.iterrows():
        logger.info(
            "Q: %s | faith=%.2f | rel=%.2f | prec=%.2f | corr=%.2f",
            str(row.get("user_input", ""))[:50],
            row.get("faithfulness", float("nan")),
            row.get("answer_relevancy", float("nan")),
            row.get("context_precision", float("nan")),
            row.get("answer_correctness", float("nan")),
        )

    def safe_mean(df, col):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            n_nan = series.isna().sum()
            if n_nan > 0:
                logger.warning("⚠️ %s: %d NaN de %d amostras", col, n_nan, len(series))
            return round(series.mean(), 4)
        return float("nan")

    scores = {
        "faithfulness": safe_mean(df_results, "faithfulness"),
        "answer_relevancy": safe_mean(df_results, "answer_relevancy"),
        "context_precision": safe_mean(df_results, "context_precision"),
        "answer_correctness": safe_mean(df_results, "answer_correctness"),
        "n_samples": len(samples),
        "n_skipped": skipped,
    }

    logger.info("Scores finais: %s", scores)
    return scores


if __name__ == "__main__":
    try:
        scores = evaluate_rag()
        print("\n=== RAGAS Results (Gemini Judge) ===")
        for k, v in scores.items():
            if isinstance(v, float):
                status = "✅" if v >= 0.7 else "⚠️"
                print(f"{status} {k}: {v}")
            else:
                print(f"   {k}: {v}")

        with open("evaluation/ragas_results.json", "w") as f:
            json.dump(scores, f, indent=2)
        print("\nRelatório salvo em evaluation/ragas_results.json")
        print("Detalhamento por amostra: evaluation/ragas_detailed_results.csv")

    except Exception as e:
        logger.error("Erro na execução: %s", e)
        raise