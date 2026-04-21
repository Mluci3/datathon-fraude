"""
Avaliação RAGAS do pipeline RAG — 4 métricas obrigatórias.
LLM juiz: Gemini 2.5 Flash (Google)
Embeddings: Google Generative AI Embeddings
"""
import json
import logging
import os
import time
import requests
from dotenv import load_dotenv

# Importações para o ecossistema Google
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/chat"
GOLDEN_SET_PATH = "data/golden_set/golden_set.json"

def query_agent(question: str, retries: int = 2) -> str:
    """Envia pergunta para o agente via API."""
    for attempt in range(retries):
        try:
            # Note: Verifique se sua API espera "message" ou "input"
            r = requests.post(API_URL, json={"message": question}, timeout=60)
            if r.status_code == 200:
                ans = r.json()["response"]
                if "stopped due to iteration" not in ans:
                    return ans
        except Exception as e:
            logger.warning("Tentativa %d erro: %s", attempt + 1, e)
        time.sleep(5) # Diminuído para o Gemini
    return ""

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
    for i, item in enumerate(golden_set):
        # AJUSTE DE CHAVES: De 'query' para 'question'
        question = item["question"]
        logger.info("Pergunta %d/%d: %s", i + 1, len(golden_set), question[:60])
        
        answer = query_agent(question)
        
        if answer:
            samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=item["contexts"],
                reference=item["ground_truth"], # AJUSTE DE CHAVE: ground_truth
            ))
            logger.info("✅ %d chars recebidos", len(answer))
        else:
            logger.warning("⚠️ Pergunta %d ignorada (vazia)", i + 1)
        
        # O Gemini é rápido, 2 segundos é o suficiente para não "atropelar"
        if i < len(golden_set) - 1:
            time.sleep(2)

    logger.info("%d pares válidos para avaliação", len(samples))

    # Configuração do Juiz Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", # Modelo de embedding padrão do Google
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    dataset = EvaluationDataset(samples=samples)
    
    # Adicionamos AnswerCorrectness para validar contra seu Ground Truth
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm)
    ]

    logger.info("Iniciando cálculo das métricas RAGAS com Gemini...")
    result = evaluate(dataset=dataset, metrics=metrics)

    # Arrumando os scores para o dicionário final
    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "answer_correctness": round(float(result["answer_correctness"]), 4),
        "n_samples": len(samples),
    }
    logger.info("Scores Finais: %s", scores)
    return scores

if __name__ == "__main__":
    try:
        scores = evaluate_rag()
        print("\n=== RAGAS Results (Gemini Judge) ===")
        for k, v in scores.items():
            status = "✅" if isinstance(v, (int, float)) and v >= 0.7 else "⚠️"
            print(f"{status} {k}: {v}")

        with open("evaluation/ragas_results.json", "w") as f:
            json.dump(scores, f, indent=2)
        print("\nRelatório salvo com sucesso!")
    except Exception as e:
        logger.error("Erro na execução: %s", e)