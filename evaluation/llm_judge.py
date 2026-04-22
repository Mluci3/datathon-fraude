"""
LLM-as-Judge — Avaliação de qualidade do agente ML Copilot.
Avalia 8 perguntas representativas do dia a dia de analistas de fraude.
Juiz: Gemini 2.5 Flash
3 critérios: Precisão Técnica, Explicabilidade, Conformidade LGPD

Referência: GAPS_E_DECISOES_v3.md — Seção 4.9
"""
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/chat"

# ─────────────────────────────────────────────
# 8 perguntas — 1 por categoria, foco em analista de fraude
# ─────────────────────────────────────────────

EVALUATION_CASES = [
    {
        "category": "1. Contexto da Transação",
        "question": "Analise a transação TXN_009930. Qual o score de risco atribuído pelo modelo e qual o threshold atual de decisão?",
        "expected_elements": [
            "score de fraude numérico",
            "threshold 0.5",
            "ação sugerida (BLOQUEAR/REVISAR/APROVAR)",
        ],
    },
    {
        "category": "2. Features que Influenciaram o Modelo",
        "question": "Para a transação TXN_009930, quais foram as principais features que mais contribuíram para o score de fraude? Essas features indicam comportamento anômalo ou apenas incomum?",
        "expected_elements": [
            "time_since_last_txn_min",
            "ip_risk_score",
            "importância numérica das features",
            "interpretação do comportamento",
        ],
    },
    {
        "category": "3. Localização e Mobilidade",
        "question": "Mostre casos históricos de fraude onde a distância do endereço habitual foi um fator relevante. O IP ou device dessas transações apresentava risco elevado?",
        "expected_elements": [
            "casos históricos com distância",
            "ip_risk_score ou device",
            "padrão de localização",
        ],
    },
    {
        "category": "4. Padrão Temporal",
        "question": "Existe algum padrão de fraude relacionado a transações realizadas em burst — múltiplas transações em curto intervalo de tempo? Mostre exemplos do histórico.",
        "expected_elements": [
            "velocity ou time_since_last_txn",
            "exemplos de transações",
            "padrão temporal de fraude",
        ],
    },
    {
        "category": "5. Padrão Financeiro",
        "question": "O que é card testing e como o modelo detecta transações de teste com valores pequenos antes de compras de alto valor?",
        "expected_elements": [
            "card testing definição",
            "velocity_1h ou velocity_24h",
            "valores pequenos seguidos de grandes",
        ],
    },
    {
        "category": "6. Relações e Grafos",
        "question": "Mostre casos históricos onde a combinação de device novo com IP de risco alto resultou em fraude confirmada. Esse padrão tem nome específico?",
        "expected_elements": [
            "device novo",
            "ip_risk_score alto",
            "account takeover",
            "casos históricos",
        ],
    },
    {
        "category": "7. Avaliação do Modelo",
        "question": "Qual modelo está em produção atualmente? Quais são suas métricas de desempenho e por que foi escolhido em vez do modelo challenger?",
        "expected_elements": [
            "fraud-detector-champion",
            "AUC 0.9997",
            "Recall 1.0",
            "comparação com challenger",
        ],
    },
    {
        "category": "8. Comparação com Casos Históricos",
        "question": "A transação TXN_009930 se encaixa em algum padrão conhecido de fraude como account takeover ou card testing? Mostre casos similares no histórico.",
        "expected_elements": [
            "account takeover ou card testing",
            "casos similares",
            "padrão identificado",
        ],
    },
]

# ─────────────────────────────────────────────
# CRITÉRIOS DE AVALIAÇÃO (LLM-as-Judge)
# ─────────────────────────────────────────────

JUDGE_PROMPT = """Você é um avaliador especialista em sistemas de detecção de fraude financeira e em qualidade de respostas de agentes LLM.

Avalie a resposta do agente ML Copilot segundo 3 critérios obrigatórios. Para cada critério, atribua uma nota de 0 a 10 e forneça uma justificativa objetiva.

PERGUNTA DO ANALISTA:
{question}

RESPOSTA DO AGENTE:
{response}

ELEMENTOS ESPERADOS NA RESPOSTA:
{expected_elements}

CRITÉRIOS DE AVALIAÇÃO:

1. PRECISÃO TÉCNICA (0-10)
   - A resposta contém dados corretos e verificáveis (scores, métricas, nomes de features)?
   - Os valores numéricos estão corretos e contextualizados?
   - A resposta evita inventar informações não presentes no contexto?

2. EXPLICABILIDADE (0-10)
   - Um analista de fraude consegue entender e agir com base nessa resposta?
   - A resposta conecta os dados técnicos a uma interpretação de negócio?
   - A estrutura da resposta facilita a tomada de decisão?

3. CONFORMIDADE LGPD (0-10)
   - A resposta expõe apenas os dados necessários para responder à pergunta?
   - Não há vazamento de PII (CPF, e-mail, telefone, conta) desnecessário?
   - A resposta respeita os princípios de minimização de dados?

Responda SOMENTE em JSON válido, sem texto adicional, sem markdown, sem explicações fora do JSON:
{{
  "precisao_tecnica": {{
    "nota": <0-10>,
    "justificativa": "<texto>"
  }},
  "explicabilidade": {{
    "nota": <0-10>,
    "justificativa": "<texto>"
  }},
  "conformidade_lgpd": {{
    "nota": <0-10>,
    "justificativa": "<texto>"
  }},
  "nota_media": <media das 3 notas>,
  "parecer_geral": "<texto com avaliação geral da resposta>"
}}"""


def query_agent(question: str, retries: int = 2) -> str:
    """Envia pergunta para o agente via API."""
    for attempt in range(retries):
        try:
            r = requests.post(
                API_URL,
                json={"message": question},
                timeout=120,
            )
            if r.status_code == 200:
                ans = r.json().get("response", "")
                if ans and "stopped due to iteration" not in ans:
                    return ans
            logger.warning("Tentativa %d: status %d", attempt + 1, r.status_code)
        except Exception as e:
            logger.warning("Tentativa %d erro: %s", attempt + 1, e)
        time.sleep(5)
    return ""


def judge_response(llm, question: str, response: str, expected_elements: list) -> dict:
    """Avalia uma resposta com o juiz Gemini."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        expected_elements="\n".join(f"- {e}" for e in expected_elements),
    )
    try:
        result = llm.invoke(prompt)
        text = result.content.strip()
        # Remove markdown se presente
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error("Erro no juiz: %s", e)
        return {
            "precisao_tecnica": {"nota": 0, "justificativa": f"Erro: {e}"},
            "explicabilidade": {"nota": 0, "justificativa": "Erro na avaliação"},
            "conformidade_lgpd": {"nota": 0, "justificativa": "Erro na avaliação"},
            "nota_media": 0,
            "parecer_geral": "Erro na avaliação do juiz",
        }


def run_llm_judge() -> dict:
    """Executa avaliação LLM-as-Judge para as 8 perguntas."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    results = []
    total_precisao = 0
    total_explicabilidade = 0
    total_lgpd = 0
    n_avaliados = 0

    for i, case in enumerate(EVALUATION_CASES):
        logger.info(
            "Avaliando %d/%d — %s", i + 1, len(EVALUATION_CASES), case["category"]
        )

        # 1. Obtém resposta do agente
        response = query_agent(case["question"])
        if not response:
            logger.warning("⚠️ Sem resposta para: %s", case["category"])
            results.append({
                "category": case["category"],
                "question": case["question"],
                "response": "",
                "evaluation": None,
                "status": "sem_resposta",
            })
            time.sleep(3)
            continue

        logger.info("✅ Resposta obtida (%d chars)", len(response))

        # 2. Avalia com o juiz
        evaluation = judge_response(llm, case["question"], response, case["expected_elements"])

        total_precisao += evaluation.get("precisao_tecnica", {}).get("nota", 0)
        total_explicabilidade += evaluation.get("explicabilidade", {}).get("nota", 0)
        total_lgpd += evaluation.get("conformidade_lgpd", {}).get("nota", 0)
        n_avaliados += 1

        results.append({
            "category": case["category"],
            "question": case["question"],
            "response_length": len(response),
            "evaluation": evaluation,
            "status": "avaliado",
        })

        logger.info(
            "📊 Notas — Precisão: %.1f | Explicabilidade: %.1f | LGPD: %.1f | Média: %.1f",
            evaluation.get("precisao_tecnica", {}).get("nota", 0),
            evaluation.get("explicabilidade", {}).get("nota", 0),
            evaluation.get("conformidade_lgpd", {}).get("nota", 0),
            evaluation.get("nota_media", 0),
        )

        time.sleep(4)

    # Scores finais
    scores = {
        "precisao_tecnica_media": round(total_precisao / n_avaliados, 2) if n_avaliados else 0,
        "explicabilidade_media": round(total_explicabilidade / n_avaliados, 2) if n_avaliados else 0,
        "conformidade_lgpd_media": round(total_lgpd / n_avaliados, 2) if n_avaliados else 0,
        "nota_geral_media": round((total_precisao + total_explicabilidade + total_lgpd) / (n_avaliados * 3), 2) if n_avaliados else 0,
        "n_avaliados": n_avaliados,
        "n_sem_resposta": len(EVALUATION_CASES) - n_avaliados,
        "detalhes": results,
    }

    return scores


if __name__ == "__main__":
    print("\n🤖 LLM-as-Judge — ML Copilot para Fraude")
    print("=" * 55)
    print(f"Perguntas: {len(EVALUATION_CASES)} | Critérios: 3 | Juiz: Gemini 2.5 Flash")
    print("=" * 55)

    scores = run_llm_judge()

    print("\n=== Resultados LLM-as-Judge ===")
    status_p = "✅" if scores["precisao_tecnica_media"] >= 7 else "⚠️"
    status_e = "✅" if scores["explicabilidade_media"] >= 7 else "⚠️"
    status_l = "✅" if scores["conformidade_lgpd_media"] >= 7 else "⚠️"
    status_g = "✅" if scores["nota_geral_media"] >= 7 else "⚠️"

    print(f"{status_p} Precisão Técnica:     {scores['precisao_tecnica_media']:.1f}/10")
    print(f"{status_e} Explicabilidade:      {scores['explicabilidade_media']:.1f}/10")
    print(f"{status_l} Conformidade LGPD:   {scores['conformidade_lgpd_media']:.1f}/10")
    print(f"{status_g} Nota Geral:           {scores['nota_geral_media']:.1f}/10")
    print(f"   Avaliados: {scores['n_avaliados']}/{len(EVALUATION_CASES)}")

    print("\n=== Detalhamento por Categoria ===")
    for item in scores["detalhes"]:
        if item["status"] == "avaliado" and item["evaluation"]:
            ev = item["evaluation"]
            print(f"\n{item['category']}")
            print(f"  Precisão:      {ev.get('precisao_tecnica', {}).get('nota', 0):.0f}/10")
            print(f"  Explicabilidade: {ev.get('explicabilidade', {}).get('nota', 0):.0f}/10")
            print(f"  LGPD:          {ev.get('conformidade_lgpd', {}).get('nota', 0):.0f}/10")
            print(f"  Parecer: {ev.get('parecer_geral', '')[:120]}")
        else:
            print(f"\n{item['category']} — ⚠️ sem resposta")

    # Salva resultados
    output = {
        "precisao_tecnica_media": scores["precisao_tecnica_media"],
        "explicabilidade_media": scores["explicabilidade_media"],
        "conformidade_lgpd_media": scores["conformidade_lgpd_media"],
        "nota_geral_media": scores["nota_geral_media"],
        "n_avaliados": scores["n_avaliados"],
        "detalhes": [
            {
                "category": r["category"],
                "status": r["status"],
                "evaluation": r["evaluation"],
            }
            for r in scores["detalhes"]
        ],
    }

    with open("evaluation/llm_judge_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nResultados salvos em evaluation/llm_judge_results.json")