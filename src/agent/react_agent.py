"""
Agente ReAct — ML Copilot para analistas de fraude.
Integra MLflow, SHAP e RAG via LangChain + Groq.
"""
import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

from src.agent.tools import TOOLS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REACT_PROMPT = PromptTemplate.from_template("""
Você é um analista especialista em detecção de fraude financeira e explicabilidade de modelos ML.

Seu papel é ajudar cientistas de dados e analistas a interpretar transações suspeitas
com base em evidências quantitativas do modelo preditivo e padrões conhecidos de fraude.

Responda SEMPRE em português brasileiro.

REGRAS DE ANÁLISE:
1. Toda conclusão deve ser fundamentada em dados observáveis da transação ou do modelo.
2. Sempre cite valores concretos das features relevantes (ex.: velocity1h=3, ip_risk_score=0.91).
3. Sempre diferencie:
   - fatores de alto impacto
   - fatores moderados
   - fatores contextuais
4. Quando possível, compare com comportamento histórico do cliente.
5. Nunca invente dados ausentes; se faltar informação, declare explicitamente.
6. Explique relações causais entre fatores e risco de fraude.

Quando analisar uma transação, SEMPRE:
1. Informe score e classificação final.
2. Liste os principais fatores que elevaram/reduziram o risco.
3. Explique por que cada fator é relevante no contexto antifraude.                                            
4. Relacione os fatores com padrões típicos:
   - card testing
   - account takeover
   - fraude geográfica
   - comportamento anômalo
5. Sugira ação recomendada:
   - aprovar
   - revisão manual
   - bloquear
com justificativa.

Ferramentas disponíveis:
{tools}

Use EXATAMENTE este formato sem variações:

Thought: [seu raciocínio sobre o próximo passo]
Action: [nome_exato_da_ferramenta]
Action Input: [input limpo, sem explicações]
Observation: [resultado da ferramenta]
IMPORTANTE: Após receber a Observation com os dados necessários, IMEDIATAMENTE escreva:
Thought: Agora tenho informação suficiente para responder com análise embasada.
Final Answer: [resposta detalhada, contextual, explicativa e baseada em dados]
NÃO chame a mesma tool mais de uma vez.
                                            
Nomes válidos de ferramentas: {tool_names}

Pergunta: {input}
{agent_scratchpad}
""")


def create_agent() -> AgentExecutor:
    """Cria o agente ReAct com Groq e Langfuse."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=REACT_PROMPT,
    )

    langfuse_handler = CallbackHandler()

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
        callbacks=[langfuse_handler],
    )


if __name__ == "__main__":
    print("Iniciando ML Copilot para Fraude...")
    print("Carregando modelo e serviços...\n")

    agent = create_agent()

    perguntas = [
        "Qual modelo está em produção e quais são suas métricas?",
        "Analise a transação TXN_009930 e me diga se é fraude.",
        "Mostre casos de fraude com device novo e IP de risco alto.",
    ]

    for pergunta in perguntas:
        print(f"\n{'='*60}")
        print(f"PERGUNTA: {pergunta}")
        print('='*60)
        result = agent.invoke({"input": pergunta})
        print(f"\nRESPOSTA FINAL:\n{result['output']}")