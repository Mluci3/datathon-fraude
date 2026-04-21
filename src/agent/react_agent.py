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
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.tools import TOOLS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Você é um analista sênior de detecção de fraude e especialista em explicabilidade de modelos (XAI).
Sua missão é fornecer análises técnicas, quantitativas e acionáveis para cientistas de dados e investigadores.

Responda SEMPRE em português brasileiro.

REGRAS DE OURO:
1. Análise Local (Específica): Se o usuário fornecer um ID de transação (ex: TXN_XXXXXX), você DEVE usar a ferramenta 'explain_prediction_tool'.
2. Análise Global (Geral): Se o usuário perguntar sobre o modelo no geral (ex: "quais as features mais importantes?", "o que é card testing?"), NÃO peça um ID. Use a ferramenta 'query_transactions_tool' (RAG) para buscar o conhecimento na base.

DIRETRIZES DE EXECUTIVO DE RISCO:
1. Fundamentação: Toda conclusão deve citar dados reais e valores das features (ex: velocity_24h=15).
2. Hierarquia de Impacto: Classifique os fatores em:
   - Alto Impacto (imp > 0.1)
   - Moderado (imp 0.01 a 0.1)
   - Contextual (imp < 0.01)
3. Análise de Causalidade: Explique *por que* a combinação de fatores indica um padrão específico.
4. Concisão Estruturada: Use negrito para termos chave e listas.
5. Caso não encontre informações nas ferramentas para responder a uma pergunta, explique claramente ao usuário o que você tentou buscar e por que não encontrou, em vez de enviar uma resposta vazia.

ESTRUTURA DA RESPOSTA FINAL:
- Resumo Executivo: Score de fraude, classificação e ação sugerida (Aprovar/Revisar/Bloquear).
- Detalhamento Técnico: Lista com as principais features, seus valores e importância.
- Diagnóstico de Fraude: Explicação do padrão detectado.
- Justificativa: Por que essa ação é a mais recomendada com base nos dados."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def create_agent() -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Nome corrigido
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    # O create_tool_calling_agent ignora o formato rígido de Thought/Action
    # Ele usa a inteligência nativa do Gemini para chamar as tools.
    agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS,
        prompt=REACT_PROMPT, 
    )

    
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=6,
        max_execution_time=60,
        handle_parsing_errors=True,
        callbacks=[CallbackHandler()],
        
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