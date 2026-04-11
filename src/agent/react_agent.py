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

REACT_PROMPT = PromptTemplate.from_template("""Você é um assistente especializado em detecção de fraude financeira.
Ajude analistas de dados a entender transações suspeitas, consultar modelos e buscar casos similares.

Responda sempre em português brasileiro.
Seja objetivo e direto. Cite os dados que encontrar.
Quando encontrar uma transação suspeita, sempre mencione a ação sugerida.

Ferramentas disponíveis:
{tools}

Use EXATAMENTE este formato sem variações:
Thought: [seu raciocínio]
Action: [nome_da_ferramenta]
Action Input: [input para a ferramenta]
Observation: [resultado da ferramenta]
Thought: Agora tenho informação suficiente para responder.
Final Answer: [resposta completa para o analista em português]

Nomes válidos de ferramentas: {tool_names}

Pergunta: {input}
{agent_scratchpad}""")


def create_agent() -> AgentExecutor:
    """Cria o agente ReAct com Groq e Langfuse."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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
        max_iterations=8,
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