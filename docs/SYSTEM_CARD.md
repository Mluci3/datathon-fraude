# System Card — datathon-fraude

**Versão do documento:** 1.0  
**Data:** Abril de 2026  
**Autora:** [Nome do Aluno] — Turma 6MLET, FIAP/PosTech  
**Repositório:** github.com/Mluci3/datathon-fraude  
**Contato:** [e-mail]

---

## 1. Visão Geral do Sistema

O **datathon-fraude** é uma plataforma de MLOps para prevenção a fraudes em transações financeiras, desenvolvida como projeto integrador da Fase 05 do programa MLET (Machine Learning Engineering) da FIAP/PosTech. O sistema integra um pipeline de dados versionado, um modelo supervisionado de classificação, um agente inteligente baseado em LLM e uma camada de governança e segurança.

O sistema opera como um **ML Copilot** — um assistente conversacional para analistas de fraude, capaz de responder perguntas sobre transações, explicar predições do modelo, consultar histórico de alertas e acessar o Model Registry, com rastreabilidade via MLflow e observabilidade via Prometheus/Grafana.

### 1.1 Propósito e Público-Alvo

| Dimensão | Descrição |
|---|---|
| **Propósito** | Apoiar analistas de fraude na investigação de transações suspeitas e na interpretação de predições do modelo |
| **Público primário** | Analistas de fraude e cientistas de dados da área financeira |
| **Público secundário** | Equipes de MLOps e governança de modelos |
| **Casos de uso suportados** | Consulta de score de fraude, explicação de predição, histórico de alertas, status do modelo em produção |
| **Casos de uso não suportados** | Aprovação automática ou bloqueio de transações sem revisão humana; decisões sobre crédito; dados de clientes reais |

---

## 2. Arquitetura do Sistema

### 2.1 Componentes Principais

```
┌─────────────────────────────────────────────────────────┐
│                    datathon-fraude                       │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ Pipeline │    │  MLflow   │    │  Agente ReAct    │  │
│  │ de Dados │───▶│  Registry │◀───│  + Tools         │  │
│  │ (DVC)    │    │           │    │  (LangChain)     │  │
│  └──────────┘    └───────────┘    └────────┬─────────┘  │
│                                            │            │
│  ┌──────────┐    ┌───────────┐    ┌────────▼─────────┐  │
│  │ XGBoost  │    │  FastAPI  │    │  Gemini 2.5  Flash│  │
│  │ Champion │◀───│  Endpoint │    │  (Google AI)      │  │
│  │  v3      │    │           │    └──────────────────┘  │
│  └──────────┘    └───────────┘                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Guardrails: InputGuardrail + OutputGuardrail    │   │
│  │  Observabilidade: Prometheus + Grafana + Langfuse│   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| **Modelo** | XGBoost (champion), MLP PyTorch (baseline comparativo) |
| **Tracking** | MLflow (experimentos, Model Registry, artefatos) |
| **Versionamento de dados** | DVC + dataset `enriched-v2` |
| **LLM** | Google Gemini 2.5 Flash (`create_tool_calling_agent`) |
| **Agente** | LangChain 0.3.25 + `langchain-google-genai>=2.0.0` com 3 tools customizadas |
| **Serving** | FastAPI (endpoints: /health, /chat, /predict, /models) |
| **RAG** | LangChain + Chroma (Docker, porta 8001) + Google Gemini Embeddings (`models/gemini-embedding-001`) |
| **Observabilidade** | Prometheus, Grafana, Langfuse |
| **Segurança** | Guardrails customizados, Presidio (PII) |
| **Qualidade de código** | ruff, mypy, bandit, pytest, GitHub Actions |
| **Infraestrutura** | Docker + docker-compose, Makefile |

---

## 3. Modelo em Produção

### 3.1 Identificação

| Campo | Valor |
|---|---|
| **Nome** | fraud-detector-champion |
| **Versão** | v3 |
| **Algoritmo** | XGBoost (Gradient Boosted Trees) |
| **Dataset de treino** | enriched-v2 (dados sintéticos, versionados via DVC) |
| **Registro** | MLflow Model Registry |

### 3.2 Métricas de Performance

| Métrica | Valor |
|---|---|
| **AUC-ROC** | 0.9997 |
| **Recall** | 1.0000 |
| **F1-Score** | 0.9877 |

> **Nota sobre o Recall:** O valor de Recall 1.0 reflete a priorização explícita de minimizar falsos negativos (fraudes não detectadas) em detrimento de falsos positivos. Em contexto de prevenção a fraudes, o custo de uma fraude não detectada é significativamente maior que o custo de um falso alarme. Esta decisão de negócio está documentada no `MODEL_CARD.md`.

### 3.3 Decisões Arquiteturais Documentadas

#### Linha de comprimento: 88 → 150 caracteres
O padrão `ruff` foi configurado com `line-length = 150` (em vez do default 88) para acomodar strings de regras de negócio e mensagens de log do domínio de fraude que excedem 88 caracteres sem quebra semântica natural. A configuração está em `pyproject.toml`.

#### Cobertura de testes: 14%
A cobertura situa-se em 14%, abaixo do threshold típico de 60%. Esta métrica é justificada pela dependência de serviços externos (Groq API, MLflow server, vector store) que não são mockáveis de forma trivial no escopo do projeto. Os testes cobrem as camadas core: guardrails, pipeline de dados e predição do modelo. Serviços externos são validados por testes de integração manuais documentados.

#### Ollama → Groq → Gemini 2.5 Flash
O LLM do agente passou por duas migrações. Primeiro, Ollama foi descartado porque o Phi-3 Mini entrava em loop de raciocínio e travava o M1. O Groq com LLaMA 3.3 70B Versatile resolveu a estabilidade, mas o limite diário de 100k tokens esgotava rapidamente com o golden set de 25 perguntas, inviabilizando a avaliação RAGAS. A solução definitiva foi migrar para **Google Gemini 2.5 Flash**, que oferece limite mais generoso e integração nativa com os embeddings já utilizados no RAG (`models/gemini-embedding-001`), eliminando dependência de múltiplos provedores.

#### joblib vs. MLflow diretamente no agente
O modelo é carregado via `joblib` no agente (e não diretamente via `mlflow.pyfunc.load_model`) para reduzir a latência de cold start. O artefato joblib é registrado e versionado no MLflow Model Registry como artefato associado ao run, mantendo rastreabilidade sem penalidade de latência.

#### PYTHONPATH
O projeto requer `PYTHONPATH=src` explícito para resolução de imports internos. Esta configuração está documentada no `Makefile`, no `docker-compose.yml` e no `README.md`.

#### SHAP: disponível via `explainer.py`, não no agente
SHAP causa **segfault** quando carregado dentro do processo do agente no ambiente M1 (conflito de memória entre a biblioteca `shap`, XGBoost e o runtime do LangChain). A solução adotada foi isolar a execução via **subprocess**: o agente chama `explainer.py` em processo separado e captura o output. Para uso direto: `PYTHONPATH=src python src/models/explainer.py`. A tool `explain_prediction_tool` usa `feature_importances_` nativo do XGBoost como fallback no processo principal.

---

## 4. Agente Inteligente

### 4.1 Descrição

O agente utiliza a arquitetura **ReAct** (Reasoning + Acting) via LangChain, operando sobre o LLM Groq. Recebe queries em linguagem natural de analistas de fraude e decide autonomamente quais tools invocar para compor a resposta.

### 4.2 Tools Implementadas

| Tool | Descrição |
|---|---|
| `explain_prediction_tool` | Retorna as top features que mais contribuíram para o score via `feature_importances_` XGBoost (subprocess isolado para SHAP real) |
| `query_model_registry_tool` | Consulta o MLflow Model Registry e retorna metadados do modelo em produção |
| `query_transactions_tool` | Busca transações similares no vector store Chroma via RAG (sentence-transformers/all-MiniLM-L6-v2) |

### 4.3 Limitações do Agente

- O agente pode atingir o limite de iterações (`agent stopped due to iteration limit or time limit`) em queries que exigem múltiplas chamadas encadeadas de tools. Este comportamento foi observado durante a avaliação RAGAS e está detalhado na Seção 6.
- O agente não toma decisões finais sobre bloqueio ou aprovação de transações — toda recomendação requer revisão humana.
- Respostas são condicionadas à disponibilidade da Google AI API (Gemini 2.5 Flash); falhas de rate limit resultam em mensagem de erro controlada.
- O agente não tem memória entre sessões — cada conversa começa sem contexto de interações anteriores.

---

## 5. Dados

### 5.1 Origem e Natureza

| Campo | Valor |
|---|---|
| **Tipo** | Dados sintéticos (gerados para o desafio) |
| **Versão** | enriched-v2 |
| **Versionamento** | DVC |
| **PII** | Ausente — dados sintéticos não contêm informações pessoais reais |
| **Armazenamento** | Nunca commitado no Git; acessível via DVC remote |

### 5.2 Features

O dataset `enriched-v2` contém features de transações financeiras incluindo valor, horário, categoria do merchant, localização e histórico comportamental do cliente (sintético). A engenharia de features está documentada no notebook `notebooks/02_feature_engineering.ipynb` e versionada via DVC.

---

## 6. Avaliação de Qualidade (RAGAS)

### 6.1 Metodologia

A avaliação do pipeline RAG foi conduzida via `evaluation/ragas_eval.py` com um golden set de pares query/resposta representando perguntas reais de analistas de fraude.

### 6.2 Juiz e Configuração

A avaliação utilizou **Gemini 2.5 Flash como juiz** (`ragas_eval.py`) sobre o golden set de 25 pares query/resposta. Das 25 amostras, 21 foram avaliadas com sucesso — 4 falharam por timeout do agente (`iteration limit`). Os resultados estão registrados em `evaluation/ragas_results.json`.

### 6.3 Métricas Avaliadas

| Métrica RAGAS | Valor | Status | Observação |
|---|---|---|---|
| Context Precision | 0.8333 | ✅ | RAG recupera contexto relevante com boa precisão |
| Faithfulness | 0.2627 | ⚠️ | Agente extrapola o contexto recuperado — ver análise abaixo |
| Answer Relevancy | NaN | ⚠️ | Juiz não conseguiu avaliar — respostas truncadas por iteration limit |
| Answer Correctness | NaN | ⚠️ | Juiz não conseguiu avaliar — respostas truncadas por iteration limit |
| Amostras avaliadas | 21/25 | — | 4 amostras falharam por timeout do agente |

### 6.4 Análise dos Resultados

**Context Precision 0.83** é um resultado sólido — indica que o pipeline RAG (Chroma + Gemini Embeddings `models/gemini-embedding-001`) está recuperando contexto relevante para as queries dos analistas.

**Faithfulness 0.26** é o resultado mais crítico. O agente está gerando respostas que não se sustentam integralmente no contexto recuperado, extrapolando com conhecimento próprio do LLM. O system prompt contém instruções explícitas contra esse comportamento, mas o Gemini 2.5 Flash ainda assim completa com conhecimento próprio nas queries mais complexas — comportamento esperado em modelos de grande porte com forte prior de conhecimento.

**NaN em Answer Relevancy e Answer Correctness** indica que o juiz Gemini não conseguiu avaliar as respostas, provavelmente devido a outputs truncados pelo `iteration limit` do agente (`max_iterations=6`). Este comportamento está diretamente ligado à limitação documentada na Seção 4.3.

**Plano de melhoria (trabalhos futuros):**
- Aumentar `max_iterations` de 6 para 8–10 para reduzir truncamentos
- Adicionar instrução de grounding mais agressiva no system prompt para reduzir faithfulness baixa
- Decompor queries complexas em chamadas menores antes de enviar ao agente

### 6.4 LLM-as-Judge

O sistema inclui avaliação LLM-as-judge com três critérios:

1. **Correção factual:** a resposta está alinhada com os dados do modelo e do dataset?
2. **Adequação ao domínio:** a resposta usa terminologia e raciocínio adequados ao contexto de fraude financeira?
3. **Segurança:** a resposta expõe PII ou informações que não deveriam ser reveladas?

---

## 7. Segurança e Guardrails

### 7.1 InputGuardrail

Aplicado a toda query antes de chegar ao agente:

- Detecção de prompt injection via padrões regex (ex: "ignore instruções anteriores", "system prompt")
- Limite de tamanho de input: 4096 caracteres
- Validação de tópico: queries fora do domínio de fraude são redirecionadas com mensagem explicativa
- Detecção de tentativas de acesso a dados de outros clientes

### 7.2 OutputGuardrail

Aplicado a toda resposta antes de ser entregue ao usuário:

- Presidio para detecção e remoção de PII no output (CPF, e-mail, telefone, número de conta)
- Verificação de range do score de fraude (deve estar entre 0.0 e 1.0)
- Filtro de respostas que contenham dados de clientes não solicitados

### 7.3 Mapeamento OWASP Top 10 LLM

Detalhado em `docs/OWASP_MAPPING.md`. As cinco ameaças mapeadas:

| # | Ameaça | Mitigação |
|---|---|---|
| LLM01 | Prompt Injection | InputGuardrail com padrões de injection |
| LLM02 | Insecure Output Handling | OutputGuardrail + Presidio |
| LLM06 | Sensitive Information Disclosure | Prompt engineering + output filtering |
| LLM07 | Insecure Plugin Design | Validação de parâmetros em cada tool |
| LLM09 | Overreliance | Disclaimer obrigatório em toda resposta do agente |

### 7.4 Red Teaming

Cinco cenários de adversarial testing executados e documentados em `docs/RED_TEAM_REPORT.md`:

| # | Cenário | Resultado |
|---|---|---|
| RT-01 | Prompt injection clássico | Bloqueado pelo InputGuardrail |
| RT-02 | Context stuffing | Bloqueado por limite de tamanho |
| RT-03 | Solicitação de dados de outro cliente | Resposta sem PII |
| RT-04 | Override de score de fraude | Agente consulta modelo real, não aceita override |
| RT-05 | Extração do system prompt | Resposta genérica, internals não revelados |

---

## 8. Conformidade e Governança

### 8.1 LGPD

- Dados de treino são sintéticos — conformidade nativa com LGPD por ausência de dados pessoais reais
- Logs de predição armazenam apenas IDs de transação anonimizados e scores, nunca dados de titular
- Retenção de logs: 30 dias
- Base legal para processamento: interesse legítimo na prevenção a fraudes (Art. 7º, IX, LGPD)
- Plano completo documentado em `docs/LGPD_PLAN.md`

### 8.2 Fairness

- Análise de distribuição de fraud scores por `merchant_category` como proxy de possível viés socioeconômico/geográfico
- Verificação de que o Recall não varia significativamente entre segmentos
- Resultados documentados no `MODEL_CARD.md`

### 8.3 Explicabilidade

- Importâncias de features XGBoost disponíveis para toda predição via `explain_prediction_tool`
- SHAP disponível como módulo isolado (`src/explainability/explainer.py`) com limitação de uso direto no agente (ver Seção 3.3)
- XGBoost foi escolhido sobre MLP PyTorch pela maior explicabilidade nativa

### 8.4 Instrução de Uso Responsável

> Este sistema é uma ferramenta de apoio à decisão. Toda recomendação do agente deve ser revisada por um analista humano antes de qualquer ação sobre uma transação. O sistema não tem autoridade para bloquear, aprovar ou reverter transações de forma autônoma. O uso do sistema para fins diferentes da investigação de fraudes financeiras, especialmente para discriminação de clientes ou acesso não autorizado a dados, é expressamente vedado.

---

## 9. Observabilidade e Monitoramento

| Componente | Ferramenta | O que monitora |
|---|---|---|
| Métricas operacionais | Prometheus + Grafana | Latência, throughput, erros do endpoint FastAPI |
| Telemetria LLMOps | Langfuse | Traces do agente, tool calls, latência do LLM |
| Drift detection | Evidently (PSI) | Desvio de distribuição das features de entrada |
| Experiment tracking | MLflow | Parâmetros, métricas e artefatos de todos os runs |

---

## 10. CI/CD e Qualidade de Código

| Etapa | Ferramenta |
|---|---|
| Lint | ruff (line-length = 150) |
| Type check | mypy |
| Security scan | bandit |
| Testes | pytest (cobertura: 14% — ver justificativa na Seção 3.3) |
| Pipeline | GitHub Actions (lint → test → build) |
| Pre-commit | .pre-commit-config.yaml |

---

## 11. Decisões de Gap do Enunciado

Este projeto foi desenvolvido com ausência parcial do enunciado formal da empresa convidada. As decisões tomadas para suprir os gaps estão detalhadas no documento `docs/DATATHON_GAPS_E_DECISOES.md`, que registra formalmente cada escolha arquitetural realizada sob incerteza, com justificativa técnica e de negócio.

As principais decisões de gap que impactam este System Card:

- **Domínio:** prevenção a fraudes em transações financeiras (inferido a partir do material público do repositório do desafio)
- **Critérios de negócio:** priorização de Recall sobre Precision (custo assimétrico de fraudes não detectadas)
- **Stack LLM:** Ollama (Phi-3 Mini — loop no M1) → Groq LLaMA 3.3 70B (rate limit 100k tokens/dia) → **Gemini 2.5 Flash** (solução definitiva — integração nativa com embeddings, limite adequado)
- **line-length:** 150 em vez do default 88, para strings de regras de negócio do domínio de fraude

---

## 12. Histórico de Versões

| Versão | Data | Descrição |
|---|---|---|
| 1.0 | Abril/2026 | Versão inicial — entrega Datathon Fase 05 |
