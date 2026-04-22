# Datathon Fase 05 — Registro Formal de Gaps do Enunciado e Decisões Arquiteturais

**Aluno:** Maria Lucilene Fernandes de Araujo  
**Turma:** 6MLET  
**Data de elaboração:** Abril de 2026  
**Versão:** v3.0 — Alinhada com implementação real (atualização final)  
**Documento:** Registro de interpretação do desafio, gaps identificados no enunciado e decisões tomadas para supri-los  
**Finalidade:** Documentar formalmente as escolhas realizadas antes e durante o desenvolvimento, com base nas informações disponíveis, para fins de contestação de avaliação e rastreabilidade de decisões.

---

## 1. Contexto e Descrição do Desafio

O Tech Challenge da Fase 05 do programa MLET (Machine Learning Engineering) da FIAP/PosTech é denominado **Datathon** e consiste em uma competição técnica baseada em um problema real fornecido por uma empresa convidada do setor financeiro. O desafio é classificado como **Projeto Integrador**, devendo demonstrar competências acumuladas ao longo das cinco fases do programa.

O material oficial disponibilizado define a estrutura do projeto em quatro etapas:

- **Etapa 1:** Pipeline de dados e modelo baseline
- **Etapa 2:** LLM serving e agente inteligente
- **Etapa 3:** Avaliação de qualidade e observabilidade
- **Etapa 4:** Segurança, governança e conformidade

A avaliação é composta por **banca mista** (empresa convidada + corpo docente), com a seguinte distribuição de pesos:

| Critério | Peso |
|---|---|
| Critérios de Negócio da Empresa | **30%** |
| LLM Serving + Agente | 15% |
| Pipeline de Dados + Baseline | 10% |
| Avaliação de Qualidade | 10% |
| Observabilidade + Monitoramento | 10% |
| Segurança + Guardrails | 10% |
| PyTorch + MLflow | 5% |
| Governança + Conformidade | 5% |
| Documentação + Arquitetura | 5% |

---

## 2. Gaps Identificados no Enunciado

### 2.1 Gap Principal: Ausência do Enunciado da Empresa Convidada

O gap mais crítico identificado é a **ausência do enunciado formal do problema real** fornecido pela empresa convidada. Esse enunciado corresponde a **30% da nota final** — o maior peso individual do Datathon — porém não foi disponibilizado formalmente aos alunos dentro do prazo necessário para planejamento arquitetural adequado.

**Impacto direto identificado:**

- Impossibilidade de definir as tools do agente ReAct com aderência ao domínio real antes do início do desenvolvimento
- Impossibilidade de construir o golden set (≥ 20 pares) com perguntas representativas do problema real
- Impossibilidade de definir critérios de negócio do LLM-as-judge com precisão
- Impossibilidade de mapear métricas técnicas para métricas de negócio conforme exigido no checklist da Etapa 1
- Impossibilidade de definir cenários de red teaming e mapeamento OWASP aderentes ao contexto real

Este gap foi formalmente comunicado aos professores responsáveis em mensagens registradas em **31/03/2026** e **09/04/2026**, solicitando orientação sobre prazo de disponibilização e possível flexibilidade na avaliação dos critérios de negócio caso o enunciado chegasse com pouco tempo antes da entrega.

### 2.2 Gap Secundário: Ausência de Dados Reais

Não houve colaboração formal da empresa para fornecimento de dados reais. Isso impacta a EDA, o feature engineering e o treino do modelo baseline, além de comprometer a autenticidade do golden set.

### 2.3 Gap Terciário: Ambiguidade sobre o Escopo do Agente

O guia técnico não especifica se o agente deve ser voltado para usuários de negócio, sistemas automatizados ou perfis técnicos intermediários. Essa ambiguidade impacta decisões de design de interface, tools e tom das respostas geradas.

### 2.4 Gap Quaternário: Ambiguidade sobre "Modernizar/Otimizar"

O desafio descreve o objetivo como "modernizar/otimizar a plataforma de ML" sem especificar se isso significa substituir regras por ML, adicionar LLM sobre sistema existente, ou construir do zero.

---

## 3. Informações Parciais Obtidas e Interpretação

Com base nas informações parcialmente disponíveis no repositório de materiais do programa e em comunicações informais com colegas, foi possível identificar que:

- A empresa convidada pertence ao **setor financeiro**
- O domínio central é **prevenção/detecção de fraude**
- O objetivo é a **modernização da plataforma de ML** para esse domínio

Essas informações, embora parciais e não formalizadas no enunciado oficial, foram utilizadas como base para as decisões arquiteturais descritas na Seção 4.

---

## 4. Decisões Tomadas para Suprir os Gaps

### 4.1 Decisão sobre Dados: Geração Sintética

**Gap suprido:** Gap 2.2

**Decisão:** Utilização exclusiva de **dados sintéticos** gerados com SDV, calibrados para representar transações financeiras com padrões de fraude realistas (~2% de taxa de fraude, 10.000 transações, seed=42).

**Justificativa:** Na ausência de dados reais, dados sintéticos são a única alternativa que garante reprodutibilidade total, conformidade com LGPD sem anonimização adicional, e controle sobre a distribuição de classes. Os dados são versionados com DVC (`enriched-v2`).

---

### 4.2 Decisão sobre o Escopo do Agente: ML Copilot para Cientistas de Dados

**Gap suprido:** Gap 2.3

**Decisão:** O agente foi desenhado como **ML Copilot para cientistas de dados** que servem ao negócio, não como sistema de decisão automatizada nem interface direta para usuários finais.

**Produto resultante:**
> Plataforma MLOps para prevenção a fraude com agente que funciona como copiloto para cientistas de dados — permitindo consultar experimentos, interpretar decisões e monitorar drift via linguagem natural.

---

### 4.3 Decisão sobre o Objetivo de Modernização: Construção do Zero

**Gap suprido:** Gap 2.4

**Decisão:** "Modernizar" foi interpretado como **construção de plataforma nova do zero**, seguindo padrões de maturidade MLOps Nível 2 do Microsoft MLOps Maturity Model.

---

### 4.4 Decisão sobre o Modelo Baseline: Champion-Challenger

**Decisão:** Implementação de dois modelos registrados no MLflow Model Registry:
- **XGBoost champion v3** — AUC 0.9997, Recall 1.0, F1 0.9877, Precision 0.9756 (Production)
- **MLP PyTorch challenger v1** — AUC 0.9903, Recall 1.0 (Staging)

**Justificativa do champion:** XGBoost foi escolhido sobre MLP por melhor equilíbrio entre Precision e Recall — o MLP atingiu Recall 1.0 mas com Precision 0.51 (F1 0.68), indicando classificação indiscriminada. A decisão de priorizar Recall sobre Precision é explícita: o custo de uma fraude não detectada supera o custo de um falso alarme.

**Limitação registrada:** retraining automatizado foi descartado por inviabilidade de prazo em projeto solo — registrado como limitação conhecida no SYSTEM_CARD.

---

### 4.5 Decisão sobre as Tools do Agente ReAct: 3 Tools Core

**Gap suprido:** Gap 2.1

**Decisão:** O agente implementa exatamente 3 tools:

| Tool | Funcionalidade | Competência coberta |
|---|---|---|
| `explain_prediction_tool` | Score de fraude + top features via `feature_importances_` XGBoost (subprocess isolado para evitar segfault M1) | Explicabilidade, MLflow |
| `query_model_registry_tool` | Consulta versões, métricas e lineage no MLflow Model Registry | Model Registry, rastreabilidade |
| `query_transactions_tool` | Busca semântica no Chroma — transações + regras de fraude + knowledge base conceitual | RAG, embedding, retrieval |

**Itens descartados com justificativa:**
- `run_experiment_tool` — risco alto de integração (subprocess de treino dentro do agente), ganho marginal na rubrica
- `drift_report_tool` — drift é coberto pelo Evidently fora do agente em `src/monitoring/drift.py`

---

### 4.6 Decisão sobre LLM Serving: Ollama → Groq → Gemini 2.5 Flash

**Gap suprido:** contexto de desenvolvimento solo sem GPU dedicada

**Histórico de migrações:**

**Tentativa 1 — Ollama + Phi-3 Mini (descartado):**
O Phi-3 Mini via Ollama entrava em loop de raciocínio ao encadear tools, travando o processo e consumindo toda a memória do M1. Inviável para o padrão ReAct com múltiplas tools.

**Tentativa 2 — Groq + LLaMA 3.3 70B Versatile (superado):**
Excelente qualidade de análise e estabilidade. Descartado porque o limite diário de 100.000 tokens esgotava rapidamente com o golden set de 25 perguntas + avaliação RAGAS, tornando inviável a Etapa 3.

**Decisão final — Google Gemini 2.5 Flash (produção):**
Migração para Gemini 2.5 Flash via `langchain-google-genai` com `create_tool_calling_agent`. Vantagens: sem limite diário de tokens, integração nativa com Google Gemini Embeddings (`models/gemini-embedding-001`) já adotados no RAG, qualidade de análise equivalente ao LLaMA 70B.

**Sobre quantização:** o Gemini 2.5 Flash aplica quantização e otimização internamente na infraestrutura Google. O controle explícito de quantização (como GGUF no Ollama) não é exposto via API — decisão documentada como trade-off aceito entre controle de infraestrutura e viabilidade operacional para projeto solo.

---

### 4.7 Decisão sobre Observabilidade: Langfuse + Evidently

**Decisão:** Stack de observabilidade composta por:
- **Langfuse** — telemetria LLMOps com traces, latência, tool calls e scores de qualidade
- **Evidently** — drift detection com PSI implementado em `src/monitoring/drift.py`

**Prometheus + Grafana descartados com justificativa:** o Langfuse já entrega telemetria e dashboard end-to-end para o componente LLM — que é o componente principal do sistema. Prometheus + Grafana adicionariam métricas operacionais HTTP (latência, throughput) que o FastAPI já expõe via logs estruturados. O custo de configuração (~2 dias) não se justifica no contexto de projeto solo com prazo de 30/04/2026. Esta decisão está registrada formalmente como item descartado com justificativa técnica.

---

### 4.8 Decisão sobre Segurança: Implementação Completa

**Decisão:** A implementação de segurança superou o escopo mínimo planejado:

**Implementado:**
- `InputGuardrail` — detecção de prompt injection via regex, limite de 4096 chars, validação de tópico
- `OutputGuardrail` com **Presidio** — remoção de PII (CPF, e-mail, telefone, conta) em todos os outputs
- `OWASP_MAPPING.md` — 5 ameaças mapeadas com cenários reais e mitigações implementadas
- `RED_TEAM_REPORT.md` — 5 cenários de adversarial testing executados e documentados

**Nota:** o guardrail de output com Presidio foi inicialmente planejado como item descartado mas foi implementado por seu valor real na conformidade LGPD e na robustez do sistema.

---

### 4.9 Decisão sobre LLM-as-Judge: 3 Critérios com Gemini

**Decisão:** Implementação de LLM-as-judge em `evaluation/llm_judge.py` com Gemini 2.5 Flash como juiz e 3 critérios:

1. **Precisão técnica** — resposta alinhada com métricas e dados reais do modelo
2. **Explicabilidade** — analista consegue entender e agir com base na resposta
3. **Conformidade LGPD** — resposta não expõe PII ou dados além do necessário

**Nota:** o juiz foi migrado de Groq (planejado originalmente) para Gemini 2.5 Flash, alinhando com o stack definitivo do projeto.

---

### 4.10 Decisão sobre o Corpus RAG: 3 Camadas

**Decisão:** O vector store Chroma foi indexado com 3 camadas de conhecimento:

| Camada | Conteúdo | Arquivo fonte |
|---|---|---|
| Transações | 10.000 transações sintéticas | `data/raw/transactions.csv` |
| Regras de fraude | 10 regras de negócio do domínio | `src/agent/rag_pipeline.py` |
| Knowledge base | 10 documentos conceituais (features, protocolos, glossário) | `data/knowledge_base/knowledge_base.json` |

**Justificativa da knowledge base:** adicionada após análise dos resultados RAGAS — o agente respondia "Informação não disponível" para perguntas conceituais (ex: `is_urgent`, `protocolo de score`) porque esse conhecimento existia apenas no golden set, não no Chroma. A indexação da knowledge base corrige o problema de faithfulness baixo.

---

### 4.11 Decisão sobre Desenvolvimento Solo e Prazo

**Contexto:** Projeto realizado individualmente com prazo de entrega em 30/04/2026.

**Princípio adotado:** entrega coerente e funcional de ponta a ponta tem mais valor avaliativo do que componentes isolados sofisticados. Todos os itens do checklist obrigatório foram cobertos. Itens classificados como diferenciais opcionais foram descartados explicitamente com justificativa registrada.

---

## 5. Itens Descartados — Decisão Final

Os itens abaixo foram avaliados e definitivamente descartados. Não serão implementados antes da entrega em 30/04/2026.

| Item | Motivo do descarte | Impacto na rubrica |
|---|---|---|
| Prometheus + Grafana | Coberto pelo Langfuse — custo de configuração não justificado no prazo | Nenhum — decisão documentada |
| Retraining automatizado | Diferencial opcional, não item do checklist | Nenhum (perda de diferencial) |
| Feature store avançada | RAG com upsert incremental via Chroma cobre GAP 03 | Nenhum |
| `run_experiment_tool` | Risco de integração alto, ganho marginal na rubrica | Nenhum |
| `drift_report_tool` no agente | Drift coberto pelo Evidently externo em `src/monitoring/drift.py` | Nenhum |
| Múltiplos ambientes (dev/staging/prod) | Inviável em projeto solo no prazo | Pequeno |
| Tuning avançado do MLP | PyTorch obrigatório — arquitetura mínima funcional é suficiente | Nenhum |

---

## 6. O Que Foi Entregue Além do Planejado

Os itens abaixo foram implementados além do escopo mínimo planejado, fortalecendo o projeto:

| Item | Descrição |
|---|---|
| OutputGuardrail com Presidio | Inicialmente descartado — implementado por valor real na conformidade LGPD |
| Knowledge base conceitual no RAG | Terceira camada do Chroma — adicionada após análise dos resultados RAGAS |
| Contexto dinâmico no `/chat` | `ContextCollectorCallback` captura outputs reais das tools para avaliação RAGAS |
| 5 cenários de red teaming | Planejado 1 cenário mínimo — entregues 5 cenários completos |
| OWASP com 5 ameaças completas | Planejado mínimo — entregue com cenários reais e mitigações implementadas |
| Documentação completa de governança | SYSTEM_CARD, MODEL_CARD, OWASP, RED_TEAM, LGPD, CHANGELOG, BENCHMARK |

---

## 7. Pendências para Entrega (até 30/04/2026)

| Item | Status | Prazo |
|---|---|---|
| `evaluation/llm_judge.py` | ⏳ Pendente | 22/04 |
| RAGAS novo run (com knowledge base) | ⏳ Pendente | 22/04 |
| `src/monitoring/drift.py` com Evidently | ⏳ Pendente | 23/04 |
| README com instruções de serviços | ⏳ Pendente | 24/04 |
| Atualizar SYSTEM_CARD seção 6.3 com métricas RAGAS reais | ⏳ Pendente | após RAGAS |
| Tabela fairness por `merchant_category` no MODEL_CARD | ⏳ Pendente | 24/04 |

---

## 8. Registro de Comunicações Formais

| Data | Destinatário | Conteúdo |
|---|---|---|
| 31/03/2026 | Professores da disciplina | Questionamento formal sobre ausência do enunciado e impacto nos 30% de critérios de negócio |
| 09/04/2026 | Professores da disciplina | Reiteração do questionamento e solicitação de orientação sobre prazo |
| Abril/2026 | Colegas e outros grupos | Alinhamento sobre estratégias adotadas na ausência do enunciado |

---

## 9. Declaração Final

Este documento registra formalmente que as decisões arquiteturais, tecnológicas e de escopo descritas acima foram tomadas com base nas informações disponíveis até a data de elaboração, considerando os gaps identificados no enunciado e as restrições de prazo e recursos de um projeto desenvolvido individualmente.

O projeto entregou todos os itens obrigatórios do checklist e superou o escopo mínimo em segurança, documentação de governança e qualidade do pipeline RAG. Os itens descartados foram avaliados individualmente com justificativa técnica e de prazo registrada neste documento.

Qualquer divergência entre este documento e os critérios de avaliação aplicados pela banca deverá considerar o contexto de informação incompleta no qual o projeto foi desenvolvido, conforme registrado nas comunicações formais da Seção 8.

---

*Versão v3.0 — Abril de 2026. Versão final alinhada com implementação real em 21/04/2026.*
