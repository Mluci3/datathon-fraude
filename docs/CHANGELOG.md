# Changelog — datathon-fraude

Todas as mudanças relevantes do projeto estão documentadas aqui.  
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [1.0.0] — Abril/2026 — Entrega Datathon Fase 05

### Etapa 4 — Segurança, Governança e Conformidade

**Documentação de governança produzida:**
- `docs/SYSTEM_CARD.md` — descrição completa do sistema, decisões arquiteturais, limitações e instrução de uso responsável
- `docs/MODEL_CARD.md` — especificação do modelo champion v3, métricas, feature importance, análise de fairness e rastreabilidade de runs MLflow
- `docs/OWASP_MAPPING.md` — mapeamento das 5 ameaças OWASP Top 10 LLM com cenários reais e mitigações implementadas
- `docs/RED_TEAM_REPORT.md` — 5 cenários de adversarial testing com resultados documentados
- `docs/LGPD_PLAN.md` — plano de conformidade LGPD com os 10 princípios do Art. 6º, direitos dos titulares e recomendações para produção real

**Segurança implementada:**
- `InputGuardrail` com detecção de prompt injection via regex
- `OutputGuardrail` com Presidio para remoção de PII (CPF, e-mail, telefone, conta)
- Validação Pydantic em todas as tools do agente

---

### Etapa 3 — Avaliação de Qualidade e Observabilidade

**Adicionado:**
- Golden set com 25 pares query/resposta representando perguntas reais de analistas de fraude (`data/golden_set/golden_set.json`)
- Script de avaliação RAGAS (`evaluation/ragas_eval.py`) com Gemini 2.5 Flash como juiz
- Knowledge base conceitual indexada no Chroma (`data/knowledge_base/knowledge_base.json`) — 10 documentos com definições de features, protocolos e glossário de fraude
- Callback `ContextCollectorCallback` no FastAPI para captura dinâmica de contextos das tools — melhora avaliação de faithfulness
- Output detalhado por amostra em `evaluation/ragas_detailed_results.csv`

**Resultados RAGAS (run inicial):**
- `context_precision`: 0.8333 ✅
- `faithfulness`: 0.2627 ⚠️ — melhoria esperada após indexação da knowledge base
- `answer_relevancy`: NaN — investigação em andamento
- `answer_correctness`: NaN — investigação em andamento
- Amostras avaliadas: 21/25 (4 falharam por iteration limit do agente)

**Benchmark de configurações** (`docs/BENCHMARK.md`):
- 3 configurações do agente avaliadas em latência, qualidade e custo

---

### Etapa 2 — LLM Serving e Agente Inteligente

**Adicionado:**
- Agente ReAct com LangChain 0.3.25 + `create_tool_calling_agent` (`src/agent/react_agent.py`)
- 3 tools customizadas (`src/agent/tools.py`):
  - `explain_prediction_tool` — scoring e explicabilidade via subprocess isolado (contorna segfault M1)
  - `query_model_registry_tool` — consulta ao MLflow Model Registry
  - `query_transactions_tool` — busca semântica no Chroma (transações + regras + knowledge base)
- RAG pipeline (`src/agent/rag_pipeline.py`):
  - Vector store: Chroma (Docker, porta 8001)
  - Embeddings: Google Gemini (`models/gemini-embedding-001`)
  - Corpus: 10.000 transações + 10 regras de fraude + 10 documentos conceituais
  - Resiliência: `tenacity` com exponential backoff para rate limits
- FastAPI (`src/serving/app.py`):
  - Endpoints: `/health`, `/chat`, `/predict`, `/models`
  - Swagger UI disponível em `/docs`
  - `ContextCollectorCallback` para captura de contextos dinâmicos
- Dockerfile para containerização do serving
- CI/CD via GitHub Actions (lint → test → build) — verde ✅
- 8 testes unitários cobrindo API e features

**Decisões arquiteturais:**
- Ollama (Phi-3 Mini) → Groq LLaMA 3.3 70B → **Gemini 2.5 Flash** (migrações documentadas no SYSTEM_CARD)
- `joblib` para carregamento do modelo no agente (evita segfault M1 com mlflow.pyfunc)
- `subprocess` isolado para SHAP (evita segfault M1 com langchain + xgboost)
- `PYTHONPATH=src` obrigatório para resolução de imports internos

---

### Etapa 1 — Pipeline de Dados e Modelo Baseline

**Adicionado:**
- Estrutura inicial do projeto com `pyproject.toml`, `.pre-commit-config`, `Makefile` (`fe21900`)
- Dados sintéticos gerados com SDV, versionados via DVC (`fe21900`)
- Feature engineering com 22 features e validação de schema via Pandera (`794db2f`)
- Configuração dos serviços: MLflow (porta 5001), Chroma (porta 8001), Langfuse, Ollama (`dd698c4`)
- Pipeline de treino XGBoost com MLflow tracking (`3def97e`):
  - Run: `xgboost-champion` — AUC 0.987, Recall 0.975
- MLP PyTorch challenger com `pos_weight` para desbalanceamento (`468f211`):
  - Run: `mlp-challenger-v2` — AUC 0.990, Recall 1.0
- MLflow Model Registry configurado (`6eb7758`):
  - `fraud-detector-champion` v1 → Production
  - `fraud-detector-challenger` v1 → Staging
- SHAP explicabilidade via `src/models/explainer.py` (`07170f7`)
- Dataset enriquecido com features de device, IP risk e velocity temporal (`8f03801`, `32faf67`):
  - Run: `xgboost-champion-v2` com dataset enriched → AUC 0.9997, Recall 1.0 ✅
- RAG pipeline inicial com Chroma + 10k transações + 10 regras de fraude (`caa468e`)

**Histórico de runs MLflow:**

| Run | Algorithm | Dataset | AUC | Recall | F1 | Run ID |
|---|---|---|---|---|---|---|
| mlp-challenger | MLP PyTorch | base | 0.5000 | 0.0000 | 0.0000 | `9723e431` |
| xgboost-champion | XGBoost | base | 0.9870 | 0.9750 | 0.9630 | `9723e431` |
| mlp-challenger-v2 | MLP PyTorch | base | 0.9903 | 1.0000 | 0.6780 | — |
| xgboost-champion-v2 (base) | XGBoost | base | 0.9622 | 0.9250 | 0.9487 | `8ba5fa13` |
| xgboost-champion-v2 (enriched) | XGBoost | enriched | 0.9997 | 1.0000 | 0.9877 | `8ba5fa13` |
| **xgboost-champion-v3** | **XGBoost** | **enriched-v2** | **0.9997** | **1.0000** | **0.9877** | **`fe09a6fc`** |

---

## Pendências e Ajustes Futuros

- [ ] Rodar RAGAS novamente após indexação da knowledge base — atualizar métricas no SYSTEM_CARD seção 6.3
- [ ] Preencher tabela de fairness por `merchant_category` no MODEL_CARD
- [ ] Completar README com instruções de como subir os serviços
- [ ] Commitar todos os documentos de governança no repositório
- [ ] Commitar ajustes no `app.py` (ContextCollectorCallback) e `tools.py` (knowledge base)
- [ ] Commitar `data/knowledge_base/knowledge_base.json` e função `index_knowledge_base` no `rag_pipeline.py`
