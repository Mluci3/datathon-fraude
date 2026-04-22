# Model Card — fraud-detector-champion

**Versão do modelo:** v3  
**Data:** Abril de 2026  
**Autora:** [Nome do Aluno] — Turma 6MLET, FIAP/PosTech  
**Repositório:** github.com/Mluci3/datathon-fraude  
**Registro:** MLflow Model Registry — `fraud-detector-champion` stage: Production

---

## 1. Descrição do Modelo

O **fraud-detector-champion** é um modelo de classificação binária treinado para detecção de transações financeiras fraudulentas. Utiliza o algoritmo **XGBoost (Gradient Boosted Trees)** e é o modelo campeão da plataforma `datathon-fraude`, promovido ao stage `Production` no MLflow Model Registry após superar os modelos challenger (incluindo MLP PyTorch baseline) em todas as métricas de avaliação.

### 1.1 Objetivo

Dado o vetor de features de uma transação financeira, o modelo retorna:
- **Fraud score:** probabilidade de fraude no intervalo [0.0, 1.0]
- **Label binário:** `1` (fraude) ou `0` (legítima), com threshold padrão de 0.5

### 1.2 Caso de Uso Pretendido

| Dimensão | Descrição |
|---|---|
| **Uso primário** | Scoring de transações em pipeline de prevenção a fraudes |
| **Usuários pretendidos** | Sistemas automatizados de alerta + analistas de fraude |
| **Contexto de uso** | Ferramenta de apoio à decisão — não substitui revisão humana |

### 1.3 Casos de Uso Fora do Escopo

- Decisões autônomas de bloqueio ou aprovação de transações sem revisão humana
- Aplicação em dados com distribuição significativamente diferente do dataset `enriched-v2`
- Avaliação de crédito ou scoring de perfil de cliente (modelo treinado exclusivamente para detecção de fraude transacional)
- Produção em ambiente com dados reais sem revisão da conformidade LGPD

---

## 2. Dados de Treinamento

| Campo | Valor |
|---|---|
| **Dataset** | enriched-v2 |
| **Tipo** | Dados sintéticos — gerados para o desafio Datathon Fase 05 |
| **Versionamento** | DVC (hash registrado no MLflow run associado ao modelo v3) |
| **PII** | Ausente — dataset sintético sem informações pessoais reais |
| **Armazenamento** | DVC remote — nunca commitado no Git |

### 2.1 Características do Dataset

O dataset `enriched-v2` é uma versão enriquecida com features de engenharia derivadas do dataset base. As principais transformações aplicadas estão documentadas em `notebooks/02_feature_engineering.ipynb`.

---

## 3. Features do Modelo

### 3.1 Feature Importance (XGBoost — gain)

| Feature | Importância | Interpretação |
|---|---|---|
| `time_since_last_txn_min` | **0.9367** | Tempo em minutos desde a última transação do cliente — feature dominante |
| `ip_risk_score` | 0.0417 | Score de risco do IP de origem da transação |
| `velocity_24h` | 0.0055 | Número de transações do cliente nas últimas 24 horas |
| `distance_from_home` | 0.0044 | Distância geográfica em relação ao endereço cadastrado do cliente |

> **Nota sobre a feature dominante:** `time_since_last_txn_min` concentra 93,67% da importância do modelo. Isso indica que o padrão temporal entre transações é o sinal mais forte de fraude no dataset `enriched-v2`. Esta concentração é esperada em dados sintéticos onde padrões temporais são gerados com separabilidade alta. Em produção com dados reais, recomenda-se monitorar a estabilidade desta feature via PSI (Population Stability Index).

### 3.2 Engenharia de Features

As features foram derivadas de dados brutos de transações. As transformações incluem:

- Cálculo de `time_since_last_txn_min` a partir do timestamp da transação e do histórico do cliente
- `ip_risk_score` derivado de lookup em base de IPs de risco conhecidos (sintética)
- `velocity_24h` calculado como contagem de transações em janela deslizante de 24h
- `distance_from_home` calculado via haversine entre coordenadas da transação e endereço cadastrado

---

## 4. Métricas de Performance

### 4.1 Resultados — fraud-detector-champion v3

| Métrica | Valor |
|---|---|
| **AUC-ROC** | 0.9997 |
| **Recall** | 1.0000 |
| **F1-Score** | 0.9877 |
| **Precision** | 0.9756 |

### 4.2 Decisão sobre o Threshold

O threshold padrão de classificação é **0.5**. A escolha de otimizar para Recall 1.0 (em vez de maximizar F1 ou Precision) é uma decisão de negócio explícita: no contexto de prevenção a fraudes, o custo de um **falso negativo** (fraude não detectada = prejuízo financeiro real) é significativamente maior que o custo de um **falso positivo** (transação legítima bloqueada = atrito com o cliente, reversível).

Esta decisão está registrada no `DATATHON_GAPS_E_DECISOES.md` como escolha de critério de negócio sob gap do enunciado.

### 4.3 Comparação com Baseline

| Modelo | AUC-ROC | Precision | Recall | F1 | Run ID | Status |
|---|---|---|---|---|---|---|
| mlp-challenger | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 9723e431e03e48c0bcdf48541ebb70b0 | Descartado |
| mlp-challenger-v2 | 0.9903 | 0.5128 | 1.0000 | 0.6780 | — | Challenger |
| xgboost-champion | 0.9870 | 0.9512 | 0.9750 | 0.9630 | 9723e431e03e48c0bcdf48541ebb70b0 | Superado |
| xgboost-champion-v2 (dataset base) | 0.9622 | 0.9737 | 0.9250 | 0.9487 | 8ba5fa1325e8437c8d2e9ad59a5b17e7 | Superado |
| xgboost-champion-v2 (enriched) | 0.9997 | 0.9756 | 1.0000 | 0.9877 | 8ba5fa1325e8437c8d2e9ad59a5b17e7 | Superado |
| **xgboost-champion-v3 (enriched-v2)** | **0.9997** | **0.9756** | **1.0000** | **0.9877** | **fe09a6fca4ef499f86ba57cda879d8fe** | **Production** |

---

## 5. Análise de Fairness

### 5.1 Metodologia

A análise de fairness foi conduzida avaliando a distribuição dos fraud scores por `merchant_category`, utilizada como proxy de possível viés socioeconômico ou geográfico (categorias de merchant podem correlacionar com perfil de renda ou região do cliente).

### 5.2 Critério de Fairness

O critério adotado é que o **Recall não deve variar significativamente entre segmentos de `merchant_category`** — ou seja, o modelo não deve ser sistematicamente pior em detectar fraudes em determinadas categorias de estabelecimento.

### 5.3 Resultados

| Segmento (`merchant_category`) | Recall | Observação |
|---|---|---|
| [categoria 1] | [valor] | [preencher após análise] |
| [categoria 2] | [valor] | [preencher após análise] |
| [categoria N] | [valor] | [preencher após análise] |

> **Nota:** Os valores desta tabela devem ser preenchidos com os resultados da análise de segmentação executada em `notebooks/04_fairness_analysis.ipynb`. O resultado geral esperado, dado o Recall global de 1.0, é de ausência de disparidade significativa entre segmentos.

---

## 6. Limitações Conhecidas

### 6.1 Limitações do Modelo

- **Dominância de uma única feature:** `time_since_last_txn_min` representa 93,67% da importância. O modelo é altamente dependente da qualidade e disponibilidade desta feature em inferência. Ausência ou atraso no cálculo desta feature degrada severamente a performance.

- **Dataset sintético:** O modelo foi treinado exclusivamente em dados sintéticos. A generalização para dados reais não foi validada e deve ser avaliada cuidadosamente antes de qualquer uso em produção real.

- **Distribuição estática:** Não há garantia de que padrões aprendidos no `enriched-v2` reflitam padrões reais de fraude, que evoluem ao longo do tempo (concept drift). O sistema de monitoramento com Evidently/PSI deve ser ativado em qualquer deploy real.

### 6.2 Limitações de Escopo

- O modelo classifica transações individualmente, sem modelagem de sequências de transações ou grafos de relacionamento entre contas.
- Não há suporte a explicações contrafactuais ("o que mudaria para esta transação ser legítima").

---

## 7. Explicabilidade

### 7.1 Feature Importance Global

Disponível via `src/models/explainer.py` usando importâncias nativas do XGBoost (gain). Os valores estão registrados como artefato no MLflow run associado ao modelo v3.

### 7.2 Explicação por Predição

A tool `explain_prediction_tool` do agente retorna as top-N features que mais contribuíram para o score de uma transação específica, usando `feature_importances_` XGBoost no processo principal.

### 7.3 SHAP

SHAP causa **segfault** quando carregado dentro do processo do agente no M1 (conflito de memória com LangChain + XGBoost). A solução é execução via **subprocess isolado**. Para uso direto:

```bash
PYTHONPATH=src python src/models/explainer.py --transaction_id <id>
```

---

## 8. Registro e Rastreabilidade

| Campo | Valor |
|---|---|
| **MLflow Experiment** | `fraud-detection` |
| **Run ID (v3 — Production)** | `fe09a6fca4ef499f86ba57cda879d8fe` |
| **Run ID (v2)** | `8ba5fa1325e8437c8d2e9ad59a5b17e7` |
| **Run ID (v1)** | `9723e431e03e48c0bcdf48541ebb70b0` |
| **Model Registry** | `fraud-detector-champion`, version 3, stage: Production |
| **Dataset versão** | enriched-v2 (hash DVC registrado no run) |
| **Artefatos registrados** | modelo joblib, feature importance chart, métricas de avaliação |

---

## 9. Instruções de Uso

### 9.1 Carregamento via MLflow

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/fraud-detector-champion/Production")
score = model.predict(transaction_df)
```

### 9.2 Carregamento via joblib (agente — baixa latência)

```python
import joblib

model = joblib.load("models/fraud_detector_champion_v3.joblib")
score = model.predict_proba(transaction_df)[:, 1]
```

### 9.3 Requisitos de Ambiente

```bash
PYTHONPATH=src  # obrigatório para resolução de imports internos
```

---

## 10. Histórico de Versões

| Versão | Data | Run ID | Descrição |
|---|---|---|---|
| v1 | Abril/2026 | `9723e431e03e48c0bcdf48541ebb70b0` | Modelos iniciais — mlp-challenger e xgboost-champion (dataset base) |
| v2 | Abril/2026 | `8ba5fa1325e8437c8d2e9ad59a5b17e7` | xgboost-champion-v2 com dataset enriquecido — AUC 0.9997 pela primeira vez |
| v3 | Abril/2026 | `fe09a6fca4ef499f86ba57cda879d8fe` | Champion — dataset enriched-v2 consolidado, Recall 1.0, promovido a Production |
