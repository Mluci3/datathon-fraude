# Plano de Conformidade LGPD — datathon-fraude

**Versão do documento:** 1.0  
**Data:** Abril de 2026  
**Autora:** [Nome do Aluno] — Turma 6MLET, FIAP/PosTech  
**Repositório:** github.com/Mluci3/datathon-fraude  
**Referência legal:** Lei Geral de Proteção de Dados Pessoais — Lei nº 13.709/2018

---

## 1. Contexto e Escopo

Este documento descreve o plano de conformidade com a LGPD aplicado ao sistema `datathon-fraude` — uma plataforma de MLOps para prevenção a fraudes em transações financeiras.

O escopo cobre:
- Pipeline de dados e modelo de ML (Etapa 1)
- Agente ReAct e pipeline RAG (Etapa 2)
- Avaliação de qualidade (Etapa 3)
- Logs de observabilidade e rastreabilidade (Langfuse, MLflow)

---

## 2. Natureza dos Dados

### 2.1 Dados de Treinamento

| Atributo | Valor |
|---|---|
| **Origem** | Dados sintéticos gerados para o Datathon Fase 05 |
| **Contém PII?** | Não — dataset sintético sem informações pessoais reais |
| **Titulares de dados** | Não aplicável — não há pessoas físicas identificáveis |
| **Versionamento** | DVC — dataset `enriched-v2` |
| **Armazenamento** | DVC remote — nunca commitado no Git |

> **Conformidade nativa:** por se tratar de dados inteiramente sintéticos, o dataset de treinamento não está sujeito aos requisitos de consentimento, minimização e direitos dos titulares da LGPD. Esta é uma decisão arquitetural documentada no `DATATHON_GAPS_E_DECISOES.md`.

### 2.2 Dados em Inferência

Em um cenário de produção real, o sistema processaria dados de transações financeiras reais que podem conter informações pessoais dos titulares (nome, CPF, localização, histórico de compras). As medidas descritas neste plano devem ser aplicadas integralmente neste cenário.

### 2.3 Dados de Logs e Observabilidade

| Sistema | Dados armazenados | Contém PII? |
|---|---|---|
| Langfuse | Traces do agente, inputs/outputs das tools | Potencial — ver Seção 5 |
| MLflow | Parâmetros, métricas, artefatos de modelos | Não |
| Logs FastAPI | Timestamps, status HTTP, primeiros 100 chars da query | Potencial — ver Seção 5 |

---

## 3. Base Legal para Processamento

Em conformidade com o Art. 7º da LGPD, o processamento de dados pessoais no contexto de prevenção a fraudes financeiras se enquadra nas seguintes bases legais:

| Base Legal | Artigo LGPD | Aplicação |
|---|---|---|
| **Interesse legítimo** | Art. 7º, IX | Prevenção a fraudes é interesse legítimo da instituição financeira e do titular — protege ambos |
| **Cumprimento de obrigação legal** | Art. 7º, II | Instituições financeiras são obrigadas por regulação do Banco Central a implementar sistemas de prevenção a fraudes |
| **Proteção ao crédito** | Art. 7º, X | Análise de risco transacional para proteção do sistema de crédito |

> **Nota:** A base de interesse legítimo (Art. 7º, IX) requer que o controlador realize o Teste de Balanceamento — verificando que o interesse legítimo não prevalece indevidamente sobre os direitos e liberdades do titular. No contexto de prevenção a fraudes, o interesse do titular em ter suas transações protegidas é convergente com o interesse da instituição.

---

## 4. Princípios LGPD Aplicados

### 4.1 Finalidade (Art. 6º, I)
Os dados são processados exclusivamente para detecção e prevenção de fraudes em transações financeiras. Não há uso secundário dos dados para marketing, enriquecimento de perfil ou compartilhamento com terceiros.

### 4.2 Adequação (Art. 6º, II)
O processamento é compatível com a finalidade declarada. As features utilizadas (tempo entre transações, score de IP, velocidade, distância) são diretamente relacionadas à detecção de fraude.

### 4.3 Necessidade (Art. 6º, III)
O modelo utiliza apenas as features mínimas necessárias para a detecção de fraude — 17 features técnicas derivadas do comportamento transacional. Não há coleta de dados sensíveis (Art. 11) como origem étnica, saúde, biometria ou orientação sexual.

### 4.4 Livre Acesso (Art. 6º, IV)
Em produção real, o titular tem direito de consultar quais dados seus foram processados e qual foi o resultado da análise de risco. O endpoint `/predict` e o agente fornecem essa transparência.

### 4.5 Qualidade dos Dados (Art. 6º, V)
O dataset `enriched-v2` é versionado via DVC. Em produção real, dados desatualizados ou incorretos devem ser corrigidos antes de impactar decisões sobre o titular.

### 4.6 Transparência (Art. 6º, VI)
O sistema fornece explicabilidade das decisões via `explain_prediction_tool` — o titular pode saber quais features influenciaram o score da sua transação. O `MODEL_CARD.md` documenta as limitações e o comportamento esperado do modelo.

### 4.7 Segurança (Art. 6º, VII)
Medidas técnicas implementadas:
- `OutputGuardrail` com Presidio para remoção de PII em outputs do agente
- Guardrails de input para prevenção de extração de dados
- Logs sem dados de transação — apenas IDs anonimizados e scores

### 4.8 Prevenção (Art. 6º, VIII)
O sistema foi projetado para minimizar riscos de dano ao titular — o agente não toma decisões autônomas de bloqueio, toda ação requer revisão humana.

### 4.9 Não Discriminação (Art. 6º, IX)
A análise de fairness documentada no `MODEL_CARD.md` verifica que o modelo não discrimina por `merchant_category` como proxy de perfil socioeconômico.

### 4.10 Responsabilização (Art. 6º, X)
Este documento, o `SYSTEM_CARD.md` e o `MODEL_CARD.md` constituem a evidência de conformidade e responsabilização do controlador.

---

## 5. Tratamento de PII em Logs

### 5.1 Risco Identificado
Logs do FastAPI e traces do Langfuse podem capturar inputs do usuário que contenham IDs de transação, CPFs ou outros dados pessoais inseridos pelos analistas.

### 5.2 Mitigações Implementadas

| Camada | Mitigação |
|---|---|
| **FastAPI logs** | Limitado aos primeiros 100 caracteres da query — reduz exposição |
| **OutputGuardrail** | Presidio remove CPF, e-mail, telefone e número de conta de todos os outputs |
| **MLflow** | Armazena apenas métricas, parâmetros e artefatos — sem dados de transação |
| **Langfuse** | Traces armazenam inputs/outputs — em produção real, configurar mascaramento de PII via SDK |

### 5.3 Política de Retenção

| Sistema | Retenção | Justificativa |
|---|---|---|
| Logs FastAPI | 30 dias | Mínimo para diagnóstico operacional |
| Traces Langfuse | 30 dias | Mínimo para auditoria de decisões do agente |
| Artefatos MLflow | Indefinido | Rastreabilidade de modelos em produção |
| Dataset DVC | Enquanto modelo ativo | Necessário para reprodutibilidade |

---

## 6. Direitos dos Titulares

Em conformidade com os Arts. 17 a 22 da LGPD, os direitos dos titulares em um cenário de produção real são:

| Direito | Art. | Como atender no sistema |
|---|---|---|
| Confirmação de tratamento | Art. 18, I | Endpoint `/predict` retorna se a transação foi analisada |
| Acesso aos dados | Art. 18, II | Analista pode consultar features usadas via `explain_prediction_tool` |
| Correção de dados | Art. 18, III | Dados incorretos devem ser corrigidos no sistema de origem antes da inferência |
| Eliminação | Art. 18, VI | Logs retidos por 30 dias e então eliminados automaticamente |
| Explicação | Art. 20 | `explain_prediction_tool` fornece as features e importâncias que geraram o score |
| Revisão humana | Art. 20, § 1º | O sistema não toma decisões autônomas — toda ação requer validação humana |

---

## 7. Decisões Automatizadas (Art. 20)

O Art. 20 da LGPD garante ao titular o direito de solicitar revisão de decisões tomadas unicamente por tratamento automatizado.

**Posição do sistema:** o `datathon-fraude` é classificado como **ferramenta de apoio à decisão**, não como sistema de decisão automatizada. O agente fornece recomendações (Aprovar / Revisar / Bloquear) mas toda ação final requer validação de um analista humano. Esta arquitetura está documentada no `SYSTEM_CARD.md` Seção 8.4 e é reforçada pelo disclaimer obrigatório em toda resposta do agente.

---

## 8. Transferência Internacional de Dados

| Serviço | País | Dado transferido | Base legal |
|---|---|---|---|
| Google AI (Gemini API) | EUA | Queries do agente (podem conter IDs de transação) | Art. 33, II — país com nível adequado de proteção (decisão pendente ANPD) |
| Langfuse Cloud (`us.cloud.langfuse.com`) | EUA | Traces do agente | Art. 33, II |
| Groq API | EUA | Queries (versão anterior — não mais em produção) | — |

> **Recomendação para produção real:** avaliar uso de instâncias on-premise do Langfuse e modelos locais para eliminar transferência internacional de dados de transações reais.

---

## 9. Encarregado de Dados (DPO)

Para fins acadêmicos deste projeto, a autora assume o papel de encarregada de dados. Em produção real, a instituição financeira deve designar um DPO conforme Art. 41 da LGPD.

| Campo | Valor |
|---|---|
| **Encarregada** | [Nome do Aluno] |
| **Contato** | [e-mail] |
| **Organização** | FIAP/PosTech — 6MLET |

---

## 10. Lacunas e Recomendações para Produção Real

| Lacuna | Recomendação |
|---|---|
| Langfuse sem mascaramento de PII | Configurar `mask` no SDK do Langfuse para campos sensíveis |
| Sem RIPD (Relatório de Impacto) | Elaborar RIPD conforme Art. 38 antes de deploy em produção |
| Transferência internacional não formalizada | Firmar cláusulas contratuais padrão com Google e Langfuse |
| Sem canal formal para exercício de direitos | Implementar endpoint dedicado para requisições de titulares |
| Logs FastAPI sem mascaramento automático | Implementar middleware de anonimização de PII nos logs |

---

## 11. Histórico de Versões

| Versão | Data | Descrição |
|---|---|---|
| 1.0 | Abril/2026 | Versão inicial — entrega Datathon Fase 05 |
