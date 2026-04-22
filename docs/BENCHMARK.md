# Benchmark — Configurações do Agente ReAct

## Objetivo

Documentar o processo de seleção do LLM para o agente ML Copilot, comparando 3 configurações testadas ao longo do desenvolvimento. O benchmark registra os trade-offs observados entre estabilidade, qualidade de resposta e viabilidade operacional.

## Metodologia

Pergunta padrão usada em todos os testes:
> "Analise a transação TXN_009930 e me diga se é fraude."

Métricas coletadas:
- Tempo de resposta (segundos)
- Número de tool calls realizadas
- Qualidade da análise (1–5, avaliada manualmente)
- Estabilidade operacional
- Custo

---

## Config A — Ollama + Phi-3 Mini (descartada)

```yaml
model: phi3:mini
temperature: 0.0
max_iterations: 4
provider: ollama (local, M1 Mac)
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | — (não concluía) |
| Tool calls | Loop infinito |
| Qualidade análise | 0/5 |
| Estabilidade | ❌ Inviável |
| Custo | Gratuito |

**Observação:** O Phi-3 Mini entrava em loop de raciocínio ao tentar encadear tools, travando o processo e consumindo toda a memória disponível do M1. A arquitetura ReAct com múltiplas tools exige um modelo com capacidade de raciocínio sequencial que o Phi-3 Mini não demonstrou no ambiente local. Configuração descartada na fase inicial de desenvolvimento.

---

## Config B — Groq + LLaMA 3.3 70B Versatile (superada)

```yaml
model: llama-3.3-70b-versatile
temperature: 0.0
max_iterations: 4
provider: groq
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | ~15s |
| Tool calls | 1–2 por query |
| Qualidade análise | 5/5 |
| Estabilidade | ✅ Estável |
| Custo | Gratuito (100k tokens/dia) |

**Observação:** Excelente qualidade de análise — respostas completas com valores concretos das features, identificação de padrões (account takeover, card testing) e justificativa da ação recomendada. Temperatura 0.0 garante determinismo, essencial para sistema financeiro.

**Limitação crítica:** o limite diário de 100.000 tokens do Groq esgotava rapidamente com o golden set de 25 perguntas + avaliação RAGAS, tornando inviável a execução completa da Etapa 3. Configuração superada pela migração para Gemini.

---

## Config C — Gemini 2.5 Flash (produção atual)

```yaml
model: gemini-2.5-flash
temperature: 0.0
max_iterations: 6
provider: google-ai (gemini api)
framework: create_tool_calling_agent (langchain-google-genai>=2.0.0)
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | ~20–40s |
| Tool calls | 1–3 por query |
| Qualidade análise | 5/5 |
| Estabilidade | ✅ Estável |
| Custo | Pago — ~R$0,10–0,30 por sessão de avaliação completa |

**Observação:** Qualidade de análise equivalente ao LLaMA 3.3 70B com maior capacidade de raciocínio em queries complexas que exigem encadeamento de múltiplas tools. A migração para `create_tool_calling_agent` (em vez de `create_react_agent`) resolveu problemas de parsing de output que ocorriam com o formato ReAct clássico.

**Vantagens sobre Config B:**
- Sem limite diário de tokens — viabiliza execução completa do RAGAS
- Integração nativa com Google Gemini Embeddings (`models/gemini-embedding-001`) — stack homogênea
- `max_iterations=6` com `max_execution_time=60s` oferece mais ciclos de raciocínio para queries complexas

**Limitação:** custo por requisição — em ambiente de avaliação intensiva (RAGAS com 25 perguntas), o consumo pode ser significativo. Monitorar via Google AI Studio.

---

## Conclusão

| Critério | Config A (Ollama/Phi-3) | Config B (Groq/LLaMA) | Config C (Gemini 2.5 Flash) |
|---|---|---|---|
| Estabilidade | ❌ | ✅ | ✅ |
| Qualidade análise | ❌ | ✅ | ✅ |
| Limite de tokens | — | ❌ 100k/dia | ✅ Sem limite diário |
| Integração RAG | — | ⚠️ Provider diferente | ✅ Stack homogênea |
| Custo | Gratuito | Gratuito | Pago por uso |
| **Seleção** | Descartada | Superada | **Produção** |

**Config C (Gemini 2.5 Flash, temperature=0.0)** foi selecionada para produção por:
- Estabilidade operacional comprovada
- Qualidade de análise máxima com respostas determinísticas
- Viabilidade para execução completa da avaliação RAGAS sem restrição de tokens
- Integração nativa com o stack de embeddings Google já adotado no RAG pipeline
