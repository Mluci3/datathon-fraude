# Benchmark — Configurações do Agente ReAct

## Objetivo
Comparar 3 configurações do agente para documentar trade-offs entre
velocidade, qualidade de resposta e custo.

## Metodologia
Pergunta padrão usada em todos os testes:
> "Analise a transação TXN_009930 e me diga se é fraude."

Métricas coletadas:
- Tempo de resposta (segundos)
- Número de tool calls
- Qualidade da análise (1-5, avaliada manualmente)

---

## Config A — LLaMA 3.1 8B Instant (baseline)

```yaml
model: llama-3.1-8b-instant
temperature: 0.0
max_iterations: 4
provider: groq
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | ~8s |
| Tool calls | 1 |
| Qualidade análise | 3/5 |
| Custo | Gratuito |

**Observação:** Resposta funcional mas superficial. Não identifica padrões de fraude por nome.

---

## Config B — LLaMA 3.3 70B Versatile (produção)

```yaml
model: llama-3.3-70b-versatile
temperature: 0.0
max_iterations: 4
provider: groq
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | ~15s |
| Tool calls | 1 |
| Qualidade análise | 5/5 |
| Custo | Gratuito |

**Observação:** Análise completa com valores concretos, identificação de padrões
(account takeover, card testing) e justificativa da ação. Modelo escolhido para produção.

---

## Config C — LLaMA 3.3 70B com temperatura 0.1

```yaml
model: llama-3.3-70b-versatile
temperature: 0.1
max_iterations: 4
provider: groq
```

| Métrica | Resultado |
|---|---|
| Tempo de resposta | ~15s |
| Tool calls | 1 |
| Qualidade análise | 4/5 |
| Custo | Gratuito |

**Observação:** Leve variação nas respostas. Mais criativo na contextualização mas
menos determinístico — indesejável para sistema financeiro.

---

## Conclusão

Config B (LLaMA 3.3 70B, temperature=0.0) foi selecionada para produção por:
- Melhor qualidade de análise
- Respostas determinísticas (temperatura zero)
- Identificação correta de padrões de fraude
- Sem custo adicional (Groq gratuito)