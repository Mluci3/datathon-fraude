"""
RAG Pipeline — indexa corpus no Chroma e provê busca semântica.
Corpus: transações históricas + regras de fraude + relatórios.
"""
import logging
import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.HttpClient:
    """Retorna cliente Chroma conectado."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    logger.info("Chroma conectado em %s:%s", CHROMA_HOST, CHROMA_PORT)
    return client


def get_embedder() -> SentenceTransformer:
    """Carrega modelo de embedding."""
    logger.info("Carregando modelo de embedding: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def transaction_to_text(row: dict) -> str:
    """
    Converte uma transação em texto descritivo para indexação.
    Quanto mais rico o texto, melhor a busca semântica.
    """
    status = "FRAUDE CONFIRMADA" if row.get("is_fraud") == 1 else "TRANSAÇÃO LEGÍTIMA"
    device = "device NOVO" if row.get("is_new_device") == 1 else "device conhecido"
    channel = row.get("channel", "N/A")
    night = "madrugada" if row.get("is_night") == 1 else "horário comercial"

    return (
        f"{status}. "
        f"Transação {row.get('transaction_id')} do cliente {row.get('customer_id')}. "
        f"Valor: R${row.get('amount', 0):.2f}. "
        f"Distância de casa: {row.get('distance_from_home', 0):.1f}km. "
        f"Canal: {channel}. "
        f"Horário: {night}. "
        f"Device: {device}. "
        f"IP risk score: {row.get('ip_risk_score', 0):.2f}. "
        f"Tentativas falhas 24h: {row.get('failed_txns_last_24h', 0)}. "
        f"Transações na última hora: {row.get('velocity_1h', 0)}. "
        f"Tempo desde última transação: {row.get('time_since_last_txn_min', 0)} minutos."
    )


def index_transactions(
    client: chromadb.HttpClient,
    embedder: SentenceTransformer,
    path: str = "data/raw/transactions.csv",
    batch_size: int = 100,
) -> None:
    """
    Indexa transações históricas no Chroma.
    Usa upsert incremental — nunca apaga tudo.
    """
    collection = client.get_or_create_collection(
        name="transactions",
        metadata={"description": "historico de transacoes financeiras"},
    )

    df = pd.read_csv(path)
    logger.info("Indexando %d transações...", len(df))

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        texts = [transaction_to_text(row) for row in batch.to_dict("records")]
        ids = batch["transaction_id"].tolist()
        embeddings = embedder.encode(texts).tolist()
        metadatas = [
            {
                "transaction_id": str(row["transaction_id"]),
                "customer_id": str(row["customer_id"]),
                "is_fraud": int(row["is_fraud"]),
                "amount": float(row["amount"]),
                "status": "FRAUDE" if row["is_fraud"] == 1 else "LEGITIMA",
            }
            for row in batch.to_dict("records")
        ]

        # upsert incremental — não apaga dados existentes
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if (i // batch_size) % 10 == 0:
            logger.info("Indexadas %d/%d transações...", min(i + batch_size, len(df)), len(df))

    logger.info("Indexação de transações concluída! Total: %d", collection.count())


def index_fraud_rules(client: chromadb.HttpClient, embedder: SentenceTransformer) -> None:
    """Indexa regras de fraude conhecidas."""
    collection = client.get_or_create_collection(
        name="fraud_rules",
        metadata={"description": "regras e padroes de fraude conhecidos"},
    )

    rules = [
        {
            "id": "rule_001",
            "text": "Device novo combinado com transação em menos de 30 minutos após acesso é forte indicador de fraude. Dispositivos não reconhecidos são usados em 85% dos casos de fraude por roubo de credenciais.",
        },
        {
            "id": "rule_002",
            "text": "IP com score de risco acima de 0.7 indica origem suspeita. IPs de proxies, VPNs e redes TOR são frequentemente usados por fraudadores para mascarar localização real.",
        },
        {
            "id": "rule_003",
            "text": "Transações realizadas entre 00h e 06h têm probabilidade 3x maior de serem fraudes. Fraudadores operam na madrugada quando equipes de monitoramento são menores.",
        },
        {
            "id": "rule_004",
            "text": "Mais de 2 tentativas falhas nas últimas 24h antes de uma transação aprovada é padrão clássico de ataque de força bruta em cartões. Risco deve ser escalado imediatamente.",
        },
        {
            "id": "rule_005",
            "text": "Transação realizada a mais de 50km do endereço habitual do cliente sem notificação prévia de viagem é suspeita. Distância média em fraudes confirmadas é de 80km.",
        },
        {
            "id": "rule_006",
            "text": "Velocity acima de 3 transações na mesma hora é anormal. Fraudadores testam cartões clonados com transações pequenas antes de realizar compras de alto valor.",
        },
        {
            "id": "rule_007",
            "text": "Valor acima de 5x a média dos últimos 30 dias do cliente é forte sinal de fraude. Clientes legítimos raramente desviam mais de 3x do padrão habitual.",
        },
        {
            "id": "rule_008",
            "text": "Compras em categorias de entretenimento e varejo online à madrugada com device novo combinam múltiplos fatores de risco. Probabilidade de fraude superior a 90%.",
        },
        {
            "id": "rule_009",
            "text": "Transações online com cartão de crédito têm maior incidência de fraude que presenciais. O canal online representa 50% das fraudes mas apenas 30% das transações legítimas.",
        },
        {
            "id": "rule_010",
            "text": "Tempo menor que 15 minutos entre transações consecutivas em localidades diferentes é fisicamente impossível e indica fraude por clonagem de cartão ou roubo de credenciais.",
        },
    ]

    texts = [r["text"] for r in rules]
    ids = [r["id"] for r in rules]
    embeddings = embedder.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"rule_id": r["id"]} for r in rules],
    )

    logger.info("Regras de fraude indexadas: %d", collection.count())


def search_similar_transactions(
    query: str,
    client: chromadb.HttpClient,
    embedder: SentenceTransformer,
    n_results: int = 5,
    filter_fraud_only: bool = False,
) -> list:
    """
    Busca transações similares no Chroma.

    Args:
        query: descrição da transação a buscar.
        client: cliente Chroma.
        embedder: modelo de embedding.
        n_results: número de resultados.
        filter_fraud_only: se True, retorna apenas fraudes confirmadas.

    Returns:
        Lista de transações similares com metadados.
    """
    collection = client.get_collection("transactions")
    query_embedding = embedder.encode([query]).tolist()

    where = {"is_fraud": 1} if filter_fraud_only else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    similar = []
    for i in range(len(results["ids"][0])):
        similar.append({
            "transaction_id": results["metadatas"][0][i]["transaction_id"],
            "customer_id": results["metadatas"][0][i]["customer_id"],
            "status": results["metadatas"][0][i]["status"],
            "amount": results["metadatas"][0][i]["amount"],
            "similarity": round(1 - results["distances"][0][i], 3),
            "description": results["documents"][0][i][:150] + "...",
        })

    return similar


def search_fraud_rules(
    query: str,
    client: chromadb.HttpClient,
    embedder: SentenceTransformer,
    n_results: int = 3,
) -> list:
    """Busca regras de fraude relevantes para a query."""
    collection = client.get_collection("fraud_rules")
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances"],
    )

    return [
        {
            "rule": results["documents"][0][i],
            "relevance": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


if __name__ == "__main__":
    client = get_chroma_client()
    embedder = get_embedder()

    print("\n1. Indexando transações históricas...")
    index_transactions(client, embedder)

    print("\n2. Indexando regras de fraude...")
    index_fraud_rules(client, embedder)

    print("\n3. Testando busca...")
    query = "transação online madrugada device novo ip risco alto tentativas falhas"
    results = search_similar_transactions(query, client, embedder, n_results=3)

    print(f"\nTop 3 casos similares para: '{query}'")
    for r in results:
        print(f"  {r['transaction_id']} | {r['status']} | R${r['amount']:.2f} | similaridade: {r['similarity']}")

    print("\nRegras relevantes:")
    rules = search_fraud_rules(query, client, embedder)
    for r in rules:
        print(f"  [{r['relevance']}] {r['rule'][:100]}...")