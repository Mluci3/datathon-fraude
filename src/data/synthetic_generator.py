"""
Gerador de dados sintéticos para transações financeiras.
Simula um dataset de detecção de fraude com distribuição realista.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# seed fixo — garante reprodutibilidade total
SEED = 42
np.random.seed(SEED)

# configurações do dataset
N_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02  # 2% de fraude — realista para o domínio


def generate_transactions(n: int = N_TRANSACTIONS) -> pd.DataFrame:
    """
    Gera transações financeiras sintéticas com padrões de fraude realistas.

    Args:
        n: número de transações a gerar.

    Returns:
        DataFrame com transações geradas.
    """
    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    logger.info("Gerando %d transações legítimas e %d fraudulentas...", n_legit, n_fraud)

    # --- transações legítimas ---
    legit = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit)],
        "customer_id": [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_legit)],
        "amount": np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit).round(2),
        "hour": np.random.choice(range(24), size=n_legit, p=_hour_distribution()),
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(
            ["alimentacao", "transporte", "saude", "educacao", "varejo", "entretenimento"],
            size=n_legit,
            p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
        ),
        "distance_from_home": np.abs(np.random.normal(loc=5, scale=10, size=n_legit)).round(1),
        "velocity_1h": np.random.poisson(lam=1, size=n_legit),
        "velocity_24h": np.random.poisson(lam=5, size=n_legit),
        "avg_amount_30d": np.random.lognormal(mean=4.5, sigma=0.8, size=n_legit).round(2),
        "is_fraud": 0,
    })

    # --- transações fraudulentas ---
    # fraudes têm padrões distintos: valores altos, madrugada, longe de casa
    fraud = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit, n)],
        "customer_id": [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_fraud)],
        "amount": np.random.lognormal(mean=6.5, sigma=1.0, size=n_fraud).round(2),
        "hour": np.random.choice(range(24), size=n_fraud, p=_hour_distribution_fraud()),
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(
            ["alimentacao", "transporte", "saude", "educacao", "varejo", "entretenimento"],
            size=n_fraud,
            p=[0.10, 0.10, 0.05, 0.05, 0.40, 0.30],
        ),
        "distance_from_home": np.abs(np.random.normal(loc=80, scale=50, size=n_fraud)).round(1),
        "velocity_1h": np.random.poisson(lam=4, size=n_fraud),
        "velocity_24h": np.random.poisson(lam=12, size=n_fraud),
        "avg_amount_30d": np.random.lognormal(mean=4.0, sigma=0.8, size=n_fraud).round(2),
        "is_fraud": 1,
    })

    # junta e embaralha
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info("Dataset gerado: %d transações, %.1f%% fraude.", len(df), df["is_fraud"].mean() * 100)
    return df


def _hour_distribution() -> list:
    """Distribuição de horas para transações legítimas — pico comercial."""
    probs = np.zeros(24)
    probs[8:20] = 3.0   # horário comercial — alta probabilidade
    probs[20:23] = 1.5  # noite — média
    probs[0:8] = 0.2    # madrugada — baixa
    probs[23] = 0.5
    return (probs / probs.sum()).tolist()


def _hour_distribution_fraud() -> list:
    """Distribuição de horas para fraudes — concentrada na madrugada."""
    probs = np.zeros(24)
    probs[0:6] = 4.0    # madrugada — alta probabilidade de fraude
    probs[6:24] = 0.5   # resto do dia — baixa
    return (probs / probs.sum()).tolist()


def save_transactions(df: pd.DataFrame, path: str = "data/raw/transactions.csv") -> None:
    """
    Salva o dataset gerado em CSV.

    Args:
        df: DataFrame com transações.
        path: caminho de destino.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset salvo em: %s", output_path)


if __name__ == "__main__":
    df = generate_transactions()
    save_transactions(df)

    # resumo rápido
    print("\n--- Resumo do Dataset ---")
    print(f"Total de transações: {len(df):,}")
    print(f"Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Legítimas: {(df['is_fraud']==0).sum():,}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nEstatísticas de amount:")
    print(df.groupby("is_fraud")["amount"].describe().round(2))
