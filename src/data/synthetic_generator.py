"""
Gerador de dados sintéticos enriquecido para transações financeiras.
Simula dataset de detecção de fraude com campos próximos à realidade bancária.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02

CITIES = [
    "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba",
    "Porto Alegre", "Salvador", "Fortaleza", "Recife", "Manaus", "Brasilia"
]

TERMINALS = [f"TERM_{i:04d}" for i in range(1, 501)]


def _random_timestamps(n: int, fraud: bool = False) -> list:
    """Gera timestamps realistas."""
    base = datetime(2024, 1, 1)
    timestamps = []
    for _ in range(n):
        days = random.randint(0, 364)
        if fraud:
            hour = random.choices(range(24), weights=_hour_distribution_fraud())[0]
        else:
            hour = random.choices(range(24), weights=_hour_distribution())[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = base + timedelta(days=days, hours=hour, minutes=minute, seconds=second)
        timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
    return timestamps


def _hour_distribution() -> list:
    probs = np.zeros(24)
    probs[8:20] = 3.0
    probs[20:23] = 1.5
    probs[0:8] = 0.2
    probs[23] = 0.5
    return (probs / probs.sum()).tolist()


def _hour_distribution_fraud() -> list:
    probs = np.zeros(24)
    probs[0:6] = 4.0
    probs[6:24] = 0.5
    return (probs / probs.sum()).tolist()


def generate_transactions(n: int = N_TRANSACTIONS) -> pd.DataFrame:
    """
    Gera transações financeiras sintéticas enriquecidas.

    Args:
        n: número de transações.

    Returns:
        DataFrame com transações.
    """
    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    logger.info("Gerando %d transações legítimas e %d fraudulentas...", n_legit, n_fraud)

    # --- legítimas ---
    legit_cities = np.random.choice(CITIES, size=n_legit)
    legit = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit)],
        "customer_id": [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_legit)],
        "timestamp": _random_timestamps(n_legit, fraud=False),
        "amount": np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit).round(2),
        "hour": [int(t[11:13]) for t in _random_timestamps(n_legit, fraud=False)],
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(
            ["alimentacao", "transporte", "saude", "educacao", "varejo", "entretenimento"],
            size=n_legit, p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
        ),
        "channel": np.random.choice(
            ["presencial", "online", "app"],
            size=n_legit, p=[0.50, 0.30, 0.20],
        ),
        "card_type": np.random.choice(
            ["debito", "credito"],
            size=n_legit, p=[0.45, 0.55],
        ),
        "terminal_id": np.random.choice(TERMINALS, size=n_legit),
        "city": legit_cities,
        "distance_from_home": np.abs(np.random.normal(loc=5, scale=10, size=n_legit)).round(1),
        "account_balance": np.random.lognormal(mean=8.0, sigma=1.0, size=n_legit).round(2),
        "velocity_1h": np.random.poisson(lam=1, size=n_legit),
        "velocity_24h": np.random.poisson(lam=5, size=n_legit),
        "avg_amount_30d": np.random.lognormal(mean=4.5, sigma=0.8, size=n_legit).round(2),
        "is_fraud": 0,
    })

    # --- fraudulentas ---
    fraud_cities = np.random.choice(CITIES, size=n_fraud)
    fraud = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit, n)],
        "customer_id": [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_fraud)],
        "timestamp": _random_timestamps(n_fraud, fraud=True),
        "amount": np.random.lognormal(mean=6.5, sigma=1.0, size=n_fraud).round(2),
        "hour": [int(t[11:13]) for t in _random_timestamps(n_fraud, fraud=True)],
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(
            ["alimentacao", "transporte", "saude", "educacao", "varejo", "entretenimento"],
            size=n_fraud, p=[0.10, 0.10, 0.05, 0.05, 0.40, 0.30],
        ),
        "channel": np.random.choice(
            ["presencial", "online", "app"],
            size=n_fraud, p=[0.20, 0.50, 0.30],
        ),
        "card_type": np.random.choice(
            ["debito", "credito"],
            size=n_fraud, p=[0.30, 0.70],
        ),
        "terminal_id": np.random.choice(TERMINALS, size=n_fraud),
        "city": fraud_cities,
        "distance_from_home": np.abs(np.random.normal(loc=80, scale=50, size=n_fraud)).round(1),
        "account_balance": np.random.lognormal(mean=7.0, sigma=1.2, size=n_fraud).round(2),
        "velocity_1h": np.random.poisson(lam=4, size=n_fraud),
        "velocity_24h": np.random.poisson(lam=12, size=n_fraud),
        "avg_amount_30d": np.random.lognormal(mean=4.0, sigma=0.8, size=n_fraud).round(2),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info("Dataset gerado: %d transações, %.1f%% fraude.", len(df), df["is_fraud"].mean() * 100)
    return df


def save_transactions(df: pd.DataFrame, path: str = "data/raw/transactions.csv") -> None:
    """Salva o dataset gerado."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset salvo em: %s", output_path)


if __name__ == "__main__":
    df = generate_transactions()
    save_transactions(df)

    print("\n--- Resumo do Dataset Enriquecido ---")
    print(f"Total: {len(df):,} transações")
    print(f"Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nExemplo de transação fraudulenta:")
    print(df[df['is_fraud']==1].head(1).to_string())
