"""
Gerador de dados sintéticos enriquecido para transações financeiras.
Schema próximo ao mercado real — máxima explicabilidade via SHAP.
"""
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

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
DEVICES = [f"DEV_{i:06d}" for i in range(1, 5001)]


def _random_timestamps(n: int, fraud: bool = False) -> list:
    base = datetime(2024, 1, 1)
    timestamps = []
    for _ in range(n):
        days = random.randint(0, 364)
        hour = random.choices(range(24), weights=(
            _hour_distribution_fraud() if fraud else _hour_distribution()
        ))[0]
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
    Gera transações financeiras sintéticas com schema moderno.

    Inclui features de device, IP risk, velocity temporal e
    tentativas falhas — padrão de plataformas de fraude reais.

    Args:
        n: número de transações.

    Returns:
        DataFrame com transações enriquecidas.
    """
    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    logger.info("Gerando %d legítimas e %d fraudulentas...", n_legit, n_fraud)

    # --- legítimas ---
    legit_customers = [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_legit)]
    legit_devices = [f"DEV_{np.random.randint(1, 3000):06d}" for _ in range(n_legit)]

    legit = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit)],
        "customer_id": legit_customers,
        "timestamp": _random_timestamps(n_legit, fraud=False),
        "amount": np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit).round(2),
        "hour": [int(t[11:13]) for t in _random_timestamps(n_legit)],
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
            ["debito", "credito"], size=n_legit, p=[0.45, 0.55],
        ),
        "terminal_id": np.random.choice(TERMINALS, size=n_legit),
        "city": np.random.choice(CITIES, size=n_legit),
        "device_id": legit_devices,
        # clientes legítimos raramente usam device novo
        "is_new_device": np.random.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "distance_from_home": np.abs(np.random.normal(loc=5, scale=10, size=n_legit)).round(1),
        "account_balance": np.random.lognormal(mean=8.0, sigma=1.0, size=n_legit).round(2),
        "velocity_1h": np.random.poisson(lam=1, size=n_legit),
        "velocity_24h": np.random.poisson(lam=5, size=n_legit),
        "avg_amount_30d": np.random.lognormal(mean=4.5, sigma=0.8, size=n_legit).round(2),
        # tempo desde última transação — legítimo: distribuição normal (horas)
        "time_since_last_txn_min": np.abs(np.random.normal(loc=480, scale=300, size=n_legit)).round(0).astype(int),
        # tentativas falhas — legítimo: quase zero
        "failed_txns_last_24h": np.random.poisson(lam=0.1, size=n_legit),
        # IP risk score — legítimo: baixo (0.0 a 0.3)
        "ip_risk_score": np.random.beta(a=1.5, b=8, size=n_legit).round(3),
        "is_fraud": 0,
    })

    # --- fraudulentas ---
    fraud_customers = [f"CUST_{np.random.randint(1, 1000):04d}" for _ in range(n_fraud)]
    # fraudes usam devices novos ou desconhecidos
    fraud_devices = [f"DEV_{np.random.randint(3001, 5000):06d}" for _ in range(n_fraud)]

    fraud = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n_legit, n)],
        "customer_id": fraud_customers,
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
            ["debito", "credito"], size=n_fraud, p=[0.30, 0.70],
        ),
        "terminal_id": np.random.choice(TERMINALS, size=n_fraud),
        "city": np.random.choice(CITIES, size=n_fraud),
        "device_id": fraud_devices,
        # fraudes quase sempre usam device novo
        "is_new_device": np.random.choice([0, 1], size=n_fraud, p=[0.15, 0.85]),
        "distance_from_home": np.abs(np.random.normal(loc=80, scale=50, size=n_fraud)).round(1),
        "account_balance": np.random.lognormal(mean=7.0, sigma=1.2, size=n_fraud).round(2),
        "velocity_1h": np.random.poisson(lam=4, size=n_fraud),
        "velocity_24h": np.random.poisson(lam=12, size=n_fraud),
        "avg_amount_30d": np.random.lognormal(mean=4.0, sigma=0.8, size=n_fraud).round(2),
        # fraudes acontecem rapidamente após acesso — poucos minutos
        "time_since_last_txn_min": np.abs(np.random.normal(loc=15, scale=20, size=n_fraud)).round(0).astype(int),
        # fraudes têm tentativas falhas antes de conseguir
        "failed_txns_last_24h": np.random.poisson(lam=3, size=n_fraud),
        # IP risk score — fraude: alto (0.6 a 1.0)
        "ip_risk_score": np.random.beta(a=6, b=2, size=n_fraud).round(3),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info("Dataset: %d transações, %.1f%% fraude.", len(df), df["is_fraud"].mean() * 100)
    return df


def save_transactions(df: pd.DataFrame, path: str = "data/raw/transactions.csv") -> None:
    """Salva o dataset."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset salvo em: %s", output_path)


if __name__ == "__main__":
    df = generate_transactions()
    save_transactions(df)

    print("\n--- Dataset Moderno ---")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nExemplo de fraude:")
    print(df[df["is_fraud"] == 1].head(1).to_string())
    print(f"\nComparação legítimo vs fraude:")
    cols = ["is_new_device", "time_since_last_txn_min",
            "failed_txns_last_24h", "ip_risk_score", "distance_from_home"]
    print(df.groupby("is_fraud")[cols].mean().round(3))