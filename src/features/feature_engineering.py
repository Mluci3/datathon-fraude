"""
Feature engineering atualizado — inclui novos campos do dataset enriquecido.
"""
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

logger = logging.getLogger(__name__)

INPUT_SCHEMA = DataFrameSchema({
    "transaction_id": Column(str),
    "customer_id": Column(str),
    "amount": Column(float, pa.Check.gt(0)),
    "hour": Column(int, pa.Check.in_range(0, 23)),
    "day_of_week": Column(int, pa.Check.in_range(0, 6)),
    "merchant_category": Column(str),
    "channel": Column(str),
    "card_type": Column(str),
    "distance_from_home": Column(float, pa.Check.ge(0)),
    "account_balance": Column(float, pa.Check.ge(0)),
    "velocity_1h": Column(int, pa.Check.ge(0)),
    "velocity_24h": Column(int, pa.Check.ge(0)),
    "avg_amount_30d": Column(float, pa.Check.gt(0)),
    "is_fraud": Column(int, pa.Check.isin([0, 1])),
})

CATEGORY_MAP = {
    "alimentacao": 0, "transporte": 1, "saude": 2,
    "educacao": 3, "varejo": 4, "entretenimento": 5,
}
CHANNEL_MAP = {"presencial": 0, "online": 1, "app": 2}
CARD_MAP = {"debito": 0, "credito": 1}

# colunas de identificacao — preservadas para o agente
ID_COLS = ["transaction_id", "customer_id"]

# colunas que o modelo usa para treinar
FEATURE_COLS = [
    "amount", "distance_from_home", "velocity_1h", "velocity_24h",
    "avg_amount_30d", "account_balance", "amount_ratio", "is_night",
    "high_velocity", "is_online", "is_credit", "merchant_category_encoded",
    "is_fraud",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering nos dados brutos enriquecidos.

    Args:
        df: DataFrame com transações brutas.

    Returns:
        DataFrame com features prontas — inclui transaction_id e customer_id.
    """
    logger.info("Validando schema de entrada...")
    INPUT_SCHEMA.validate(df)

    logger.info("Calculando features...")
    result = df.copy()

    # features derivadas
    result["amount_ratio"] = (result["amount"] / result["avg_amount_30d"]).round(4)
    result["is_night"] = result["hour"].apply(lambda h: 1 if h >= 22 or h <= 6 else 0)
    result["high_velocity"] = (result["velocity_1h"] > 3).astype(int)
    result["is_online"] = result["channel"].map(CHANNEL_MAP).apply(lambda x: 1 if x == 1 else 0)
    result["is_credit"] = result["card_type"].map(CARD_MAP)
    result["merchant_category_encoded"] = result["merchant_category"].map(CATEGORY_MAP)

    # preserva IDs + features do modelo
    final_cols = ID_COLS + FEATURE_COLS
    result = result[final_cols]

    logger.info("Features calculadas: %d colunas, %d linhas.", len(result.columns), len(result))
    return result


def save_features(df: pd.DataFrame, path: str = "data/processed/features.csv") -> None:
    """Salva as features processadas."""
    from pathlib import Path
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Features salvas em: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = pd.read_csv("data/raw/transactions.csv")
    features = compute_features(raw)
    save_features(features)

    print("\n--- Features Geradas ---")
    print(f"Colunas: {list(features.columns)}")
    print(f"Shape: {features.shape}")
    print(f"\nExemplo de transação fraudulenta:")
    print(features[features["is_fraud"]==1].head(1).to_string())
    print(f"\nCorrelação com is_fraud:")
    numeric = features.drop(columns=ID_COLS)
    print(numeric.corr()["is_fraud"].sort_values(ascending=False).round(3))
