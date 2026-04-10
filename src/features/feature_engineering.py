"""
Feature engineering para detecção de fraude.
Transforma dados brutos em features relevantes para o modelo.
"""
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

logger = logging.getLogger(__name__)

# schema de entrada — valida que os dados brutos estão corretos
INPUT_SCHEMA = DataFrameSchema({
    "transaction_id": Column(str),
    "customer_id": Column(str),
    "amount": Column(float, pa.Check.gt(0)),
    "hour": Column(int, pa.Check.in_range(0, 23)),
    "day_of_week": Column(int, pa.Check.in_range(0, 6)),
    "merchant_category": Column(str),
    "distance_from_home": Column(float, pa.Check.ge(0)),
    "velocity_1h": Column(int, pa.Check.ge(0)),
    "velocity_24h": Column(int, pa.Check.ge(0)),
    "avg_amount_30d": Column(float, pa.Check.gt(0)),
    "is_fraud": Column(int, pa.Check.isin([0, 1])),
})

# schema de saída — garante que as features criadas estão corretas
OUTPUT_SCHEMA = DataFrameSchema({
    "amount": Column(float, pa.Check.gt(0)),
    "distance_from_home": Column(float, pa.Check.ge(0)),
    "velocity_1h": Column(int, pa.Check.ge(0)),
    "velocity_24h": Column(int, pa.Check.ge(0)),
    "avg_amount_30d": Column(float, pa.Check.gt(0)),
    "amount_ratio": Column(float, pa.Check.ge(0)),
    "is_night": Column(int, pa.Check.isin([0, 1])),
    "high_velocity": Column(int, pa.Check.isin([0, 1])),
    "merchant_category_encoded": Column(int, pa.Check.ge(0)),
    "is_fraud": Column(int, pa.Check.isin([0, 1])),
})

# mapeamento de categorias para números
CATEGORY_MAP = {
    "alimentacao": 0,
    "transporte": 1,
    "saude": 2,
    "educacao": 3,
    "varejo": 4,
    "entretenimento": 5,
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering nos dados brutos.

    Args:
        df: DataFrame com transações brutas.

    Returns:
        DataFrame com features prontas para o modelo.
    """
    logger.info("Validando schema de entrada...")
    INPUT_SCHEMA.validate(df)

    logger.info("Calculando features...")
    result = df.copy()

    # feature 1: razão entre valor da transação e média do cliente
    # valores muito acima da média do cliente são suspeitos
    result["amount_ratio"] = (result["amount"] / result["avg_amount_30d"]).round(4)

    # feature 2: flag de madrugada (22h às 6h)
    # fraudes concentram-se na madrugada
    result["is_night"] = result["hour"].apply(
        lambda h: 1 if h >= 22 or h <= 6 else 0
    )

    # feature 3: flag de velocidade alta (mais de 3 transações na última hora)
    # múltiplas transações rápidas são padrão de fraude
    result["high_velocity"] = (result["velocity_1h"] > 3).astype(int)

    # feature 4: encoding da categoria do estabelecimento
    result["merchant_category_encoded"] = result["merchant_category"].map(CATEGORY_MAP)

    # seleciona apenas as colunas que o modelo vai usar
    feature_cols = [
        "amount",
        "distance_from_home",
        "velocity_1h",
        "velocity_24h",
        "avg_amount_30d",
        "amount_ratio",
        "is_night",
        "high_velocity",
        "merchant_category_encoded",
        "is_fraud",
    ]

    result = result[feature_cols]

    logger.info("Validando schema de saída...")
    OUTPUT_SCHEMA.validate(result)

    logger.info("Features calculadas: %d colunas, %d linhas.", len(result.columns), len(result))
    return result


def save_features(df: pd.DataFrame, path: str = "data/processed/features.csv") -> None:
    """
    Salva as features processadas.

    Args:
        df: DataFrame com features.
        path: caminho de destino.
    """
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
    print(features.head())
    print(f"\nShape: {features.shape}")
    print(f"\nCorrelação com is_fraud:")
    print(features.corr()["is_fraud"].sort_values(ascending=False).round(3))
