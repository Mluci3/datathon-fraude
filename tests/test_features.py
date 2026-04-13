"""Testes de feature engineering."""
import pandas as pd
import pytest
import sys
sys.path.insert(0, ".")

from src.features.feature_engineering import compute_features


@pytest.fixture
def sample_raw():
    return pd.DataFrame([{
        "transaction_id": "TXN_000001",
        "customer_id": "CUST_0001",
        "amount": 150.0,
        "hour": 14,
        "day_of_week": 2,
        "merchant_category": "alimentacao",
        "channel": "presencial",
        "card_type": "debito",
        "distance_from_home": 3.0,
        "account_balance": 2000.0,
        "velocity_1h": 1,
        "velocity_24h": 3,
        "avg_amount_30d": 120.0,
        "is_new_device": 0,
        "time_since_last_txn_min": 300,
        "failed_txns_last_24h": 0,
        "ip_risk_score": 0.1,
        "is_fraud": 0,
    }])


def test_compute_features_shape(sample_raw):
    result = compute_features(sample_raw)
    assert result.shape[0] == 1
    assert "is_fraud" in result.columns


def test_compute_features_no_nulls(sample_raw):
    result = compute_features(sample_raw)
    assert result.isnull().sum().sum() == 0


def test_amount_ratio(sample_raw):
    result = compute_features(sample_raw)
    expected = round(150.0 / 120.0, 4)
    assert result.iloc[0]["amount_ratio"] == expected


def test_is_night_day(sample_raw):
    result = compute_features(sample_raw)
    assert result.iloc[0]["is_night"] == 0


def test_preserves_ids(sample_raw):
    result = compute_features(sample_raw)
    assert result.iloc[0]["transaction_id"] == "TXN_000001"
    assert result.iloc[0]["customer_id"] == "CUST_0001"