"""
Pipeline de treinamento com MLflow tracking padronizado.
XGBoost (champion) e MLP PyTorch (challenger).
"""
import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "datathon-fraude"


def train_xgboost(df: pd.DataFrame) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": int((y == 0).sum() / (y == 1).sum()),
        "random_state": 42,
        "eval_metric": "auc",
    }

    with mlflow.start_run(run_name="xgboost-champion") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("test_size", 0.2)

        mlflow.set_tag("model_name", "xgboost-fraud-detector")
        mlflow.set_tag("model_version", "1.0.0")
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("owner", "datathon-6mlet")
        mlflow.set_tag("risk_level", "high")
        mlflow.set_tag("fairness_checked", "false")
        mlflow.set_tag("phase", "datathon-fase05")

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "auc": roc_auc_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        logger.info("XGBoost treinado — AUC: %.4f | Recall: %.4f | F1: %.4f",
                    metrics["auc"], metrics["recall"], metrics["f1"])

        return run.info.run_id


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    run_id = train_xgboost(df)
    print(f"\nTreino concluído! Run ID: {run_id}")
    print(f"Visualize em: {MLFLOW_URI}")
