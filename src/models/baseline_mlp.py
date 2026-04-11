"""
Modelo MLP PyTorch — challenger para detecção de fraude.
Arquitetura simples: 3 camadas lineares com ReLU.
Correção: pos_weight para lidar com desbalanceamento de classes.
"""
import logging
import os

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "datathon-fraude"


class FraudMLP(nn.Module):
    """Rede neural simples para detecção de fraude."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_mlp(df: pd.DataFrame) -> str:
    """
    Treina MLP PyTorch e loga no MLflow.

    Args:
        df: DataFrame com features prontas.

    Returns:
        run_id do experimento.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X = df.drop(columns=["is_fraud"]).values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # normaliza features — importante para redes neurais
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # converte para tensores PyTorch
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)

    # calcula peso para classe minoritária (fraude)
    # se temos 98% legítimo e 2% fraude, o peso é 98/2 = 49
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    pos_weight = torch.tensor([n_legit / n_fraud], dtype=torch.float32)
    logger.info("pos_weight para fraude: %.1f", pos_weight.item())

    params = {
        "epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_1": 64,
        "hidden_2": 32,
        "pos_weight": round(pos_weight.item(), 2),
    }

    with mlflow.start_run(run_name="mlp-challenger-v2") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])

        # tags obrigatórias
        mlflow.set_tag("model_name", "mlp-fraud-detector")
        mlflow.set_tag("model_version", "1.1.0")
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("owner", "datathon-6mlet")
        mlflow.set_tag("risk_level", "high")
        mlflow.set_tag("fairness_checked", "false")
        mlflow.set_tag("phase", "datathon-fase05")

        # treino com BCEWithLogitsLoss + pos_weight
        model = FraudMLP(input_dim=X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCELoss(weight=None)

        # aplica pos_weight manualmente via weighted loss
        def weighted_loss(output, target):
            loss = -pos_weight * target * torch.log(output + 1e-7)                    - (1 - target) * torch.log(1 - output + 1e-7)
            return loss.mean()

        model.train()
        for epoch in range(params["epochs"]):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = weighted_loss(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — Loss: %.4f", epoch+1, params["epochs"], loss.item())
                mlflow.log_metric("loss", loss.item(), step=epoch)

        # avaliação
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test_t).squeeze().numpy()
            y_pred = (y_pred_prob >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")

        logger.info("MLP v2 treinado — AUC: %.4f | Recall: %.4f | F1: %.4f",
                    metrics["auc"], metrics["recall"], metrics["f1"])

        return run.info.run_id


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    run_id = train_mlp(df)
    print(f"\nTreino concluído! Run ID: {run_id}")
    print(f"Visualize em: {MLFLOW_URI}")
