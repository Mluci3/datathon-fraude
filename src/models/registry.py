"""
Registra os modelos treinados no MLflow Model Registry.
"""
import logging
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def register_models() -> None:
    """Registra champion v2 no Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=["1"],
        order_by=["start_time DESC"],
        max_results=10,
    )

    champion_run = next(
        (r for r in runs if r.info.run_name == "xgboost-champion-v2"), None
    )

    if not champion_run:
        logger.error("Run xgboost-champion-v2 nao encontrado!")
        return

    logger.info("Registrando XGBoost v2 como champion...")
    champion_mv = mlflow.register_model(
        model_uri=f"runs:/{champion_run.info.run_id}/model",
        name="fraud-detector-champion",
    )

    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "stage", "production")
    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "algorithm", "xgboost")
    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "dataset", "enriched-v2")
    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "auc", str(round(champion_run.data.metrics.get("auc", 0), 4)))
    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "recall", str(round(champion_run.data.metrics.get("recall", 0), 4)))
    client.set_model_version_tag("fraud-detector-champion", champion_mv.version, "f1", str(round(champion_run.data.metrics.get("f1", 0), 4)))

    print("\n=== Model Registry Atualizado ===")
    print(f"CHAMPION: fraud-detector-champion v{champion_mv.version}")
    print(f"  Stage:   production")
    print(f"  Dataset: enriched-v2")
    print(f"  AUC:     {champion_run.data.metrics.get('auc', 0):.4f}")
    print(f"  Recall:  {champion_run.data.metrics.get('recall', 0):.4f}")
    print(f"  F1:      {champion_run.data.metrics.get('f1', 0):.4f}")
    print(f"\nVisualize em: {MLFLOW_URI}/#/models")


if __name__ == "__main__":
    register_models()