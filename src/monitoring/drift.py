"""
Drift Detection — Monitoramento de Data Drift com Evidently 0.7.x.
Compara distribuição de referência (treino) com distribuição atual (produção simulada).
Métrica principal: PSI (Population Stability Index)
Thresholds: PSI > 0.1 = warning | PSI > 0.2 = retrain trigger

Referência: GAP 06 do guia técnico Datathon Fase 05
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Features monitoradas — as mais importantes para o modelo
MONITORED_FEATURES = [
    "amount",
    "distance_from_home",
    "velocity_1h",
    "velocity_24h",
    "ip_risk_score",
    "time_since_last_txn_min",
    "failed_txns_last_24h",
    "is_new_device",
    "is_night",
    "is_urgent",
]

# Thresholds PSI
PSI_WARNING = 0.1
PSI_RETRAIN = 0.2


def load_data(
    features_path: str = "data/processed/features.csv",
    reference_frac: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega dados e divide em referência (treino) e atual (produção simulada).

    Args:
        features_path: Caminho para o dataset de features processadas.
        reference_frac: Proporção usada como referência.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (reference_df, current_df) com as features monitoradas.
    """
    df = pd.read_csv(features_path)
    logger.info("Dataset carregado: %d transações", len(df))

    # Seleciona apenas features monitoradas disponíveis
    available = [f for f in MONITORED_FEATURES if f in df.columns]
    logger.info("Features monitoradas: %d features", len(available))

    df_features = df[available].copy()

    # Divide em referência e atual sem overlap
    reference_df = df_features.sample(frac=reference_frac, random_state=random_state)
    current_df = df_features.drop(reference_df.index)

    reference_df = reference_df.reset_index(drop=True)
    current_df = current_df.reset_index(drop=True)

    logger.info("Referência: %d | Atual: %d", len(reference_df), len(current_df))
    return reference_df, current_df


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calcula o PSI (Population Stability Index) entre duas distribuições.

    PSI = sum((current% - reference%) * ln(current% / reference%))

    Args:
        reference: Série de referência (treino).
        current: Série atual (produção).
        bins: Número de bins para discretização.

    Returns:
        Valor do PSI.
    """
    epsilon = 1e-10

    if reference.nunique() <= 2:
        # Feature binária — usa proporção direta
        ref_pct = reference.mean() + epsilon
        cur_pct = current.mean() + epsilon
        psi = abs((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    else:
        # Feature contínua — usa percentis da referência como breakpoints
        breakpoints = np.percentile(reference.dropna(), np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference.dropna(), bins=breakpoints)[0]
        cur_counts = np.histogram(current.dropna(), bins=breakpoints)[0]

        ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon)
        cur_pct = (cur_counts + epsilon) / (len(current) + epsilon)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    return round(abs(psi), 4)


def run_drift_detection(
    features_path: str = "data/processed/features.csv",
    output_dir: str = "evaluation",
    log_to_mlflow: bool = True,
) -> dict:
    """
    Executa detecção de drift com Evidently 0.7.x e PSI por feature.

    Args:
        features_path: Caminho para o dataset de features.
        output_dir: Diretório para salvar relatórios.
        log_to_mlflow: Se True, loga métricas no MLflow.

    Returns:
        Dicionário com métricas de drift.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carrega dados
    reference_df, current_df = load_data(features_path)

    # ─────────────────────────────────────────────
    # Relatório Evidently 0.7.x
    # ─────────────────────────────────────────────
    logger.info("Gerando relatório Evidently...")

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    # Salva HTML
    html_path = f"{output_dir}/drift_report_{timestamp}.html"
    snapshot.save_html(html_path)
    logger.info("Relatório salvo: %s", html_path)

    # Extrai share of drifted columns
    drift_dict = snapshot.dict()
    share_drifted = 0.0
    for metric in drift_dict.get("metrics", []):
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            share_drifted = metric.get("value", {}).get("share", 0.0)
            break

    logger.info("Share of drifted columns: %.1f%%", share_drifted * 100)

    # ─────────────────────────────────────────────
    # PSI por feature
    # ─────────────────────────────────────────────
    logger.info("Calculando PSI por feature...")

    psi_results = {}
    for feature in reference_df.columns:
        try:
            psi = calculate_psi(reference_df[feature], current_df[feature])
            psi_results[feature] = psi

            if psi > PSI_RETRAIN:
                status = "🔴 RETRAIN"
            elif psi > PSI_WARNING:
                status = "🟡 WARNING"
            else:
                status = "🟢 STABLE"

            logger.info("  %s | PSI: %.4f | %s", feature.ljust(30), psi, status)

        except Exception as e:
            logger.warning("Erro calculando PSI para %s: %s", feature, e)
            psi_results[feature] = None

    # ─────────────────────────────────────────────
    # Resultado consolidado
    # ─────────────────────────────────────────────
    valid_psi = [v for v in psi_results.values() if v is not None]
    max_psi = max(valid_psi) if valid_psi else 0.0
    avg_psi = round(sum(valid_psi) / len(valid_psi), 4) if valid_psi else 0.0

    n_stable = sum(1 for v in valid_psi if v <= PSI_WARNING)
    n_warning = sum(1 for v in valid_psi if PSI_WARNING < v <= PSI_RETRAIN)
    n_retrain = sum(1 for v in valid_psi if v > PSI_RETRAIN)

    results = {
        "timestamp": datetime.now().isoformat(),
        "reference_size": len(reference_df),
        "current_size": len(current_df),
        "share_drifted_evidently": round(share_drifted, 4),
        "psi_max": max_psi,
        "psi_avg": avg_psi,
        "n_features_stable": n_stable,
        "n_features_warning": n_warning,
        "n_features_retrain_trigger": n_retrain,
        "psi_by_feature": psi_results,
        "html_report": html_path,
        "recommendation": (
            "RETRAIN TRIGGER — degradação significativa detectada"
            if max_psi > PSI_RETRAIN
            else "WARNING — monitorar nas próximas execuções"
            if max_psi > PSI_WARNING
            else "STABLE — distribuição dentro dos limites esperados"
        ),
    }

    # ─────────────────────────────────────────────
    # Log no MLflow
    # ─────────────────────────────────────────────
    if log_to_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("datathon-fraude")
            with mlflow.start_run(run_name=f"drift_detection_{timestamp}"):
                mlflow.set_tag("type", "drift_monitoring")
                mlflow.set_tag("dataset", "enriched-v2")
                mlflow.log_metric("psi_max", max_psi)
                mlflow.log_metric("psi_avg", avg_psi)
                mlflow.log_metric("share_drifted", share_drifted)
                mlflow.log_metric("n_features_stable", n_stable)
                mlflow.log_metric("n_features_warning", n_warning)
                mlflow.log_metric("n_features_retrain_trigger", n_retrain)
                for feature, psi in psi_results.items():
                    if psi is not None:
                        mlflow.log_metric(f"psi_{feature}", psi)
                mlflow.log_artifact(html_path)
                logger.info("Métricas logadas no MLflow")
        except Exception as e:
            logger.warning("MLflow indisponível: %s", e)

    return results


if __name__ == "__main__":
    print("\n📊 Drift Detection — datathon-fraude")
    print("=" * 55)
    print(f"Thresholds: PSI > {PSI_WARNING} = WARNING | PSI > {PSI_RETRAIN} = RETRAIN")
    print("=" * 55)

    results = run_drift_detection()

    print("\n=== Resultados de Drift ===")
    print(f"Referência: {results['reference_size']} amostras")
    print(f"Atual:      {results['current_size']} amostras")
    print(f"\nShare drifted (Evidently): {results['share_drifted_evidently']:.1%}")
    print(f"PSI máximo:  {results['psi_max']:.4f}")
    print(f"PSI médio:   {results['psi_avg']:.4f}")
    print(f"\n🟢 Stable:  {results['n_features_stable']} features")
    print(f"🟡 Warning: {results['n_features_warning']} features")
    print(f"🔴 Retrain: {results['n_features_retrain_trigger']} features")
    print(f"\n📋 Recomendação: {results['recommendation']}")
    print(f"📄 Relatório HTML: {results['html_report']}")

    print("\n=== PSI por Feature ===")
    for feature, psi in sorted(
        results["psi_by_feature"].items(),
        key=lambda x: x[1] or 0,
        reverse=True,
    ):
        if psi is not None:
            icon = "🔴" if psi > PSI_RETRAIN else "🟡" if psi > PSI_WARNING else "🟢"
            print(f"  {icon} {feature:<30} PSI: {psi:.4f}")

    output_path = "evaluation/drift_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {output_path}")