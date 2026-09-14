

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent.parent / "models" / "artifacts"


def load_artifacts():
    return {
        "scaler": joblib.load(ARTIFACT_DIR / "scaler.joblib"),
        "anomaly_model": joblib.load(ARTIFACT_DIR / "anomaly_model.joblib"),
        "status_model": joblib.load(ARTIFACT_DIR / "status_model.joblib"),
        "status_encoder": joblib.load(ARTIFACT_DIR / "status_encoder.joblib"),
        "status_report": joblib.load(ARTIFACT_DIR / "status_report.joblib"),
        "sensor_baselines": joblib.load(ARTIFACT_DIR / "sensor_baselines.joblib"),
        "feature_cols": joblib.load(ARTIFACT_DIR / "feature_cols.joblib"),
        "sensor_list": joblib.load(ARTIFACT_DIR / "sensor_list.joblib"),
    }


def build_feature_row(reading: dict) -> pd.DataFrame:
    """reading must contain: pressure_bar, flow_rate_lps, temperature_c"""
    row = {
        "pressure_bar": reading["pressure_bar"],
        "flow_rate_lps": reading["flow_rate_lps"],
        "temperature_c": reading["temperature_c"],
    }
    return pd.DataFrame([row])


def estimate_impact(pressure_bar: float, sensor_id: str, baselines: dict) -> dict:
    """Non-ML heuristic: how far is this reading's pressure below that
    sensor's own historical median? Bigger drop -> more likely a real
    water-loss event, on the physical logic that a leak/burst bleeds
    pressure out of the line. This is NOT a trained model — there's no
    water-loss ground truth in this dataset to train one on."""
    baseline = baselines.get(sensor_id, baselines["_global"])
    drop = max(0.0, baseline - pressure_bar)
    drop_pct = drop / baseline if baseline else 0.0

    if drop_pct < 0.10:
        level = "Low"
    elif drop_pct < 0.30:
        level = "Moderate"
    else:
        level = "High"

    return {"level": level, "drop_pct": round(drop_pct * 100, 1),
            "baseline_pressure": round(baseline, 2)}


def run_prediction(artifacts: dict, reading: dict) -> dict:
    feature_cols = artifacts["feature_cols"]
    X_raw = build_feature_row(reading)[feature_cols]
    X_scaled = artifacts["scaler"].transform(X_raw)

    # 1. Anomaly detection
    anomaly_pred = artifacts["anomaly_model"].predict(X_scaled)[0]
    anomaly_score = artifacts["anomaly_model"].decision_function(X_scaled)[0]
    is_anomaly = anomaly_pred == -1

    # 2. Status classification (normal / leak / burst)
    status_proba = artifacts["status_model"].predict_proba(X_scaled)[0]
    status_idx = np.argmax(status_proba)
    status = artifacts["status_encoder"].inverse_transform([status_idx])[0]
    status_confidence = float(status_proba[status_idx])

    # 3. Impact estimate (heuristic, not ML)
    impact = estimate_impact(reading["pressure_bar"], reading["sensor_id"], artifacts["sensor_baselines"])

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(anomaly_score),
        "status": status,
        "status_confidence": status_confidence,
        "status_proba": dict(zip(artifacts["status_encoder"].classes_, status_proba.tolist())),
        "impact": impact,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
    }
