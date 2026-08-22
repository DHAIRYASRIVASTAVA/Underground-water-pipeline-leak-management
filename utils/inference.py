"""
AquaGuard AI — Inference Utility
------------------------------------
Loads all trained models once (cached by Streamlit) and exposes a single
`run_prediction()` function that both dashboards call.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent.parent / "models" / "artifacts"

SEGMENT_INFO = {
    "SEG1": {"diameter": 8.0, "length": 250, "stations": ("S1", "S2")},
    "SEG2": {"diameter": 6.0, "length": 300, "stations": ("S2", "S3")},
    "SEG3": {"diameter": 6.0, "length": 275, "stations": ("S3", "S4")},
    "SEG4": {"diameter": 4.0, "length": 200, "stations": ("S4", "S5")},
}


def load_artifacts():
    return {
        "scaler": joblib.load(ARTIFACT_DIR / "scaler.joblib"),
        "anomaly_model": joblib.load(ARTIFACT_DIR / "anomaly_model.joblib"),
        "severity_model": joblib.load(ARTIFACT_DIR / "severity_model.joblib"),
        "severity_encoder": joblib.load(ARTIFACT_DIR / "severity_encoder.joblib"),
        "location_model": joblib.load(ARTIFACT_DIR / "location_model.joblib"),
        "location_encoder": joblib.load(ARTIFACT_DIR / "location_encoder.joblib"),
        "waterloss_model": joblib.load(ARTIFACT_DIR / "waterloss_model.joblib"),
        "feature_cols": joblib.load(ARTIFACT_DIR / "feature_cols.joblib"),
    }


def build_feature_row(reading: dict) -> pd.DataFrame:
    """reading must contain: segment, pressure_upstream_psi, pressure_downstream_psi,
    flow_upstream_lps, flow_downstream_lps, vibration_g, acoustic_db"""
    seg_info = SEGMENT_INFO[reading["segment"]]
    row = {
        "pipe_diameter_in": seg_info["diameter"],
        "pipe_length_m": seg_info["length"],
        "pressure_upstream_psi": reading["pressure_upstream_psi"],
        "pressure_downstream_psi": reading["pressure_downstream_psi"],
        "pressure_drop_psi": reading["pressure_upstream_psi"] - reading["pressure_downstream_psi"],
        "flow_upstream_lps": reading["flow_upstream_lps"],
        "flow_downstream_lps": reading["flow_downstream_lps"],
        "flow_diff_lps": reading["flow_upstream_lps"] - reading["flow_downstream_lps"],
        "vibration_g": reading["vibration_g"],
        "acoustic_db": reading["acoustic_db"],
    }
    return pd.DataFrame([row])


def run_prediction(artifacts: dict, reading: dict) -> dict:
    feature_cols = artifacts["feature_cols"]
    X_raw = build_feature_row(reading)[feature_cols]
    X_scaled = artifacts["scaler"].transform(X_raw)

    # 1. Anomaly detection
    anomaly_pred = artifacts["anomaly_model"].predict(X_scaled)[0]  # -1 anomaly, 1 normal
    anomaly_score = artifacts["anomaly_model"].decision_function(X_scaled)[0]
    is_anomaly = anomaly_pred == -1

    # 2. Severity classification
    sev_proba = artifacts["severity_model"].predict_proba(X_scaled)[0]
    sev_idx = np.argmax(sev_proba)
    severity = artifacts["severity_encoder"].inverse_transform([sev_idx])[0]
    severity_confidence = float(sev_proba[sev_idx])

    # 3. Location classification (only meaningful if a leak is suspected)
    loc_proba = artifacts["location_model"].predict_proba(X_scaled)[0]
    loc_idx = np.argmax(loc_proba)
    location = artifacts["location_encoder"].inverse_transform([loc_idx])[0]
    location_confidence = float(loc_proba[loc_idx])

    # 4. Water loss estimate
    water_loss = float(artifacts["waterloss_model"].predict(X_scaled)[0])
    if severity == "none":
        water_loss = min(water_loss, 0.3)  # clamp tiny noise for "no leak" case

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(anomaly_score),
        "severity": severity,
        "severity_confidence": severity_confidence,
        "severity_proba": dict(zip(artifacts["severity_encoder"].classes_, sev_proba.tolist())),
        "location": location if severity != "none" else reading["segment"],
        "location_confidence": location_confidence,
        "water_loss_lpm": round(water_loss, 3),
        "X_raw": X_raw,
        "X_scaled": X_scaled,
    }
