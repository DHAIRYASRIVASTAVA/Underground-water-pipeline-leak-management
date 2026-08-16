"""
AquaGuard AI — Model Training Pipeline
----------------------------------------
Trains four models on the synthetic pipeline dataset:

1. IsolationForest        -> unsupervised anomaly detector (pressure/flow pattern)
2. RandomForestClassifier -> leak severity classifier (none/small/medium/severe)
3. RandomForestClassifier -> leak location classifier (which segment is leaking)
4. RandomForestRegressor  -> water loss estimator (L/min)

All models + a fitted StandardScaler + LabelEncoders are saved as .joblib
files inside models/artifacts/ for the Streamlit app to load.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, RandomForestRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_PATH = BASE_DIR.parent / "data" / "pipeline_dataset.csv"

FEATURE_COLS = [
    "pipe_diameter_in", "pipe_length_m",
    "pressure_upstream_psi", "pressure_downstream_psi", "pressure_drop_psi",
    "flow_upstream_lps", "flow_downstream_lps", "flow_diff_lps",
    "vibration_g", "acoustic_db",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def train_anomaly_model(df, scaler):
    X = scaler.transform(df[FEATURE_COLS])
    model = IsolationForest(
        n_estimators=200, contamination=0.18, random_state=42
    )
    model.fit(X)
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    anomaly_rate = (preds == -1).mean()
    print(f"[Anomaly Model] Flagged {anomaly_rate:.1%} of readings as anomalous")
    return model


def train_severity_model(df, scaler):
    X = scaler.transform(df[FEATURE_COLS])
    le = LabelEncoder()
    y = le.fit_transform(df["severity"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=80, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Severity Model] Test accuracy: {acc:.3f}")
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))
    return model, le


def train_location_model(df, scaler):
    """Only trained on leak rows — predicts WHICH segment, given leak-pattern features."""
    leak_df = df[df["is_leak"] == 1].copy()
    X = scaler.transform(leak_df[FEATURE_COLS])
    le = LabelEncoder()
    y = le.fit_transform(leak_df["segment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Location Model] Test accuracy: {acc:.3f}")
    return model, le


def train_waterloss_model(df, scaler):
    X = scaler.transform(df[FEATURE_COLS])
    y = df["water_loss_lpm"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"[Water Loss Model] Test MAE: {mae:.3f} L/min")
    return model


def main():
    df = load_data()

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS])

    anomaly_model = train_anomaly_model(df, scaler)
    severity_model, severity_encoder = train_severity_model(df, scaler)
    location_model, location_encoder = train_location_model(df, scaler)
    waterloss_model = train_waterloss_model(df, scaler)

    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(anomaly_model, ARTIFACT_DIR / "anomaly_model.joblib")
    joblib.dump(severity_model, ARTIFACT_DIR / "severity_model.joblib")
    joblib.dump(severity_encoder, ARTIFACT_DIR / "severity_encoder.joblib")
    joblib.dump(location_model, ARTIFACT_DIR / "location_model.joblib")
    joblib.dump(location_encoder, ARTIFACT_DIR / "location_encoder.joblib")
    joblib.dump(waterloss_model, ARTIFACT_DIR / "waterloss_model.joblib")
    joblib.dump(FEATURE_COLS, ARTIFACT_DIR / "feature_cols.joblib")

    print(f"\nAll models saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
