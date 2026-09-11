"""
AquaGuard AI — Model Training (Real Dataset)
------------------------------------------------
Trains on the cleaned Kaggle dataset (data/pipeline_dataset.csv). Only two
models now, versus the four in the synthetic-data version:

1. IsolationForest        -> anomaly detector (unsupervised)
2. RandomForestClassifier -> status classifier (normal / leak / burst)

Location and water-loss models from the synthetic version are DROPPED —
this dataset has no segment/pipe-network structure (each row is a single
point-sensor reading, not an upstream/downstream pair) and no water-loss
ground truth column, so there's nothing to train those on. Sensor location
is handled directly from the input (the user picks a sensor), and water
loss is estimated with a transparent, non-ML heuristic at inference time
(see utils/inference.py) — not presented as a model prediction.

Also computes and saves per-sensor pressure baselines (median), used by
that heuristic.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_PATH = BASE_DIR.parent / "data" / "pipeline_dataset.csv"

FEATURE_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c"]


def load_data():
    return pd.read_csv(DATA_PATH)


def train_anomaly_model(df, scaler):
    X = scaler.transform(df[FEATURE_COLS])
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    print(f"[Anomaly Model] Flagged {(preds == -1).mean():.1%} of readings as anomalous")
    return model


def train_status_model(df, scaler):
    X = scaler.transform(df[FEATURE_COLS])
    le = LabelEncoder()
    y = le.fit_transform(df["status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Status Model] Test accuracy: {acc:.3f}")
    print("NOTE: leak/burst are ~2% and ~1% of this 1,000-row dataset, so the")
    print("test split only has a handful of each — treat minority-class")
    print("metrics below as indicative, not statistically solid.")
    report_dict = classification_report(y_test, model.predict(X_test), target_names=le.classes_,
                                         zero_division=0, output_dict=True)
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_, zero_division=0))
    return model, le, report_dict


def compute_sensor_baselines(df):
    """Median pressure per sensor + a global fallback — used by the
    heuristic water-loss/impact estimate at inference time."""
    baselines = df.groupby("sensor_id")["pressure_bar"].median().to_dict()
    baselines["_global"] = float(df["pressure_bar"].median())
    return baselines


def main():
    df = load_data()

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS])

    anomaly_model = train_anomaly_model(df, scaler)
    status_model, status_encoder, status_report = train_status_model(df, scaler)
    baselines = compute_sensor_baselines(df)

    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(anomaly_model, ARTIFACT_DIR / "anomaly_model.joblib")
    joblib.dump(status_model, ARTIFACT_DIR / "status_model.joblib")
    joblib.dump(status_encoder, ARTIFACT_DIR / "status_encoder.joblib")
    joblib.dump(status_report, ARTIFACT_DIR / "status_report.joblib")
    joblib.dump(baselines, ARTIFACT_DIR / "sensor_baselines.joblib")
    joblib.dump(FEATURE_COLS, ARTIFACT_DIR / "feature_cols.joblib")
    joblib.dump(sorted(df["sensor_id"].unique().tolist()), ARTIFACT_DIR / "sensor_list.joblib")

    print(f"\nAll models saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
