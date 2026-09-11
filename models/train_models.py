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

Class imbalance handling: the dataset is 97.1% normal / 1.9% leak / 1.0%
burst. Two things address this, both standard, defensible ML practice
(not data fabrication):

  1. Reported performance comes from 5-fold STRATIFIED cross-validation,
     with SMOTE oversampling applied ONLY inside each fold's training
     split (never touching the validation fold) — this avoids the classic
     leakage mistake of oversampling before splitting, which would let
     synthetic near-duplicates of validation rows leak into training.
  2. The final deployed model is trained on the full dataset after SMOTE
     oversampling (real feature space interpolation between existing
     minority-class points — not invented data), then saved as a plain
     RandomForestClassifier (not wrapped in a Pipeline) so SHAP's
     TreeExplainer keeps working directly on it, same as before.

Also computes and saves per-sensor pressure baselines (median), used by
the impact-estimate heuristic.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

    print(f"[Status Model] Class counts before SMOTE: "
          f"{dict(zip(le.classes_, [sum(y == i) for i in range(len(le.classes_))]))}")

    # --- Honest performance estimate: 5-fold stratified CV, SMOTE inside
    #     training folds only (via imblearn's Pipeline + cross_val_predict,
    #     which refits SMOTE fresh per fold on that fold's train split) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ])
    y_pred_cv = cross_val_predict(cv_pipeline, X, y, cv=skf)
    report_dict = classification_report(y, y_pred_cv, target_names=le.classes_,
                                         zero_division=0, output_dict=True)
    print("\n=== 5-Fold Stratified CV Report (SMOTE applied only within training folds) ===")
    print(classification_report(y, y_pred_cv, target_names=le.classes_, zero_division=0))

    # --- Final deployed model: SMOTE the full dataset once, fit a plain
    #     RandomForestClassifier (not a Pipeline) so SHAP TreeExplainer
    #     and .feature_importances_ keep working exactly as before ---
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"[Status Model] Class counts after SMOTE: "
          f"{dict(zip(le.classes_, [sum(y_resampled == i) for i in range(len(le.classes_))]))}")

    final_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    final_model.fit(X_resampled, y_resampled)

    return final_model, le, report_dict


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
