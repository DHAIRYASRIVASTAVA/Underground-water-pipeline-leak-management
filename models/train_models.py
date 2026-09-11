"""
AquaGuard AI — Model Training (Augmented Dataset)
------------------------------------------------------
Two things happen here, kept deliberately separate:

1. REPORTED PERFORMANCE (Analysis dashboard): 5-fold stratified
   cross-validation over the FULL real 1,000-row dataset. Inside each
   fold, augmentation statistics are computed fresh from ONLY that
   fold's training partition (never the validation rows), so nothing
   synthetic ever leaks information about a row it will later be
   evaluated on. Every real row gets predicted exactly once (as that
   fold's validation data), so the aggregated report covers all 1,000
   real rows — far more statistically robust than a single 200-row
   train/test split.

2. THE DEPLOYED MODEL (what the live app actually uses): trained on the
   persisted data/pipeline_dataset_augmented.csv (see augment_dataset.py)
   — real training rows + distribution-matched synthetic leak/burst rows,
   a literal, inspectable file rather than something that only exists
   transiently inside a training script.

Location and water-loss models remain dropped — no segment/pipe-network
structure or water-loss ground truth in this dataset to train them on.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR.parent / "data"
TRAIN_PATH = DATA_DIR / "pipeline_dataset_augmented.csv"
FULL_REAL_PATH = DATA_DIR / "pipeline_dataset.csv"

FEATURE_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c"]
AUGMENT_TARGETS = {"leak": 750, "burst": 750}
CLIP_BOUNDS = {
    "pressure_bar": (0.5, 4.2),
    "flow_rate_lps": (1.0, 320.0),
    "temperature_c": (8.0, 27.0),
}
RNG = np.random.default_rng(42)


def augment_fold(train_fold: pd.DataFrame, sensor_list: list) -> pd.DataFrame:
    """Same distribution-matched augmentation logic as augment_dataset.py,
    applied fresh to just this fold's training rows — computed here (not
    imported from the persisted file) so each fold's augmentation is
    strictly isolated to that fold's own training partition."""
    parts = [train_fold]
    for status, target in AUGMENT_TARGETS.items():
        class_rows = train_fold[train_fold["status"] == status]
        n_to_generate = max(0, target - len(class_rows))
        if n_to_generate == 0:
            continue
        stats = {col: (class_rows[col].mean(), class_rows[col].std()) for col in FEATURE_COLS}
        rows = []
        for _ in range(n_to_generate):
            row = {}
            for col in FEATURE_COLS:
                mean, std = stats[col]
                val = RNG.normal(mean, std if std > 0 else 0.05)
                lo, hi = CLIP_BOUNDS[col]
                row[col] = float(np.clip(val, lo, hi))
            row["status"] = status
            rows.append(row)
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def evaluate_with_cv():
    """5-fold stratified CV over all 1,000 real rows, augmenting only
    within each fold's training partition. Returns a classification
    report covering every real row exactly once."""
    df = pd.read_csv(FULL_REAL_PATH)
    le = LabelEncoder()
    y_full = le.fit_transform(df["status"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for train_idx, val_idx in skf.split(df, y_full):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)

        augmented_train = augment_fold(train_fold, sorted(df["sensor_id"].unique().tolist()))

        fold_scaler = StandardScaler().fit(augmented_train[FEATURE_COLS])
        X_train = fold_scaler.transform(augmented_train[FEATURE_COLS])
        y_train = le.transform(augmented_train["status"])
        X_val = fold_scaler.transform(val_fold[FEATURE_COLS])
        y_val = le.transform(val_fold["status"])

        fold_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                             class_weight="balanced", random_state=42)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        y_true_all.extend(y_val.tolist())
        y_pred_all.extend(y_pred.tolist())

    report_dict = classification_report(y_true_all, y_pred_all, target_names=le.classes_,
                                         zero_division=0, output_dict=True)
    print("=== 5-Fold CV performance across all 1,000 REAL rows ===")
    print("(augmentation computed fresh per fold, only from that fold's training data)")
    print(classification_report(y_true_all, y_pred_all, target_names=le.classes_, zero_division=0))
    return report_dict


def load_deployed_training_data():
    return pd.read_csv(TRAIN_PATH)


def train_anomaly_model(train_df, scaler):
    """Trained on REAL rows only — an anomaly detector should learn what
    a genuine real reading looks like, not a synthetic approximation of one."""
    real_rows = train_df[train_df["source"] == "real"]
    X = scaler.transform(real_rows[FEATURE_COLS])
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    print(f"[Anomaly Model] Trained on {len(real_rows)} real rows. "
          f"Flagged {(preds == -1).mean():.1%} as anomalous.")
    return model


def train_deployed_status_model(train_df, scaler):
    X_train = scaler.transform(train_df[FEATURE_COLS])
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["status"])

    print(f"[Status Model] Deployed-model training class counts: "
          f"{train_df['status'].value_counts().to_dict()}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model, le


def compute_sensor_baselines():
    """Computed from the FULL real dataset (all 1,000 original rows) —
    a descriptive statistic per sensor, not part of predictive evaluation."""
    df = pd.read_csv(FULL_REAL_PATH)
    baselines = df.groupby("sensor_id")["pressure_bar"].median().to_dict()
    baselines["_global"] = float(df["pressure_bar"].median())
    return baselines


def main():
    status_report = evaluate_with_cv()

    train_df = load_deployed_training_data()
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    anomaly_model = train_anomaly_model(train_df, scaler)
    status_model, status_encoder = train_deployed_status_model(train_df, scaler)
    baselines = compute_sensor_baselines()
    sensor_list = sorted(pd.read_csv(FULL_REAL_PATH)["sensor_id"].unique().tolist())

    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(anomaly_model, ARTIFACT_DIR / "anomaly_model.joblib")
    joblib.dump(status_model, ARTIFACT_DIR / "status_model.joblib")
    joblib.dump(status_encoder, ARTIFACT_DIR / "status_encoder.joblib")
    joblib.dump(status_report, ARTIFACT_DIR / "status_report.joblib")
    joblib.dump(baselines, ARTIFACT_DIR / "sensor_baselines.joblib")
    joblib.dump(FEATURE_COLS, ARTIFACT_DIR / "feature_cols.joblib")
    joblib.dump(sensor_list, ARTIFACT_DIR / "sensor_list.joblib")

    print(f"\nAll models saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
