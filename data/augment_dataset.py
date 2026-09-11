"""
AquaGuard AI — Dataset Augmentation
----------------------------------------
Expands the real Kaggle dataset's minority classes (leak, burst) using
distribution-matched sampling — NOT invented data, NOT hand-tuned formulas.
Each synthetic row is drawn from a Normal distribution whose mean and
std come directly from that class's REAL examples (computed from the
training split only — see below), so every generated row follows the
same physical pattern already present in the real data (leak readings
run lower-pressure/higher-flow than normal; burst more so).

Critical ordering, to keep evaluation honest:

  1. Split the real 1,000 rows into train (80%) / test (20%) FIRST,
     stratified by status.
  2. Compute augmentation statistics (mean/std per class per feature)
     from the TRAINING split ONLY — the test split is never touched,
     never seen, never used to inform augmentation.
  3. Generate synthetic leak/burst rows from those training-only stats,
     append them to the training split -> data/pipeline_dataset_augmented.csv.
  4. Save the untouched real test split separately -> data/pipeline_dataset_test.csv.

Final reported model performance (Analysis dashboard) is evaluated ONLY
on that pure-real test split — so what the user sees is genuine held-out
performance on real sensor readings, not on anything synthetic. The
training set's class ratio is deliberately rebalanced (so the model has
enough leak/burst examples to learn from) — the test set's class ratio
stays exactly as imbalanced as the real world is (this dataset's own
~97% / 2% / 1% split), because that's what an honest evaluation should
reflect.

Every row keeps a `source` column: "real" or "augmented", so the two are
never ambiguous if someone inspects the file directly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).parent
REAL_PATH = DATA_DIR / "pipeline_dataset.csv"
TRAIN_OUT_PATH = DATA_DIR / "pipeline_dataset_augmented.csv"
TEST_OUT_PATH = DATA_DIR / "pipeline_dataset_test.csv"

FEATURE_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c"]

# How many total training rows each minority class should have after
# augmentation — brought close to the real "normal" training count
# (~777 rows) so the model trains on a roughly balanced set. This is a
# deliberate choice for training only: the held-out test split (see
# below) stays 100% real and keeps the genuine ~97/2/1 imbalance, so
# reported performance is never evaluated against an artificially
# balanced world.
AUGMENT_TARGETS = {"leak": 750, "burst": 750}

# Physical bounds (from the real data's own observed min/max per class,
# with a small margin) — keeps generated rows physically plausible
# instead of letting Gaussian sampling drift into impossible values.
CLIP_BOUNDS = {
    "pressure_bar": (0.5, 4.2),
    "flow_rate_lps": (1.0, 320.0),
    "temperature_c": (8.0, 27.0),
}


def augment_class(train_real: pd.DataFrame, sensor_list: list, status: str, target_count: int) -> pd.DataFrame:
    class_rows = train_real[train_real["status"] == status]
    n_existing = len(class_rows)
    n_to_generate = max(0, target_count - n_existing)
    if n_to_generate == 0:
        return pd.DataFrame(columns=train_real.columns)

    stats = {col: (class_rows[col].mean(), class_rows[col].std()) for col in FEATURE_COLS}

    rows = []
    for _ in range(n_to_generate):
        row = {}
        for col in FEATURE_COLS:
            mean, std = stats[col]
            val = RNG.normal(mean, std if std > 0 else 0.05)
            lo, hi = CLIP_BOUNDS[col]
            row[col] = float(np.clip(val, lo, hi))
        row["sensor_id"] = RNG.choice(sensor_list)
        row["status"] = status
        row["leak_status"] = 1 if status == "leak" else 0
        row["burst_status"] = 1 if status == "burst" else 0
        row["source"] = "augmented"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(REAL_PATH)
    df["source"] = "real"

    train_real, test_real = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["status"]
    )

    sensor_list = sorted(df["sensor_id"].unique().tolist())

    augmented_parts = [train_real]
    for status, target in AUGMENT_TARGETS.items():
        augmented_parts.append(augment_class(train_real, sensor_list, status, target))

    train_augmented = pd.concat(augmented_parts, ignore_index=True)
    train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    train_augmented.to_csv(TRAIN_OUT_PATH, index=False)
    test_real.to_csv(TEST_OUT_PATH, index=False)

    print("=== Training set (real + augmented) ===")
    print(train_augmented.groupby(["status", "source"]).size())
    print(f"\nTotal training rows: {len(train_augmented)}")

    print("\n=== Test set (100% real, untouched, held out) ===")
    print(test_real["status"].value_counts())
    print(f"Total test rows: {len(test_real)}")


if __name__ == "__main__":
    main()
