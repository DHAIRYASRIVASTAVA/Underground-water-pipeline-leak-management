"""
AquaGuard AI — Real Dataset Preparation
--------------------------------------------
Loads the Kaggle "water leak detection" CSV (raw_kaggle_dataset.csv) and
cleans it into the format the rest of the app expects: snake_case column
names, and a single 3-class `status` label derived from the dataset's two
binary columns (Leak Status, Burst Status).

Why status is 3-class, not the 4-tier none/small/medium/severe used in
the earlier synthetic-data version: this dataset only provides binary
leak/burst flags, no severity gradient. Burst is treated as the more
severe event. In this dataset Leak Status and Burst Status never overlap
(a burst row always has Leak Status = 0), so the mapping is clean:

    Burst Status == 1        -> "burst"   (most severe)
    Leak Status == 1         -> "leak"
    both 0                   -> "normal"

Note on scale: this dataset has 1,000 rows with only 19 leak and 10 burst
examples (1.9% and 1.0% of the data). That's a small, heavily imbalanced
sample — good enough for a working demo, but the trained model's accuracy
on the minority classes should be read with that caveat in mind (the test
split ends up with only a handful of leak/burst examples to evaluate on).
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw_kaggle_dataset.csv"
CLEAN_PATH = Path(__file__).parent / "pipeline_dataset.csv"

COLUMN_MAP = {
    "Timestamp": "timestamp",
    "Sensor_ID": "sensor_id",
    "Pressure (bar)": "pressure_bar",
    "Flow Rate (L/s)": "flow_rate_lps",
    "Temperature (°C)": "temperature_c",
    "Leak Status": "leak_status",
    "Burst Status": "burst_status",
}

FEATURE_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c"]


def derive_status(row):
    if row["burst_status"] == 1:
        return "burst"
    if row["leak_status"] == 1:
        return "leak"
    return "normal"


def prepare():
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing after rename: {missing}")

    df["status"] = df.apply(derive_status, axis=1)
    df = df.dropna(subset=FEATURE_COLS + ["status"])

    df.to_csv(CLEAN_PATH, index=False)
    return df


if __name__ == "__main__":
    df = prepare()
    print(f"Cleaned dataset -> {CLEAN_PATH}")
    print(f"Rows: {len(df)}")
    print(df["status"].value_counts())
    print(f"\nSensors: {sorted(df['sensor_id'].unique())}")
