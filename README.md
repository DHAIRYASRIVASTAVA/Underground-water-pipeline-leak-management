# 💧 AquaGuard AI

**AI-based water pipeline leak/burst detection**, trained on a real Kaggle sensor
dataset — pressure, flow rate, and temperature readings from a 10-sensor
monitoring network — with a documented, leakage-safe augmentation step to
handle severe class imbalance.

## Dataset pipeline

```
data/raw_kaggle_dataset.csv          (source: 1,000 real readings, 10 sensors)
        ↓  data/prepare_data.py       (cleans columns, derives 3-class status)
data/pipeline_dataset.csv             (1,000 real rows: normal/leak/burst)
        ↓  data/augment_dataset.py    (80/20 stratified split, then augments TRAIN only)
data/pipeline_dataset_augmented.csv   (real train rows + synthetic leak/burst rows)
data/pipeline_dataset_test.csv        (pure real, held-out — for manual inspection)
```

**The real numbers:** 1,000 readings, of which only 19 are labeled `leak` and
10 are labeled `burst` — a 97% / 2% / 1% split. That's a genuine, honest
limitation of the source data, not something further engineering can erase.

**What augmentation does about it:** `data/augment_dataset.py` generates
additional synthetic `leak`/`burst` rows by sampling from a Normal
distribution whose mean/std are computed from the REAL examples of that class
— so every synthetic row follows the same physical pattern already present in
the real data (leak readings run lower-pressure/higher-flow than normal,
burst more so). The training set is brought to ~750 examples of each class
(roughly balanced with the ~780 real "normal" training rows) — this measurably
improves burst detection (recall up to 90% in cross-validation) but trades
some leak precision for higher recall, a known effect of pushing training
data further from its real-world class proportions. Every row keeps a
`source` column (`real` or `augmented`) so the two are never ambiguous if you
open the file.

**Why this is honest, not fabrication:** the reported performance (Analysis
dashboard → Model Performance) comes from 5-fold stratified cross-validation
over all 1,000 REAL rows. Augmentation statistics are recomputed fresh
inside each fold, using ONLY that fold's training partition — the validation
rows are never touched by augmentation, directly or indirectly. Every
prediction shown in that report was made on a real, non-synthetic reading.

## What it does

| Model | Type | Purpose |
|---|---|---|
| Anomaly Detector | `IsolationForest` | Flags unusual pressure/flow/temperature patterns |
| Status Classifier | `RandomForestClassifier` | normal / leak / burst |

Two models, not four — this dataset has no pipe-segment topology (each row is a
single point-sensor reading, not an upstream/downstream pair) and no water-loss
ground truth, so there's no meaningful "location" or "water-loss regression"
model to train. Instead:
- **Location** = the sensor ID itself (known directly from the input, not predicted)
- **Impact estimate** = a transparent, non-ML heuristic (pressure drop vs. that
  sensor's own training-data baseline), clearly labeled as such in the UI —
  not presented as a model prediction

Every prediction is explained with **SHAP** and logged to **SQLite**.

## Project structure

```
aquaguard_ai/
├── Home.py
├── pages/
│   ├── 1_📥_Data_Entry.py            # Input readings, run predictions, view SHAP
│   └── 2_📊_Analysis.py              # History, trends, model performance
├── data/
│   ├── raw_kaggle_dataset.csv        # Original Kaggle CSV
│   ├── prepare_data.py               # Cleans it into pipeline_dataset.csv
│   ├── pipeline_dataset.csv          # Cleaned, real, 1,000 rows
│   ├── augment_dataset.py            # Splits + augments -> the two files below
│   ├── pipeline_dataset_augmented.csv # Real train rows + synthetic leak/burst rows
│   └── pipeline_dataset_test.csv     # Pure real, held-out split
├── models/
│   ├── train_models.py               # CV evaluation + trains the deployed models
│   └── artifacts/                    # Saved .joblib models + scaler/baselines
├── db/
│   └── database.py                   # SQLite schema + read/write helpers
├── utils/
│   ├── inference.py                  # Feature-building + prediction logic
│   ├── recommendations.py            # Rule-based action guidance per status
│   ├── ui.py                         # Design-system components
│   └── js_components.py              # JS-powered interactive widgets
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# 1. Clean the raw Kaggle CSV (already included, re-run to regenerate)
python data/prepare_data.py

# 2. Split + augment the training set (already included, re-run to regenerate)
python data/augment_dataset.py

# 3. Train the models — runs the 5-fold CV evaluation, then trains the
#    deployed models on the augmented training file (already included)
python models/train_models.py

# 4. Launch the app
streamlit run Home.py
```

## Swapping in your own dataset later

Update `data/prepare_data.py`'s `COLUMN_MAP` and `derive_status()` to match
your new columns, drop the raw CSV in as `data/raw_kaggle_dataset.csv`, then
re-run `prepare_data.py` → `augment_dataset.py` → `train_models.py` in order.
The rest of the app doesn't need to change as long as the cleaned CSV still
has `sensor_id`, `status`, and the three feature columns.
