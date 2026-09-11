# 💧 AquaGuard AI

**AI-based water pipeline leak/burst detection**, trained on a real (if small) Kaggle
sensor dataset — pressure, flow rate, and temperature readings from a 10-sensor
monitoring network.

## Dataset

`data/raw_kaggle_dataset.csv` — 1,000 readings from 10 sensors (S001–S010),
each with:
- `Pressure (bar)`, `Flow Rate (L/s)`, `Temperature (°C)`
- `Leak Status` (binary), `Burst Status` (binary)

`data/prepare_data.py` cleans this into `data/pipeline_dataset.csv` with a single
3-class `status` label (`normal` / `leak` / `burst`).

**⚠️ Honest caveat:** this is a small, heavily imbalanced dataset — only 19 leak
and 10 burst rows out of 1,000. That's enough for a working demo, but the
trained model's precision/recall on those two minority classes should be read
as indicative, not production-grade. See the **Analysis dashboard → Model
Performance** panel for the actual test-set numbers, pulled straight from the
training run (not cherry-picked).

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
│   ├── 1_📥_Data_Entry.py       # Input readings, run predictions, view SHAP
│   └── 2_📊_Analysis.py         # History, trends, model performance
├── data/
│   ├── raw_kaggle_dataset.csv   # Original Kaggle CSV
│   ├── prepare_data.py          # Cleans it into pipeline_dataset.csv
│   └── pipeline_dataset.csv     # Cleaned, ready for training
├── models/
│   ├── train_models.py          # Trains anomaly + status models
│   └── artifacts/               # Saved .joblib models + scaler/baselines
├── db/
│   └── database.py              # SQLite schema + read/write helpers
├── utils/
│   ├── inference.py             # Feature-building + prediction logic
│   ├── recommendations.py       # Rule-based action guidance per status
│   ├── ui.py                    # Design-system components
│   └── js_components.py         # JS-powered interactive widgets
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# 1. Clean the raw Kaggle CSV (already included, re-run to regenerate)
python data/prepare_data.py

# 2. Train the models (already included, re-run to retrain)
python models/train_models.py

# 3. Launch the app
streamlit run Home.py
```

## Swapping in your own dataset later

If you get a bigger or better-labeled dataset, update `data/prepare_data.py`'s
`COLUMN_MAP` and `derive_status()` to match its columns, drop your raw CSV in
as `data/raw_kaggle_dataset.csv`, then re-run `prepare_data.py` and
`train_models.py`. The rest of the app (inference, SHAP, dashboards) doesn't
need to change as long as the cleaned CSV still has `sensor_id`, `status`, and
the three feature columns.
