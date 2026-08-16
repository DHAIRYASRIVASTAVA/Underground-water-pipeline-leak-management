# 💧 AquaGuard AI

**AI-Based Underground Water Pipeline Leak Detection and Localization**
Software-only demo — no hardware required.

## What it does

Simulates a 5-station pipeline network (S1 → S2 → S3 → S4 → S5, 4 monitored
segments) and runs sensor readings through four ML models:

| Model | Type | Purpose |
|---|---|---|
| Anomaly Detector | `IsolationForest` | Flags unusual pressure/flow patterns |
| Severity Classifier | `RandomForestClassifier` | none / small / medium / severe leak |
| Location Classifier | `RandomForestClassifier` | Which segment is leaking |
| Water Loss Estimator | `RandomForestRegressor` | Estimated loss in L/min |

Every prediction is explained with **SHAP** feature-contribution charts and
logged to a local **SQLite** database.

## Project structure

```
aquaguard_ai/
├── Home.py                      # Streamlit entry point
├── pages/
│   ├── 1_📥_Data_Entry.py       # Input readings, run predictions, view SHAP
│   └── 2_📊_Analysis.py         # History, trends, network-wide stats
├── data/
│   ├── generate_data.py         # Physics-inspired synthetic data generator
│   └── pipeline_dataset.csv     # Generated dataset (16,000 rows)
├── models/
│   ├── train_models.py          # Trains all 4 models
│   └── artifacts/               # Saved .joblib models + scaler/encoders
├── db/
│   └── database.py               # SQLite schema + read/write helpers
├── utils/
│   └── inference.py              # Shared feature-building + prediction logic
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset (already included, re-run to regenerate)
python data/generate_data.py

# 2. Train the models (already included, re-run to retrain)
python models/train_models.py

# 3. Launch the app
streamlit run Home.py
```

The app opens two dashboards in the sidebar:

- **Data Entry** — pick a segment, simulate or manually enter sensor values
  (pressure, flow, vibration, acoustic), run AquaGuard's prediction, inspect
  the SHAP explanation, and save the reading to the database.
- **Analysis** — filterable charts (severity distribution, leak-by-segment,
  water-loss trend, pressure-drop vs flow-diff scatter) plus a full history
  table with CSV export.

## Notes on the physics-inspired simulation

- Pressure loss along a segment follows a simplified Hazen-Williams-style
  friction model (`dp ∝ Q^1.85`).
- A leak injects an **extra abrupt pressure drop** and a **flow imbalance**
  (`flow_in − flow_out` = water loss) proportional to severity.
- Vibration and acoustic readings get a small severity-linked bump to mimic
  the physical signature of a leak (turbulence/noise at the leak point).

This keeps the dataset internally consistent so the trained models learn a
genuine leak "signature" rather than memorizing noise.
