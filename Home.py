import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import joblib
from utils.ui import inject_css, hero, section_head, feature_card, legend_card, render_html
from utils.js_components import interactive_sensor_map, counter_grid, live_ticker

st.set_page_config(page_title="AquaGuard AI", page_icon="💧", layout="wide")
inject_css()

ARTIFACT_DIR = Path(__file__).parent / "models" / "artifacts"
sensor_list = joblib.load(ARTIFACT_DIR / "sensor_list.joblib")
baselines = joblib.load(ARTIFACT_DIR / "sensor_baselines.joblib")

hero(
    eyebrow="SYSTEM ONLINE · TRAINED ON REAL SENSOR DATA",
    title_html='💧 AquaGuard <span>AI</span>',
    subtitle="AI-based water pipeline leak detection, trained on real pressure/flow/temperature "
              "sensor readings from a 10-sensor monitoring network. Flags anomalies and classifies "
              "each reading as normal, leak, or burst.",
)

st.write("")
interactive_sensor_map(sensor_list, baselines)

st.write("")
live_ticker(label="LIVE MONITORING FEED (SIMULATED)", start_value=48210, unit=" readings")

st.write("")
counter_grid([
    {"label": "Sensors Monitored", "value": len(sensor_list), "sub": "S001 – S010"},
    {"label": "Training Readings", "value": 1000, "sub": "real Kaggle dataset"},
    {"label": "ML Models Active", "value": 2, "sub": "anomaly · status classifier"},
    {"label": "SHAP Explanations", "value": 100, "suffix": "%", "sub": "per-prediction attribution"},
])

st.write("")
st.write("")
section_head("01", "How the system works")

c1, c2, c3 = st.columns(3)
with c1:
    render_html(feature_card(
        "📥", "Log a reading",
        "Pick a sensor, enter or simulate pressure, flow rate, and temperature values.",
        "DATA ENTRY",
    ))
with c2:
    render_html(feature_card(
        "🧠", "Two models decide",
        "IsolationForest flags unusual readings; a Random Forest classifies the "
        "reading as normal, leak, or burst.",
        "INFERENCE",
    ))
with c3:
    render_html(feature_card(
        "📊", "Trace the reasoning",
        "SHAP breaks down which sensor reading pushed the prediction — and every "
        "result is logged to SQLite for trend analysis.",
        "ANALYSIS",
    ))

st.write("")
st.write("")
section_head("02", "Status legend")

c1, c2, c3 = st.columns(3)
for col, status, label in zip(
    [c1, c2, c3],
    ["none", "medium", "severe"],
    ["Normal reading", "Leak detected", "Burst detected"],
):
    with col:
        render_html(legend_card(status, label))

st.write("")
st.warning(
    "⚠️ **About the training data:** this model is trained on a real but small Kaggle dataset — "
    "1,000 readings, of which only 19 are labeled 'leak' and 10 are labeled 'burst'. That's enough "
    "to build a working demo, but predictions on the minority classes (leak/burst) should be read "
    "as indicative, not production-grade — see the Analysis dashboard for the model's actual "
    "test-set performance on each class."
)

st.write("")
st.info("👈 Open **Data Entry** in the sidebar to log your first reading, then check **Analysis** for network-wide trends.")
