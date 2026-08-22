import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
from utils.ui import inject_css, hero, section_head, readout_grid, pipeline_schematic, feature_card, legend_card, render_html

st.set_page_config(page_title="AquaGuard AI", page_icon="💧", layout="wide")
inject_css()

hero(
    eyebrow="SYSTEM ONLINE · SOFTWARE-ONLY MONITORING",
    title_html='💧 AquaGuard <span>AI</span>',
    subtitle="AI-based underground water pipeline leak detection and localization. "
              "Four machine-learning models watch pressure, flow, vibration, and acoustic "
              "signatures across a simulated network — no hardware required.",
)

st.write("")
pipeline_schematic()

st.write("")
readout_grid([
    {"label": "Sensor Stations", "value": "05", "sub": "S1 – S5"},
    {"label": "Monitored Segments", "value": "04", "sub": "SEG1 – SEG4"},
    {"label": "ML Models Active", "value": "04", "sub": "anomaly · severity · location · loss"},
    {"label": "Explainability", "value": "SHAP", "sub": "per-prediction attribution", "accent": True},
])

st.write("")
st.write("")
section_head("01", "How the system works")

c1, c2, c3 = st.columns(3)
with c1:
    render_html(feature_card(
        "📥", "Log a reading",
        "Enter live sensor values — or simulate a scenario — for any segment: "
        "pressure, flow, vibration, and acoustic level.",
        "DATA ENTRY",
    ))
with c2:
    render_html(feature_card(
        "🧠", "Four models decide",
        "IsolationForest flags anomalies; Random Forests classify severity and "
        "pinpoint the leaking segment; a regressor estimates water loss in L/min.",
        "INFERENCE",
    ))
with c3:
    render_html(feature_card(
        "📊", "Trace the reasoning",
        "SHAP breaks down exactly which sensor readings pushed the prediction "
        "toward a leak — and every result is logged to SQLite for trend analysis.",
        "ANALYSIS",
    ))

st.write("")
st.write("")
section_head("02", "Network legend")

c1, c2, c3, c4 = st.columns(4)
for col, status, label in zip(
    [c1, c2, c3, c4],
    ["none", "small", "medium", "severe"],
    ["Normal operation", "Minor leak", "Moderate leak", "Severe leak"],
):
    with col:
        render_html(legend_card(status, label))

st.write("")
st.info("👈 Open **Data Entry** in the sidebar to log your first reading, then check **Analysis** for network-wide trends.")
