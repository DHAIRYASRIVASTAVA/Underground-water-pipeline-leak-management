import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import shap
import plotly.graph_objects as go

from utils.inference import load_artifacts, run_prediction
from utils.ui import inject_css, hero, section_head, readout_grid, status_line, themed_layout, recommendation_panel
from utils.js_components import interactive_sensor_map, radial_gauge
from utils.recommendations import get_recommendation
from db.database import insert_prediction

st.set_page_config(page_title="Data Entry — AquaGuard AI", page_icon="📥", layout="wide")
inject_css()

hero(
    eyebrow="CONSOLE · MANUAL READING",
    title_html="📥 Data Entry",
    subtitle="Feed a sensor reading into AquaGuard's model stack — simulate a scenario "
              "or enter live values — and get a fully explained prediction back.",
)
st.write("")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_shap_explainer(_artifacts):
    return shap.TreeExplainer(_artifacts["status_model"])


artifacts = get_artifacts()
explainer = get_shap_explainer(artifacts)

# Real per-status stats from the training data, used to simulate plausible readings
SCENARIO_STATS = {
    "Normal":       {"pressure": (3.26, 0.43), "flow": (124.2, 43.1), "temp": (17.4, 4.3)},
    "Leak":         {"pressure": (2.19, 0.45), "flow": (148.7, 57.6), "temp": (17.9, 4.4)},
    "Burst":        {"pressure": (1.35, 0.43), "flow": (166.0, 76.8), "temp": (18.6, 3.8)},
}

# ---------------------------------------------------------------------
# Input console
# ---------------------------------------------------------------------
section_head("01", "Sensor Input Console")

left, right = st.columns([1, 1.1])

with left:
    sensor_id = st.selectbox("Sensor", artifacts["sensor_list"])
    baseline = artifacts["sensor_baselines"].get(sensor_id, artifacts["sensor_baselines"]["_global"])
    st.caption(f"This sensor's baseline pressure (training data median): {baseline:.2f} bar")

    simulate = st.toggle("🎲 Simulate a scenario", value=True,
                          help="Auto-fill values sampled from this dataset's real per-class statistics.")

    if simulate:
        rng = np.random.default_rng()
        scenario = st.radio("Scenario", list(SCENARIO_STATS.keys()), horizontal=True)
        stats = SCENARIO_STATS[scenario]
        p_default = round(max(rng.normal(*stats["pressure"]), 0.1), 3)
        f_default = round(max(rng.normal(*stats["flow"]), 1.0), 2)
        t_default = round(rng.normal(*stats["temp"]), 2)
    else:
        p_default, f_default, t_default = 3.2, 125.0, 17.4

with right:
    pressure = st.number_input("Pressure (bar)", value=p_default, step=0.05)
    flow_rate = st.number_input("Flow Rate (L/s)", value=f_default, step=1.0)
    temperature = st.number_input("Temperature (°C)", value=t_default, step=0.5)

st.write("")
run_clicked = st.button("▶  RUN PREDICTION", type="primary", width='stretch')

if run_clicked:
    reading = {
        "sensor_id": sensor_id,
        "pressure_bar": pressure,
        "flow_rate_lps": flow_rate,
        "temperature_c": temperature,
    }
    result = run_prediction(artifacts, reading)
    st.session_state["last_result"] = result
    st.session_state["last_reading"] = reading

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    reading = st.session_state["last_reading"]

    color_map = {"normal": "none", "leak": "medium", "burst": "severe"}
    status_color = color_map.get(result["status"], "none")

    st.write("")
    section_head("02", "Prediction Output")

    interactive_sensor_map(
        artifacts["sensor_list"], artifacts["sensor_baselines"],
        active_sensor=sensor_id,
        alert_sensor=sensor_id if result["status"] != "normal" else None,
        alert_color=status_color,
    )

    st.write("")
    status_line(result["status"])

    readout_grid([
        {"label": "Status", "value": result["status"].upper(),
         "sub": f"{result['status_confidence']:.0%} confidence", "status": status_color},
        {"label": "Sensor", "value": sensor_id, "sub": f"baseline {baseline:.2f} bar"},
        {"label": "Anomaly Check", "value": "FLAGGED" if result["is_anomaly"] else "CLEAR",
         "sub": f"score {result['anomaly_score']:.3f}", "status": "severe" if result["is_anomaly"] else "none"},
    ])

    st.write("")
    gcol1, gcol2 = st.columns(2)
    gauge_color = {"none": "#3ECF8E", "medium": "#F0883E", "severe": "#EF5350"}[status_color]
    with gcol1:
        radial_gauge("Status Confidence", result["status_confidence"] * 100,
                     max_value=100, unit="%", color=gauge_color)
    with gcol2:
        impact = result["impact"]
        impact_color = {"Low": "#3ECF8E", "Moderate": "#F0883E", "High": "#EF5350"}[impact["level"]]
        radial_gauge("Est. Pressure Drop vs Baseline", impact["drop_pct"],
                     max_value=100, unit="%", color=impact_color)

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        proba_fig = go.Figure(go.Bar(
            x=list(result["status_proba"].values()),
            y=list(result["status_proba"].keys()),
            orientation="h",
            marker_color=["#F0883E" if k == "leak" else "#EF5350" if k == "burst" else "#3ECF8E"
                          for k in result["status_proba"].keys()],
        ))
        proba_fig.update_layout(**themed_layout("Status Class Probabilities",
                                                  height=280, xaxis_title="Probability"))
        st.plotly_chart(proba_fig, width='stretch')

    with col2:
        shap_values = explainer.shap_values(result["X_scaled"])
        status_idx = list(artifacts["status_encoder"].classes_).index(result["status"])
        if isinstance(shap_values, list):
            vals = shap_values[status_idx][0]
        else:
            vals = shap_values[0, :, status_idx] if shap_values.ndim == 3 else shap_values[0]

        feat_names = artifacts["feature_cols"]
        shap_fig = go.Figure(go.Bar(
            x=vals, y=feat_names, orientation="h",
            marker_color=["#EF5350" if v > 0 else "#3FA9E0" for v in vals],
        ))
        shap_fig.update_layout(**themed_layout(f"Why '{result['status']}'? (SHAP)",
                                                 height=280, xaxis_title="Impact on prediction"))
        st.plotly_chart(shap_fig, width='stretch')

    st.caption("🔴 Pushes toward this class · 🔵 Pushes away from it")
    st.caption("⚠️ This model is trained on only 19 leak and 10 burst examples in the training data — "
               "treat confidence on those two classes as indicative, not production-grade.")

    st.write("")
    section_head("03", "Recommended Action")

    rec = get_recommendation(
        status=result["status"],
        sensor_id=sensor_id,
        impact=result["impact"],
    )
    recommendation_panel(rec)

    st.write("")
    section_head("04", "Log to Database")
    notes = st.text_input("Notes (optional)", "")
    if st.button("💾 SAVE READING", width='stretch'):
        record = {
            "sensor_id": sensor_id,
            "pressure_bar": reading["pressure_bar"],
            "flow_rate_lps": reading["flow_rate_lps"],
            "temperature_c": reading["temperature_c"],
            "is_anomaly": int(result["is_anomaly"]),
            "predicted_status": result["status"],
            "status_confidence": result["status_confidence"],
            "impact_level": result["impact"]["level"],
            "impact_drop_pct": result["impact"]["drop_pct"],
            "notes": notes,
        }
        insert_prediction(record)
        st.success("✅ Saved. Head to the Analysis dashboard to see it in the history.")
