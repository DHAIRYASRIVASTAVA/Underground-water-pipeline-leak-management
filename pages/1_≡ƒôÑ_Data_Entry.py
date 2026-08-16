import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import shap
import plotly.graph_objects as go

from utils.inference import load_artifacts, run_prediction, SEGMENT_INFO
from db.database import insert_prediction

st.set_page_config(page_title="Data Entry — AquaGuard AI", page_icon="📥", layout="wide")
st.title("📥 Data Entry Dashboard")
st.caption("Manually enter or simulate a sensor reading and run it through AquaGuard's models.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_shap_explainer(_artifacts):
    return shap.TreeExplainer(_artifacts["severity_model"])


artifacts = get_artifacts()
explainer = get_shap_explainer(artifacts)

# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------
st.subheader("1. Sensor Reading Input")

left, right = st.columns([1, 1])

with left:
    segment = st.selectbox("Pipeline Segment", list(SEGMENT_INFO.keys()),
                            format_func=lambda s: f"{s}  ({SEGMENT_INFO[s]['stations'][0]} → {SEGMENT_INFO[s]['stations'][1]})")
    info = SEGMENT_INFO[segment]
    st.caption(f"Pipe diameter: {info['diameter']}\" · Length: {info['length']} m")

    simulate = st.toggle("🎲 Simulate a random reading", value=True,
                          help="Auto-fill plausible values you can still edit below.")

    if simulate:
        rng = np.random.default_rng()
        scenario = st.radio("Scenario to simulate", ["Normal", "Small leak", "Medium leak", "Severe leak"],
                             horizontal=True)
        base_flow = rng.uniform(4.5, 6.5)
        p_up = 55 + rng.normal(0, 1.0)
        loss_map = {"Normal": (0, 0.2), "Small leak": (0.5, 2.5),
                    "Medium leak": (2.5, 6.0), "Severe leak": (6.0, 15.0)}
        dp_map = {"Normal": (0, 0.5), "Small leak": (1.0, 3.0),
                  "Medium leak": (3.0, 7.0), "Severe leak": (7.0, 15.0)}
        loss = rng.uniform(*loss_map[scenario])
        dp_extra = rng.uniform(*dp_map[scenario])
        flow_down_default = round(max(base_flow - loss / 60, 0.05), 3)
        p_down_default = round(p_up - 1.5 - dp_extra, 2)
        vib_default = round(0.2 + (dp_extra * 0.02 if scenario != "Normal" else 0) + rng.normal(0, 0.03), 3)
        acoustic_default = round(30 + (dp_extra * 1.2 if scenario != "Normal" else 0) + rng.normal(0, 1), 2)
        p_up_default = round(p_up, 2)
        flow_up_default = round(base_flow, 3)
    else:
        p_up_default, p_down_default = 55.0, 53.0
        flow_up_default, flow_down_default = 5.0, 4.9
        vib_default, acoustic_default = 0.2, 30.0

with right:
    pressure_upstream = st.number_input("Upstream Pressure (psi)", value=p_up_default, step=0.1)
    pressure_downstream = st.number_input("Downstream Pressure (psi)", value=p_down_default, step=0.1)
    flow_upstream = st.number_input("Upstream Flow (L/s)", value=flow_up_default, step=0.05)
    flow_downstream = st.number_input("Downstream Flow (L/s)", value=flow_down_default, step=0.05)
    vibration = st.number_input("Vibration (g)", value=vib_default, step=0.01)
    acoustic = st.number_input("Acoustic Level (dB proxy)", value=acoustic_default, step=0.5)

st.divider()

if st.button("🔍 Run AquaGuard Prediction", type="primary", use_container_width=True):
    reading = {
        "segment": segment,
        "pressure_upstream_psi": pressure_upstream,
        "pressure_downstream_psi": pressure_downstream,
        "flow_upstream_lps": flow_upstream,
        "flow_downstream_lps": flow_downstream,
        "vibration_g": vibration,
        "acoustic_db": acoustic,
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

    st.subheader("2. Prediction Results")

    status_color = {"none": "🟢", "small": "🟡", "medium": "🟠", "severe": "🔴"}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", f"{status_color[result['severity']]} "
              f"{'LEAK DETECTED' if result['severity'] != 'none' else 'Normal'}")
    c2.metric("Severity", result["severity"].capitalize(),
              f"{result['severity_confidence']:.0%} confidence")
    c3.metric("Predicted Location", result["location"] if result["severity"] != "none" else "—",
              f"{result['location_confidence']:.0%} confidence" if result["severity"] != "none" else "")
    c4.metric("Est. Water Loss", f"{result['water_loss_lpm']} L/min")

    if result["is_anomaly"]:
        st.warning(f"⚠️ Anomaly detector flagged this reading as unusual "
                   f"(score: {result['anomaly_score']:.3f})")
    else:
        st.success(f"✅ Anomaly detector: pattern looks normal (score: {result['anomaly_score']:.3f})")

    # Severity probability bar chart
    proba_fig = go.Figure(go.Bar(
        x=list(result["severity_proba"].values()),
        y=list(result["severity_proba"].keys()),
        orientation="h",
        marker_color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
    ))
    proba_fig.update_layout(title="Severity Class Probabilities", height=280,
                             xaxis_title="Probability", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(proba_fig, use_container_width=True)

    # SHAP explanation
    st.subheader("3. Why did the model decide this? (SHAP)")
    shap_values = explainer.shap_values(result["X_scaled"])
    sev_idx = list(artifacts["severity_encoder"].classes_).index(result["severity"])

    if isinstance(shap_values, list):
        vals = shap_values[sev_idx][0]
    else:
        vals = shap_values[0, :, sev_idx] if shap_values.ndim == 3 else shap_values[0]

    feat_names = artifacts["feature_cols"]
    shap_fig = go.Figure(go.Bar(
        x=vals,
        y=feat_names,
        orientation="h",
        marker_color=["#e74c3c" if v > 0 else "#3498db" for v in vals],
    ))
    shap_fig.update_layout(
        title=f"Feature contributions toward '{result['severity']}' prediction",
        height=350, xaxis_title="SHAP value (impact on prediction)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.caption("🔴 Red bars push the prediction toward this class · 🔵 Blue bars push away from it.")

    # Save to DB
    st.subheader("4. Log this reading")
    notes = st.text_input("Notes (optional)", "")
    if st.button("💾 Save to Database"):
        record = {
            "segment": segment,
            "station_upstream": SEGMENT_INFO[segment]["stations"][0],
            "station_downstream": SEGMENT_INFO[segment]["stations"][1],
            "pipe_diameter_in": SEGMENT_INFO[segment]["diameter"],
            "pipe_length_m": SEGMENT_INFO[segment]["length"],
            "pressure_upstream_psi": reading["pressure_upstream_psi"],
            "pressure_downstream_psi": reading["pressure_downstream_psi"],
            "pressure_drop_psi": reading["pressure_upstream_psi"] - reading["pressure_downstream_psi"],
            "flow_upstream_lps": reading["flow_upstream_lps"],
            "flow_downstream_lps": reading["flow_downstream_lps"],
            "flow_diff_lps": reading["flow_upstream_lps"] - reading["flow_downstream_lps"],
            "vibration_g": reading["vibration_g"],
            "acoustic_db": reading["acoustic_db"],
            "is_anomaly": int(result["is_anomaly"]),
            "predicted_severity": result["severity"],
            "severity_confidence": result["severity_confidence"],
            "predicted_location": result["location"],
            "location_confidence": result["location_confidence"],
            "estimated_water_loss_lpm": result["water_loss_lpm"],
            "notes": notes,
        }
        insert_prediction(record)
        st.success("Saved! Head to the Analysis Dashboard to see it in the history.")
