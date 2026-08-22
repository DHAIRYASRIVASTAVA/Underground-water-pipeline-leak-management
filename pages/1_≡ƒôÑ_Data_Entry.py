import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import shap
import plotly.graph_objects as go

from utils.inference import load_artifacts, run_prediction, SEGMENT_INFO
from utils.ui import inject_css, hero, section_head, readout_grid, status_line, pipeline_schematic, themed_layout
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
    return shap.TreeExplainer(_artifacts["severity_model"])


artifacts = get_artifacts()
explainer = get_shap_explainer(artifacts)

# ---------------------------------------------------------------------
# Input console
# ---------------------------------------------------------------------
section_head("01", "Sensor Input Console")

left, right = st.columns([1, 1.1])

with left:
    segment = st.selectbox("Pipeline Segment", list(SEGMENT_INFO.keys()),
                            format_func=lambda s: f"{s}  ({SEGMENT_INFO[s]['stations'][0]} → {SEGMENT_INFO[s]['stations'][1]})")
    info = SEGMENT_INFO[segment]
    st.caption(f"⌀ {info['diameter']}\" pipe · {info['length']} m length")

    simulate = st.toggle("🎲 Simulate a scenario", value=True,
                          help="Auto-fill plausible sensor values you can still edit.")

    if simulate:
        rng = np.random.default_rng()
        scenario = st.radio("Scenario", ["Normal", "Small leak", "Medium leak", "Severe leak"], horizontal=True)
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
    r1, r2 = st.columns(2)
    with r1:
        pressure_upstream = st.number_input("Upstream Pressure (psi)", value=p_up_default, step=0.1)
        flow_upstream = st.number_input("Upstream Flow (L/s)", value=flow_up_default, step=0.05)
        vibration = st.number_input("Vibration (g)", value=vib_default, step=0.01)
    with r2:
        pressure_downstream = st.number_input("Downstream Pressure (psi)", value=p_down_default, step=0.1)
        flow_downstream = st.number_input("Downstream Flow (L/s)", value=flow_down_default, step=0.05)
        acoustic = st.number_input("Acoustic Level (dB proxy)", value=acoustic_default, step=0.5)

st.write("")
run_clicked = st.button("▶  RUN PREDICTION", type="primary", width='stretch')

if run_clicked:
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

    st.write("")
    section_head("02", "Prediction Output")

    pipeline_schematic(
        active_segment=segment,
        leak_segment=segment if result["severity"] != "none" else None,
    )

    st.write("")
    status_line(result["severity"])

    readout_grid([
        {"label": "Severity", "value": result["severity"].upper(),
         "sub": f"{result['severity_confidence']:.0%} confidence", "status": result["severity"]},
        {"label": "Location", "value": result["location"] if result["severity"] != "none" else "—",
         "sub": f"{result['location_confidence']:.0%} confidence" if result["severity"] != "none" else "no leak located"},
        {"label": "Water Loss Est.", "value": f"{result['water_loss_lpm']}", "sub": "liters / minute", "accent": True},
        {"label": "Anomaly Check", "value": "FLAGGED" if result["is_anomaly"] else "CLEAR",
         "sub": f"score {result['anomaly_score']:.3f}", "status": "severe" if result["is_anomaly"] else "none"},
    ])

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        proba_fig = go.Figure(go.Bar(
            x=list(result["severity_proba"].values()),
            y=list(result["severity_proba"].keys()),
            orientation="h",
            marker_color=["#3ECF8E", "#F5B942", "#F0883E", "#EF5350"],
        ))
        proba_fig.update_layout(**themed_layout("Severity Class Probabilities",
                                                  height=280, xaxis_title="Probability"))
        st.plotly_chart(proba_fig, width='stretch')

    with col2:
        shap_values = explainer.shap_values(result["X_scaled"])
        sev_idx = list(artifacts["severity_encoder"].classes_).index(result["severity"])
        if isinstance(shap_values, list):
            vals = shap_values[sev_idx][0]
        else:
            vals = shap_values[0, :, sev_idx] if shap_values.ndim == 3 else shap_values[0]

        feat_names = artifacts["feature_cols"]
        shap_fig = go.Figure(go.Bar(
            x=vals, y=feat_names, orientation="h",
            marker_color=["#EF5350" if v > 0 else "#3FA9E0" for v in vals],
        ))
        shap_fig.update_layout(**themed_layout(f"Why '{result['severity']}'? (SHAP)",
                                                 height=280, xaxis_title="Impact on prediction"))
        st.plotly_chart(shap_fig, width='stretch')

    st.caption("🔴 Pushes toward this class · 🔵 Pushes away from it")

    st.write("")
    section_head("03", "Log to Database")
    notes = st.text_input("Notes (optional)", "")
    if st.button("💾 SAVE READING", width='stretch'):
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
        st.success("✅ Saved. Head to the Analysis dashboard to see it in the history.")
