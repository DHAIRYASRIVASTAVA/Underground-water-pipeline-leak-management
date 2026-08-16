import streamlit as st

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
)

st.title("💧 AquaGuard AI")
st.caption("AI-Based Underground Water Pipeline Leak Detection and Localization")

st.markdown("""
Welcome to **AquaGuard AI** — a software-only, ML-powered system for detecting,
localizing, and quantifying leaks in underground water pipelines.

### How it works
1. **Data Entry Dashboard** — Enter or simulate sensor readings (pressure, flow,
   vibration, acoustic) for a pipeline segment.
2. AquaGuard runs four models: **anomaly detection**, **severity classification**,
   **leak location**, and **water-loss estimation** — plus **SHAP** explanations
   for every prediction.
3. **Analysis Dashboard** — Review prediction history, trends, and network-wide
   leak statistics, all logged to SQLite.

### Pipeline network (simulated)
```
S1 ──SEG1──▶ S2 ──SEG2──▶ S3 ──SEG3──▶ S4 ──SEG4──▶ S5
```

Use the sidebar to navigate between the two dashboards.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sensor Stations", "5")
col2.metric("Monitored Segments", "4")
col3.metric("ML Models", "4")
col4.metric("Explainability", "SHAP ✅")

st.info("👈 Start with **Data Entry** to log a reading, then check **Analysis** for insights.")
