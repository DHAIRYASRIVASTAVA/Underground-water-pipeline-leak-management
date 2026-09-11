import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from db.database import fetch_all, fetch_stats, clear_all
from utils.ui import inject_css, hero, section_head, themed_layout, render_html
from utils.js_components import counter_grid
from utils.inference import load_artifacts

st.set_page_config(page_title="Analysis — AquaGuard AI", page_icon="📊", layout="wide")
inject_css()

hero(
    eyebrow="CONSOLE · NETWORK ANALYSIS",
    title_html="📊 Analysis",
    subtitle="Historical predictions, status trends, and model performance — "
              "pulled straight from the logged SQLite history and the training run.",
)
st.write("")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

cols, rows = fetch_all(limit=1000)
df = pd.DataFrame(rows, columns=cols)
stats = fetch_stats()

counter_grid([
    {"label": "Readings Logged", "value": stats["total_readings"]},
    {"label": "Leak/Burst Events", "value": stats["incident_events"], "status": "medium" if stats["incident_events"] else "none"},
    {"label": "Burst Events", "value": stats["burst_events"], "status": "severe" if stats["burst_events"] else "none"},
])

st.write("")
section_head("00", "Model Performance (5-fold cross-validation)")
st.caption(
    "These numbers come from 5-fold stratified cross-validation across all 1,000 REAL rows — "
    "not a single lucky test split. Inside each fold, the training data was augmented with "
    "synthetic leak/burst rows sampled from that fold's own real-class statistics (never touching "
    "the validation fold) — every prediction below was made on a real, non-synthetic reading."
)

report = artifacts["status_report"]
perf_cols = st.columns(3)
for col, cls in zip(perf_cols, ["normal", "leak", "burst"]):
    with col:
        m = report.get(cls, {})
        render_html(f"""
<div class="panel" style="padding:16px 18px;">
<span class="bl"></span><span class="br"></span>
<div class="panel-label">{cls.upper()}</div>
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:var(--text); line-height:1.9;">
Precision: <span style="color:var(--water);">{m.get('precision', 0):.2f}</span><br>
Recall: <span style="color:var(--water);">{m.get('recall', 0):.2f}</span><br>
F1-score: <span style="color:var(--water);">{m.get('f1-score', 0):.2f}</span><br>
Support: <span style="color:var(--steel);">{int(m.get('support', 0))} real rows</span>
</div>
</div>
""")

st.warning(
    "⚠️ Leak and burst are still rare in the underlying real data (19 and 10 rows out of 1,000) — "
    "the training set was augmented to ~750 examples of each (roughly balanced with the ~780 "
    "normal training rows) so the model has enough to learn from. This pushed burst recall up to "
    "90%, but also pushed leak precision down (more false positives) — a known trade-off when "
    "training data is balanced further from its real-world proportions. Every number above is "
    "still measured on real, held-out rows only — normal-class performance stays near-perfect "
    "because ~97% of the real data is normal readings."
)

st.write("")

if df.empty:
    st.info("No readings logged yet. Head to **Data Entry** to log your first reading.")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])

section_head("01", "Filters")
with st.container():
    f1, f2 = st.columns(2)
    with f1:
        sensor_filter = st.multiselect("Sensor", sorted(df["sensor_id"].unique()),
                                        default=sorted(df["sensor_id"].unique()))
    with f2:
        status_filter = st.multiselect("Status", sorted(df["predicted_status"].unique()),
                                        default=sorted(df["predicted_status"].unique()))

df_view = df[df["sensor_id"].isin(sensor_filter) & df["predicted_status"].isin(status_filter)]

st.write("")
section_head("02", "Trends")

STATUS_COLORS = {"normal": "#3ECF8E", "leak": "#F0883E", "burst": "#EF5350"}

col1, col2 = st.columns(2)
with col1:
    status_counts = df_view["predicted_status"].value_counts().reindex(
        ["normal", "leak", "burst"]).fillna(0)
    fig1 = px.bar(x=status_counts.index, y=status_counts.values, color=status_counts.index,
                  color_discrete_map=STATUS_COLORS, labels={"x": "Status", "y": "Count"})
    fig1.update_layout(**themed_layout("Status Distribution", showlegend=False, height=320))
    st.plotly_chart(fig1, width='stretch')

with col2:
    incident_by_sensor = df_view[df_view["predicted_status"] != "normal"]["sensor_id"].value_counts()
    if len(incident_by_sensor) > 0:
        fig2 = px.pie(names=incident_by_sensor.index, values=incident_by_sensor.values, hole=0.55,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(**themed_layout("Incidents by Sensor", height=320))
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No leak/burst events in the current filter — nothing to break down by sensor.")

col3, col4 = st.columns(2)
with col3:
    fig3 = px.line(df_view.sort_values("created_at"), x="created_at", y="pressure_bar",
                    color="sensor_id", markers=True)
    fig3.update_layout(**themed_layout("Pressure Over Time (bar)", height=320))
    st.plotly_chart(fig3, width='stretch')

with col4:
    fig4 = px.scatter(df_view, x="pressure_bar", y="flow_rate_lps", color="predicted_status",
                       color_discrete_map=STATUS_COLORS, hover_data=["sensor_id", "impact_level"])
    fig4.update_layout(**themed_layout("Pressure vs Flow Rate", height=320))
    st.plotly_chart(fig4, width='stretch')

st.write("")
section_head("03", "Reading History")

st.dataframe(
    df_view[[
        "created_at", "sensor_id", "predicted_status", "status_confidence",
        "impact_level", "impact_drop_pct", "is_anomaly", "notes"
    ]].sort_values("created_at", ascending=False),
    width='stretch', hide_index=True,
)

csv = df_view.to_csv(index=False).encode("utf-8")
st.download_button("⬇  DOWNLOAD FILTERED HISTORY (CSV)", csv, "aquaguard_history.csv", "text/csv",
                    width='stretch')

with st.expander("⚠️ Danger zone"):
    if st.button("🗑️ Clear all logged predictions"):
        clear_all()
        st.success("Database cleared. Refresh the page.")
        st.rerun()
