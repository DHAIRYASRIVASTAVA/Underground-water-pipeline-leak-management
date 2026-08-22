import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from db.database import fetch_all, fetch_stats, clear_all
from utils.ui import inject_css, hero, section_head, readout_grid, themed_layout

st.set_page_config(page_title="Analysis — AquaGuard AI", page_icon="📊", layout="wide")
inject_css()

hero(
    eyebrow="CONSOLE · NETWORK ANALYSIS",
    title_html="📊 Analysis",
    subtitle="Historical predictions, severity trends, and network-wide leak "
              "statistics — pulled straight from the logged SQLite history.",
)
st.write("")

cols, rows = fetch_all(limit=1000)
df = pd.DataFrame(rows, columns=cols)
stats = fetch_stats()

readout_grid([
    {"label": "Readings Logged", "value": str(stats["total_readings"])},
    {"label": "Leak Events", "value": str(stats["leak_events"]), "status": "severe" if stats["leak_events"] else "none"},
    {"label": "Total Water Loss", "value": f"{stats['total_water_loss_lpm']:.2f}", "sub": "liters / minute", "accent": True},
])

st.write("")

if df.empty:
    st.info("No readings logged yet. Head to **Data Entry** to log your first reading.")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])

section_head("01", "Filters")
with st.container():
    f1, f2 = st.columns(2)
    with f1:
        seg_filter = st.multiselect("Segment", sorted(df["segment"].unique()),
                                     default=sorted(df["segment"].unique()))
    with f2:
        sev_filter = st.multiselect("Severity", sorted(df["predicted_severity"].unique()),
                                     default=sorted(df["predicted_severity"].unique()))

df_view = df[df["segment"].isin(seg_filter) & df["predicted_severity"].isin(sev_filter)]

st.write("")
section_head("02", "Trends")

STATUS_COLORS = {"none": "#3ECF8E", "small": "#F5B942", "medium": "#F0883E", "severe": "#EF5350"}

col1, col2 = st.columns(2)
with col1:
    sev_counts = df_view["predicted_severity"].value_counts().reindex(
        ["none", "small", "medium", "severe"]).fillna(0)
    fig1 = px.bar(x=sev_counts.index, y=sev_counts.values, color=sev_counts.index,
                  color_discrete_map=STATUS_COLORS, labels={"x": "Severity", "y": "Count"})
    fig1.update_layout(**themed_layout("Severity Distribution", showlegend=False, height=320))
    st.plotly_chart(fig1, width='stretch')

with col2:
    seg_leak = df_view[df_view["predicted_severity"] != "none"]["segment"].value_counts()
    if len(seg_leak) > 0:
        fig2 = px.pie(names=seg_leak.index, values=seg_leak.values, hole=0.55,
                      color_discrete_sequence=["#3FA9E0", "#1B6FA8", "#F0883E", "#EF5350"])
        fig2.update_layout(**themed_layout("Leak Events by Segment", height=320))
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No leak events in the current filter — nothing to break down by segment.")

col3, col4 = st.columns(2)
with col3:
    fig3 = px.line(df_view.sort_values("created_at"), x="created_at", y="estimated_water_loss_lpm",
                    color="segment", markers=True,
                    color_discrete_sequence=["#3FA9E0", "#3ECF8E", "#F5B942", "#EF5350"])
    fig3.update_layout(**themed_layout("Water Loss Over Time (L/min)", height=320))
    st.plotly_chart(fig3, width='stretch')

with col4:
    fig4 = px.scatter(df_view, x="pressure_drop_psi", y="flow_diff_lps", color="predicted_severity",
                       color_discrete_map=STATUS_COLORS, hover_data=["segment", "estimated_water_loss_lpm"])
    fig4.update_layout(**themed_layout("Pressure Drop vs Flow Difference", height=320))
    st.plotly_chart(fig4, width='stretch')

st.write("")
section_head("03", "Reading History")

st.dataframe(
    df_view[[
        "created_at", "segment", "predicted_severity", "severity_confidence",
        "predicted_location", "estimated_water_loss_lpm", "is_anomaly", "notes"
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
