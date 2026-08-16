import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from db.database import fetch_all, fetch_stats, clear_all

st.set_page_config(page_title="Analysis — AquaGuard AI", page_icon="📊", layout="wide")
st.title("📊 Analysis Dashboard")
st.caption("Historical predictions, trends, and network-wide leak statistics.")

cols, rows = fetch_all(limit=1000)
df = pd.DataFrame(rows, columns=cols)

stats = fetch_stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total Readings Logged", stats["total_readings"])
c2.metric("Leak Events Detected", stats["leak_events"])
c3.metric("Total Estimated Water Loss", f"{stats['total_water_loss_lpm']:.2f} L/min")

st.divider()

if df.empty:
    st.info("No readings logged yet. Go to the **Data Entry** dashboard to add your first reading.")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])

# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
with st.expander("🔎 Filters", expanded=False):
    seg_filter = st.multiselect("Segment", sorted(df["segment"].unique()),
                                 default=sorted(df["segment"].unique()))
    sev_filter = st.multiselect("Severity", sorted(df["predicted_severity"].unique()),
                                 default=sorted(df["predicted_severity"].unique()))

df_view = df[df["segment"].isin(seg_filter) & df["predicted_severity"].isin(sev_filter)]

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    sev_counts = df_view["predicted_severity"].value_counts().reindex(
        ["none", "small", "medium", "severe"]).fillna(0)
    fig1 = px.bar(
        x=sev_counts.index, y=sev_counts.values,
        color=sev_counts.index,
        color_discrete_map={"none": "#2ecc71", "small": "#f1c40f",
                             "medium": "#e67e22", "severe": "#e74c3c"},
        labels={"x": "Severity", "y": "Count"}, title="Severity Distribution",
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    seg_leak = df_view[df_view["predicted_severity"] != "none"]["segment"].value_counts()
    fig2 = px.pie(names=seg_leak.index, values=seg_leak.values,
                  title="Leak Events by Segment")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.line(
        df_view.sort_values("created_at"),
        x="created_at", y="estimated_water_loss_lpm", color="segment",
        markers=True, title="Water Loss Over Time (L/min)",
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.scatter(
        df_view, x="pressure_drop_psi", y="flow_diff_lps",
        color="predicted_severity",
        color_discrete_map={"none": "#2ecc71", "small": "#f1c40f",
                             "medium": "#e67e22", "severe": "#e74c3c"},
        hover_data=["segment", "estimated_water_loss_lpm"],
        title="Pressure Drop vs Flow Difference",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------
# Table + management
# ---------------------------------------------------------------------
st.subheader("📋 Reading History")
st.dataframe(
    df_view[[
        "created_at", "segment", "predicted_severity", "severity_confidence",
        "predicted_location", "estimated_water_loss_lpm", "is_anomaly", "notes"
    ]].sort_values("created_at", ascending=False),
    use_container_width=True, hide_index=True,
)

csv = df_view.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered history as CSV", csv, "aquaguard_history.csv", "text/csv")

with st.expander("⚠️ Danger zone"):
    if st.button("🗑️ Clear all logged predictions"):
        clear_all()
        st.success("Database cleared. Refresh the page.")
        st.rerun()
