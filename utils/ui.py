"""
AquaGuard AI — UI helper components
--------------------------------------
Injects the design-system CSS and renders reusable HTML components
(readout cards, status badges, feature cards, the animated pipeline
schematic) used across all three pages.
"""

import streamlit as st
from pathlib import Path

CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"

STATUS_LABEL = {"none": "NORMAL", "small": "MINOR LEAK", "medium": "MODERATE LEAK", "severe": "SEVERE LEAK"}


def inject_css():
    css = CSS_PATH.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(eyebrow: str, title_html: str, subtitle: str):
    st.markdown(f"""
    <div class="eyebrow">{eyebrow}</div>
    <div class="hero-title">{title_html}</div>
    <p class="hero-sub">{subtitle}</p>
    """, unsafe_allow_html=True)


def section_head(num: str, title: str):
    st.markdown(f"""
    <div class="section-head">
        <span class="num">{num}</span>
        <span class="title">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def readout(label: str, value: str, sub: str = "", status: str = "", accent: bool = False):
    cls = f"s-{status}" if status else ("accent" if accent else "")
    value_cls = "value accent" if accent and not status else "value"
    return f"""
    <div class="readout {cls}">
        <div class="label">{label}</div>
        <div class="{value_cls}">{value}</div>
        {f'<div class="sub">{sub}</div>' if sub else ''}
    </div>
    """


def readout_grid(items: list):
    """items: list of dicts with keys label, value, sub(optional), status(optional), accent(optional)"""
    html = '<div class="readout-grid">'
    for it in items:
        html += readout(
            it.get("label", ""), it.get("value", ""),
            it.get("sub", ""), it.get("status", ""), it.get("accent", False),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def status_line(severity: str):
    st.markdown(f"""
    <div class="status-line">
        <span class="led s-{severity}"></span>
        <span class="status-text s-{severity}">{STATUS_LABEL.get(severity, severity.upper())}</span>
    </div>
    """, unsafe_allow_html=True)


def panel_open(label: str = ""):
    label_html = f'<div class="panel-label">{label}</div>' if label else ""
    st.markdown(f'<div class="panel"><span class="bl"></span><span class="br"></span>{label_html}', unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str, tag: str = ""):
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    return f"""
    <div class="feature-card">
        <span class="icon">{icon}</span>
        <div class="title">{title}</div>
        <div class="desc">{desc}</div>
        {tag_html}
    </div>
    """


def pipeline_schematic(active_segment: str = None, leak_segment: str = None):
    """Animated SVG: 5 sensor stations, 4 segments, flowing-water dash
    animation, pulsing nodes. If leak_segment is set, that segment glows
    red/amber instead of blue."""
    stations = [("S1", 60), ("S2", 260), ("S3", 460), ("S4", 660), ("S5", 860)]
    segs = [("SEG1", 60, 260), ("SEG2", 260, 460), ("SEG3", 460, 660), ("SEG4", 660, 860)]

    seg_paths = ""
    for name, x1, x2 in segs:
        is_leak = leak_segment == name
        is_active = active_segment == name
        color = "var(--signal-severe)" if is_leak else "var(--water)"
        width = "4" if (is_leak or is_active) else "2.5"
        dash_speed = "0.6s" if is_leak else "1.6s"
        seg_paths += f"""
        <line x1="{x1+22}" y1="90" x2="{x2-22}" y2="90"
              stroke="{color}" stroke-width="{width}" stroke-dasharray="10 8"
              opacity="{0.95 if (is_leak or is_active) else 0.55}">
            <animate attributeName="stroke-dashoffset" from="36" to="0" dur="{dash_speed}" repeatCount="indefinite" />
        </line>
        """
        if is_leak:
            mx = (x1 + x2) / 2
            seg_paths += f"""
            <circle cx="{mx}" cy="90" r="7" fill="var(--signal-severe)" opacity="0.85">
                <animate attributeName="r" values="6;12;6" dur="1s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.85;0.15;0.85" dur="1s" repeatCount="indefinite" />
            </circle>
            <text x="{mx}" y="70" text-anchor="middle" font-family="IBM Plex Mono" font-size="11"
                  fill="var(--signal-severe)" letter-spacing="1">LEAK</text>
            """

    node_svgs = ""
    for name, x in stations:
        node_svgs += f"""
        <circle cx="{x}" cy="90" r="9" fill="var(--ink)" stroke="var(--water)" stroke-width="2.5" />
        <circle cx="{x}" cy="90" r="3" fill="var(--water)" />
        <text x="{x}" y="122" text-anchor="middle" font-family="IBM Plex Mono" font-size="13"
              fill="var(--steel)" font-weight="500">{name}</text>
        """

    svg = f"""
    <div class="panel" style="padding:24px 10px 14px 10px;">
        <span class="bl"></span><span class="br"></span>
        <div class="panel-label" style="padding-left:14px;">Live Network Schematic — S1 → S5</div>
        <svg viewBox="0 0 920 150" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto;">
            {seg_paths}
            {node_svgs}
        </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E7EEF6", size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=15, color="#E7EEF6")),
        xaxis=dict(gridcolor="#1E2E45", zerolinecolor="#1E2E45", linecolor="#1E2E45"),
        yaxis=dict(gridcolor="#1E2E45", zerolinecolor="#1E2E45", linecolor="#1E2E45"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=["#3FA9E0", "#3ECF8E", "#F5B942", "#F0883E", "#EF5350"],
        margin=dict(l=10, r=10, t=45, b=10),
    )
)
