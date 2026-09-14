

import streamlit as st
from pathlib import Path

CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"

STATUS_LABEL = {"none": "NORMAL", "normal": "NORMAL", "leak": "LEAK DETECTED", "burst": "BURST DETECTED"}


def render_html(html: str):
    """Render raw HTML safely: strips leading whitespace from every line
    so Streamlit's Markdown pass doesn't turn indented lines into a code
    block."""
    cleaned = "\n".join(line.lstrip() for line in html.strip("\n").split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css():
    css = CSS_PATH.read_text()
    render_html(f"<style>\n{css}\n</style>")


def hero(eyebrow: str, title_html: str, subtitle: str):
    render_html(f"""
    <div class="eyebrow">{eyebrow}</div>
    <div class="hero-title">{title_html}</div>
    <p class="hero-sub">{subtitle}</p>
    """)


def section_head(num: str, title: str):
    render_html(f"""
    <div class="section-head">
        <span class="num">{num}</span>
        <span class="title">{title}</span>
    </div>
    """)


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
    render_html(html)


def status_line(severity: str):
    render_html(f"""
    <div class="status-line">
        <span class="led s-{severity}"></span>
        <span class="status-text s-{severity}">{STATUS_LABEL.get(severity, severity.upper())}</span>
    </div>
    """)


def panel_open(label: str = ""):
    label_html = f'<div class="panel-label">{label}</div>' if label else ""
    render_html(f'<div class="panel"><span class="bl"></span><span class="br"></span>{label_html}')


def panel_close():
    render_html("</div>")


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


def legend_card(status: str, label: str) -> str:
    return f"""
    <div class="panel" style="padding:14px 16px; text-align:center;">
        <span class="bl"></span><span class="br"></span>
        <span class="led s-{status}" style="display:inline-block; margin-bottom:6px;"></span>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:var(--steel);">{label}</div>
    </div>
    """


def recommendation_panel(rec: dict):
    """Renders the action-recommendation card: urgency tier, response
    window, location, water-loss note, and a checklist of concrete next
    steps. `rec` comes from utils.recommendations.get_recommendation()."""
    actions_html = "".join(f'<li>{a}</li>' for a in rec["actions"])
    loss_html = f'<div class="rec-loss">{rec["loss_note"]}</div>' if rec.get("loss_note") else ""
    html = (
        f'<div class="panel rec-panel s-{rec["color"]}">'
        '<span class="bl"></span><span class="br"></span>'
        f'<div class="rec-tier-row">'
        f'<span class="led s-{rec["color"]}"></span>'
        f'<span class="rec-tier s-{rec["color"]}">{rec["tier"]}</span>'
        f'<span class="rec-window">{rec["response_window"]}</span>'
        f'</div>'
        f'<div class="rec-headline">{rec["headline"]}</div>'
        f'<div class="rec-location">📍 {rec["location_text"]}</div>'
        f'{loss_html}'
        f'<ul class="rec-actions">{actions_html}</ul>'
        '</div>'
    )
    render_html(html)


def sensor_network_map(sensor_list: list, active_sensor: str = None, alert_sensor: str = None, alert_color: str = "severe"):
    """10 independent point-sensor nodes laid out in a row. Unlike the
    earlier synthetic-data version, this dataset has no known pipe
    topology connecting sensors (each row is one sensor's own reading,
    not an upstream/downstream pair) — so nodes are drawn standalone,
    not linked by animated segment lines. The selected sensor pulses
    blue; if it's flagged leak/burst, it glows the alert color instead."""
    n = len(sensor_list)
    width = 900
    margin = 60
    spacing = (width - 2 * margin) / max(n - 1, 1)

    nodes = ""
    for i, sid in enumerate(sensor_list):
        x = margin + i * spacing
        is_active = sid == active_sensor
        is_alert = sid == alert_sensor
        color = f"var(--signal-{alert_color})" if is_alert else "var(--water)"
        radius = "12" if (is_active or is_alert) else "9"
        pulse = ""
        if is_alert:
            pulse = (
                f'<circle cx="{x}" cy="70" r="18" fill="none" stroke="{color}" stroke-width="2" opacity="0.6">'
                f'<animate attributeName="r" values="14;26;14" dur="1.1s" repeatCount="indefinite" />'
                f'<animate attributeName="opacity" values="0.6;0;0.6" dur="1.1s" repeatCount="indefinite" />'
                f'</circle>'
            )
        elif is_active:
            pulse = (
                f'<circle cx="{x}" cy="70" r="16" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.5">'
                f'<animate attributeName="r" values="12;20;12" dur="1.8s" repeatCount="indefinite" />'
                f'<animate attributeName="opacity" values="0.5;0;0.5" dur="1.8s" repeatCount="indefinite" />'
                f'</circle>'
            )
        nodes += (
            f'{pulse}'
            f'<circle cx="{x}" cy="70" r="{radius}" fill="var(--ink)" stroke="{color}" stroke-width="2.5" />'
            f'<circle cx="{x}" cy="70" r="3.5" fill="{color}" />'
            f'<text x="{x}" y="102" text-anchor="middle" font-family="IBM Plex Mono" font-size="12" '
            f'fill="var(--steel)" font-weight="500">{sid}</text>'
        )

    svg = (
        '<div class="panel" style="padding:24px 10px 14px 10px;">'
        '<span class="bl"></span><span class="br"></span>'
        '<div class="panel-label" style="padding-left:14px;">Sensor Network — 10 independent monitoring points</div>'
        '<svg viewBox="0 0 900 120" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto;">'
        f'{nodes}'
        '</svg>'
        '</div>'
    )
    render_html(svg)


def themed_layout(title: str = None, **overrides):
    """Merge the shared Plotly theme with a chart-specific title and any
    other layout overrides. PLOTLY_TEMPLATE['layout'] already carries a
    'title' key (for font styling only), so passing title="..." directly
    into fig.update_layout(**PLOTLY_TEMPLATE['layout'], title=...) collides
    with it — always build the layout dict through this helper instead."""
    layout = dict(PLOTLY_TEMPLATE["layout"])
    if title is not None:
        layout["title"] = dict(text=title, font=layout["title"]["font"])
    layout.update(overrides)
    return layout


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
