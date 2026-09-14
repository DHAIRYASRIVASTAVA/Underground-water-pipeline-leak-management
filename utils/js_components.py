
import streamlit as st

FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700"
               "&family=IBM+Plex+Mono:wght@400;500;600&display=swap');")

COLORS = {
    "ink": "#0A1420", "panel": "#101B2C", "border": "#1E2E45",
    "water": "#3FA9E0", "steel": "#8A9AB0", "steel-dim": "#566579", "text": "#E7EEF6",
    "none": "#3ECF8E", "normal": "#3ECF8E",
    "medium": "#F0883E", "leak": "#F0883E",
    "severe": "#EF5350", "burst": "#EF5350",
}


def counter_grid(items, height=115):
    """items: list of dicts {label, value(float), suffix(optional), sub(optional), status(optional)}
    Renders instrument-style cards whose numbers count up from 0 on load
    via requestAnimationFrame — a small satisfying 'reveal' instead of a
    static number appearing instantly."""
    cards_html = ""
    for i, it in enumerate(items):
        status = it.get("status", "")
        color = COLORS.get(status, COLORS["water"])
        cards_html += f"""<div class="ctr-card" style="border-left-color:{color}">
<div class="ctr-label">{it['label']}</div>
<div class="ctr-value" id="ctr-{i}" style="color:{color}" data-target="{it['value']}" data-suffix="{it.get('suffix', '')}">0</div>
<div class="ctr-sub">{it.get('sub', '')}</div>
</div>"""

    html = f"""<style>
{FONT_IMPORT}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:transparent; font-family:'IBM Plex Mono',monospace; }}
.ctr-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }}
.ctr-card {{ background:{COLORS['panel']}; border:1px solid {COLORS['border']}; border-left:3px solid {COLORS['water']}; border-radius:4px; padding:14px 16px; }}
.ctr-label {{ font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; color:{COLORS['steel-dim']}; margin-bottom:6px; }}
.ctr-value {{ font-size:1.55rem; font-weight:600; }}
.ctr-sub {{ font-size:0.75rem; color:{COLORS['steel']}; margin-top:2px; }}
</style>
<div class="ctr-grid">{cards_html}</div>
<script>
document.querySelectorAll('.ctr-value').forEach(function(el) {{
    var target = parseFloat(el.getAttribute('data-target'));
    var suffix = el.getAttribute('data-suffix') || '';
    var isInt = Number.isInteger(target);
    var duration = 900;
    var start = null;
    function step(ts) {{
        if (!start) start = ts;
        var progress = Math.min((ts - start) / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        var current = target * eased;
        el.textContent = (isInt ? Math.round(current) : current.toFixed(2)) + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
}});
</script>"""
    st.iframe(html, height=height)


def interactive_sensor_map(sensor_list, baselines, active_sensor=None, alert_sensor=None, alert_color="severe", height=170):
    """10 independent sensor nodes with real JS hover tooltips showing
    each sensor's own baseline pressure (from training data). No segment
    lines between nodes — this dataset has no known pipe topology linking
    sensors, unlike the earlier synthetic-data version, so drawing fake
    connections would misrepresent the data."""
    n = len(sensor_list)
    width, margin = 900, 60
    spacing = (width - 2 * margin) / max(n - 1, 1)

    nodes = ""
    for i, sid in enumerate(sensor_list):
        x = margin + i * spacing
        is_active = sid == active_sensor
        is_alert = sid == alert_sensor
        color = COLORS.get(alert_color, COLORS['burst']) if is_alert else COLORS['water']
        radius = 12 if (is_active or is_alert) else 9
        baseline = baselines.get(sid, baselines.get('_global', 0))
        info = f"{sid} · baseline {baseline:.2f} bar"
        nodes += (
            f'<g class="hit" data-info="{info}">'
            f'<circle cx="{x}" cy="70" r="{radius}" fill="{COLORS["ink"]}" stroke="{color}" stroke-width="2.5" />'
            f'<circle cx="{x}" cy="70" r="3.5" fill="{color}" />'
            f'<circle cx="{x}" cy="70" r="20" fill="transparent" />'
            f'<text x="{x}" y="102" text-anchor="middle" font-size="12" fill="{COLORS["steel"]}" font-weight="500">{sid}</text>'
            f'</g>'
        )

    html = f"""<style>
{FONT_IMPORT}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:transparent; font-family:'IBM Plex Mono',monospace; }}
.wrap {{ position:relative; padding:14px 10px 4px 10px; background:{COLORS['panel']}; border:1px solid {COLORS['border']}; border-radius:4px; }}
.wrap-label {{ font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; color:{COLORS['steel-dim']}; padding-left:8px; margin-bottom:6px; }}
svg text {{ font-family:'IBM Plex Mono',monospace; }}
.hit {{ cursor:pointer; }}
#tooltip {{
    position:absolute; pointer-events:none; opacity:0; transition:opacity 0.12s ease;
    background:{COLORS['ink']}; border:1px solid {COLORS['water']}; border-radius:4px;
    padding:5px 10px; font-size:12px; color:{COLORS['text']}; white-space:nowrap;
    transform:translate(-50%,-135%); z-index:10;
}}
</style>
<div class="wrap">
<div class="wrap-label">Sensor Network — 10 independent points &nbsp;·&nbsp; hover for baseline</div>
<svg viewBox="0 0 900 120" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; display:block;">
{nodes}
</svg>
<div id="tooltip"></div>
</div>
<script>
var tip = document.getElementById('tooltip');
var wrap = document.querySelector('.wrap');
document.querySelectorAll('.hit').forEach(function(el) {{
    el.addEventListener('mouseenter', function() {{
        tip.textContent = el.getAttribute('data-info');
        tip.style.opacity = 1;
    }});
    el.addEventListener('mousemove', function(e) {{
        var rect = wrap.getBoundingClientRect();
        tip.style.left = (e.clientX - rect.left) + 'px';
        tip.style.top = (e.clientY - rect.top) + 'px';
    }});
    el.addEventListener('mouseleave', function() {{
        tip.style.opacity = 0;
    }});
}});
</script>"""
    st.iframe(html, height=height)


def radial_gauge(label, value, max_value, unit="", color="#3FA9E0", height=190):
    """Animated semicircular gauge — needle sweeps from 0 to the target
    value on load. Good for a single headline number like water-loss
    estimate."""
    pct = max(0.0, min(value / max_value, 1.0)) if max_value else 0.0
    html = f"""<style>
{FONT_IMPORT}
body {{ margin:0; background:transparent; font-family:'IBM Plex Mono',monospace; text-align:center; }}
.g-label {{ font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:{COLORS['steel-dim']}; margin-top:2px; }}
.g-value {{ font-size:1.3rem; font-weight:600; color:{color}; }}
</style>
<svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:220px;">
<path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="{COLORS['border']}" stroke-width="14" stroke-linecap="round"/>
<path id="arc" d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
      stroke-dasharray="251.2" stroke-dashoffset="251.2"/>
<line id="needle" x1="100" y1="110" x2="100" y2="40" stroke="{COLORS['text']}" stroke-width="3" stroke-linecap="round"
      transform="rotate(-90 100 110)"/>
<circle cx="100" cy="110" r="6" fill="{COLORS['text']}"/>
</svg>
<div class="g-value" id="gval">0{unit}</div>
<div class="g-label">{label}</div>
<script>
var arc = document.getElementById('arc');
var needle = document.getElementById('needle');
var gval = document.getElementById('gval');
var target = {value};
var targetPct = {pct};
var duration = 900;
var start = null;
function step(ts) {{
    if (!start) start = ts;
    var progress = Math.min((ts - start) / duration, 1);
    var eased = 1 - Math.pow(1 - progress, 3);
    var curPct = targetPct * eased;
    var curAngle = -90 + curPct * 180;
    var curVal = target * eased;
    arc.style.strokeDashoffset = 251.2 * (1 - curPct);
    needle.setAttribute('transform', 'rotate(' + curAngle + ' 100 110)');
    gval.textContent = curVal.toFixed(2) + '{unit}';
    if (progress < 1) requestAnimationFrame(step);
}}
requestAnimationFrame(step);
</script>"""
    st.iframe(html, height=height)


def live_ticker(label="LIVE NETWORK THROUGHPUT", start_value=48210, unit=" L", height=70):
    """An ambient, continuously-incrementing counter (setInterval, not a
    one-shot animation) — gives the homepage a 'system is live' feel while
    idle. Purely cosmetic / illustrative, resets on every page load."""
    html = f"""<style>
{FONT_IMPORT}
body {{ margin:0; background:transparent; font-family:'IBM Plex Mono',monospace; }}
.tick-wrap {{ background:{COLORS['panel']}; border:1px solid {COLORS['border']}; border-left:3px solid {COLORS['water']};
              border-radius:4px; padding:12px 16px; display:flex; align-items:baseline; justify-content:space-between; }}
.tick-label {{ font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:{COLORS['steel-dim']}; }}
.tick-value {{ font-size:1.3rem; font-weight:600; color:{COLORS['water']}; }}
.dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{COLORS['none']};
        margin-right:6px; box-shadow:0 0 8px 2px rgba(62,207,142,0.55); animation:blip 1.4s ease-in-out infinite; }}
@keyframes blip {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.35; }} }}
</style>
<div class="tick-wrap">
<span class="tick-label"><span class="dot"></span>{label}</span>
<span class="tick-value" id="tv">0{unit}</span>
</div>
<script>
var v = {start_value};
var el = document.getElementById('tv');
el.textContent = v.toLocaleString() + '{unit}';
setInterval(function() {{
    v += Math.floor(Math.random() * 4) + 1;
    el.textContent = v.toLocaleString() + '{unit}';
}}, 400);
</script>"""
    st.iframe(html, height=height)
