"""
AI-Based Campus E-Rickshaw Demand & Intelligent Dispatch System — Deployment App
Single Streamlit app with three role-based views (Student, Driver, Admin).
Run with:  streamlit run app/streamlit_app.py

Extra (optional) for the clickable/interactive map:  pip install plotly
Without plotly the app still works and shows the static campus map.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import RAW_DIR, MODELS_DIR, FIGURES_DIR, load_locations, zone_distance_matrix  # noqa: E402
from dispatch import DispatchEngine  # noqa: E402

st.set_page_config(
    page_title="Campus E-Rickshaw Dispatch",
    page_icon="🛺",
    layout="wide",
    initial_sidebar_state="expanded"
)# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer,[data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 4rem; max-width: 1280px; }

.stApp { background: radial-gradient(900px 500px at 0% -10%, rgba(14,110,255,0.16), transparent 60%),
         radial-gradient(900px 500px at 100% 0%, rgba(139,92,246,0.14), transparent 55%), #0a0f1d; color: #e8ecf5; }
section[data-testid="stSidebar"] { background: #0d1324; border-right: 1px solid rgba(255,255,255,0.07); }

h1, h2, h3, h4 { color: #f1f5f9; font-weight: 700; }
h1 { background: linear-gradient(90deg,#ffffff,#7dd3fc 60%,#c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h4 { margin: 0 0 .6rem 0; font-size: 1.05rem; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: #141b2d; border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 22px !important; padding: .5rem .7rem;
}
[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  padding: .9rem 1.1rem; border-radius: 16px; }
[data-testid="stMetricLabel"] { color: #7dd3fc; text-transform: uppercase; letter-spacing: .06em; font-size: .75rem; }
[data-testid="stMetricValue"] { font-weight: 800; color: #fff; }

.stButton > button { border: 0; border-radius: 12px; padding: .7rem 1.4rem; font-weight: 700; color: #fff; width: 100%;
  background: linear-gradient(90deg,#0a6cff,#6d5dfc); box-shadow: 0 6px 20px rgba(10,108,255,.35); transition: all .15s ease; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(10,108,255,.55); color: #fff; }
[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stAlert"] { border-radius: 14px; }

.kpi { background: #141b2d; border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; padding: 1rem 1.2rem;
  display: flex; align-items: center; gap: .9rem; }
.kpi .ico { width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-size: 1.4rem;
  background: linear-gradient(135deg, rgba(10,108,255,.35), rgba(139,92,246,.35)); }
.kpi .lbl { font-size: .72rem; color: #8fa3c7; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.kpi .val { font-size: 1.55rem; font-weight: 800; color: #fff; line-height: 1.15; }

.ride-card { background: linear-gradient(160deg,#0a6cff,#2a46d6); border-radius: 22px; padding: 1.4rem 1.5rem; color: #fff;
  box-shadow: 0 16px 40px rgba(10,108,255,.35); height: 100%; }
.ride-card .tag { font-size: .72rem; letter-spacing: .1em; opacity: .85; font-weight: 700; }
.ride-card .big { font-size: 2.2rem; font-weight: 800; margin: .2rem 0 .8rem 0; }
.ride-card .row { display: flex; justify-content: space-between; padding: .55rem 0; border-top: 1px solid rgba(255,255,255,.25); font-size: .92rem; }
.ride-card .row b { font-weight: 700; }

.info-card { background: #1a2338; border: 1px solid rgba(255,255,255,0.07); border-radius: 22px; padding: 1.2rem 1.4rem; height: 100%; }
.info-card .hd { font-size: .72rem; color: #8fa3c7; letter-spacing: .12em; font-weight: 700; margin-bottom: .6rem; }
.info-card .row { display: flex; justify-content: space-between; gap: 1rem; padding: .5rem 0; border-bottom: 1px solid rgba(255,255,255,0.07); font-size: .92rem; color: #aab8d4; }
.info-card .row b { color: #fff; text-align: right; }

.drv-list { max-height: 720px; overflow-y: auto; padding-right: 4px; }
.drv { display: flex; align-items: center; gap: .8rem; background: #1a2338; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding: .7rem .9rem; margin-bottom: .6rem; transition: transform .15s ease, border-color .15s ease; }
.drv:hover { transform: translateY(-2px); border-color: rgba(125,211,252,.45); }
.drv .av { width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; font-size: .8rem; color: #fff; flex: none; }
.drv .meta { flex: 1; min-width: 0; line-height: 1.25; }
.drv .meta b { color: #fff; font-size: .92rem; }
.drv .meta span { display: block; color: #8fa3c7; font-size: .75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pill { font-size: .68rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; flex: none; }
.pill.ok { background: rgba(74,222,128,.18); color: #4ade80; border: 1px solid rgba(74,222,128,.45); }
.pill.busy { background: rgba(251,191,36,.15); color: #fbbf24; border: 1px solid rgba(251,191,36,.45); }

.table-wrap { overflow-x: auto; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); margin-bottom: 1rem; }
.pro-table { width: 100%; border-collapse: collapse; }
.pro-table th { text-align: left; padding: .8rem 1rem; color: #7dd3fc; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;
  font-weight: 700; background: linear-gradient(90deg, rgba(10,108,255,0.2), rgba(139,92,246,0.18)); }
.pro-table td { padding: .7rem 1rem; color: #e8ecf5; border-top: 1px solid rgba(255,255,255,0.06); }
.pro-table tr:nth-child(even) td { background: rgba(255,255,255,0.025); }
.pro-table tr:hover td { background: rgba(125,211,252,0.08); }

/* radar (driver-matching animation) */
.radar-box { background: #1a2338; border: 1px solid rgba(255,255,255,0.07); border-radius: 22px; padding: 1rem 1.4rem; text-align: center; margin: .8rem 0; }
.radar { position: relative; height: 170px; overflow: hidden; }
.radar .ring { position: absolute; left: 50%; top: 50%; width: 20px; height: 20px; margin: -10px 0 0 -10px; border-radius: 50%;
  border: 2px solid rgba(56,189,248,.7); animation: ping 2s ease-out infinite; }
.radar .r2 { animation-delay: .65s; } .radar .r3 { animation-delay: 1.3s; }
.radar .core { position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%); font-size: 2rem; }
@keyframes ping { 0% { transform: scale(1); opacity: .9; } 100% { transform: scale(9); opacity: 0; } }
.radar-txt { color: #7dd3fc; font-weight: 600; letter-spacing: .03em; }

/* live trip progress */
.trip-prog { background: #1a2338; border: 1px solid rgba(255,255,255,.07); border-radius: 18px; padding: 1rem 1.2rem; margin: .8rem 0; }
.tp-top { display: flex; justify-content: space-between; margin-bottom: .6rem; color: #fff; }
.tp-top span { color: #8fa3c7; font-size: .85rem; }
.bar { height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }
.fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#0ea5e9,#6d5dfc,#c084fc); transition: width .2s ease; }
.steps { display: flex; justify-content: space-between; margin-top: .7rem; font-size: .72rem; color: #6b7a99; }
.steps .on { color: #7dd3fc; font-weight: 700; }

@keyframes fadeUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform:none; } }
.kpi, .ride-card, .info-card, .table-wrap, [data-testid="stVerticalBlockBorderWrapper"] { animation: fadeUp .55s ease both; }
@keyframes drive { 0% { left:-60px; } 100% { left:100%; } }
.road { position: relative; height: 56px; overflow: hidden; border-radius: 16px; margin: .2rem 0 1.1rem;
  background: linear-gradient(135deg, rgba(10,108,255,0.12), rgba(139,92,246,0.12)); border: 1px solid rgba(255,255,255,0.08); }
.road::after { content: ""; position: absolute; left: 0; right: 0; bottom: 12px; border-top: 2px dashed rgba(255,255,255,0.22); }
.rick { position: absolute; bottom: 16px; font-size: 1.8rem; animation: drive 9s linear infinite; }
.rick.two { animation-delay: -4.5s; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#141b2d", "axes.facecolor": "#141b2d", "savefig.facecolor": "#141b2d",
    "text.color": "#e8ecf5", "axes.labelcolor": "#e8ecf5", "axes.edgecolor": "#3a4566",
    "xtick.color": "#b6c0d6", "ytick.color": "#b6c0d6",
})


# =============================================================================
# UI HELPERS
# =============================================================================
def show_table(df):
    html = df.to_html(index=False, classes="pro-table", border=0)
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)


def page_title(emoji, text):
    st.markdown(f'<h1><span style="-webkit-text-fill-color: initial;">{emoji}</span> {text}</h1>', unsafe_allow_html=True)


def road_animation():
    st.markdown('<div class="road"><span class="rick">🛺</span><span class="rick two">🛺</span></div>', unsafe_allow_html=True)


def spacer(h=".8rem"):
    st.markdown(f"<div style='height:{h}'></div>", unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (ico, lbl, val) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="ico">{ico}</div><div><div class="lbl">{lbl}</div>'
                     f'<div class="val">{val}</div></div></div>', unsafe_allow_html=True)
    spacer(".9rem")


def driver_list_html(df):
    rows = []
    for _, r in df.iterrows():
        hue = (sum(ord(c) for c in r["driver_id"]) * 47) % 360
        ok = r["status"] == "available"
        rows.append(
            f'<div class="drv"><div class="av" style="background:linear-gradient(135deg,hsl({hue},80%,55%),hsl({(hue+50)%360},80%,45%))">'
            f'{r["driver_id"][-3:]}</div><div class="meta"><b>{r["driver_id"]}</b>'
            f'<span>{r["home_zone"]} · {r["shift_type"]} · P={r["p_available"]:.0%}</span></div>'
            f'<span class="pill {"ok" if ok else "busy"}">{"Available" if ok else "Busy"}</span></div>')
    return '<div class="drv-list">' + "".join(rows) + '</div>'


def progress_html(pct, label, sub, stage):
    names = ["Matched", "On the way", "Arrived", "On trip", "Completed"]
    dots = "".join(f'<span class="{"on" if i <= stage else ""}">{n}</span>' for i, n in enumerate(names))
    return (f'<div class="trip-prog"><div class="tp-top"><b>{label}</b><span>{sub}</span></div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div><div class="steps">{dots}</div></div>')


def radar_html(text):
    return ('<div class="radar-box"><div class="radar"><div class="ring"></div><div class="ring r2"></div>'
            '<div class="ring r3"></div><div class="core">🛺</div></div>'
            f'<div class="radar-txt">{text}</div></div>')


# =============================================================================
# DATA / MODELS
# =============================================================================
@st.cache_resource
def load_everything():
    locations = load_locations()
    zone_names, dist_matrix = zone_distance_matrix(locations)
    drivers = pd.read_csv(RAW_DIR / "drivers.csv")
    rides = pd.read_csv(RAW_DIR / "ride_requests.csv", parse_dates=["timestamp", "date"])
    engine = DispatchEngine()
    demand_model = joblib.load(MODELS_DIR / "demand_model.joblib")
    avail_model = joblib.load(MODELS_DIR / "availability_model.joblib")
    return locations, zone_names, dist_matrix, drivers, rides, engine, demand_model, avail_model


locations, zone_names, dist_matrix, drivers_df, rides_hist, engine, demand_model, avail_model = load_everything()
ZONE_TYPE = locations.set_index("zone_name")["zone_type"].to_dict()
ZONE_POS = {n: (float(x), float(y)) for n, x, y in zip(locations["zone_name"], locations["x_km"], locations["y_km"])}


def cyc(val, period):
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


def zone_stats():
    d = locations.copy()
    d["requests"] = d["zone_name"].map(rides_hist.groupby("pickup_zone").size()).fillna(0)
    d["fail"] = d["zone_name"].map(rides_hist.groupby("pickup_zone")["request_failed"].mean()).fillna(0)
    return d


# =============================================================================
# MAPS
# =============================================================================
def campus_map(pickup=None, drop=None, driver_zones=None, matched_zone=None, height=5.4, **_):
    """Static matplotlib map (fallback when plotly is not installed)."""
    d = zone_stats()
    X, Y = d["x_km"].to_numpy(float), d["y_km"].to_numpy(float)
    pos = ZONE_POS
    BG = "#141b2d"
    fig, ax = plt.subplots(figsize=(9, height))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    seen = set()
    for i in range(len(d)):
        for j in np.argsort(np.hypot(X - X[i], Y - Y[i]))[1:3]:
            k = tuple(sorted((i, j)))
            if k in seen:
                continue
            seen.add(k)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#2a3656", lw=5, zorder=1, solid_capstyle="round")
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#3b4a72", lw=1, ls=(0, (4, 4)), zorder=1)
    size = 140 + 800 * d["requests"].to_numpy() / max(d["requests"].max(), 1)
    cmap = LinearSegmentedColormap.from_list("f", ["#22d3ee", "#8b5cf6", "#f43f5e"])
    colors = cmap(d["fail"].to_numpy() / max(d["fail"].max(), 1e-9))
    for mult, a in [(5.0, 0.05), (3.0, 0.09), (1.8, 0.16)]:
        ax.scatter(X, Y, s=size * mult, c=colors, alpha=a, linewidths=0, zorder=2)
    ax.scatter(X, Y, s=size, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
    for n, x, y, s in zip(d["zone_name"], X, Y, size):
        ax.annotate(n, (x, y), xytext=(0, -(np.sqrt(s) / 2 + 7)), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color="#c7d2e8", zorder=6)
    if pickup in pos and drop in pos:
        (x1, y1), (x2, y2) = pos[pickup], pos[drop]
        ax.plot([x1, x2], [y1, y2], color="#38bdf8", lw=3.2, zorder=4, solid_capstyle="round")
        ax.scatter([x1], [y1], s=520, facecolors="none", edgecolors="#4ade80", linewidths=2.6, zorder=5)
        ax.scatter([x2], [y2], s=520, facecolors="none", edgecolors="#fb7185", linewidths=2.6, zorder=5)
    for z in (driver_zones or []):
        if z in pos:
            ax.scatter([pos[z][0] + 0.04], [pos[z][1] + 0.05], marker="^", s=70, c="#fbbf24",
                       edgecolors="#0b1020", linewidths=0.8, zorder=7)
    if matched_zone in pos:
        ax.scatter([pos[matched_zone][0]], [pos[matched_zone][1]], marker="*", s=520, c="#facc15",
                   edgecolors="#0b1020", linewidths=1, zorder=8)
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.12)
    ax.text(0.01, 0.985, "●  size = demand      ●  cyan → red = failure rate      ▲  available driver      ★  matched driver",
            transform=ax.transAxes, fontsize=7.5, color="#8b9bb8", va="top")
    fig.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close(fig)


ZONE_TRACE = 2                      # index of the clickable zones trace in the plotly figure
PLOT_CFG = {"displayModeBar": False, "scrollZoom": False}


def plotly_map(pickup=None, drop=None, driver_zones=None, matched_zone=None, mover=None, height=540):
    """Interactive campus map (hover for details, click a zone to select it)."""
    d = zone_stats()
    names = d["zone_name"].tolist()
    X, Y = d["x_km"].to_numpy(float), d["y_km"].to_numpy(float)
    req = d["requests"].to_numpy(float)
    fail = d["fail"].to_numpy(float)
    sizes = 18 + 34 * req / max(req.max(), 1)
    cs = [[0, "#22d3ee"], [0.5, "#8b5cf6"], [1, "#f43f5e"]]
    cmax = max(float(fail.max()), 1e-9)

    rx, ry, seen = [], [], set()
    for i in range(len(names)):
        for j in np.argsort(np.hypot(X - X[i], Y - Y[i]))[1:3]:
            k = tuple(sorted((i, j)))
            if k in seen:
                continue
            seen.add(k)
            rx += [X[i], X[j], None]; ry += [Y[i], Y[j], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="#2d3a5c", width=6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", hoverinfo="skip",
                             marker=dict(size=sizes * 2.2, color=fail, colorscale=cs, cmin=0, cmax=cmax, opacity=0.14)))
    custom = np.stack([names, req.astype(int).astype(str), (fail * 100).round(1).astype(str)], axis=-1)
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers+text", text=names, textposition="bottom center",
        textfont=dict(size=10, color="#c7d2e8"), customdata=custom,
        marker=dict(size=sizes, color=fail, colorscale=cs, cmin=0, cmax=cmax, line=dict(color="white", width=1.2)),
        hovertemplate="<b>%{customdata[0]}</b><br>Requests: %{customdata[1]}<br>Failure rate: %{customdata[2]}%<extra></extra>"))

    if pickup in ZONE_POS and drop in ZONE_POS:
        (x1, y1), (x2, y2) = ZONE_POS[pickup], ZONE_POS[drop]
        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", hoverinfo="skip",
                                 line=dict(color="#38bdf8", width=4)))
        fig.add_trace(go.Scatter(x=[x1], y=[y1], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=50, line=dict(color="#4ade80", width=3.5))))
        fig.add_trace(go.Scatter(x=[x2], y=[y2], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=50, line=dict(color="#fb7185", width=3.5))))
    dz = [z for z in (driver_zones or []) if z in ZONE_POS]
    if dz:
        fig.add_trace(go.Scatter(x=[ZONE_POS[z][0] + 0.04 for z in dz], y=[ZONE_POS[z][1] + 0.05 for z in dz],
                                 mode="markers", text=dz, hovertemplate="Available driver · %{text}<extra></extra>",
                                 marker=dict(symbol="triangle-up", size=12, color="#fbbf24", line=dict(color="#0b1020", width=1))))
    if matched_zone in ZONE_POS:
        fig.add_trace(go.Scatter(x=[ZONE_POS[matched_zone][0]], y=[ZONE_POS[matched_zone][1]], mode="markers",
                                 hovertemplate="Matched driver<extra></extra>",
                                 marker=dict(symbol="star", size=28, color="#facc15", line=dict(color="#0b1020", width=1.5))))
    if mover is not None:
        fig.add_trace(go.Scatter(x=[mover[0]], y=[mover[1]], mode="text", text=["🛺"], textfont=dict(size=30), hoverinfo="skip"))

    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
        paper_bgcolor="#141b2d", plot_bgcolor="#141b2d", dragmode="pan", hoverlabel=dict(bgcolor="#1a2338", font_color="#fff"),
        xaxis=dict(visible=False, range=[X.min() - 0.5, X.max() + 0.5]),
        yaxis=dict(visible=False, scaleanchor="x", range=[Y.min() - 0.5, Y.max() + 0.5]),
        annotations=[dict(text="size = demand · cyan → red = failure rate · ▲ available driver · ★ matched driver",
                          xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, xanchor="left", yanchor="top",
                          font=dict(size=11, color="#8b9bb8"))])
    return fig


def render_map(key, **kw):
    """Static (non-clickable) map: plotly if available, otherwise matplotlib."""
    if HAS_PLOTLY:
        st.plotly_chart(plotly_map(**kw), key=key, config=PLOT_CFG)
    else:
        campus_map(**kw)


def zone_from_click(point):
    cd = point.get("customdata")
    z = cd[0] if isinstance(cd, (list, tuple)) else cd
    if z in ZONE_POS:
        return z
    x, y = point.get("x"), point.get("y")
    if x is None or y is None:
        return None
    return min(ZONE_POS, key=lambda n: (ZONE_POS[n][0] - x) ** 2 + (ZONE_POS[n][1] - y) ** 2)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🛺 Campus E-Rickshaw Dispatch")
role = st.sidebar.radio("View as:", ["Student", "Driver", "Admin"])

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario context")
hour = st.sidebar.slider("Hour of day", 0, 23, 13)
weekday_name = st.sidebar.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                                     "Friday", "Saturday", "Sunday"], index=0)
weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(weekday_name)
is_weekend = weekday >= 5
is_exam = st.sidebar.checkbox("Exam period", value=False)
is_raining = st.sidebar.checkbox("Raining", value=False)
st.sidebar.caption("These controls simulate the live context a real deployment would "
                    "read from the system clock + a weather/calendar feed.")


def time_block_of(h):
    if h >= 23 or h < 6:
        return "late_night"
    if 6 <= h < 8:
        return "early_morning"
    if 8 <= h < 12:
        return "morning"
    if 12 <= h < 14:
        return "lunch"
    if 14 <= h < 17:
        return "afternoon"
    if 17 <= h < 20:
        return "evening"
    return "night"


def simulate_current_drivers():
    hs, hc = cyc(hour, 24)
    rows = []
    rng = np.random.default_rng(hash((hour, weekday, is_exam, is_raining)) % (2**32))
    for _, d in drivers_df.iterrows():
        feat = pd.DataFrame([{
            "hour_sin": hs, "hour_cos": hc, "prev_slot_available": 0.5,
            "is_weekend": int(is_weekend), "is_exam_period": int(is_exam),
            "is_raining": int(is_raining), "shift_type": d["shift_type"], "home_zone": d["home_zone"],
        }])
        p_avail = avail_model.predict_proba(feat)[0, 1]
        status = "available" if rng.random() < p_avail else "busy"
        rows.append({"driver_id": d["driver_id"], "home_zone": d["home_zone"],
                      "shift_type": d["shift_type"], "status": status, "p_available": round(p_avail, 2)})
    return pd.DataFrame(rows)


current_drivers = simulate_current_drivers()
available_now = current_drivers[current_drivers["status"] == "available"][["driver_id", "home_zone"]]
n_avail = len(available_now)


# =============================================================================
# STUDENT-VIEW HELPERS (matching animation, trip cards, live trip)
# =============================================================================
def play_matching_animation(slot, trip):
    steps = trip["expansions"] + 1
    pause = min(0.9, 2.6 / steps)
    for k in range(1, steps + 1):
        txt = "Scanning nearby drivers…" if k == 1 else f"No driver yet — expanding search area ({k - 1}/{trip['expansions']})…"
        slot.markdown(radar_html(txt), unsafe_allow_html=True)
        time.sleep(pause)
    slot.markdown(radar_html("✅ Driver found!" if trip["matched"] else "⚠️ No driver within the maximum radius"),
                  unsafe_allow_html=True)
    time.sleep(0.6)
    slot.empty()


def show_trip_cards(t):
    if not t["matched"]:
        st.error(f"⚠️ No driver available within the maximum search radius ({t['radius']} km). "
                 f"Estimated failure risk was {t['risk']:.0%}. Please try again shortly, or walk to a "
                 f"nearby pickup point with better coverage.")
        return
    a, b = st.columns([1, 1.25])
    a.markdown(f'<div class="ride-card"><div class="tag">DRIVER FOUND ✅</div>'
               f'<div class="big">{t["eta"]:.1f} min</div>'
               f'<div class="row"><span>Assigned driver</span><b>{t["driver"]}</b></div>'
               f'<div class="row"><span>Driver zone</span><b>{t["driver_zone"]}</b></div>'
               f'<div class="row"><span>Distance</span><b>{t["dist"]:.2f} km</b></div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="info-card"><div class="hd">TRIP INFORMATION</div>'
               f'<div class="row"><span>Route</span><b>{t["pickup"]} → {t["drop"]}</b></div>'
               f'<div class="row"><span>Search radius used</span><b>{t["radius"]} km</b></div>'
               f'<div class="row"><span>Radius expansions</span><b>{t["expansions"]}</b></div>'
               f'<div class="row"><span>Failure risk (a-priori)</span><b>{t["risk"]:.0%}</b></div></div>',
               unsafe_allow_html=True)


def run_live_trip(t, map_slot, prog_slot):
    """Server-driven simulation: the rickshaw moves on the map while the progress bar fills."""
    a, b, c = np.array(ZONE_POS[t["driver_zone"]]), np.array(ZONE_POS[t["pickup"]]), np.array(ZONE_POS[t["drop"]])
    n1, hold, n2 = 10, 3, 14
    frames = []
    for k in range(n1 + 1):
        s = k / n1
        frames.append((a + (b - a) * s, 40 * s, "Driver on the way", f"ETA {t['eta'] * (1 - s):.1f} min", 1))
    for _ in range(hold):
        frames.append((b, 40, "Driver arrived — boarding", "Pickup point", 2))
    for k in range(1, n2 + 1):
        s = k / n2
        frames.append((b + (c - b) * s, 40 + 60 * s, "Trip in progress", f"{(1 - s) * t['dist']:.2f} km to go", 3))
    frames.append((c, 100, "Trip completed ✅", "You have reached your destination", 4))
    for idx, (xy, pct, label, sub, stage) in enumerate(frames):
        prog_slot.markdown(progress_html(pct, label, sub, stage), unsafe_allow_html=True)
        if HAS_PLOTLY:
            map_slot.plotly_chart(plotly_map(pickup=t["pickup"], drop=t["drop"], mover=tuple(xy)),
                                  key=f"trip_frame_{idx}", config=PLOT_CFG)
        time.sleep(0.16)


# =============================================================================
# STUDENT VIEW
# =============================================================================
if role == "Student":
    page_title("🎓", "Student — Request a Ride")
    kpi_row([("🛺", "Drivers available", f"{n_avail} / {len(current_drivers)}"),
             ("⏱️", "Avg historical wait", f"{rides_hist['wait_time_minutes'].mean():.1f} min"),
             ("✅", "Fulfilment rate", f"{1 - rides_hist['request_failed'].mean():.1%}"),
             ("📍", "Campus zones", f"{len(zone_names)}")])

    # --- selection state (so clicks on the map can set pickup / drop-off) -----
    ss = st.session_state
    ss.setdefault("pickup_sel", "Central Library" if "Central Library" in zone_names else zone_names[0])
    ss.setdefault("drop_sel", "Boys Hostel Zone 1" if "Boys Hostel Zone 1" in zone_names else zone_names[1])
    ss.update(ss.pop("_pending", {}))
    if ss["drop_sel"] == ss["pickup_sel"]:
        ss["drop_sel"] = next(z for z in zone_names if z != ss["pickup_sel"])

    left, right = st.columns([2.3, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Plan your ride")
            c1, c2 = st.columns(2)
            pickup_zone = c1.selectbox("Pickup location", zone_names, key="pickup_sel")
            dropoff_options = [z for z in zone_names if z != pickup_zone]
            dropoff_zone = c2.selectbox("Drop-off location", dropoff_options, key="drop_sel")
            go_btn = st.button("🔍 Request Ride")

        anim_slot = st.empty()

        if go_btn:
            block = time_block_of(hour)
            i, j = zone_names.index(pickup_zone), zone_names.index(dropoff_zone)
            distance_km = float(dist_matrix[i, j])
            nearby = len(available_now[available_now["home_zone"].map(
                lambda z: dist_matrix[zone_names.index(z), i] <= 1.5)])
            zone_hist = rides_hist[rides_hist["pickup_zone"] == pickup_zone]
            hs, hc = cyc(hour, 24); ws, wc = cyc(weekday, 7)
            req_feat = pd.DataFrame([{
                "hour_sin": hs, "hour_cos": hc, "weekday_sin": ws, "weekday_cos": wc,
                "distance_km": distance_km, "nearby_available_drivers": nearby,
                "rolling_zone_wait_avg": zone_hist["wait_time_minutes"].mean(),
                "rolling_zone_fail_rate": zone_hist["request_failed"].mean(),
                "is_weekend": int(is_weekend), "is_exam_period": int(is_exam), "is_raining": int(is_raining),
                "pickup_zone_type": ZONE_TYPE[pickup_zone], "dropoff_zone_type": ZONE_TYPE[dropoff_zone],
                "time_block": block,
            }])
            failure_risk = engine.predict_failure_risk(req_feat)[0]
            result = engine.match_driver(pickup_zone, available_now, failure_risk=failure_risk)
            trip = dict(pickup=pickup_zone, drop=dropoff_zone, matched=bool(result["matched"]), dist=distance_km,
                        radius=result["radius_used_km"], expansions=int(result["expansions_used"]),
                        risk=float(failure_risk), driver=None, driver_zone=None, eta=None)
            if trip["matched"]:
                trip["driver"] = result["driver_id"]
                trip["driver_zone"] = current_drivers.set_index("driver_id").loc[result["driver_id"], "home_zone"]
                trip["eta"] = float(engine.predict_eta(req_feat)[0])
            ss["trip"] = trip
            play_matching_animation(anim_slot, trip)

        trip = ss.get("trip")
        show = bool(trip) and trip["pickup"] == pickup_zone and trip["drop"] == dropoff_zone
        if show:
            show_trip_cards(trip)
        else:
            st.info("Pick your route (use the dropdowns **or click zones on the map below**), then press **Request Ride**.")

        prog_slot = st.empty()
        track = False
        if show and trip["matched"]:
            prog_slot.markdown(progress_html(0, "Driver assigned", f"ETA {trip['eta']:.1f} min", 0), unsafe_allow_html=True)
            track = st.button("▶ Track live trip")

        spacer()
        with st.container(border=True):
            st.markdown("#### Trip map")
            map_slot = st.empty()
            matched_zone = trip["driver_zone"] if show and trip["matched"] else None
            if track:
                run_live_trip(trip, map_slot, prog_slot)
            elif HAS_PLOTLY:
                fig = plotly_map(pickup=pickup_zone, drop=dropoff_zone,
                                 driver_zones=available_now["home_zone"].tolist(), matched_zone=matched_zone)
                try:
                    event = map_slot.plotly_chart(fig, key="student_map", config=PLOT_CFG,
                                                  on_select="rerun", selection_mode="points")
                    pts = event.selection.points if event else []
                    if not pts:
                        ss["_last_zone"] = None
                    else:
                        zone = zone_from_click(pts[0]) if pts[0].get("curve_number", ZONE_TRACE) == ZONE_TRACE else None
                        if zone and zone != ss.get("_last_zone"):
                            ss["_last_zone"] = zone
                            target = ss.get("_next", "pickup")
                            if target == "pickup":
                                pend = {"pickup_sel": zone}
                                if ss["drop_sel"] == zone:
                                    pend["drop_sel"] = next(z for z in zone_names if z != zone)
                                ss["_next"] = "drop"
                            elif zone != ss["pickup_sel"]:
                                pend = {"drop_sel": zone}
                                ss["_next"] = "pickup"
                            else:
                                pend = None
                            if pend:
                                ss["_pending"] = pend
                                st.rerun()
                    st.caption("👆 Click a zone to set the **pickup**, then click another zone to set the **drop-off**. "
                               "Hover any zone for its demand and failure rate.")
                except TypeError:   # older Streamlit without on_select support
                    map_slot.plotly_chart(fig, key="student_map_static", config=PLOT_CFG)
                    st.caption("Tip: upgrade Streamlit (`pip install -U streamlit`) to enable clicking zones on the map.")
            else:
                with map_slot.container():
                    campus_map(pickup=pickup_zone, drop=dropoff_zone,
                               driver_zones=available_now["home_zone"].tolist(), matched_zone=matched_zone)
                st.caption("Install plotly (`pip install plotly`) for the interactive, clickable map.")

    with right:
        with st.container(border=True):
            st.markdown("#### Live drivers")
            st.markdown(driver_list_html(current_drivers.sort_values("status")), unsafe_allow_html=True)

# =============================================================================
# DRIVER VIEW
# =============================================================================
elif role == "Driver":
    page_title("🧑‍✈️", "Driver — Nearby Requests & Status")
    road_animation()
    my_driver_id = st.sidebar.selectbox("You are:", drivers_df["driver_id"].tolist())
    my_row = drivers_df[drivers_df["driver_id"] == my_driver_id].iloc[0]
    my_status_row = current_drivers[current_drivers["driver_id"] == my_driver_id].iloc[0]

    a, b = st.columns([1, 1.6])
    a.markdown(f'<div class="ride-card"><div class="tag">YOU ARE {my_status_row["status"].upper()}</div>'
               f'<div class="big">{my_driver_id}</div>'
               f'<div class="row"><span>Home zone</span><b>{my_row["home_zone"]}</b></div>'
               f'<div class="row"><span>Shift</span><b>{my_row["shift_type"]}</b></div>'
               f'<div class="row"><span>P(available) this slot</span><b>{my_status_row["p_available"]:.0%}</b></div>'
               f'</div>', unsafe_allow_html=True)

    i = zone_names.index(my_row["home_zone"])
    nearby_zones = [z for z, d in zip(zone_names, dist_matrix[i]) if d <= 2.0]
    demand_estimates = []
    for z in nearby_zones:
        hs, hc = cyc(hour, 24); ws, wc = cyc(weekday, 7)
        zh = rides_hist[(rides_hist["pickup_zone"] == z) & (rides_hist["hour"] == hour)]
        lag1 = zh["request_id"].count() / max(rides_hist["date"].nunique(), 1)
        feat = pd.DataFrame([{
            "hour_sin": hs, "hour_cos": hc, "weekday_sin": ws, "weekday_cos": wc,
            "lag_1h_same_zone": lag1, "lag_24h_same_zone_hour": lag1,
            "rolling7_avg_zone_hour": lag1, "is_weekend": int(is_weekend),
            "is_exam_period": int(is_exam), "is_raining": int(is_raining),
            "pickup_zone": z, "pickup_zone_type": ZONE_TYPE[z],
        }])
        pred = max(0, demand_model.predict(feat)[0])
        demand_estimates.append({"Zone": z, "Distance (km)": round(float(dist_matrix[i, zone_names.index(z)]), 2),
                                  "Predicted requests (next hr)": round(pred, 2)})
    demand_df_view = pd.DataFrame(demand_estimates).sort_values("Predicted requests (next hr)", ascending=False)

    with b:
        with st.container(border=True):
            st.markdown("#### Nearby demand right now")
            show_table(demand_df_view)
            st.caption("Ranks nearby zones by predicted demand so you can position yourself before requests arrive.")

    spacer()
    with st.container(border=True):
        st.markdown("#### Campus map — your position ★")
        render_map("driver_map", driver_zones=available_now["home_zone"].tolist(), matched_zone=my_row["home_zone"])

# =============================================================================
# ADMIN VIEW
# =============================================================================
else:
    page_title("🛠️", "Admin — Operations Dashboard")
    kpi_row([("📊", "Rides analysed", f"{len(rides_hist):,}"),
             ("✅", "Fulfilment rate", f"{1 - rides_hist['request_failed'].mean():.1%}"),
             ("⏱️", "Avg wait", f"{rides_hist['wait_time_minutes'].mean():.1f} min"),
             ("🛺", "Drivers online now", f"{n_avail} / {len(current_drivers)}")])

    tab1, tab2, tab3, tab4 = st.tabs(["Demand Heatmap", "Failed-Request / Route Analysis",
                                       "Model Performance", "Explainability (SHAP)"])

    with tab1:
        with st.container(border=True):
            st.markdown("#### Campus demand map")
            render_map("admin_map", driver_zones=available_now["home_zone"].tolist())
        with st.container(border=True):
            st.markdown("#### Demand by zone and hour")
            pivot = rides_hist.pivot_table(index="pickup_zone", columns="hour",
                                            values="request_id", aggfunc="count", fill_value=0)
            fig, ax = plt.subplots(figsize=(13, 6))
            sns.heatmap(pivot, cmap="magma", ax=ax, linewidths=0.3, linecolor="#141b2d")
            ax.set_title("Ride Requests by Pickup Zone × Hour of Day (full history)")
            st.pyplot(fig)
            plt.close(fig)
        with st.container(border=True):
            st.markdown("#### Live driver availability")
            show_table(current_drivers)

    with tab2:
        with st.container(border=True):
            st.markdown("#### Worst-served pickup → dropoff routes")
            show_table(engine.route_analysis(rides_hist, top_n=15))
            st.caption("`service_score` blends failure rate (70%) and normalized average wait time (30%) — "
                       "prioritise these routes for extra driver coverage or a dedicated shuttle.")
        with st.container(border=True):
            st.markdown("#### Failed-request rate by pickup zone")
            fail_by_zone = rides_hist.groupby("pickup_zone")["request_failed"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 5))
            fail_by_zone.plot(kind="barh", ax=ax, color="#6d5dfc")
            ax.set_xlabel("Failure rate")
            st.pyplot(fig)
            plt.close(fig)
        with st.container(border=True):
            st.markdown("#### Dispatch policy comparison")
            st.caption("Evaluated on the held-out test period (`src/dispatch.py :: evaluate_policy`).")
            show_table(pd.DataFrame({
                "Policy": ["Static Radius (baseline, used to generate historical data)",
                           "ML-Informed Dynamic Radius (this system)"],
                "Fulfilment Rate": ["92.2%", "93.6%"],
            }))

    with tab3:
        model_files = {
            "Demand Prediction (regression)": "demand_model_metrics.json",
            "Driver Availability (classification)": "availability_model_metrics.json",
            "ETA Prediction (regression)": "eta_model_metrics.json",
            "Failed-Request Prediction (classification)": "failed_model_metrics.json",
        }
        fig_map = {
            "Demand Prediction (regression)": "demand_model_comparison.png",
            "Driver Availability (classification)": "availability_model_comparison.png",
            "ETA Prediction (regression)": "eta_model_comparison.png",
            "Failed-Request Prediction (classification)": "failed_model_comparison.png",
        }
        with st.container(border=True):
            st.markdown("#### Model comparison & final test metrics")
            chosen = st.selectbox("Select model:", list(model_files.keys()))
            with open(MODELS_DIR / model_files[chosen]) as f:
                m = json.load(f)
            st.markdown(f"**Best model selected:** `{m['best_model']}`")
            st.markdown("**Validation comparison across candidate models:**")
            st.dataframe(pd.DataFrame(m["validation_results"]).T, width='stretch')
            st.markdown("**Final held-out test metrics (best model, evaluated once):**")
            show_table(pd.DataFrame([m["test_metrics"]]))
            st.image(str(FIGURES_DIR / fig_map[chosen]), width='stretch')

    with tab4:
        shap_map = {
            "Demand Prediction": "demand_shap_summary.png",
            "Driver Availability": "availability_shap_summary.png",
            "ETA Prediction": "eta_shap_summary.png",
            "Failed-Request Prediction": "failed_shap_summary.png",
        }
        with st.container(border=True):
            st.markdown("#### SHAP feature-importance explanations")
            chosen_shap = st.selectbox("Select model for explanation:", list(shap_map.keys()))
            st.image(str(FIGURES_DIR / shap_map[chosen_shap]), width='stretch')
            st.caption("Each point is one test-set prediction; red = high feature value, blue = low. "
                       "Position on the x-axis shows whether that feature pushed the prediction up or down.")

st.sidebar.markdown("---")
st.sidebar.caption("INT395 Supervised Learning — Project 04. All data is simulated; see "
                    "`data/README.md` for full documentation of assumptions and limitations.")