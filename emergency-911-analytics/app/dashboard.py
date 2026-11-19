"""911 Emergency Analytics dashboard.
Run the pipeline first (python run_pipeline.py), then: streamlit run app/dashboard.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.config import HANDLE_HOURS, REPORTS, TARGET_UTILIZATION
from src.data import load_processed
from src.eda import DOW, hour_dow_matrix
from src.staffing import units_needed

ACCENT = "#E4572E"
CAT_COLORS = {"EMS": "#E4572E", "Fire": "#F2A541", "Traffic": "#2E86AB"}

st.set_page_config(page_title="911 Emergency Analytics", page_icon="🚨", layout="wide")
st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.2rem; max-width: 1320px;}}
.hero {{background: linear-gradient(120deg,#101828 0%,#1D2939 60%,{ACCENT} 160%); color:#fff;
        padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}}
.hero h1 {{margin:0; font-size:1.7rem; color:#fff;}}
.hero p {{margin:4px 0 0; color:#D0D5DD; font-size:.95rem;}}
.kpi {{background:#fff; border:1px solid #E6E8EC; border-left:4px solid {ACCENT}; border-radius:10px;
       padding:14px 18px; box-shadow:0 1px 2px rgba(16,24,40,.05);}}
.kpi .l {{font-size:.72rem; color:#667085; text-transform:uppercase; letter-spacing:.06em;}}
.kpi .v {{font-size:1.7rem; font-weight:700; color:#101828; line-height:1.2;}}
.kpi .s {{font-size:.8rem; color:#667085;}}
.insight {{background:#FFF6F3; border:1px solid #FBD5C8; border-radius:8px; padding:10px 14px;
           margin:6px 0 14px; color:#7A2E18; font-size:.92rem;}}
button[data-baseweb="tab"] {{font-weight:600;}}
</style>""",
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(
        template="simple_white",
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        font=dict(size=13),
    )
    return fig


def read_csv(name, **kw):
    p = REPORTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def get_data():
    return load_processed()


@st.cache_data
def hourly_counts(cat):
    d = get_data()
    s = d[d["category"] == cat].set_index("timeStamp").resample("h").size()
    t = pd.DataFrame({"calls": s})
    t["dow"], t["hour"] = t.index.dayofweek, t.index.hour
    return t


try:
    df = get_data()
except Exception:
    st.error("Processed data not found. Run `python run_pipeline.py --data data/911.csv` first.")
    st.stop()

# ---------- sidebar ----------
st.sidebar.markdown("### 🎛️ Filters")
all_cats = sorted(df["category"].unique())
cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)
d_min, d_max = df["date"].min().date(), df["date"].max().date()
rng = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
st.sidebar.caption("Filters apply to Overview, Patterns and Geospatial. Forecast and Staffing use the full history.")

f = df[df["category"].isin(cats)]
if isinstance(rng, (list, tuple)) and len(rng) == 2:
    f = f[(f["date"] >= pd.Timestamp(rng[0])) & (f["date"] <= pd.Timestamp(rng[1]))]
if f.empty:
    st.warning("No calls match the current filters.")
    st.stop()

st.markdown(
    f"""<div class="hero"><h1>🚨 911 Emergency Call Analytics</h1>
<p>From call patterns to staffing decisions · {len(df):,} calls · {d_min:%b %Y} – {d_max:%b %Y}</p></div>""",
    unsafe_allow_html=True,
)

tab_over, tab_pat, tab_geo, tab_fc, tab_staff, tab_meth = st.tabs(
    ["📊 Overview", "🕒 Demand patterns", "🗺️ Geospatial", "📈 Forecast", "🧮 Staffing planner", "📘 Method & caveats"]
)

# ---------- Overview ----------
with tab_over:
    daily_avg = f.groupby("date").size().mean()
    peak_hour = int(f["hour"].mode().iat[0])
    top_reason = f["reason"].value_counts().idxmax()
    ems_share = (f["category"] == "EMS").mean() * 100
    c = st.columns(4)
    kpi(c[0], "Total calls", f"{len(f):,}", "in selected range")
    kpi(c[1], "Avg calls / day", f"{daily_avg:,.0f}")
    kpi(c[2], "Peak hour", f"{peak_hour}:00", f"top reason: {top_reason.title()[:28]}")
    kpi(c[3], "EMS share", f"{ems_share:.1f}%", "of selected calls")
    st.write("")

    m = f.set_index("timeStamp").groupby("category").resample("MS").size().rename("calls").reset_index()
    if m["timeStamp"].nunique() > 3:
        m = m[(m["timeStamp"] > m["timeStamp"].min()) & (m["timeStamp"] < m["timeStamp"].max())]
    left, right = st.columns([2, 1])
    left.plotly_chart(
        style(
            px.line(
                m,
                x="timeStamp",
                y="calls",
                color="category",
                markers=True,
                color_discrete_map=CAT_COLORS,
                title="Monthly volume by category",
            )
        ),
        width="stretch",
    )
    mix = f["category"].value_counts().reset_index()
    mix.columns = ["category", "calls"]
    right.plotly_chart(
        style(
            px.pie(
                mix,
                names="category",
                values="calls",
                hole=0.55,
                color="category",
                color_discrete_map=CAT_COLORS,
                title="Category mix",
            )
        ),
        width="stretch",
    )

    a, b = st.columns(2)
    top = f.groupby(["category", "reason"]).size().rename("calls").reset_index().nlargest(12, "calls")
    a.plotly_chart(
        style(
            px.bar(
                top.sort_values("calls"),
                x="calls",
                y="reason",
                color="category",
                orientation="h",
                color_discrete_map=CAT_COLORS,
                title="Top 12 call reasons",
            )
        ),
        width="stretch",
    )
    an = read_csv("daily_anomalies.csv", parse_dates=["date"])
    if an is not None:
        flagged = an[an["anomaly"]]
        fig = go.Figure()
        fig.add_scatter(
            x=an["date"], y=an["calls"], mode="lines", line=dict(color="#98A2B3", width=1), name="Daily calls"
        )
        fig.add_scatter(
            x=flagged["date"],
            y=flagged["calls"],
            mode="markers",
            marker=dict(color=ACCENT, size=7),
            name=f"Anomalous days ({len(flagged)})",
        )
        b.plotly_chart(style(fig.update_layout(title="Anomaly detection (robust z-score)")), width="stretch")

# ---------- Demand patterns ----------
with tab_pat:
    mtx = hour_dow_matrix(f)
    fig = px.imshow(
        mtx.values,
        x=list(range(24)),
        y=DOW,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="Hour of day", y="", color="Avg calls/hr"),
        title="Average calls per hour slot",
    )
    st.plotly_chart(style(fig, 360), width="stretch")
    peak_idx = np.unravel_index(mtx.values.argmax(), mtx.shape)
    insight(
        f"Busiest slot: <b>{DOW[peak_idx[0]]} at {peak_idx[1]}:00</b> (≈{mtx.values.max():.1f} calls/hour on average)."
    )

    counts = f.groupby(["is_weekend", "hour"]).size()
    ndays = f.groupby("is_weekend")["date"].nunique()
    prof = counts.div(ndays, level=0).rename("avg").reset_index()
    prof["type"] = np.where(prof["is_weekend"], "Weekend", "Weekday")
    lc, rc = st.columns(2)
    lc.plotly_chart(
        style(
            px.line(
                prof,
                x="hour",
                y="avg",
                color="type",
                title="Daily rhythm: weekday vs weekend",
                color_discrete_sequence=["#2E86AB", ACCENT],
            ).update_layout(yaxis_title="Avg calls/hour")
        ),
        width="stretch",
    )
    stats_p = REPORTS / "stats_results.json"
    with rc:
        st.markdown("##### Statistical evidence")
        if stats_p.exists():
            s = json.loads(stats_p.read_text())
            w, t = s["weekday_vs_weekend"], s["township_mix"]
            st.markdown(f"""
**Weekday vs weekend** · Mann-Whitney U  
Weekdays average **{w["weekday_mean"]:.0f}** calls/day vs **{w["weekend_mean"]:.0f}** on weekends
(**{w["pct_difference"]:+.1f}%**). Rank-biserial = **{w["rank_biserial"]:.2f}**, Cohen's d = **{w["cohens_d"]:.2f}**, p = {w["p_value"]:.2g}.

**Category mix across townships** · chi-square  
Cramér's V = **{t["cramers_v"]:.3f}**, p = {t["p_value"]:.2g}.  
*With this many rows p-values are almost always tiny, so the effect size is what matters.*
""")
        else:
            st.info("Run the pipeline to generate statistical results.")

# ---------- Geospatial ----------
with tab_geo:
    ctrl = st.columns([1, 1, 2])
    radius = ctrl[0].slider("Hexagon radius (m)", 300, 2500, 800, 100)
    extrude = ctrl[1].toggle("3D columns", value=True)
    pts = f.sample(min(40_000, len(f)), random_state=1)[["lat", "lng"]]
    layer = pdk.Layer(
        "HexagonLayer",
        data=pts,
        get_position="[lng, lat]",
        radius=radius,
        elevation_scale=6,
        extruded=extrude,
        pickable=True,
        coverage=0.92,
        auto_highlight=True,
        color_range=[[255, 237, 213], [253, 196, 140], [247, 140, 80], [228, 87, 46], [160, 40, 20]],
    )
    view = pdk.ViewState(
        latitude=float(pts["lat"].mean()), longitude=float(pts["lng"].mean()), zoom=9.3, pitch=45 if extrude else 0
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_style="light",
            tooltip={"text": "{elevationValue} calls in this cell (sample)"},
        ),
        width="stretch",
    )
    lc, rc = st.columns(2)
    hs = read_csv("ems_hotspots.csv")
    with lc:
        st.markdown("##### EMS hotspots (DBSCAN, 500 m radius)")
        if hs is not None and len(hs):
            st.dataframe(
                hs.head(10).rename(columns={"est_calls": "Est. calls", "top_township": "Township"})[
                    ["cluster", "Township", "Est. calls", "lat", "lng"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No hotspot file found.")
    cv = read_csv("station_coverage_curve.csv")
    with rc:
        if cv is not None:
            fig = px.line(
                cv, x="stations", y="mean_km_to_nearest", markers=True, title="Diminishing returns of adding stations"
            )
            fig.update_traces(line_color=ACCENT)
            st.plotly_chart(style(fig, 330).update_layout(yaxis_title="Mean km to nearest station"), width="stretch")

    stab = read_csv("hotspot_stability.csv")
    if stab is not None and len(stab):
        st.markdown("##### Are hotspots stable over time?")
        st.dataframe(
            stab.rename(
                columns={
                    "year_a": "From",
                    "year_b": "To",
                    "hotspots_a": "Hotspots (from)",
                    "hotspots_b": "Hotspots (to)",
                    "pct_a_persisting_in_b": "% persisting within 1 km",
                }
            ),
            hide_index=True,
            width="stretch",
        )

# ---------- Forecast ----------
with tab_fc:
    pred = read_csv("forecast_predictions.csv", parse_dates=["timestamp"])
    hold = read_csv("forecast_holdout_metrics.csv", index_col=0)
    itv_p = REPORTS / "forecast_interval.json"
    if pred is None or hold is None:
        st.info("Run the pipeline to generate forecasts.")
    else:
        naive, gbm = hold.iloc[0], hold.iloc[1]
        itv = json.loads(itv_p.read_text()) if itv_p.exists() else {}
        c = st.columns(4)
        kpi(c[0], "Model MAE", f"{gbm['MAE']:.2f}", "calls / hour")
        kpi(c[1], "Baseline MAE", f"{naive['MAE']:.2f}", "same hour last week")
        kpi(c[2], "Error reduction", f"{(1 - gbm['MAE'] / naive['MAE']) * 100:.1f}%", "vs baseline")
        kpi(
            c[3],
            "80% interval coverage",
            f"{itv.get('empirical_coverage_pct', float('nan')):.1f}%",
            "calibration check",
        )
        st.write("")
        days = st.slider("Days of hold-out to display", 3, 60, 14)
        p = pred[pred["timestamp"] > pred["timestamp"].max() - pd.Timedelta(days=days)]
        fig = go.Figure()
        fig.add_scatter(
            x=p["timestamp"], y=p["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
        fig.add_scatter(
            x=p["timestamp"],
            y=p["lo"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(228,87,46,.18)",
            name="80% interval",
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=p["timestamp"], y=p["actual"], mode="lines", name="Actual", line=dict(color="#101828", width=1.6)
        )
        fig.add_scatter(
            x=p["timestamp"], y=p["gbm"], mode="lines", name="GBM forecast", line=dict(color=ACCENT, width=1.6)
        )
        fig.add_scatter(
            x=p["timestamp"],
            y=p["naive"],
            mode="lines",
            name="Seasonal naive",
            line=dict(color="#98A2B3", width=1, dash="dot"),
        )
        st.plotly_chart(style(fig.update_layout(title="Hourly calls: actual vs forecast"), 420), width="stretch")
        pred["hour"] = pred["timestamp"].dt.hour
        pred["abs_err"] = (pred["actual"] - pred["gbm"]).abs()
        e = pred.groupby("hour")["abs_err"].mean().reset_index()
        st.plotly_chart(
            style(
                px.bar(
                    e,
                    x="hour",
                    y="abs_err",
                    title="Where is the model weakest? MAE by hour of day",
                    color_discrete_sequence=[ACCENT],
                ).update_layout(yaxis_title="MAE"),
                300,
            ),
            width="stretch",
        )
        abl = read_csv("forecast_ablation.csv")
        imp = read_csv("forecast_importance.csv")
        if abl is not None and imp is not None:
            lc, rc = st.columns(2)
            lc.plotly_chart(
                style(
                    px.bar(
                        abl,
                        x="model",
                        y="MAE",
                        text_auto=".2f",
                        color_discrete_sequence=[ACCENT],
                        title="Ablation: what each feature group buys",
                    ),
                    320,
                ),
                width="stretch",
            )
            rc.plotly_chart(
                style(
                    px.bar(
                        imp.head(8).sort_values("mae_increase"),
                        x="mae_increase",
                        y="feature",
                        orientation="h",
                        color_discrete_sequence=[ACCENT],
                        title="Permutation importance (MAE increase)",
                    ),
                    320,
                ),
                width="stretch",
            )
        insight(
            "Validation is chronological (last 60 days held out + 4 rolling-origin folds), and every feature is "
            "lagged ≥24 h, so the forecast could really have been issued a day ahead."
        )

# ---------- Staffing planner ----------
with tab_staff:
    st.markdown(
        "Adjust the planning assumptions and see how the roster changes. "
        "These parameters are **assumptions**, not measurements."
    )
    k = st.columns(4)
    cat = k[0].selectbox("Category", all_cats, index=0)
    pct = k[1].slider("Demand percentile", 50, 99, 90, help="Plan to cover this share of hours")
    handle = k[2].slider("Hours a unit is busy per call", 0.25, 3.0, float(HANDLE_HOURS.get(cat, 1.0)), 0.25)
    util = k[3].slider("Target utilization", 0.3, 0.9, float(TARGET_UTILIZATION), 0.05)

    t = hourly_counts(cat)
    q = t.groupby(["dow", "hour"])["calls"].quantile(pct / 100).unstack().reindex(index=range(7), columns=range(24))
    units = units_needed(q.values, handle, util)
    fig = px.imshow(
        units,
        x=list(range(24)),
        y=DOW,
        aspect="auto",
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Hour of day", y="", color="Units"),
        title=f"{cat} units on duty (p{pct} demand)",
    )
    st.plotly_chart(style(fig, 400), width="stretch")

    flat = units.max() * 168
    needed = units.sum()
    c = st.columns(3)
    kpi(c[0], "Peak units", f"{units.max()}", "busiest hour of the week")
    kpi(c[1], "Quietest hour", f"{units.min()}", "minimum on duty")
    kpi(
        c[2],
        "Savings vs flat peak roster",
        f"{(1 - needed / flat) * 100:.1f}%",
        f"{needed:,} vs {flat:,} unit-hours/week",
    )
    out = pd.DataFrame(units, index=DOW, columns=range(24)).stack().rename("units").reset_index()
    out.columns = ["weekday", "hour", "units"]
    st.download_button("⬇️ Download roster (CSV)", out.to_csv(index=False), f"roster_{cat.lower()}.csv", "text/csv")

# ---------- Method ----------
with tab_meth:
    st.markdown("""
### Approach
1. **Clean:** dedupe, parse timestamps, filter GPS outliers, split `title` into category and reason.
2. **Explore and test:** hour × weekday structure, Mann-Whitney and chi-square tests with effect sizes.
3. **Locate:** DBSCAN hotspots (haversine) and a KMeans station-coverage curve.
4. **Forecast:** 24-hour-ahead hourly model vs a seasonal-naive baseline, with quantile prediction intervals.
5. **Decide:** convert p-percentile demand into units on duty per weekday and hour.

### Caveats
- Staffing output depends on **assumed** handling time and utilization; the data has no unit or response-time fields.
- KMeans station placement ignores roads and existing stations.
- Anomalies are flagged statistically; explaining them (weather, holidays, events) needs outside data.
- One county, one time span: findings may not generalise.
""")

    qp = REPORTS / "data_quality.json"
    if qp.exists():
        q = json.loads(qp.read_text())
        st.markdown("### Data quality audit")
        c = st.columns(4)
        kpi(c[0], "Raw rows", f"{q['raw_rows']:,}")
        kpi(c[1], "Retained", f"{q['pct_retained']}%", f"{q['raw_rows'] - q['rows_after_cleaning']:,} removed")
        kpi(c[2], "Duplicates", f"{q['duplicate_rows']:,}")
        kpi(c[3], "Days with no calls", f"{q['calendar_days_with_no_calls']}", f"longest gap {q['longest_gap_days']} d")
