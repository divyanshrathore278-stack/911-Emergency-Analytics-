"""Geospatial analysis: hotspots (DBSCAN), station placement (KMeans), interactive map."""

import matplotlib

matplotlib.use("Agg")
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import haversine_distances

from .config import FIGURES, REPORTS, SEED

EARTH_KM = 6371.0088


def find_hotspots(df, category="EMS", sample=60_000, eps_km=0.5, min_samples=40) -> pd.DataFrame:
    """DBSCAN on lat/lng with haversine distance. Returns clusters ranked by estimated calls."""
    d = df[df["category"] == category]
    total = len(d)
    d = d.sample(min(sample, total), random_state=SEED)
    coords = np.radians(d[["lat", "lng"]].values)
    labels = DBSCAN(
        eps=eps_km / EARTH_KM, min_samples=min_samples, metric="haversine", algorithm="ball_tree"
    ).fit_predict(coords)
    d = d.assign(cluster=labels)
    d = d[d["cluster"] >= 0]
    if d.empty:
        return pd.DataFrame(columns=["cluster", "est_calls", "lat", "lng", "top_township"])
    scale = total / min(sample, total)  # scale sample counts back up to the full data
    out = (
        d.groupby("cluster")
        .agg(
            sample_calls=("lat", "size"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
            top_township=("twp", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    out["est_calls"] = (out["sample_calls"] * scale).round().astype(int)
    return out.sort_values("est_calls", ascending=False).drop(columns="sample_calls").reset_index(drop=True)


def _mean_nearest_km(points_rad, centers_rad) -> float:
    return float(haversine_distances(points_rad, centers_rad).min(axis=1).mean() * EARTH_KM)


def station_coverage_curve(df, category="EMS", k_range=range(2, 13), sample=40_000):
    """Mean distance from a call to its nearest station for different numbers of stations."""
    d = df[df["category"] == category]
    d = d.sample(min(sample, len(d)), random_state=SEED)
    X = d[["lat", "lng"]].values
    Xr = np.radians(X)
    rows, centers_by_k = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
        rows.append({"stations": k, "mean_km_to_nearest": _mean_nearest_km(Xr, np.radians(km.cluster_centers_))})
        centers_by_k[k] = km.cluster_centers_
    curve = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["stations"], curve["mean_km_to_nearest"], marker="o")
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("Mean km from call to nearest station")
    ax.set_title(f"Diminishing returns of adding {category} stations")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_station_coverage_curve.png", dpi=150)
    plt.close(fig)
    return curve, centers_by_k


def build_map(df, hotspots, centers=None, category="EMS", sample=30_000):
    d = df[df["category"] == category]
    d = d.sample(min(sample, len(d)), random_state=SEED)
    m = folium.Map(location=[d["lat"].mean(), d["lng"].mean()], zoom_start=10)
    HeatMap(d[["lat", "lng"]].values.tolist(), radius=8, blur=10, name="Call density").add_to(m)
    for _, r in hotspots.head(15).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lng"]],
            radius=8,
            color="red",
            fill=True,
            tooltip=f"Hotspot {int(r['cluster'])}: ~{int(r['est_calls']):,} calls, {r['top_township']}",
        ).add_to(m)
    if centers is not None:
        for i, (lat, lng) in enumerate(centers):
            folium.Marker(
                [lat, lng], tooltip=f"Suggested station {i + 1}", icon=folium.Icon(color="blue", icon="plus")
            ).add_to(m)
    REPORTS.mkdir(parents=True, exist_ok=True)
    m.save(REPORTS / "hotspot_map.html")


def hotspot_stability(df, category="EMS", sample=30_000, eps_km=0.5, min_samples=25, match_km=1.0) -> pd.DataFrame:
    """Do hotspots persist? For each pair of consecutive years, the share of year-A hotspots that
    have a year-B hotspot centroid within `match_km`."""
    d = df[df["category"] == category]
    per_year = {}
    for year in sorted(d["timeStamp"].dt.year.unique()):
        sub = d[d["timeStamp"].dt.year == year]
        if len(sub) >= 2000:  # skip tiny partial years
            per_year[year] = find_hotspots(sub, category, sample, eps_km, min_samples)
    years, rows = list(per_year), []
    for a, b in zip(years[:-1], years[1:], strict=True):
        ha, hb = per_year[a], per_year[b]
        pct = np.nan
        if len(ha) and len(hb):
            dist = (
                haversine_distances(np.radians(ha[["lat", "lng"]].values), np.radians(hb[["lat", "lng"]].values))
                * EARTH_KM
            )
            pct = float((dist.min(axis=1) <= match_km).mean() * 100)
        rows.append(
            {"year_a": a, "year_b": b, "hotspots_a": len(ha), "hotspots_b": len(hb), "pct_a_persisting_in_b": pct}
        )
    return pd.DataFrame(rows)
