"""Flag unusual days using a robust (median/MAD) z-score on deviations from a weekday baseline."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, REPORTS


def detect_daily_anomalies(df, window_weeks=8, z_thresh=3.5) -> pd.DataFrame:
    """Expected calls = median of the same weekday over the previous `window_weeks` weeks.
    Robust z = 0.6745 * (residual - median) / MAD, so outliers do not distort their own threshold."""
    daily = df.groupby("date").size()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    out = pd.DataFrame({"calls": daily.reindex(full, fill_value=0).iloc[1:-1]})  # drop partial edge days
    out["expected"] = out.groupby(out.index.dayofweek)["calls"].transform(
        lambda s: s.shift(1).rolling(window_weeks, min_periods=4).median()
    )
    out = out.dropna()
    resid = out["calls"] - out["expected"]
    mad = np.median(np.abs(resid - resid.median())) or 1.0
    out["robust_z"] = 0.6745 * (resid - resid.median()) / mad
    out["anomaly"] = out["robust_z"].abs() > z_thresh
    out.index.name = "date"
    return out


def save_anomalies(daily):
    REPORTS.mkdir(parents=True, exist_ok=True)
    daily.to_csv(REPORTS / "daily_anomalies.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["calls"], lw=0.8, color="#98A2B3", label="Daily calls")
    a = daily[daily["anomaly"]]
    ax.scatter(a.index, a["calls"], color="#E4572E", s=22, zorder=3, label=f"Anomalous days ({len(a)})")
    ax.set_title("Daily volume with robust anomaly detection")
    ax.legend()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_anomalies.png", dpi=150)
    plt.close(fig)
