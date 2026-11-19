"""Turn demand patterns into a staffing recommendation (assumption-driven, documented)."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES, HANDLE_HOURS, REPORTS, TARGET_UTILIZATION
from .eda import DOW


def units_needed(demand, handle_hours, utilization):
    """Units required to serve `demand` calls/hour if each call ties a unit up for `handle_hours`."""
    return np.ceil(np.asarray(demand) * handle_hours / utilization).astype(int)


def staffing_table(df) -> pd.DataFrame:
    """Units needed per (category, weekday, hour).

    units = ceil( p90 calls/hour * avg hours a unit is busy per call / target utilization )
    Planning to the 90th percentile of demand covers 9 of 10 such hours.
    """
    out = []
    for cat, g in df.groupby("category"):
        s = g.set_index("timeStamp").resample("h").size()
        t = pd.DataFrame({"calls": s})
        t["dow_num"], t["hour"] = t.index.dayofweek, t.index.hour
        agg = (
            t.groupby(["dow_num", "hour"])["calls"]
            .agg(mean_calls="mean", p90_calls=lambda x: x.quantile(0.9))
            .reset_index()
        )
        handle = HANDLE_HOURS.get(cat, 1.0)
        agg["units_needed"] = units_needed(agg["p90_calls"], handle, TARGET_UTILIZATION)
        agg.insert(0, "category", cat)
        out.append(agg)
    table = pd.concat(out, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "staffing_recommendation.csv", index=False)
    return table


def plot_staffing_heatmap(table, category="EMS"):
    t = table[table["category"] == category].pivot(index="dow_num", columns="hour", values="units_needed")
    t.index = DOW
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(t, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"Recommended {category} units on duty (p90 demand)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"08_staffing_{category.lower()}.png", dpi=150)
    plt.close(fig)
