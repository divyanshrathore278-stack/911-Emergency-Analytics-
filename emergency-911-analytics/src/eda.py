"""Exploratory plots (Seaborn). Every function saves a PNG into reports/figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def plot_category_mix(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="category", order=df["category"].value_counts().index, ax=ax)
    ax.set_title("911 calls by category")
    ax.set_xlabel("")
    _save(fig, "01_category_mix.png")


def plot_top_reasons(df, n=12):
    top = df.groupby(["category", "reason"]).size().rename("calls").reset_index().nlargest(n, "calls")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, y="reason", x="calls", hue="category", dodge=False, ax=ax)
    ax.set_title(f"Top {n} call reasons")
    ax.set_ylabel("")
    _save(fig, "02_top_reasons.png")


def hour_dow_matrix(df) -> pd.DataFrame:
    """Average calls per hour-slot (total calls / number of such weekdays in the data)."""
    counts = df.groupby(["dow_num", "hour"]).size().unstack(fill_value=0)
    n_days = df.groupby("dow_num")["date"].nunique()
    return counts.div(n_days, axis=0).reindex(index=range(7), columns=range(24), fill_value=0)


def plot_hour_dow_heatmap(df, category=None):
    d = df if category is None else df[df["category"] == category]
    m = hour_dow_matrix(d)
    m.index = DOW
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(m, cmap="YlOrRd", ax=ax, cbar_kws={"label": "avg calls per hour"})
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.set_title(f"When do calls happen? ({category or 'all categories'})")
    _save(fig, f"03_heatmap_{(category or 'all').lower()}.png")


def plot_monthly_trend(df):
    m = df.set_index("timeStamp").groupby("category").resample("MS").size().rename("calls").reset_index()
    first, last = m["timeStamp"].min(), m["timeStamp"].max()
    m = m[(m["timeStamp"] > first) & (m["timeStamp"] < last)]  # drop partial months
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=m, x="timeStamp", y="calls", hue="category", marker="o", ax=ax)
    ax.set_title("Monthly call volume (partial first/last months excluded)")
    ax.set_xlabel("")
    _save(fig, "04_monthly_trend.png")


def plot_top_townships(df, n=15):
    top = df["twp"].value_counts().head(n).reset_index()
    top.columns = ["twp", "calls"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, y="twp", x="calls", color="steelblue", ax=ax)
    ax.set_title(f"Top {n} townships by call volume")
    ax.set_ylabel("")
    _save(fig, "05_top_townships.png")


def run_all(df):
    plot_category_mix(df)
    plot_top_reasons(df)
    plot_hour_dow_heatmap(df)
    for cat in df["category"].unique():
        plot_hour_dow_heatmap(df, cat)
    plot_monthly_trend(df)
    plot_top_townships(df)
