"""Statistical testing with effect sizes (not just p-values)."""

import numpy as np
import pandas as pd
from scipy import stats


def weekday_vs_weekend(df) -> dict:
    """Do weekends get a different number of calls per day than weekdays?

    Daily counts are not guaranteed to be normal, so we use Mann-Whitney U and report
    rank-biserial correlation (effect size) plus Cohen's d for interpretability.
    """
    daily = df.groupby("date").size()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0).iloc[1:-1]  # drop partial first/last day
    wk = daily[daily.index.dayofweek < 5].values
    we = daily[daily.index.dayofweek >= 5].values

    u, p = stats.mannwhitneyu(wk, we, alternative="two-sided")
    rank_biserial = 2 * u / (len(wk) * len(we)) - 1  # >0 means weekdays higher
    pooled = np.sqrt(((len(wk) - 1) * wk.var(ddof=1) + (len(we) - 1) * we.var(ddof=1)) / (len(wk) + len(we) - 2))
    return {
        "test": "Mann-Whitney U (weekday vs weekend daily calls)",
        "weekday_mean": float(wk.mean()),
        "weekend_mean": float(we.mean()),
        "pct_difference": float((wk.mean() - we.mean()) / we.mean() * 100),
        "p_value": float(p),
        "rank_biserial": float(rank_biserial),
        "cohens_d": float((wk.mean() - we.mean()) / pooled),
    }


def category_mix_by_township(df, top_n=10) -> dict:
    """Is the mix of EMS/Fire/Traffic the same across townships? Chi-square + Cramer's V."""
    top = df["twp"].value_counts().head(top_n).index
    sub = df[df["twp"].isin(top)]
    ct = pd.crosstab(sub["twp"], sub["category"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return {
        "test": f"Chi-square (category x top {top_n} townships)",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(cramers_v),
        "note": "With very large n, p is almost always ~0; read Cramer's V for practical size.",
    }
