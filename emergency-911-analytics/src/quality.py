"""Data-quality audit of the raw file. Run BEFORE trusting any analysis."""

import json

import pandas as pd

from .config import REPORTS
from .data import LAT_RANGE, LNG_RANGE


def quality_report(raw: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(raw["timeStamp"], errors="coerce")
    bad_gps = ~(raw["lat"].between(*LAT_RANGE) & raw["lng"].between(*LNG_RANGE))

    daily = clean_df.groupby("date").size()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing = pd.Series(full.difference(daily.index))
    longest_gap = 0
    if len(missing):
        runs = (missing.diff() != pd.Timedelta(days=1)).cumsum()
        longest_gap = int(missing.groupby(runs).size().max())

    nulls = raw.isna().mean() * 100
    return {
        "raw_rows": int(len(raw)),
        "duplicate_rows": int(raw.duplicated().sum()),
        "unparseable_timestamps": int(ts.isna().sum()),
        "missing_or_out_of_county_gps": int(bad_gps.sum()),
        "rows_after_cleaning": int(len(clean_df)),
        "pct_retained": round(len(clean_df) / len(raw) * 100, 2),
        "null_pct_by_column": {c: round(float(v), 2) for c, v in nulls.items() if v > 0},
        "date_range": [str(clean_df["timeStamp"].min()), str(clean_df["timeStamp"].max())],
        "calendar_days_with_no_calls": int(len(missing)),
        "longest_gap_days": longest_gap,
    }


def save_quality(report: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
