import numpy as np
import pandas as pd

from src.anomalies import detect_daily_anomalies
from src.data import clean
from src.forecast import make_features
from src.staffing import units_needed


def _raw():
    return pd.DataFrame(
        {
            "lat": [40.1, 40.1, 0.0, 40.2],
            "lng": [-75.3, -75.3, 0.0, -75.2],
            "title": [
                "EMS: BACK PAINS/INJURY",
                "EMS: BACK PAINS/INJURY",
                "Fire: FIRE ALARM",
                "Traffic: VEHICLE ACCIDENT -",
            ],
            "timeStamp": ["2018-01-01 10:00:00", "2018-01-01 10:00:00", "2018-01-01 11:00:00", "2018-01-02 12:30:00"],
            "twp": ["ABINGTON", "ABINGTON", None, "LOWER MERION"],
        }
    )


def test_clean_splits_title_dedupes_and_filters_gps():
    df = clean(_raw())
    assert len(df) == 2  # 1 duplicate + 1 bad GPS row removed
    assert set(df["category"]) == {"EMS", "Traffic"}
    assert df.loc[df["category"] == "Traffic", "reason"].iat[0] == "VEHICLE ACCIDENT"  # trailing " -" stripped


def test_features_have_no_future_leakage():
    idx = pd.date_range("2020-01-01", periods=600, freq="h")
    s = pd.Series(np.random.default_rng(0).poisson(5, 600), index=idx, name="calls")
    k = 400
    s2 = s.copy()
    s2.iloc[k:] += 1000  # change the "future" from time k onward
    a, b = make_features(s), make_features(s2)
    feats = [c for c in a.columns if c != "y"]
    t = idx[k]
    assert (a.loc[t, feats] == b.loc[t, feats]).all()  # features at t must not see s[t:]


def test_units_needed_monotonic():
    assert units_needed(10, 1.0, 0.6) == 17
    assert units_needed(10, 1.0, 0.8) < units_needed(10, 1.0, 0.6)
    assert units_needed(20, 1.0, 0.6) > units_needed(10, 1.0, 0.6)


def test_anomaly_detector_flags_spike():
    days = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for d in days:
        n = 50 + (d.dayofyear % 3) + (400 if d == pd.Timestamp("2020-04-01") else 0)
        rows += [{"date": d}] * n
    out = detect_daily_anomalies(pd.DataFrame(rows))
    assert out.loc["2020-04-01", "anomaly"]


def test_holiday_flag_and_quality_report():
    from src.quality import quality_report

    idx = pd.date_range("2020-12-20", periods=24 * 30, freq="h")
    s = pd.Series(np.random.default_rng(1).poisson(5, len(idx)), index=idx)
    X = make_features(s)
    assert X.loc["2021-01-01 12:00", "is_holiday"] == 1
    assert X.loc["2021-01-05 12:00", "is_holiday"] == 0

    raw = _raw()
    rep = quality_report(raw, clean(raw))
    assert rep["raw_rows"] == 4 and rep["duplicate_rows"] == 1 and rep["missing_or_out_of_county_gps"] == 1
