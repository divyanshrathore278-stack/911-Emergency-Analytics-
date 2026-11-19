"""Generate SYNTHETIC data in the Kaggle 911.csv schema to test the pipeline without downloading.
Never publish results from this file as real findings."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
hours = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h")
hod = hours.hour.values
base = 5 + 4 * np.exp(-((hod - 14) ** 2) / 40) + 1.5 * np.exp(-((hod - 8) ** 2) / 6)
base = base * np.where(hours.dayofweek >= 5, 0.85, 1.0) * (1 + 0.1 * np.sin(2 * np.pi * hours.dayofyear / 365))
counts = rng.poisson(base)
ts = np.repeat(hours.values, counts) + rng.integers(0, 3600, counts.sum()).astype("timedelta64[s]")

twps = [
    "LOWER MERION",
    "ABINGTON",
    "NORRISTOWN",
    "UPPER MORELAND",
    "CHELTENHAM",
    "POTTSTOWN",
    "UPPER MERION",
    "PLYMOUTH",
    "HORSHAM",
    "LANSDALE",
    "WHITEMARSH",
    "LIMERICK",
]
centers = {t: (40.0 + 0.03 * i, -75.5 + 0.04 * (i % 5)) for i, t in enumerate(twps)}
weights = rng.dirichlet(np.ones(len(twps)) * 3)
n = len(ts)
twp = rng.choice(twps, n, p=weights)
lat = np.array([centers[t][0] for t in twp]) + rng.normal(0, 0.01, n)
lng = np.array([centers[t][1] for t in twp]) + rng.normal(0, 0.01, n)

titles = {
    "EMS": ["EMS: BACK PAINS/INJURY", "EMS: FALL VICTIM", "EMS: RESPIRATORY EMERGENCY", "EMS: CARDIAC EMERGENCY"],
    "Fire": ["Fire: FIRE ALARM", "Fire: VEHICLE ACCIDENT", "Fire: GAS-ODOR/LEAK"],
    "Traffic": ["Traffic: VEHICLE ACCIDENT -", "Traffic: DISABLED VEHICLE -", "Traffic: ROAD OBSTRUCTION -"],
}
mix = {t: rng.dirichlet([5, 2, 3]) for t in twps}
cats = np.array([rng.choice(["EMS", "Fire", "Traffic"], p=mix[t]) for t in twp])
title = np.array([rng.choice(titles[c]) for c in cats])

df = pd.DataFrame(
    {
        "lat": lat,
        "lng": lng,
        "desc": "synthetic",
        "zip": 19000.0,
        "title": title,
        "timeStamp": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "twp": twp,
        "addr": "SYNTHETIC ST",
        "e": 1,
    }
)
df.to_csv("data/911_sample.csv", index=False)
print(f"wrote {len(df):,} synthetic rows")
