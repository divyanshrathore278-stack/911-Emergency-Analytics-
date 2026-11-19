"""Load and clean the Montgomery County 911 dataset."""

import logging

import pandas as pd

from .config import DATA_PROCESSED

log = logging.getLogger(__name__)

# Rough bounding box around Montgomery County, PA. Drops bad GPS rows (0,0 etc.)
LAT_RANGE = (39.9, 40.5)
LNG_RANGE = (-75.8, -74.9)


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates().copy()
    df["timeStamp"] = pd.to_datetime(df["timeStamp"], errors="coerce")
    df = df.dropna(subset=["timeStamp", "lat", "lng", "title"])
    df = df[df["lat"].between(*LAT_RANGE) & df["lng"].between(*LNG_RANGE)]

    # "EMS: BACK PAINS/INJURY" -> category="EMS", reason="BACK PAINS/INJURY"
    parts = df["title"].str.split(":", n=1, expand=True)
    df["category"] = parts[0].str.strip()
    df["reason"] = parts[1].fillna("UNKNOWN").str.strip().str.rstrip("-").str.strip()

    df["twp"] = df["twp"].fillna("UNKNOWN")
    df["hour"] = df["timeStamp"].dt.hour
    df["dow_num"] = df["timeStamp"].dt.dayofweek  # Monday=0
    df["month"] = df["timeStamp"].dt.month
    df["date"] = df["timeStamp"].dt.normalize()
    df["is_weekend"] = df["dow_num"] >= 5

    df = df.sort_values("timeStamp").reset_index(drop=True)
    log.info("Cleaning: %s -> %s rows (%s removed)", f"{n0:,}", f"{len(df):,}", f"{n0 - len(df):,}")
    return df


def save_processed(df: pd.DataFrame) -> None:
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED, index=False)


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED)
