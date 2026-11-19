from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "911.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "calls_clean.parquet"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42

# Staffing assumptions (NOT from the dataset - state them in your README and tune them)
# Average hours a unit is tied up per call, by category
HANDLE_HOURS = {"EMS": 1.0, "Fire": 1.25, "Traffic": 0.75}
# Fraction of a unit's time we are willing to keep busy (headroom for surges)
TARGET_UTILIZATION = 0.6
