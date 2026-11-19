"""24-hour-ahead hourly call forecasting with honest time-based evaluation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .config import FIGURES, REPORTS, SEED

HORIZON = 24  # forecast 24h ahead, so every feature must be known 24h before the target


def hourly_series(df) -> pd.Series:
    return df.set_index("timeStamp").resample("h").size().rename("calls")


def make_features(s: pd.Series, holidays: bool = True) -> pd.DataFrame:
    X = pd.DataFrame({"y": s})
    X["lag_24"] = s.shift(24)
    X["lag_48"] = s.shift(48)
    X["lag_168"] = s.shift(168)  # same hour last week
    X["roll_168_mean"] = s.shift(HORIZON).rolling(168).mean()  # shifted -> no leakage
    X["hour"] = X.index.hour
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    if holidays:  # calendar information is known in advance, so it cannot leak
        hol = USFederalHolidayCalendar().holidays(start=s.index.min().normalize(), end=s.index.max().normalize())
        X["is_holiday"] = X.index.normalize().isin(hol).astype(int)
    return X.dropna()


def _metrics(y, pred) -> dict:
    err = np.abs(y - pred)
    return {"MAE": float(err.mean()), "WAPE_%": float(err.sum() / y.sum() * 100)}


def _model():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, random_state=SEED)


def _quantile_model(alpha):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, max_iter=200, learning_rate=0.05, random_state=SEED
    )


def evaluate(df, test_days=60):
    s = hourly_series(df)
    X = make_features(s)
    feats = [c for c in X.columns if c != "y"]

    # 1) Final hold-out: last `test_days` days are never seen in training
    split = X.index.max() - pd.Timedelta(days=test_days)
    train, test = X[X.index <= split], X[X.index > split]
    model = _model().fit(train[feats], train["y"])
    pred = model.predict(test[feats])
    naive = test["lag_168"].values

    # 80% prediction interval from 10th/90th percentile models
    lo = np.minimum(_quantile_model(0.1).fit(train[feats], train["y"]).predict(test[feats]), pred)
    hi = np.maximum(_quantile_model(0.9).fit(train[feats], train["y"]).predict(test[feats]), pred)
    y_test = test["y"].values
    interval = {
        "nominal_coverage_pct": 80.0,
        "empirical_coverage_pct": float(((y_test >= lo) & (y_test <= hi)).mean() * 100),
        "mean_interval_width": float((hi - lo).mean()),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": test.index, "actual": y_test, "gbm": pred, "naive": naive, "lo": lo, "hi": hi}).to_csv(
        REPORTS / "forecast_predictions.csv", index=False
    )

    holdout = pd.DataFrame(
        {
            "seasonal_naive (same hour last week)": _metrics(test["y"].values, naive),
            "hist_gradient_boosting": _metrics(test["y"].values, pred),
        }
    ).T

    # Feature-group ablation: what does each group of features buy us?
    calendar = ["hour", "dow", "month", "is_weekend"]
    lags = ["lag_24", "lag_48", "lag_168", "roll_168_mean"]
    groups = {"calendar only": calendar, "+ lag features": calendar + lags, "+ holiday flag": feats}
    abl = []
    for name, cols in groups.items():
        m_ = _model().fit(train[cols], train["y"])
        abl.append({"model": name, **_metrics(y_test, m_.predict(test[cols]))})
    pd.DataFrame(abl).to_csv(REPORTS / "forecast_ablation.csv", index=False)

    # Permutation importance on the hold-out (correlated lags share importance, interpret with care)
    imp = permutation_importance(
        model, test[feats], test["y"], n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error"
    )
    (
        pd.DataFrame({"feature": feats, "mae_increase": imp.importances_mean})
        .sort_values("mae_increase", ascending=False)
        .to_csv(REPORTS / "forecast_importance.csv", index=False)
    )

    # 2) Rolling-origin CV: 4 forward-chaining folds of 30 days each
    tscv = TimeSeriesSplit(n_splits=4, test_size=24 * 30)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        m = _model().fit(X.iloc[tr][feats], X.iloc[tr]["y"])
        yte = X.iloc[te]["y"].values
        rows.append(
            {
                "fold": fold,
                "naive_MAE": _metrics(yte, X.iloc[te]["lag_168"].values)["MAE"],
                "gbm_MAE": _metrics(yte, m.predict(X.iloc[te][feats]))["MAE"],
            }
        )
    cv = pd.DataFrame(rows)

    # plot last 14 days of hold-out
    mask = test.index > test.index.max() - pd.Timedelta(days=14)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index[mask], test["y"][mask], label="Actual", lw=1.5)
    ax.fill_between(test.index[mask], lo[mask], hi[mask], alpha=0.2, color="#E4572E", label="80% interval")
    ax.plot(test.index[mask], pred[mask], label="GBM forecast", lw=1.2, color="#E4572E")
    ax.plot(test.index[mask], naive[mask], label="Seasonal naive", lw=1, alpha=0.6)
    ax.set_title("Hourly calls: last 14 days of hold-out")
    ax.legend()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_forecast_holdout.png", dpi=150)
    plt.close(fig)
    return holdout, cv, interval
