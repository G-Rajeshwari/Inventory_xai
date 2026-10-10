"""Regression metrics and a paired bootstrap over weeks."""
import numpy as np
import pandas as pd


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((y - p) ** 2) / ss_tot)


def wape(y, p):
    """Sum of absolute errors / sum of actuals. Safe when some actuals are zero."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom) if denom > 0 else float("nan")


def summarize(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p), "WAPE": wape(y, p), "n": len(y)}


def paired_week_bootstrap(week, abs_err_a, abs_err_b, n_boot=2000, seed=0, alpha=0.05):
    """95% CI for mean(abs_err_a - abs_err_b), resampling WEEKS (rows in a week stay together).

    Negative difference means model A has lower error than model B.
    """
    df = pd.DataFrame({"week": np.asarray(week), "d": np.asarray(abs_err_a, float) - np.asarray(abs_err_b, float)})
    per_week = df.groupby("week")["d"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_week), size=(n_boot, len(per_week)))
    boots = per_week[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(per_week.mean()), "ci_low": float(lo), "ci_high": float(hi)}