"""Simple forecasting baselines (one week ahead)."""
import pandas as pd


def add_baselines(feat: pd.DataFrame, fit_splits=("train",)) -> pd.DataFrame:
    """Add prediction columns pred_*.

    pred_last_week / pred_ma4 / pred_ma26 come from leak-free features.
    pred_train_mean: each product's mean units over the fit splits only.
    Use fit_splits=('train',) for validation; for the final test run, refit on
    ('train', 'validation') and state this in the paper.
    """
    out = feat.copy()
    out["pred_last_week"] = out["lag_1"]
    out["pred_ma4"] = out["roll_mean_4"]
    out["pred_ma26"] = out["roll_mean_26"]
    fit = out[out["split"].isin(fit_splits)]
    means = fit.groupby("product_id")["units"].mean().rename("pred_train_mean")
    out = out.merge(means, left_on="product_id", right_index=True, how="left")
    return out


BASELINE_COLS = ["pred_train_mean", "pred_ma26", "pred_ma4", "pred_last_week"]