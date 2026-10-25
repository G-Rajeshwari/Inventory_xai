"""SHAP post-processing: additivity check, global importance with a noise control,
within-product spread, and local (single-row) explanations.

SHAP values describe how a MODEL's prediction was built from its inputs. They are not causal effects.
For a model that does not beat a simple baseline they mostly describe reliance on level and noise.
"""
import numpy as np
import pandas as pd

from src import metrics

NOISE = "noise_random"


def check_additivity(values, base, raw_pred, rel_tol=1e-3):
    """SHAP values plus the base value must reproduce the model's raw prediction for every row."""
    values = np.asarray(values, dtype=float)
    recon = values.sum(axis=1) + np.asarray(base, dtype=float)
    err = np.abs(recon - np.asarray(raw_pred, dtype=float))
    atol = rel_tol * max(1.0, float(np.abs(raw_pred).mean()))
    return {"max_abs_error": float(err.max()), "tolerance": float(atol), "ok": bool(err.max() <= atol)}


def _mean_ci(x, weeks, n_boot, seed):
    b = metrics.paired_week_bootstrap(weeks, x, np.zeros(len(x)), n_boot=n_boot, seed=seed)
    return b["mean_diff"], b["ci_low"], b["ci_high"]


def global_importance(values, columns, weeks, products=None, noise=NOISE, n_boot=1000, seed=0):
    """Mean |SHAP| per feature with a 95% week-bootstrap interval; optionally also the mean absolute
    deviation of a feature's SHAP value from its product average (variation NOT explained by a
    product's level). A feature is 'above noise' if its interval lower bound exceeds the noise
    feature's upper bound."""
    values = np.asarray(values, dtype=float)
    weeks = np.asarray(weeks)
    rows = []
    for j, col in enumerate(columns):
        m, lo, hi = _mean_ci(np.abs(values[:, j]), weeks, n_boot, seed)
        row = {"feature": col, "mean_abs_shap": m, "ci_low": lo, "ci_high": hi}
        if products is not None:
            centered = values[:, j] - pd.Series(values[:, j]).groupby(np.asarray(products)).transform("mean").to_numpy()
            wm, wlo, whi = _mean_ci(np.abs(centered), weeks, n_boot, seed)
            row.update({"within_product_abs": wm, "within_ci_low": wlo, "within_ci_high": whi})
        rows.append(row)
    out = pd.DataFrame(rows)
    if noise not in set(out["feature"]):
        raise ValueError("Noise control feature not found.")
    nz = out.loc[out["feature"] == noise].iloc[0]
    out["above_noise"] = (out["ci_low"] > nz["ci_high"]) & (out["feature"] != noise)
    if products is not None:
        out["within_above_noise"] = (out["within_ci_low"] > nz["within_ci_high"]) & (out["feature"] != noise)
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(values, X_row, columns, base, raw_pred):
    """Contribution table for one row, sorted by absolute SHAP value."""
    v = np.asarray(values, dtype=float)
    tab = pd.DataFrame({"feature": list(columns), "value": list(X_row), "shap": v})
    tab = tab.reindex(tab["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    summary = {"base_value": float(base), "sum_shap": float(v.sum()), "model_prediction": float(raw_pred)}
    return tab, summary


def pick_rows(y, pred):
    """Indices of the row with the largest absolute error and of a median-error row."""
    err = np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
    order = np.argsort(err, kind="stable")
    return {"largest_error": int(order[-1]), "median_error": int(order[len(order) // 2])}


def shap_frame(meta: pd.DataFrame, values, base, raw_pred, columns):
    """Meta columns + one shap_<feature> column per feature + base and raw prediction."""
    out = meta.reset_index(drop=True).copy()
    for j, c in enumerate(columns):
        out[f"shap_{c}"] = np.asarray(values)[:, j]
    out["base_value"] = np.asarray(base, dtype=float) + np.zeros(len(out))
    out["model_prediction"] = np.asarray(raw_pred, dtype=float)
    return out