"""Permutation feature importance on held-out rows, with a random-noise control feature.

Importance = mean increase of the per-row loss when one feature column is shuffled, with a
95% interval from a paired week-bootstrap (rows of one week are resampled together).
Shuffling is either across all rows (groups=None) or only WITHIN groups (e.g. within each
product), which keeps group-level information such as a product's volume intact.
It describes what the MODEL relies on. It is NOT evidence of causation, and for a model that
does not beat a simple baseline it mostly describes reliance on noise.
"""
import numpy as np
import pandas as pd

from src import metrics

NOISE = "noise_random"


def abs_loss(y, p):
    return np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))


def sq_loss(y, p):
    return (np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2


def add_noise(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Copy of df with an independent standard-normal column NOISE (control feature)."""
    out = df.copy()
    out[NOISE] = np.random.default_rng(seed).normal(size=len(df))
    return out


def group_indices(groups):
    """List of positional index arrays, one per distinct group label."""
    g = pd.Series(np.asarray(groups))
    return [np.asarray(ix) for ix in g.groupby(g).indices.values()]


def permuted_order(n, group_idx, rng):
    """Row order for one shuffle: across all rows (group_idx None) or only inside each group."""
    if group_idx is None:
        return rng.permutation(n)
    order = np.arange(n)
    for ix in group_idx:
        order[ix] = rng.permutation(ix)
    return order


def permutation_importance(predict, X: pd.DataFrame, y, weeks, row_loss, n_repeats=20, n_boot=1000, seed=0,
                           groups=None):
    """predict: callable taking a DataFrame with the columns of X. Returns one row per feature.

    groups: optional array (same length as X); if given, values are shuffled only within groups.
    """
    if not (len(X) == len(y) == len(weeks)):
        raise ValueError("X, y and weeks must have the same length.")
    if groups is not None and len(groups) != len(X):
        raise ValueError("groups must have the same length as X.")
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    gidx = None if groups is None else group_indices(groups)
    base = row_loss(y, predict(X))
    rows = []
    for col in X.columns:
        values = X[col].to_numpy()
        losses = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = values[permuted_order(len(X), gidx, rng)]
            losses.append(row_loss(y, predict(Xp)))
        b = metrics.paired_week_bootstrap(weeks, np.mean(losses, axis=0), base, n_boot=n_boot, seed=seed)
        rows.append({"feature": col, "importance": b["mean_diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
    out = pd.DataFrame(rows)
    out["base_loss"] = float(base.mean())
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def flag_informative(imp: pd.DataFrame, noise_name: str = NOISE) -> pd.DataFrame:
    """Add above_noise: True only if the feature's interval lies above zero AND above the noise
    feature's upper bound (pre-specified rule). The noise feature itself is always False."""
    out = imp.copy()
    noise = out.loc[out["feature"] == noise_name]
    if noise.empty:
        raise ValueError("Noise control feature not found in importance table.")
    threshold = max(0.0, float(noise["ci_high"].iloc[0]))
    out["above_noise"] = (out["ci_low"] > threshold) & (out["feature"] != noise_name)
    return out