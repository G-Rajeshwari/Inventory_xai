"""Learned demand models. Library defaults first (decision rule 3); only the seed is set.

All models use the same feature columns and the same training rows.
Predictions are clipped at zero because weekly units cannot be negative; the same
rule is applied to every learned model.
"""
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS

MODEL_NAMES = ("random_forest", "xgboost", "tabpfn")


def make_model(name: str, seed: int):
    """Create an unfitted model with library defaults and a fixed seed."""
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=-1)
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(random_state=seed)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}.")


def fit_predict(name: str, seed: int, train: pd.DataFrame, evaluate: pd.DataFrame,
                factory=None, feature_cols=None) -> np.ndarray:
    """Fit on `train`, predict `evaluate`. `factory` lets tests inject a fast fake model.
    `feature_cols` selects a feature subset (ablation); default is all features."""
    model = (factory or make_model)(name, seed)
    cols = list(feature_cols or FEATURE_COLS)
    model.fit(train[cols].to_numpy(dtype=float), train["units"].to_numpy(dtype=float))
    pred = np.asarray(model.predict(evaluate[cols].to_numpy(dtype=float)), dtype=float)
    return np.clip(pred, 0, None)