"""Learned demand models. Library defaults first (decision rule 3); only the seed is set.

All models use the same feature columns and the same training rows.
Predictions are clipped at zero because weekly units cannot be negative; the same
rule is applied to every learned model.

'tabpfn_cat' is an exploratory variant added AFTER seeing validation results: the
product code is declared categorical instead of being read as an ordered number.
"""
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS

MODEL_NAMES = ("random_forest", "xgboost", "tabpfn", "tabpfn_cat")


def make_model(name: str, seed: int, cols=None):
    """Create an unfitted model with library defaults and a fixed seed.

    `cols` is the feature column list; only 'tabpfn_cat' uses it.
    """
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=-1)
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(random_state=seed)
    if name == "tabpfn_cat":
        from tabpfn import TabPFNRegressor
        cat = [list(cols).index("product_code")] if cols and "product_code" in cols else None
        return TabPFNRegressor(random_state=seed, categorical_features_indices=cat)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}.")


def fit_predict(name: str, seed: int, train: pd.DataFrame, evaluate: pd.DataFrame,
                factory=None, feature_cols=None) -> np.ndarray:
    """Fit on `train`, predict `evaluate`. `factory` lets tests inject a fast fake model.
    `feature_cols` selects a feature subset (ablation); default is all features."""
    cols = list(feature_cols or FEATURE_COLS)
    model = factory(name, seed) if factory is not None else make_model(name, seed, cols)
    model.fit(train[cols].to_numpy(dtype=float), train["units"].to_numpy(dtype=float))
    pred = np.asarray(model.predict(evaluate[cols].to_numpy(dtype=float)), dtype=float)
    return np.clip(pred, 0, None)