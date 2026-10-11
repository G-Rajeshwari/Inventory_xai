"""Cost-ratio models (Stage 7a). Library defaults; only the seed is set.

Random Forest and XGBoost get ordinal-coded categoricals (codes learned on the training
lines; unseen values -> -1). CatBoost receives the categoricals natively as strings.
Predictions are clipped to the training range of cost_ratio for every learned model.
"""
import numpy as np
import pandas as pd

from src.cost_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

MODEL_NAMES = ("catboost", "random_forest", "xgboost")


class OrdinalCoder:
    """Maps each categorical value to an integer, learned on training rows only."""

    def fit(self, df: pd.DataFrame):
        self.maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique(), key=str))}
                     for c in CATEGORICAL_FEATURES}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[c].astype(float).to_numpy() for c in NUMERIC_FEATURES]
        cols += [df[c].map(self.maps[c]).fillna(-1).to_numpy(dtype=float) for c in CATEGORICAL_FEATURES]
        return np.column_stack(cols)


def as_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype(str)
    return out


def make_model(name: str, seed: int):
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, verbose=0, thread_count=-1,
                                 allow_writing_files=False, cat_features=list(CATEGORICAL_FEATURES))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}.")


def fit_predict(name: str, seed: int, train: pd.DataFrame, evaluate: pd.DataFrame, factory=None) -> np.ndarray:
    """Fit on `train`, predict `evaluate`. `factory` lets tests inject a fast fake model."""
    model = (factory or make_model)(name, seed)
    y = train[TARGET].to_numpy(dtype=float)
    if name == "catboost":
        model.fit(as_catboost_frame(train), y)
        pred = model.predict(as_catboost_frame(evaluate))
    else:
        coder = OrdinalCoder().fit(train)
        model.fit(coder.transform(train), y)
        pred = model.predict(coder.transform(evaluate))
    return np.clip(np.asarray(pred, dtype=float), y.min(), y.max())