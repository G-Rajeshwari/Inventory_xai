"""Leak-free features. Every feature for week t uses only weeks < t."""
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4)
ROLL_WINDOWS = (4, 13, 26)
WARMUP_WEEKS = max(ROLL_WINDOWS)  # rows before this have incomplete history and are dropped

FEATURE_COLS = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["week_of_year", "product_code", "is_tier_a"]
)


def build_features(table: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
    """table: product_id, week, units (+ optional split). tiers: product_id -> tier name.

    Returns one row per product-week with complete history, including the target 'units'.
    """
    df = table.sort_values(["product_id", "week"]).reset_index(drop=True).copy()
    g = df.groupby("product_id")["units"]
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(1)  # information up to week t-1 only
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = (
            shifted.groupby(df["product_id"]).rolling(w, min_periods=w).mean()
            .reset_index(level=0, drop=True)
        )
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    codes = {p: i for i, p in enumerate(sorted(df["product_id"].unique()))}
    df["product_code"] = df["product_id"].map(codes)
    df["is_tier_a"] = (df["product_id"].map(tiers) == "A_high_volume").astype(int)
    df = df.dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    return df