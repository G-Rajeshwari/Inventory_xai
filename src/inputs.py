"""Planning inputs for the optimizer, all derived from TRAINING weeks only.

Lead time per mode reuses CUSTOMER delivery days (Days for shipping (real)) as a stand-in
for replenishment lead time: this is a modelling assumption, not an observed supplier lead time.
"""
import numpy as np
import pandas as pd

from src import config, data, split
from src.assumptions import MODES

SHIP_COLS = ["order date (DateOrders)", "Shipping Mode", "Days for shipping (real)",
             "Days for shipment (scheduled)"]


def load_shipping_lines(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.RAW_CSV, encoding="utf-8", usecols=SHIP_COLS)
    df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    if df["order_dt"].isna().any():
        raise ValueError("Unparseable order dates found.")
    df = df[(df["order_dt"] >= config.WINDOW_START) & (df["order_dt"] <= config.WINDOW_END)]
    df = df.rename(columns={"Shipping Mode": "mode", "Days for shipping (real)": "real_days",
                            "Days for shipment (scheduled)": "sched_days"})
    df["week"] = data.add_week_start(df["order_dt"])
    df = df[(df["week"] >= config.FIRST_WEEK) & (df["week"] <= config.LAST_WEEK)]
    return split.assign_split(df.reset_index(drop=True))[["week", "split", "mode", "real_days", "sched_days"]]


def mode_params(ship_lines: pd.DataFrame) -> pd.DataFrame:
    """Per mode (training lines only): mean real days, lead time in weeks, late rate."""
    tr = ship_lines[ship_lines["split"] == "train"].copy()
    tr["late"] = (tr["real_days"] > tr["sched_days"]).astype(float)
    g = tr.groupby("mode")
    out = pd.DataFrame({"days": g["real_days"].mean(), "late_rate": g["late"].mean()}).reindex(list(MODES))
    if out.isna().any().any():
        raise ValueError(f"Missing shipping modes in training data: {out[out.isna().any(axis=1)].index.tolist()}")
    out["lead_weeks"] = out["days"] / 7.0
    if (out["lead_weeks"] >= 1.0).any():
        raise ValueError("Simulation assumes lead time < 1 week.")
    return out


def demand_by_split(weekly: pd.DataFrame):
    """weekly: product_id, week, units. Returns ({split: {pid: array}}, {split: weeks array})."""
    w = split.assign_split(weekly)
    demand, weeks = {}, {}
    for sp, g in w.groupby("split"):
        g = g.sort_values(["product_id", "week"])
        demand[sp] = {pid: gg["units"].to_numpy(dtype=float) for pid, gg in g.groupby("product_id")}
        weeks[sp] = np.sort(g["week"].unique())
    return demand, weeks


def product_params(cost_lines: pd.DataFrame, demand_train: dict, cost_ratio: float) -> pd.DataFrame:
    """Per product: list price, mean discount, net price, derived unit cost, demand mean and std (train)."""
    tr = cost_lines[cost_lines["split"] == "train"]
    g = tr.groupby("product_id")
    out = pd.DataFrame({"price": g["price"].mean(), "mean_discount": g["discount_rate"].mean()})
    out = out.loc[sorted(demand_train)]
    out["net_price"] = out["price"] * (1 - out["mean_discount"])
    out["unit_cost"] = out["net_price"] * cost_ratio            # DERIVED, not an observed cost
    out["mu"] = [float(np.mean(demand_train[p])) for p in out.index]
    out["sigma"] = [float(np.std(demand_train[p], ddof=1)) for p in out.index]
    return out