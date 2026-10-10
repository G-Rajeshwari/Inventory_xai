"""Load order lines and build the dense weekly product table."""
import json

import pandas as pd

from src import config

LINE_COLS = ["order date (DateOrders)", "Product Card Id", "Order Item Quantity"]


def add_week_start(dt: pd.Series) -> pd.Series:
    """Monday 00:00 of the week containing each timestamp."""
    return dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")


def load_order_lines(path=None) -> pd.DataFrame:
    """Order lines inside the modelling window, with a Monday-start 'week' column."""
    df = pd.read_csv(path or config.RAW_CSV, encoding="utf-8", usecols=LINE_COLS)
    df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    if df["order_dt"].isna().any():
        raise ValueError("Unparseable order dates found.")
    df = df[(df["order_dt"] >= config.WINDOW_START) & (df["order_dt"] <= config.WINDOW_END)]
    df = df.rename(columns={"Product Card Id": "product_id", "Order Item Quantity": "qty"})
    df["week"] = add_week_start(df["order_dt"])
    df = df[(df["week"] >= config.FIRST_WEEK) & (df["week"] <= config.LAST_WEEK)]
    return df[["product_id", "week", "qty"]].reset_index(drop=True)


def build_weekly_table(lines: pd.DataFrame, products, first_week=None, last_week=None) -> pd.DataFrame:
    """Dense product x week grid with zero-filled weekly units.

    Returns long format: product_id, week, units (sorted by product, week).
    """
    first_week = pd.Timestamp(first_week or config.FIRST_WEEK)
    last_week = pd.Timestamp(last_week or config.LAST_WEEK)
    weeks = pd.date_range(first_week, last_week, freq="7D")
    products = sorted(set(products))
    sub = lines[lines["product_id"].isin(products)]
    grid = pd.MultiIndex.from_product([products, weeks], names=["product_id", "week"])
    units = sub.groupby(["product_id", "week"])["qty"].sum().reindex(grid, fill_value=0)
    return units.rename("units").reset_index()


def load_eligible_products() -> pd.DataFrame:
    """Frozen product list from Stage 2 (index = product id)."""
    df = pd.read_csv(config.REPORTS / "eligible_products.csv", index_col=0)
    df.index.name = "product_id"
    return df


def load_split() -> dict:
    return json.loads((config.REPORTS / "stage2_split.json").read_text())