"""Order-line data for the cost model (Stage 7a).

Target: cost_ratio = 1 - Order Item Profit Ratio. It is DERIVED from profit data,
NOT an observed supply-chain cost. Dollar cost = net price x cost_ratio.
Order lines are split chronologically with the same dates as the demand models.
"""
import pandas as pd

from src import config, data, split

RAW_TO_NAME = {
    "Product Card Id": "product_id",
    "Order Item Quantity": "qty",
    "Order Item Profit Ratio": "profit_ratio",
    "Order Item Product Price": "price",
    "Order Item Discount Rate": "discount_rate",
    "Days for shipment (scheduled)": "sched_days",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Customer Segment": "segment",
    "Type": "pay_type",
    "Order Region": "region",
    "Category Id": "category_id",
    "Department Name": "department",
}
USECOLS = ["order date (DateOrders)"] + list(RAW_TO_NAME)

NUMERIC_FEATURES = ["qty", "price", "discount_rate", "sched_days"]
CATEGORICAL_FEATURES = ["product_id", "category_id", "department", "shipping_mode",
                        "market", "segment", "pay_type", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cost_ratio"

# raw column names of everything the cost model must never see as an input
FORBIDDEN_RAW = list(config.LEAKY_FOR_COST) + list(config.LEAKY_FOR_DEMAND) + ["Order Item Profit Ratio"]


def load_cost_lines(path=None, products=None) -> pd.DataFrame:
    """Order lines in the modelling window with cost_ratio, week and split labels."""
    df = pd.read_csv(path or config.RAW_CSV, encoding="utf-8", usecols=USECOLS)
    df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    if df["order_dt"].isna().any():
        raise ValueError("Unparseable order dates found.")
    df = df[(df["order_dt"] >= config.WINDOW_START) & (df["order_dt"] <= config.WINDOW_END)]
    df = df.rename(columns=RAW_TO_NAME)
    df["week"] = data.add_week_start(df["order_dt"])
    df = df[(df["week"] >= config.FIRST_WEEK) & (df["week"] <= config.LAST_WEEK)]
    if products is not None:
        df = df[df["product_id"].isin(set(products))]
    df[TARGET] = 1.0 - df["profit_ratio"]
    df = split.assign_split(df.reset_index(drop=True))
    keep = ["order_dt", "week", "split", TARGET] + FEATURES
    return df[keep]