from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
SPLIT = json.loads(Path("reports/stage2_split.json").read_text())
OUT = Path("reports/eligible_products.csv")

FIRST_WEEK = pd.Timestamp("2015-01-05")
LAST_WEEK = pd.Timestamp("2017-09-18")
WINDOW_END = pd.Timestamp("2017-09-30 23:59:59")
TIER_A_MIN_MEAN = 50   # units per week in TRAINING weeks (judgement call, see output)

df = pd.read_csv(RAW, encoding="utf-8",
                 usecols=["order date (DateOrders)", "Product Card Id",
                          "Product Name", "Order Item Quantity"])
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df = df[(df["order_dt"] >= "2015-01-01") & (df["order_dt"] <= WINDOW_END)].copy()
df["week"] = df["order_dt"].dt.normalize() - pd.to_timedelta(df["order_dt"].dt.weekday, unit="D")
df = df[(df["week"] >= FIRST_WEEK) & (df["week"] <= LAST_WEEK)]
names = df.groupby("Product Card Id")["Product Name"].first()

all_weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
weekly = (df.groupby(["Product Card Id", "week"])["Order Item Quantity"].sum()
            .unstack(fill_value=0).reindex(columns=all_weeks, fill_value=0))

n_train, n_val = SPLIT["train"][2], SPLIT["validation"][2]
train = weekly.iloc[:, :n_train]
val = weekly.iloc[:, n_train:n_train + n_val]   # test weeks are never touched here

zero_train = (train == 0).mean(axis=1)
eligible = (zero_train <= 0.20) & (train.iloc[:, -8:].sum(axis=1) > 0)

tab = pd.DataFrame({
    "product_name": names,
    "train_mean_units": train.mean(axis=1),
    "train_var_units": train.var(axis=1, ddof=1),
    "train_zero_share": zero_train,
    "val_mean_units": val.mean(axis=1),
    "val_zero_share": (val == 0).mean(axis=1),
})[eligible]
tab["dispersion"] = tab["train_var_units"] / tab["train_mean_units"]
tab["tier"] = np.where(tab["train_mean_units"] >= TIER_A_MIN_MEAN, "A_high_volume", "B_low_volume")
tab.sort_values("train_mean_units", ascending=False).to_csv(OUT)

print("Top 15 products by training mean units/week (look for a natural gap):")
print(tab["train_mean_units"].sort_values(ascending=False).head(15).round(1).to_string())

print("\nPer tier:")
summary = tab.groupby("tier").agg(
    products=("train_mean_units", "size"),
    train_mean_units=("train_mean_units", "median"),
    train_zero_share=("train_zero_share", "median"),
    val_zero_share=("val_zero_share", "median"),
    val_mean_units=("val_mean_units", "median"),
    dispersion=("dispersion", "median"),
).round(3)
print(summary.to_string())
share = (train.loc[tab.index].sum(axis=1).groupby(tab["tier"]).sum()
         / train.loc[tab.index].to_numpy().sum()).round(3)
print("\nShare of training units by tier:")
print(share.to_string())
print(f"\nSaved to {OUT}")