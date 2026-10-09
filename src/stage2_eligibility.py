from pathlib import Path
import json
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
SPLIT = json.loads(Path("reports/stage2_split.json").read_text())
OUT = Path("reports/stage2_eligibility_trainonly.csv")

FIRST_WEEK = pd.Timestamp("2015-01-05")
LAST_WEEK = pd.Timestamp("2017-09-18")
WINDOW_END = pd.Timestamp("2017-09-30 23:59:59")

df = pd.read_csv(RAW, encoding="utf-8",
                 usecols=["order date (DateOrders)", "Product Card Id", "Order Item Quantity"])
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df = df[(df["order_dt"] >= "2015-01-01") & (df["order_dt"] <= WINDOW_END)].copy()
df["week"] = df["order_dt"].dt.normalize() - pd.to_timedelta(df["order_dt"].dt.weekday, unit="D")
df = df[(df["week"] >= FIRST_WEEK) & (df["week"] <= LAST_WEEK)]

all_weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
weekly = (df.groupby(["Product Card Id", "week"])["Order Item Quantity"].sum()
            .unstack(fill_value=0).reindex(columns=all_weeks, fill_value=0))

n_train, n_val = SPLIT["train"][2], SPLIT["validation"][2]
train = weekly.iloc[:, :n_train]
val = weekly.iloc[:, n_train:n_train + n_val]

stats = pd.DataFrame({
    "train_zero_share": (train == 0).mean(axis=1),
    "train_mean_units": train.mean(axis=1),
    "sold_last8_train": train.iloc[:, -8:].sum(axis=1) > 0,
    "val_total_units": val.sum(axis=1),
    "val_silent_last8": val.iloc[:, -8:].sum(axis=1) == 0,
})
stats.to_csv(OUT)

print(f"Products with any sale in the window: {len(stats)}")
print(f"Train weeks: {n_train} | Validation weeks: {n_val}\n")
for t in (0.10, 0.20, 0.30):
    m = (stats["train_zero_share"] <= t) & stats["sold_last8_train"]
    silent = (m & stats["val_silent_last8"]).sum()
    share = train.loc[m].to_numpy().sum() / train.to_numpy().sum()
    print(f"train_zero_share <= {t:.2f} and sold in last 8 train weeks: {m.sum():3d} | "
          f"silent in last 8 VALIDATION weeks: {silent} | share of train units: {share:.3f}")

m20 = (stats["train_zero_share"] <= 0.20) & stats["sold_last8_train"]
print("\nMean weekly units in training, products at <= 0.20:")
print(stats.loc[m20, "train_mean_units"].describe().round(1).to_string())
print(f"\nSaved to {OUT}")