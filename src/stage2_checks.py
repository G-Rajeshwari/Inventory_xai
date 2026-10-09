from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
OUT_TXT = Path("reports/stage2_checks.txt")
OUT_CSV = Path("reports/stage2_product_eligibility.csv")
OUT_SPLIT = Path("reports/stage2_split.json")
OUT_TXT.parent.mkdir(exist_ok=True)

WINDOW_START = pd.Timestamp("2015-01-01")
WINDOW_END = pd.Timestamp("2017-09-30 23:59:59")
FIRST_WEEK = pd.Timestamp("2015-01-05")   # first Monday inside the window
LAST_WEEK = pd.Timestamp("2017-09-18")    # last Monday whose full week is inside the window
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

lines = []
def w(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    w("\n" + "=" * 70)
    w(title)
    w("=" * 70)

cols = ["order date (DateOrders)", "Product Card Id", "Order Item Quantity",
        "Order Item Total", "Order Profit Per Order", "Order Item Profit Ratio",
        "Order Item Product Price", "Order Item Discount Rate", "Order Item Discount",
        "Sales", "Category Id", "Shipping Mode", "Market", "Customer Segment",
        "Type", "Order Region"]
df = pd.read_csv(RAW, encoding="utf-8", usecols=cols)
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df = df[(df["order_dt"] >= WINDOW_START) & (df["order_dt"] <= WINDOW_END)].copy()
w(f"Rows inside modelling window: {len(df)}")

# ---------------------------------------------------------------
# 1. Weekly demand table (Monday-start weeks, full weeks only)
# ---------------------------------------------------------------
df["week"] = df["order_dt"].dt.normalize() - pd.to_timedelta(df["order_dt"].dt.weekday, unit="D")
dfw = df[(df["week"] >= FIRST_WEEK) & (df["week"] <= LAST_WEEK)]
all_weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
weekly = (dfw.groupby(["Product Card Id", "week"])["Order Item Quantity"].sum()
             .unstack(fill_value=0).reindex(columns=all_weeks, fill_value=0))
N = weekly.shape[1]

section("1. Which products have a usable weekly history?")
w(f"Products: {weekly.shape[0]} | Weeks: {N} ({all_weeks[0].date()} -> {all_weeks[-1].date()})")

stats = pd.DataFrame({
    "zero_share": (weekly == 0).mean(axis=1),
    "mean_weekly_units": weekly.mean(axis=1),
    "first_sale_week": weekly.gt(0).idxmax(axis=1),
    "last_sale_week": weekly.gt(0).iloc[:, ::-1].idxmax(axis=1),
    "sold_first8": weekly.iloc[:, :8].sum(axis=1) > 0,
    "sold_last8": weekly.iloc[:, -8:].sum(axis=1) > 0,
})
w("\nProducts by maximum share of zero-sales weeks:")
for t in (0.05, 0.10, 0.20, 0.30, 0.50):
    w(f"  zero_share <= {t:.2f}: {(stats['zero_share'] <= t).sum()}")

stats["eligible"] = (stats["zero_share"] <= 0.20) & stats["sold_first8"] & stats["sold_last8"]
stats.to_csv(OUT_CSV)
elig = stats.index[stats["eligible"]]
w(f"\nELIGIBLE (zero_share <= 0.20, sold in first 8 and last 8 weeks): {len(elig)}")
w(f"Their share of all units in the window: "
  f"{weekly.loc[elig].to_numpy().sum() / weekly.to_numpy().sum():.3f}")
w("Mean weekly units of eligible products (describe):")
w(stats.loc[elig, "mean_weekly_units"].describe().round(2).to_string())

# ---------------------------------------------------------------
# 2. Chronological split (boundaries only; test set is NOT touched)
# ---------------------------------------------------------------
n_train = int(round(TRAIN_FRAC * N))
n_val = int(round(VAL_FRAC * N))
split = {
    "n_weeks": N,
    "train": [str(all_weeks[0].date()), str(all_weeks[n_train - 1].date()), n_train],
    "validation": [str(all_weeks[n_train].date()), str(all_weeks[n_train + n_val - 1].date()), n_val],
    "test": [str(all_weeks[n_train + n_val].date()), str(all_weeks[-1].date()), N - n_train - n_val],
}
OUT_SPLIT.write_text(json.dumps(split, indent=2), encoding="utf-8")
section("2. Chronological split (weeks)")
w(json.dumps(split, indent=2))

# ---------------------------------------------------------------
# 3. Predictability preview on TRAIN + VALIDATION only
# ---------------------------------------------------------------
section("3. Is weekly demand predictable at all? (train + validation only)")
if len(elig) > 0:
    arr = weekly.loc[elig].to_numpy(dtype=float)
    train = arr[:, :n_train]
    w(f"TabPFN training rows estimate (eligible products x train weeks): {len(elig) * n_train}")

    def acf(row, lag):
        return pd.Series(row).autocorr(lag=lag)
    ac1 = np.array([acf(r, 1) for r in train])
    ac52 = np.array([acf(r, 52) for r in train])
    w(f"\nMedian lag-1 autocorrelation (train):  {np.nanmedian(ac1):.3f}")
    w(f"Median lag-52 autocorrelation (train): {np.nanmedian(ac52):.3f}")
    w("(near 0 = looks like random noise around a level; clearly positive = real structure)")

    val_idx = list(range(n_train, n_train + n_val))
    actual = arr[:, n_train:n_train + n_val]
    naive = arr[:, [t - 1 for t in val_idx]]
    ma4 = np.column_stack([arr[:, t - 4:t].mean(axis=1) for t in val_idx])
    tmean = np.repeat(train.mean(axis=1, keepdims=True), n_val, axis=1)

    w("\nOne-week-ahead MAE on the VALIDATION weeks (mean over products):")
    w(f"  previous week      : {np.abs(actual - naive).mean():.3f}")
    w(f"  4-week moving avg  : {np.abs(actual - ma4).mean():.3f}")
    w(f"  training-set mean  : {np.abs(actual - tmean).mean():.3f}")
    w("(if the training mean is as good as the others, demand has no useful short-term pattern)")

# ---------------------------------------------------------------
# 4. Cost target checks
# ---------------------------------------------------------------
section("4. Cost target: is it just a formula?")
unit_cost = (df["Order Item Total"] - df["Order Profit Per Order"]) / df["Order Item Quantity"]
formula = (df["Order Item Product Price"] * (1 - df["Order Item Discount Rate"])
           * (1 - df["Order Item Profit Ratio"]))
w(f"unit_cost == price*(1-discount_rate)*(1-profit_ratio) (rtol 2%): "
  f"{np.isclose(unit_cost, formula, rtol=0.02, atol=0.1).mean():.4f}")
disc_ok = (df["Order Item Discount"] / df["Sales"] - df["Order Item Discount Rate"]).abs() <= 0.006
w(f"Discount / Sales == Discount Rate (within 0.006): {disc_ok.mean():.4f}")

df["unit_cost"] = unit_cost
df["cost_ratio"] = 1 - df["Order Item Profit Ratio"]

def r2_groups(y, g):
    mean_g = y.groupby(g).transform("mean")
    return 1 - ((y - mean_g) ** 2).sum() / ((y - y.mean()) ** 2).sum()

w("\nShare of variance explained by group averages alone (in-sample R2):")
w(f"  unit_cost  by Product Card Id : {r2_groups(df['unit_cost'], df['Product Card Id']):.3f}")
w(f"  cost_ratio by Product Card Id : {r2_groups(df['cost_ratio'], df['Product Card Id']):.3f}")
for c in ["Category Id", "Shipping Mode", "Market", "Customer Segment", "Type", "Order Region"]:
    w(f"  cost_ratio by {c:<17}: {r2_groups(df['cost_ratio'], df[c]):.3f}")
w("\ncost_ratio describe:")
w(df["cost_ratio"].describe().round(3).to_string())

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {OUT_TXT}")