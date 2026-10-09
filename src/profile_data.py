from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
OUT = Path("reports/stage1_profile.txt")
OUT.parent.mkdir(exist_ok=True)

df = pd.read_csv(RAW, encoding="utf-8")
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

lines = []
def w(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    w("\n" + "=" * 70)
    w(title)
    w("=" * 70)

# 1. Numeric columns
section("1. Key numeric columns")
num_cols = ["Order Item Quantity", "Sales", "Order Item Product Price",
            "Order Item Discount", "Order Item Total", "Order Profit Per Order",
            "Order Item Profit Ratio", "Days for shipping (real)",
            "Days for shipment (scheduled)"]
w(df[num_cols].describe().T.round(2).to_string())
w("\nNegative values (negative profit can be legitimate, a loss):")
for c in num_cols:
    w(f"  {c}: {(df[c] < 0).sum()}")

# 2. Relationships between columns
section("2. Relationships between columns (share of rows that match)")
def share_equal(a, b, tol=0.01):
    return float((np.abs(a - b) <= tol).mean())

w(f"Sales == price * quantity            : {share_equal(df['Sales'], df['Order Item Product Price'] * df['Order Item Quantity']):.4f}")
w(f"Order Item Total == Sales - discount : {share_equal(df['Order Item Total'], df['Sales'] - df['Order Item Discount']):.4f}")
w(f"Benefit per order == Order Profit    : {share_equal(df['Benefit per order'], df['Order Profit Per Order']):.4f}")
w(f"Sales per customer == Order Item Total: {share_equal(df['Sales per customer'], df['Order Item Total']):.4f}")
w(f"Profit == Total * Profit Ratio       : {share_equal(df['Order Profit Per Order'], df['Order Item Total'] * df['Order Item Profit Ratio']):.4f}")
w(f"Category Id == Product Category Id   : {share_equal(df['Category Id'], df['Product Category Id']):.4f}")

implied = df["Order Item Total"] - df["Order Profit Per Order"]
w("\nImplied cost = Order Item Total - Order Profit Per Order (hypothesis only):")
w(implied.describe().round(2).to_string())
w(f"Implied cost <= 0 rows: {(implied <= 0).sum()}")

# 3. Categorical columns
section("3. Categorical value counts")
for c in ["Order Status", "Delivery Status", "Shipping Mode", "Type", "Market", "Customer Segment"]:
    w(f"\n{c}:")
    w(df[c].value_counts().to_string())

# 4. Time-series feasibility
section("4. Time-series feasibility (weekly units per product)")
w("Monthly total units:")
w(df.groupby(df["order_dt"].dt.to_period("M"))["Order Item Quantity"].sum().to_string())

df["week"] = df["order_dt"].dt.to_period("W").dt.start_time
weekly = (df.groupby(["Product Card Id", "week"])["Order Item Quantity"]
            .sum().unstack(fill_value=0))
all_weeks = pd.date_range(weekly.columns.min(), weekly.columns.max(), freq="7D")
weekly = weekly.reindex(columns=all_weeks, fill_value=0)

w(f"\nProducts: {weekly.shape[0]}  |  Weeks: {weekly.shape[1]}")
w(f"Share of product-weeks with zero sales: {(weekly == 0).to_numpy().mean():.3f}")
active = (weekly > 0).sum(axis=1)
w("\nActive weeks per product (describe):")
w(active.describe().round(1).to_string())
last_active = weekly.gt(0).apply(lambda r: r[r].index.max(), axis=1)
cutoff = weekly.columns.max() - pd.Timedelta(weeks=8)
w(f"\nProducts with no sales in the last 8 weeks: {(last_active < cutoff).sum()}")
w("\nTotal weekly units (describe):")
w(weekly.sum(axis=0).describe().round(1).to_string())

# 5. Lead-time proxy
section("5. Delivery time by shipping mode")
g = df.groupby("Shipping Mode")[["Days for shipping (real)", "Days for shipment (scheduled)"]].mean().round(2)
g["share_late_vs_schedule"] = df.assign(
    late=df["Days for shipping (real)"] > df["Days for shipment (scheduled)"]
).groupby("Shipping Mode")["late"].mean().round(3)
w(g.to_string())

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {OUT}")