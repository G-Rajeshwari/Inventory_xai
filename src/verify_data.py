from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
OUT = Path("reports/stage1_verify.txt")
OUT.parent.mkdir(exist_ok=True)

df = pd.read_csv(RAW, encoding="utf-8")
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["month"] = df["order_dt"].dt.to_period("M")

lines = []
def w(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    w("\n" + "=" * 70)
    w(title)
    w("=" * 70)

# A. October 2017 drop: fewer products, or less demand per product?
section("A. Volume drop: catalog shrink or real demand drop?")
m = df.groupby("month").agg(order_lines=("Order Item Id", "size"),
                            orders=("Order Id", "nunique"),
                            products=("Product Card Id", "nunique"),
                            units=("Order Item Quantity", "sum"))
w(m.to_string())
late_products = set(df.loc[df["order_dt"] >= "2017-10-01", "Product Card Id"])
w(f"\nProducts selling from 2017-10 onward: {len(late_products)}")
sub = df[df["Product Card Id"].isin(late_products)]
w("Units per month for ONLY those products:")
w(sub.groupby("month")["Order Item Quantity"].sum().loc["2017-01":].to_string())

# B. Does price change over time?
section("B. Is each product's price constant?")
n_prices = df.groupby("Product Card Id")["Order Item Product Price"].nunique()
w("Distinct prices per product (index = number of prices, value = how many products):")
w(n_prices.value_counts().sort_index().to_string())
w("\nDiscount rate values (index = rate, value = row count):")
w(df["Order Item Discount Rate"].round(2).value_counts().sort_index().to_string())

# C. Category id vs name
section("C. Category id vs name")
ids_per_name = df.groupby("Category Name")["Category Id"].nunique()
names_per_id = df.groupby("Category Id")["Category Name"].nunique()
w("Names that have more than one id:")
w(ids_per_name[ids_per_name > 1].to_string())
w("Ids that have more than one name:")
w(names_per_id[names_per_id > 1].to_string())

# D. Rows where Total != Sales - Discount
section("D. Rows where Order Item Total != Sales - Discount")
diff = (df["Order Item Total"] - (df["Sales"] - df["Order Item Discount"])).abs()
bad = diff > 0.01
w(f"Rows: {bad.sum()} ({bad.mean() * 100:.2f}%) | largest gap: {diff.max():.2f}")
w(df.loc[bad, ["Sales", "Order Item Discount", "Order Item Total"]].head(5).to_string())

# E. Delivery time pattern
section("E. Real shipping days by shipping mode (row counts)")
w(pd.crosstab(df["Shipping Mode"], df["Days for shipping (real)"]).to_string())

# F. Can an implied cost be trusted?
section("F. Can an implied cost be trusted?")
lines_per_order = df.groupby("Order Id").size()
profit_vals = df.groupby("Order Id")["Order Profit Per Order"].nunique()
multi = lines_per_order[lines_per_order > 1].index
same = profit_vals.loc[multi] == 1
w(f"Multi-line orders: {len(multi)}")
w(f"...with ONE identical profit value on every line: {same.mean():.4f}")
w("(close to 0 means profit is per line; close to 1 means profit is per order)")

w("")
for name in ["Sales", "Order Item Total"]:
    ratio = df["Order Profit Per Order"] / df[name]
    ok = (ratio - df["Order Item Profit Ratio"]).abs() <= 0.006
    w(f"profit/{name} matches Order Item Profit Ratio (within 0.006): {ok.mean():.4f}")

df["implied_cost"] = df["Order Item Total"] - df["Order Profit Per Order"]
df["implied_cost_per_unit"] = df["implied_cost"] / df["Order Item Quantity"]
g = df.groupby("Product Card Id")["implied_cost_per_unit"].agg(["mean", "std"])
g["cv"] = g["std"] / g["mean"]
w("\nVariation of implied cost per unit within each product (cv = std/mean, low = stable):")
w(g["cv"].describe().round(3).to_string())

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {OUT}")