import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, cost_data, probe_cost


def make_raw_csv(path, n_per_week=20, n_weeks=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2015-01-05")
    for w in range(n_weeks):
        for _ in range(n_per_week):
            d = start + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
            rows.append({
                "order date (DateOrders)": d.strftime("%m/%d/%Y %H:%M"),
                "Product Card Id": int(rng.choice([10, 11, 99])),
                "Order Item Quantity": int(rng.integers(1, 6)),
                "Order Item Profit Ratio": round(float(rng.uniform(-1, 0.5)), 2),
                "Order Item Product Price": 50.0,
                "Order Item Discount Rate": float(rng.choice([0.0, 0.1, 0.2])),
                "Days for shipment (scheduled)": int(rng.choice([0, 2, 4])),
                "Shipping Mode": "Standard Class", "Market": "LATAM", "Customer Segment": "Consumer",
                "Type": "DEBIT", "Order Region": "Central America", "Category Id": 5,
                "Department Name": "Fan Shop",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cost_features_exclude_leaky_and_target_columns():
    renamed_forbidden = {cost_data.RAW_TO_NAME.get(c, c) for c in cost_data.FORBIDDEN_RAW}
    assert not (set(cost_data.FEATURES) & renamed_forbidden)
    assert "profit_ratio" not in cost_data.FEATURES and cost_data.TARGET not in cost_data.FEATURES
    # the only restricted column that is loaded is the one needed to build the target
    assert set(cost_data.USECOLS) & set(config.LEAKY_FOR_COST) == {"Order Item Profit Ratio"}
    assert not (set(cost_data.USECOLS) & set(config.LEAKY_FOR_DEMAND))


def test_cost_ratio_is_one_minus_profit_ratio_and_split_is_chronological():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "raw.csv"
        make_raw_csv(p)
        raw = pd.read_csv(p)
        sp = {"train": ["2015-01-05", "2015-07-27", 0], "validation": ["2015-08-03", "2015-09-28", 0],
              "test": ["2015-10-05", "2015-11-02", 0]}
        from src import split as split_mod
        orig = split_mod.data.load_split
        split_mod.data.load_split = lambda: sp
        try:
            df = cost_data.load_cost_lines(path=p, products=[10, 11])
        finally:
            split_mod.data.load_split = orig
    assert set(df["product_id"]) == {10, 11}                       # product 99 filtered out
    kept = raw[raw["Product Card Id"].isin([10, 11])]
    assert len(df) == len(kept)
    expected = np.sort((1 - kept["Order Item Profit Ratio"]).to_numpy())
    assert np.allclose(np.sort(df[cost_data.TARGET].to_numpy()), expected)
    assert "profit_ratio" not in df.columns                         # not left in as an input
    tr_max = df[df["split"] == "train"]["week"].max()
    va = df[df["split"] == "validation"]["week"]
    te = df[df["split"] == "test"]["week"]
    assert tr_max < va.min() and va.max() < te.min()


def test_oos_skill_is_near_zero_for_noise_and_high_for_a_real_signal():
    rng = np.random.default_rng(0)
    n = 4000
    g = rng.integers(0, 5, n)
    noise = pd.DataFrame({"g": g, cost_data.TARGET: rng.normal(size=n)})
    signal = pd.DataFrame({"g": g, cost_data.TARGET: g * 1.0 + rng.normal(scale=0.1, size=n)})
    for df, lo, hi in ((noise, -0.01, 0.01), (signal, 0.9, 1.0)):
        tr, va = df.iloc[:3000], df.iloc[3000:]
        s = probe_cost.oos_skill(tr, va, "g")
        assert lo <= s <= hi, s
    # unseen group falls back to the global train mean
    tr = pd.DataFrame({"g": [1, 1, 2], cost_data.TARGET: [1.0, 3.0, 5.0]})
    va = pd.DataFrame({"g": [9], cost_data.TARGET: [4.0]})
    assert np.isclose(probe_cost.oos_skill(tr, va, "g"), 0.0)