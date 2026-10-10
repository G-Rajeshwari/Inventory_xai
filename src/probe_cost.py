"""Stage 7a probe: is the cost ratio predictable from anything known at order time?

Fits single-column group means on TRAIN lines and scores them on VALIDATION lines.
skill = 1 - SSE(model) / SSE(global train mean); 0 = no better than a constant.
The test split is never loaded into the scores.
"""
import numpy as np
import pandas as pd

from src import config, cost_data, data


def oos_skill(train: pd.DataFrame, val: pd.DataFrame, col: str, target=cost_data.TARGET) -> float:
    """Out-of-sample skill of a group-mean predictor on one column."""
    gm = train[target].mean()
    means = train.groupby(col)[target].mean()
    pred = val[col].map(means).fillna(gm).to_numpy(dtype=float)
    y = val[target].to_numpy(dtype=float)
    sse_const = np.sum((y - gm) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / sse_const) if sse_const > 0 else float("nan")


def main():
    elig = data.load_eligible_products()
    df = cost_data.load_cost_lines(products=elig.index)
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    print(f"Order lines: train {len(tr)} | validation {len(va)} | (test lines exist but are not scored: "
          f"{(df['split'] == 'test').sum()})")
    print("\ncost_ratio on TRAIN lines:")
    print(tr[cost_data.TARGET].describe().round(3).to_string())
    print(f"Share of train lines with cost_ratio > 1 (negative profit): {(tr[cost_data.TARGET] > 1).mean():.3f}")

    rows = [(c, oos_skill(tr, va, c), tr[c].nunique()) for c in cost_data.FEATURES]
    res = pd.DataFrame(rows, columns=["column", "oos_skill_vs_constant", "n_groups"]).sort_values(
        "oos_skill_vs_constant", ascending=False)
    print("\nOut-of-sample skill of a single-column group-mean predictor (validation):")
    print(res.round(5).to_string(index=False))

    print("\nPearson correlation with cost_ratio on TRAIN lines:")
    for c in cost_data.NUMERIC_FEATURES:
        print(f"  {c:14s} {tr[c].corr(tr[cost_data.TARGET]):+.4f}")

    print("\nMean cost_ratio by discount_rate (TRAIN):")
    print(tr.groupby("discount_rate")[cost_data.TARGET].agg(["mean", "std", "size"]).round(3).to_string())

    res.round(6).to_csv(config.REPORTS / "cost_probe.csv", index=False)
    print(f"\nSaved to {config.REPORTS / 'cost_probe.csv'}")


if __name__ == "__main__":
    main()