"""Stage 5: build tables and evaluate the four baselines on the VALIDATION weeks only.

The test weeks are never evaluated here.
"""
import pandas as pd

from src import baselines, config, data, features, metrics, split


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    elig = data.load_eligible_products()
    lines = data.load_order_lines()
    weekly = data.build_weekly_table(lines, elig.index)
    weekly.to_csv(config.PROCESSED / "weekly_units.csv", index=False)
    print(f"Weekly table: {len(weekly)} rows | products: {weekly['product_id'].nunique()} "
          f"| weeks: {weekly['week'].nunique()}")

    feat = features.build_features(weekly, elig["tier"])
    feat = split.assign_split(feat)
    feat.to_csv(config.PROCESSED / "features.csv", index=False)
    print("\nFeature rows per split (after dropping the first "
          f"{features.WARMUP_WEEKS} warm-up weeks):")
    print(feat["split"].value_counts().reindex(["train", "validation", "test"]).to_string())

    feat = baselines.add_baselines(feat, fit_splits=("train",))
    val = feat[feat["split"] == "validation"].copy()   # test rows are NOT touched
    val["tier"] = val["is_tier_a"].map({1: "A_high_volume", 0: "B_low_volume"})

    rows = []
    for tier_name, sub in [("pooled", val)] + list(val.groupby("tier")):
        for col in baselines.BASELINE_COLS:
            m = metrics.summarize(sub["units"], sub[col])
            m.update({"tier": tier_name, "model": col.replace("pred_", "")})
            rows.append(m)
    res = pd.DataFrame(rows)
    ref = res[res["model"] == "train_mean"].set_index("tier")["MAE"]
    res["MAE_vs_train_mean"] = res.apply(lambda r: r["MAE"] / ref[r["tier"]], axis=1)
    res = res[["tier", "model", "MAE", "RMSE", "R2", "WAPE", "MAE_vs_train_mean", "n"]].round(4)
    res.to_csv(config.REPORTS / "baseline_validation.csv", index=False)
    print("\nBaselines on VALIDATION weeks (1 week ahead):")
    print(res.to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'baseline_validation.csv'}")


if __name__ == "__main__":
    main()