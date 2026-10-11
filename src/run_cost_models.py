"""Stage 7a: cost-ratio models vs baselines on the VALIDATION lines (test split not selectable).

Run:  python -m src.run_cost_models
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import config, cost_data, cost_models, data, metrics

PRED_DIR = config.REPORTS / "predictions" / "cost"
EVAL_SPLIT = "validation"
COST_SEEDS = {"catboost": [0, 1, 2], "random_forest": [0, 1, 2], "xgboost": [0]}
PRACTICAL_FLOOR = 0.005          # rule 3(b): at least 0.5% lower MSE
TARGET = cost_data.TARGET


def pred_path(model, seed):
    return PRED_DIR / f"{EVAL_SPLIT}_{model}_seed{seed}.csv"


def read_cache(model, seed, ev):
    path = pred_path(model, seed)
    if not path.exists():
        return None
    c = pd.read_csv(path, parse_dates=["order_dt"])
    if len(c) == len(ev) and (c["order_dt"].to_numpy() == ev["order_dt"].to_numpy()).all() \
            and (c["product_id"].to_numpy() == ev["product_id"].to_numpy()).all():
        return c["pred"].to_numpy()
    return None


def run_model(model, seed, df, factory=None):
    ev = df[df["split"] == EVAL_SPLIT]
    cached = read_cache(model, seed, ev)
    if cached is not None:
        print(f"  {model} seed {seed}: loaded from cache")
        return cached
    train = df[df["split"] == "train"]
    t0 = time.time()
    pred = cost_models.fit_predict(model, seed, train, ev, factory=factory)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = ev[["order_dt", "product_id"]].copy()
    out["pred"] = pred
    out.to_csv(pred_path(model, seed), index=False)
    print(f"  {model} seed {seed}: done in {time.time() - t0:.1f}s", flush=True)
    return pred


def baseline_preds(train, val):
    gm = train[TARGET].mean()
    pm = train.groupby("product_id")[TARGET].mean()
    return {"global_mean": [np.full(len(val), gm)],
            "product_mean": [val["product_id"].map(pm).fillna(gm).to_numpy(dtype=float)]}


def evaluate(train, val, model_preds, n_boot=2000):
    """One row per model: MSE-based comparison with the global training mean + dollar metrics."""
    val = val.reset_index(drop=True)
    entries = baseline_preds(train, val)
    entries.update(model_preds)
    y = val[TARGET].to_numpy(dtype=float)
    net_price = (val["price"] * (1 - val["discount_rate"])).to_numpy(dtype=float)
    weeks = val["week"].to_numpy()
    ref = entries["global_mean"][0]
    ref_se = (y - ref) ** 2
    ref_mse = ref_se.mean()

    rows = []
    for model, preds in entries.items():
        mses = [np.mean((y - p) ** 2) for p in preds]
        mean_se = np.mean([(y - p) ** 2 for p in preds], axis=0)
        b = metrics.paired_week_bootstrap(weeks, mean_se, ref_se, n_boot=n_boot, seed=0)
        rel_red = -b["mean_diff"] / ref_mse
        rows.append({
            "model": model, "n_seeds": len(preds),
            "MSE": np.mean(mses),
            "MSE_vs_global_mean": np.mean(mses) / ref_mse,
            "RMSE": np.mean([metrics.rmse(y, p) for p in preds]),
            "MAE": np.mean([metrics.mae(y, p) for p in preds]),
            "skill": 1 - np.mean(mses) / ref_mse,
            "diff_MSE": b["mean_diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "improves_on_reference": bool(b["ci_high"] < -1e-9 and rel_red >= PRACTICAL_FLOOR),
            "dollar_R2": np.mean([metrics.r2(net_price * y, net_price * p) for p in preds]),
            "dollar_MAE": np.mean([metrics.mae(net_price * y, net_price * p) for p in preds]),
            "n_rows": len(val),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(COST_SEEDS), choices=list(COST_SEEDS))
    args = ap.parse_args()

    elig = data.load_eligible_products()
    df = cost_data.load_cost_lines(products=elig.index)
    train = df[df["split"] == "train"]
    val = df[df["split"] == EVAL_SPLIT].reset_index(drop=True)
    print(f"Cost lines: train {len(train)} | validation {len(val)}")

    for model in args.models:
        print(f"{model}:")
        for seed in COST_SEEDS[model]:
            run_model(model, seed, df)

    model_preds = {}
    for model, seeds in COST_SEEDS.items():
        arrs = [a for a in (read_cache(model, s, val) for s in seeds) if a is not None]
        if len(arrs) == len(seeds):
            model_preds[model] = arrs

    res = evaluate(train, val, model_preds)
    res.round(6).to_csv(config.REPORTS / "cost_validation_results.csv", index=False)
    show = res[["model", "n_seeds", "MSE", "MSE_vs_global_mean", "RMSE", "MAE", "skill",
                "diff_MSE", "ci_low", "ci_high", "improves_on_reference", "dollar_R2", "dollar_MAE"]].round(5)
    print("\nCOST RATIO on VALIDATION lines (reference = global training mean):")
    print(show.to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'cost_validation_results.csv'}")


if __name__ == "__main__":
    main()