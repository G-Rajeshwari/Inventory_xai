"""Stage 6b: feature ablation on the VALIDATION weeks (the test split is not selectable).

Run:  python -m src.run_ablation --models random_forest xgboost
      python -m src.run_ablation --models tabpfn
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import baselines, config, demand_models, metrics
from src import run_demand_models as rdm
from src.features import FEATURE_COLS

HISTORY = [c for c in FEATURE_COLS if c.startswith("lag_") or c.startswith("roll_mean_")]
VARIANTS = {
    "full": list(FEATURE_COLS),
    "history_only": HISTORY + ["product_code", "is_tier_a"],
    "id_season": ["product_code", "week_of_year"],
    "id_only": ["product_code"],
}
# XGBoost with library defaults ignores the seed, so one run is enough.
ABLATION_SEEDS = {"random_forest": [0, 1, 2, 3, 4], "xgboost": [0], "tabpfn": [0, 1, 2]}


def abl_dir():
    return rdm.PRED_DIR / "ablation"


def cache_path(variant, model, seed):
    if variant == "full":
        return rdm.pred_path(model, seed)
    return abl_dir() / f"{rdm.EVAL_SPLIT}_{variant}_{model}_seed{seed}.csv"


def read_cache(variant, model, seed, ev):
    path = cache_path(variant, model, seed)
    if not path.exists():
        return None
    c = pd.read_csv(path, parse_dates=["week"])
    if len(c) == len(ev) and (c["product_id"].to_numpy() == ev["product_id"].to_numpy()).all() \
            and (c["week"].to_numpy() == ev["week"].to_numpy()).all():
        return c["pred"].to_numpy()
    return None


def get_preds(variant, model, seed, feat, factory=None):
    if variant == "full":
        return rdm.run_model(model, seed, feat, factory=factory)
    ev = feat[feat["split"] == rdm.EVAL_SPLIT]
    cached = read_cache(variant, model, seed, ev)
    if cached is not None:
        print(f"  {variant:13s} {model} seed {seed}: loaded from cache")
        return cached
    train = feat[feat["split"] == "train"]
    t0 = time.time()
    pred = demand_models.fit_predict(model, seed, train, ev, factory=factory,
                                     feature_cols=VARIANTS[variant])
    abl_dir().mkdir(parents=True, exist_ok=True)
    out = ev[["product_id", "week"]].copy()
    out["pred"] = pred
    out.to_csv(cache_path(variant, model, seed), index=False)
    print(f"  {variant:13s} {model} seed {seed}: done in {time.time() - t0:.1f}s", flush=True)
    return pred


def vs_full_table(val, model_preds, n_boot=2000):
    """Paired week-bootstrap of MAE(variant) - MAE(full) for the same model."""
    y = val["units"].to_numpy(dtype=float)
    weeks = val["week"].to_numpy()
    groups = [("pooled", np.ones(len(val), bool)),
              ("A_high_volume", val["is_tier_a"].to_numpy() == 1),
              ("B_low_volume", val["is_tier_a"].to_numpy() == 0)]

    def mean_abs_err(preds):
        return np.mean([np.abs(y - p) for p in preds], axis=0)

    rows = []
    for (model, variant), preds in model_preds.items():
        if variant == "full" or (model, "full") not in model_preds:
            continue
        e_var, e_full = mean_abs_err(preds), mean_abs_err(model_preds[(model, "full")])
        for gname, mask in groups:
            b = metrics.paired_week_bootstrap(weeks[mask], e_var[mask], e_full[mask], n_boot=n_boot, seed=0)
            rows.append({"group": gname, "model": f"{model}[{variant}]",
                         "diff_vs_full": b["mean_diff"], "full_ci_low": b["ci_low"], "full_ci_high": b["ci_high"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ABLATION_SEEDS), choices=list(ABLATION_SEEDS))
    args = ap.parse_args()

    feat = baselines.add_baselines(rdm.load_features(), fit_splits=("train",))
    val = feat[feat["split"] == rdm.EVAL_SPLIT].reset_index(drop=True)
    print(f"Ablation on {rdm.EVAL_SPLIT}: {len(val)} rows")

    for model in args.models:
        print(f"{model}:")
        for variant in VARIANTS:
            for seed in ABLATION_SEEDS[model]:
                get_preds(variant, model, seed, feat)

    model_preds = {}
    for model, seeds in ABLATION_SEEDS.items():
        for variant in VARIANTS:
            arrs = [a for a in (read_cache(variant, model, s, val) for s in seeds) if a is not None]
            if len(arrs) == len(seeds):
                model_preds[(model, variant)] = arrs

    named = {f"{m}[{v}]": a for (m, v), a in model_preds.items()}
    res = rdm.evaluate(val, named)
    res = res[~res["model"].isin(["ma26", "ma4", "last_week"])]
    res = res.merge(vs_full_table(val, model_preds), on=["group", "model"], how="left")
    res.round(4).to_csv(config.REPORTS / "ablation_validation_results.csv", index=False)

    show = res[["group", "model", "n_seeds", "MAE", "MAE_vs_train_mean", "diff_vs_train_mean",
                "ci_low", "ci_high", "diff_vs_full", "full_ci_low", "full_ci_high"]].round(3)
    show = show.rename(columns={"diff_vs_train_mean": "d_trainmean", "ci_low": "tm_lo", "ci_high": "tm_hi",
                                "diff_vs_full": "d_full", "full_ci_low": "full_lo", "full_ci_high": "full_hi"})
    print("\nABLATION on VALIDATION weeks (d_* < 0 = lower error; tm = vs training mean, full = vs full features):")
    print(show.to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'ablation_validation_results.csv'}")


if __name__ == "__main__":
    main()