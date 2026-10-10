"""Stage 6: Random Forest, XGBoost, TabPFN vs baselines on the VALIDATION weeks.

The test split cannot be evaluated by this script. Predictions are cached per
model and seed, so an interrupted run resumes where it stopped.

Run:  python -m src.run_demand_models --models random_forest xgboost --seeds 0 1 2 3 4
      python -m src.run_demand_models --models tabpfn --seeds 0 1 2
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import baselines, config, demand_models, metrics

PRED_DIR = config.REPORTS / "predictions"
EVAL_SPLIT = "validation"          # the test split is deliberately not selectable here
REFERENCE = "train_mean"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED / "features.csv", parse_dates=["week"])
    return df.sort_values(["product_id", "week"]).reset_index(drop=True)


def pred_path(model: str, seed: int):
    return PRED_DIR / f"{EVAL_SPLIT}_{model}_seed{seed}.csv"


def run_model(model: str, seed: int, feat: pd.DataFrame, factory=None) -> np.ndarray:
    """Return predictions for the evaluation rows, using the cache when available."""
    ev = feat[feat["split"] == EVAL_SPLIT]
    path = pred_path(model, seed)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["week"])
        if len(cached) == len(ev) and (cached["product_id"].to_numpy() == ev["product_id"].to_numpy()).all() \
                and (cached["week"].to_numpy() == ev["week"].to_numpy()).all():
            print(f"  {model} seed {seed}: loaded from cache")
            return cached["pred"].to_numpy()
    train = feat[feat["split"] == "train"]
    t0 = time.time()
    pred = demand_models.fit_predict(model, seed, train, ev, factory=factory)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = ev[["product_id", "week"]].copy()
    out["pred"] = pred
    out.to_csv(path, index=False)
    print(f"  {model} seed {seed}: done in {time.time() - t0:.1f}s", flush=True)
    return pred


def evaluate(val: pd.DataFrame, model_preds: dict, n_boot: int = 2000) -> pd.DataFrame:
    """val: validation rows with baseline prediction columns.
    model_preds: {model name: [pred array per seed]} aligned with val rows.
    Returns one row per (group, model) with mean/std over seeds and a paired
    week-bootstrap interval of MAE difference versus the training mean.
    """
    val = val.reset_index(drop=True)
    entries = {}   # model -> list of prediction arrays (one per seed)
    for col in baselines.BASELINE_COLS:
        entries[col.replace("pred_", "")] = [val[col].to_numpy(dtype=float)]
    entries.update(model_preds)

    groups = [("pooled", np.ones(len(val), bool)),
              ("A_high_volume", val["is_tier_a"].to_numpy() == 1),
              ("B_low_volume", val["is_tier_a"].to_numpy() == 0)]
    y = val["units"].to_numpy(dtype=float)
    weeks = val["week"].to_numpy()
    ref_err = np.abs(y - entries[REFERENCE][0])

    rows = []
    for gname, mask in groups:
        ref_mae = ref_err[mask].mean()
        for model, preds in entries.items():
            per_seed = [metrics.summarize(y[mask], p[mask]) for p in preds]
            mean_abs_err = np.mean([np.abs(y - p) for p in preds], axis=0)  # expected single-seed error
            boot = metrics.paired_week_bootstrap(weeks[mask], mean_abs_err[mask], ref_err[mask], n_boot=n_boot, seed=0)
            mae_seeds = [m["MAE"] for m in per_seed]
            rows.append({
                "group": gname, "model": model, "n_seeds": len(preds),
                "MAE": np.mean(mae_seeds),
                "MAE_seed_std": np.std(mae_seeds, ddof=1) if len(mae_seeds) > 1 else np.nan,
                "RMSE": np.mean([m["RMSE"] for m in per_seed]),
                "WAPE": np.mean([m["WAPE"] for m in per_seed]),
                "MAE_vs_train_mean": np.mean(mae_seeds) / ref_mae,
                "diff_vs_train_mean": boot["mean_diff"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "beats_train_mean": bool(boot["ci_high"] < -1e-6),   # tolerance: ignore float rounding
                "n_rows": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["random_forest", "xgboost"], choices=demand_models.MODEL_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()

    feat = load_features()
    feat = baselines.add_baselines(feat, fit_splits=("train",))
    val = feat[feat["split"] == EVAL_SPLIT].reset_index(drop=True)
    print(f"Evaluating on {EVAL_SPLIT}: {len(val)} rows | train rows: {(feat['split'] == 'train').sum()}")

    for model in args.models:
        print(f"{model}:")
        for seed in args.seeds:
            run_model(model, seed, feat)

    # evaluate everything that has cached predictions (all models, all available seeds)
    model_preds = {}
    for model in demand_models.MODEL_NAMES:
        preds = []
        for seed in config.SEEDS:
            p = pred_path(model, seed)
            if p.exists():
                c = pd.read_csv(p, parse_dates=["week"])
                if len(c) == len(val):
                    preds.append(c["pred"].to_numpy())
        if preds:
            model_preds[model] = preds

    res = evaluate(val, model_preds)
    res.round(4).to_csv(config.REPORTS / "demand_validation_results.csv", index=False)
    show = res[["group", "model", "n_seeds", "MAE", "MAE_seed_std", "MAE_vs_train_mean",
                "diff_vs_train_mean", "ci_low", "ci_high", "beats_train_mean"]].round(3)
    print("\nVALIDATION results (diff < 0 = lower error than training mean):")
    print(show.to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'demand_validation_results.csv'}")


if __name__ == "__main__":
    main()