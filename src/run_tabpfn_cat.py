"""Stage 6c (EXPLORATORY, added after seeing validation results):
TabPFN with product_code declared categorical. VALIDATION weeks only.

Run:  python -m src.run_tabpfn_cat
"""
import time

import numpy as np
import pandas as pd

from src import baselines, config, demand_models
from src import run_demand_models as rdm
from src.features import FEATURE_COLS

VARIANTS = {"full": list(FEATURE_COLS), "id_only": ["product_code"]}
SEEDS = [0, 1, 2]


def out_dir():
    return rdm.PRED_DIR / "tabpfn_cat"


def read_aligned(path, ev):
    """Cached predictions if the file exists and lines up with the evaluation rows."""
    if not path.exists():
        return None
    c = pd.read_csv(path, parse_dates=["week"])
    if len(c) == len(ev) and (c["product_id"].to_numpy() == ev["product_id"].to_numpy()).all() \
            and (c["week"].to_numpy() == ev["week"].to_numpy()).all():
        return c["pred"].to_numpy()
    return None


def get_preds(variant, seed, feat, factory=None):
    ev = feat[feat["split"] == rdm.EVAL_SPLIT]
    path = out_dir() / f"{rdm.EVAL_SPLIT}_{variant}_tabpfn_cat_seed{seed}.csv"
    cached = read_aligned(path, ev)
    if cached is not None:
        print(f"  {variant:8s} tabpfn_cat seed {seed}: loaded from cache")
        return cached
    train = feat[feat["split"] == "train"]
    t0 = time.time()
    pred = demand_models.fit_predict("tabpfn_cat", seed, train, ev, factory=factory,
                                     feature_cols=VARIANTS[variant])
    out_dir().mkdir(parents=True, exist_ok=True)
    out = ev[["product_id", "week"]].copy()
    out["pred"] = pred
    out.to_csv(path, index=False)
    print(f"  {variant:8s} tabpfn_cat seed {seed}: done in {time.time() - t0:.1f}s", flush=True)
    return pred


def run(feat, factory=None):
    val = feat[feat["split"] == rdm.EVAL_SPLIT].reset_index(drop=True)
    named = {}
    for variant in VARIANTS:
        named[f"tabpfn_cat[{variant}]"] = [get_preds(variant, s, feat, factory) for s in SEEDS]

    # earlier TabPFN runs (numeric product code) for comparison, if their caches exist
    for variant in VARIANTS:
        arrs = []
        for s in SEEDS:
            p = rdm.pred_path("tabpfn", s) if variant == "full" else \
                rdm.PRED_DIR / "ablation" / f"{rdm.EVAL_SPLIT}_{variant}_tabpfn_seed{s}.csv"
            a = read_aligned(p, val)
            if a is not None:
                arrs.append(a)
        if len(arrs) == len(SEEDS):
            named[f"tabpfn_numeric[{variant}]"] = arrs

    res = rdm.evaluate(val, named)
    return res[~res["model"].isin(["ma26", "ma4", "last_week"])].reset_index(drop=True)


def main():
    feat = baselines.add_baselines(rdm.load_features(), fit_splits=("train",))
    print(f"TabPFN categorical-product experiment on {rdm.EVAL_SPLIT}")
    res = run(feat)
    res.round(4).to_csv(config.REPORTS / "tabpfn_cat_validation_results.csv", index=False)
    show = res[["group", "model", "n_seeds", "MAE", "MAE_seed_std", "MAE_vs_train_mean",
                "diff_vs_train_mean", "ci_low", "ci_high", "beats_train_mean"]].round(3)
    print("\nRESULTS on VALIDATION weeks (diff < 0 = lower error than training mean):")
    print(show.to_string(index=False))
    print(f"\nSaved to {config.REPORTS / 'tabpfn_cat_validation_results.csv'}")


if __name__ == "__main__":
    main()