"""Stage 8a: permutation importance with a noise control on VALIDATION rows.

Demand models: Random Forest and XGBoost (TabPFN is excluded: one pass over the validation rows
takes about 3 minutes, and permutation importance needs hundreds of passes).
Cost models: CatBoost and XGBoost on order lines.
Models are retrained with the extra noise column (seed 0, library defaults).

Run:  python -m src.run_explain --task demand
      python -m src.run_explain --task demand --within-product
      python -m src.run_explain --task cost
"""
import argparse

import numpy as np
import pandas as pd

from src import config, cost_data, cost_models, data, demand_models, explain, metrics
from src.features import FEATURE_COLS

SEED = 0
EVAL = "validation"


def demand_importance(feat, name, seed=SEED, n_repeats=20, n_boot=1000, factory=None, within_product=False):
    tr = explain.add_noise(feat[feat["split"] == "train"], 10_000 + seed)
    va = explain.add_noise(feat[feat["split"] == EVAL], 20_000 + seed)
    cols = list(FEATURE_COLS) + [explain.NOISE]
    model = (factory or demand_models.make_model)(name, seed)
    model.fit(tr[cols].to_numpy(dtype=float), tr["units"].to_numpy(dtype=float))

    def predict(X):
        return np.clip(np.asarray(model.predict(X[cols].to_numpy(dtype=float)), dtype=float), 0, None)

    groups = va["product_id"].to_numpy() if within_product else None
    imp = explain.permutation_importance(predict, va[cols], va["units"].to_numpy(), va["week"].to_numpy(),
                                         explain.abs_loss, n_repeats=n_repeats, n_boot=n_boot, seed=seed,
                                         groups=groups)
    imp = explain.flag_informative(imp)
    imp["task"], imp["model"], imp["loss"] = "demand", name, "MAE"
    imp["scheme"] = "within_product" if within_product else "global"
    imp["model_val_MAE_with_noise_col"] = imp["base_loss"]
    return imp


def cost_importance(df, name, seed=SEED, n_repeats=5, n_boot=1000, factory=None):
    tr = explain.add_noise(df[df["split"] == "train"], 10_000 + seed)
    va = explain.add_noise(df[df["split"] == EVAL], 20_000 + seed)
    y_tr = tr[cost_data.TARGET].to_numpy(dtype=float)
    y_va = va[cost_data.TARGET].to_numpy(dtype=float)
    lo, hi = y_tr.min(), y_tr.max()
    cat = list(cost_data.CATEGORICAL_FEATURES)

    if name == "catboost":
        cols = list(cost_data.FEATURES) + [explain.NOISE]

        def frame(d):
            out = d[cols].copy()
            for c in cat:
                out[c] = out[c].astype(str)
            return out

        if factory is not None:
            model = factory(name, seed)
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(random_seed=seed, verbose=0, thread_count=-1,
                                      allow_writing_files=False, cat_features=cat)
        model.fit(frame(tr), y_tr)
        X_va = frame(va)

        def predict(X):
            return np.clip(np.asarray(model.predict(X), dtype=float), lo, hi)
    else:                                        # xgboost-style: ordinal-coded categoricals
        coder = cost_models.OrdinalCoder().fit(tr)
        cols = list(cost_data.NUMERIC_FEATURES) + cat + [explain.NOISE]

        def matrix(d):
            return pd.DataFrame(np.column_stack([coder.transform(d), d[explain.NOISE].to_numpy()]),
                                columns=cols)

        model = (factory or cost_models.make_model)(name, seed)
        model.fit(matrix(tr).to_numpy(), y_tr)
        X_va = matrix(va)

        def predict(X):
            return np.clip(np.asarray(model.predict(X[cols].to_numpy()), dtype=float), lo, hi)

    imp = explain.permutation_importance(predict, X_va, y_va, va["week"].to_numpy(), explain.sq_loss,
                                         n_repeats=n_repeats, n_boot=n_boot, seed=seed)
    imp = explain.flag_informative(imp)
    imp["task"], imp["model"], imp["loss"] = "cost", name, "MSE"
    imp["model_val_MAE_with_noise_col"] = np.nan
    return imp


def show(imp):
    cols = ["feature", "importance", "ci_low", "ci_high", "above_noise"]
    print(imp[cols].round(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["demand", "cost"], required=True)
    ap.add_argument("--within-product", action="store_true", help="demand only: shuffle within each product")
    args = ap.parse_args()
    config.REPORTS.mkdir(exist_ok=True)

    if args.task == "demand":
        feat = pd.read_csv(config.PROCESSED / "features.csv", parse_dates=["week"])
        feat = feat.sort_values(["product_id", "week"]).reset_index(drop=True)
        outs = []
        for name in ("random_forest", "xgboost"):
            imp = demand_importance(feat, name, within_product=args.within_product)
            scheme = "WITHIN each product" if args.within_product else "across all rows"
            print(f"\n{name}: permutation importance ({scheme}) on VALIDATION rows, loss = MAE "
                  f"(model MAE incl. noise column = {imp['base_loss'].iloc[0]:.3f})")
            show(imp)
            outs.append(imp)
        fname = ("explain_permutation_demand_within_product.csv" if args.within_product
                 else "explain_permutation_demand.csv")
        pd.concat(outs).round(6).to_csv(config.REPORTS / fname, index=False)
        print(f"\nSaved to {config.REPORTS / fname}")
    else:
        elig = data.load_eligible_products()
        df = cost_data.load_cost_lines(products=elig.index)
        outs = []
        for name in ("catboost", "xgboost"):
            imp = cost_importance(df, name)
            print(f"\n{name}: permutation importance on VALIDATION lines, loss = MSE "
                  f"(model MSE incl. noise column = {imp['base_loss'].iloc[0]:.5f})")
            show(imp)
            outs.append(imp)
        pd.concat(outs).round(8).to_csv(config.REPORTS / "explain_permutation_cost.csv", index=False)
        print(f"\nSaved to {config.REPORTS / 'explain_permutation_cost.csv'}")


if __name__ == "__main__":
    main()