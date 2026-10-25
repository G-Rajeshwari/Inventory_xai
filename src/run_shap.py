"""Stage 8b: SHAP (TreeSHAP) explanations on VALIDATION rows, with the noise control feature.

Demand: Random Forest (shap package, at most --max-rows rows because TreeSHAP on fully grown
forests is slow) and XGBoost (native pred_contribs). Cost: CatBoost (native ShapValues) and XGBoost.
Models are retrained exactly as in run_explain (seed 0, library defaults, extra noise column).
Every run checks additivity: SHAP values + base value must equal the model's raw prediction.

Run:  python -m src.run_shap --task demand
      python -m src.run_shap --task cost
"""
import argparse

import numpy as np
import pandas as pd

from src import config, cost_data, cost_models, data, demand_models, explain, shap_explain
from src.features import FEATURE_COLS

SEED = 0
EVAL = "validation"


def fit_demand(feat, name, seed=SEED, factory=None):
    tr = explain.add_noise(feat[feat["split"] == "train"], 10_000 + seed)
    va = explain.add_noise(feat[feat["split"] == EVAL], 20_000 + seed)
    cols = list(FEATURE_COLS) + [explain.NOISE]
    model = (factory or demand_models.make_model)(name, seed)
    model.fit(tr[cols].to_numpy(dtype=float), tr["units"].to_numpy(dtype=float))
    return model, va, cols


def fit_cost(df, name, seed=SEED, factory=None):
    """Returns model, validation rows, model input (DataFrame for CatBoost, array otherwise), column names."""
    tr = explain.add_noise(df[df["split"] == "train"], 10_000 + seed)
    va = explain.add_noise(df[df["split"] == EVAL], 20_000 + seed)
    y_tr = tr[cost_data.TARGET].to_numpy(dtype=float)
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
        return model, va, frame(va), cols
    coder = cost_models.OrdinalCoder().fit(tr)
    cols = list(cost_data.NUMERIC_FEATURES) + cat + [explain.NOISE]

    def matrix(d):
        return np.column_stack([coder.transform(d), d[explain.NOISE].to_numpy()])

    model = (factory or cost_models.make_model)(name, seed)
    model.fit(matrix(tr), y_tr)
    return model, va, matrix(va), cols


def compute_shap(name, model, X, cat_features=None):
    """Returns (values (n, p), base (n,), raw model prediction (n,)) for the rows of X."""
    if name == "random_forest":
        import shap
        ex = shap.TreeExplainer(model)
        values = np.asarray(ex.shap_values(X, check_additivity=False))
        base = np.full(len(X), float(np.ravel(ex.expected_value)[0]))
    elif name == "xgboost":
        import xgboost as xgb
        contrib = np.asarray(model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True))
        values, base = contrib[:, :-1], contrib[:, -1]
    elif name == "catboost":
        from catboost import Pool
        contrib = np.asarray(model.get_feature_importance(Pool(X, cat_features=cat_features), type="ShapValues"))
        values, base = contrib[:, :-1], contrib[:, -1]
    else:
        raise ValueError(f"Unknown model: {name}")
    raw = np.asarray(model.predict(X), dtype=float)
    return values, base, raw


def sample_rows(n, max_rows, seed=SEED):
    if max_rows is None or n <= max_rows:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))


def run_one(task, name, model, va, X, cols, y, max_rows=None, n_boot=1000, products=True):
    idx = sample_rows(len(va), max_rows)
    Xs = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    cat = list(cost_data.CATEGORICAL_FEATURES) if name == "catboost" else None
    values, base, raw = compute_shap(name, model, Xs, cat_features=cat)
    add = shap_explain.check_additivity(values, base, raw)
    vs = va.iloc[idx]
    glob = shap_explain.global_importance(values, cols, vs["week"].to_numpy(),
                                          products=vs["product_id"].to_numpy() if products else None,
                                          n_boot=n_boot)
    ys = np.asarray(y)[idx]
    picks = shap_explain.pick_rows(ys, raw)
    xdf = Xs if isinstance(Xs, pd.DataFrame) else pd.DataFrame(Xs, columns=cols)
    local = []
    for case, i in picks.items():
        tab, summ = shap_explain.local_explanation(values[i], xdf.iloc[i].tolist(), cols, base[i], raw[i])
        tab["case"] = case
        for k, v in summ.items():
            tab[k] = v
        tab["actual"] = float(ys[i])
        tab["product_id"] = vs.iloc[i]["product_id"]
        tab["week"] = vs.iloc[i]["week"]
        local.append(tab)
    meta = vs[["product_id", "week"]].copy()
    meta["actual"] = ys
    return add, glob, pd.concat(local, ignore_index=True), shap_explain.shap_frame(meta, values, base, raw, cols)


def report(task, name, add, glob, local, n_rows):
    print(f"\n{name}: SHAP on {n_rows} VALIDATION rows | additivity: max |sum(SHAP)+base-prediction| = "
          f"{add['max_abs_error']:.2e} (tolerance {add['tolerance']:.2e}) -> {'OK' if add['ok'] else 'FAILED'}")
    cols = ["feature", "mean_abs_shap", "ci_low", "ci_high", "above_noise"]
    if "within_product_abs" in glob:
        cols += ["within_product_abs", "within_ci_low", "within_ci_high", "within_above_noise"]
    print(glob[cols].round(5).to_string(index=False))
    for case, g in local.groupby("case", sort=False):
        r = g.iloc[0]
        print(f"\n  local explanation, {case}: product {r['product_id']}, week {str(r['week'])[:10]}, "
              f"actual {r['actual']:.3f}, model prediction {r['model_prediction']:.3f}, base value {r['base_value']:.3f}")
        print(g[["feature", "value", "shap"]].head(6).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["demand", "cost"], required=True)
    ap.add_argument("--max-rows", type=int, default=300,
                    help="demand Random Forest only: cap on validation rows explained")
    args = ap.parse_args()
    config.REPORTS.mkdir(exist_ok=True)

    if args.task == "demand":
        feat = pd.read_csv(config.PROCESSED / "features.csv", parse_dates=["week"])
        feat = feat.sort_values(["product_id", "week"]).reset_index(drop=True)
        for name in ("xgboost", "random_forest"):
            model, va, cols = fit_demand(feat, name)
            X = va[cols].to_numpy(dtype=float)
            cap = args.max_rows if name == "random_forest" else None
            print(f"\nComputing SHAP for demand {name} ...", flush=True)
            add, glob, local, frame = run_one("demand", name, model, va, X, cols, va["units"].to_numpy(), max_rows=cap)
            report("demand", name, add, glob, local, len(frame))
            frame.round(6).to_csv(config.REPORTS / f"shap_demand_{name}_validation.csv", index=False)
            glob.round(6).to_csv(config.REPORTS / f"shap_demand_{name}_global.csv", index=False)
            local.round(6).to_csv(config.REPORTS / f"shap_demand_{name}_local.csv", index=False)
    else:
        elig = data.load_eligible_products()
        df = cost_data.load_cost_lines(products=elig.index)
        for name in ("catboost", "xgboost"):
            model, va, X, cols = fit_cost(df, name)
            print(f"\nComputing SHAP for cost {name} ...", flush=True)
            add, glob, local, frame = run_one("cost", name, model, va, X, cols,
                                              va[cost_data.TARGET].to_numpy(), max_rows=None)
            report("cost", name, add, glob, local, len(frame))
            frame.round(6).to_csv(config.REPORTS / f"shap_cost_{name}_validation.csv", index=False)
            glob.round(6).to_csv(config.REPORTS / f"shap_cost_{name}_global.csv", index=False)
            local.round(6).to_csv(config.REPORTS / f"shap_cost_{name}_local.csv", index=False)
    print("\nSaved SHAP files to the reports folder.")


if __name__ == "__main__":
    main()