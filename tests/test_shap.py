import sys
import types
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src import cost_data, explain, features, run_shap, shap_explain, split

NOISE = shap_explain.NOISE
TIERS = pd.Series({1: "A_high_volume", 2: "B_low_volume", 3: "B_low_volume"})
SPLIT = {"train": ["2015-01-05", "2015-07-27", 0], "validation": ["2015-08-03", "2015-09-28", 0],
         "test": ["2015-10-05", "2015-11-02", 0]}


def make_feat(n_weeks=45, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-05", periods=n_weeks, freq="7D")
    rows = [(p, w, int(rng.poisson(lam))) for p, lam in ((1, 40), (2, 5), (3, 6)) for w in weeks]
    table = pd.DataFrame(rows, columns=["product_id", "week", "units"])
    return split.assign_split(features.build_features(table, TIERS), SPLIT)


def make_cost_lines(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-05", periods=30, freq="7D")
    df = pd.DataFrame({
        "product_id": rng.choice([10, 11, 12], n), "qty": rng.integers(1, 6, n).astype(float),
        "price": rng.choice([20.0, 50.0, 100.0], n), "discount_rate": rng.choice([0.0, 0.1, 0.2], n),
        "sched_days": rng.choice([0, 2, 4], n).astype(float), "shipping_mode": rng.choice(["A", "B"], n),
        "market": "LATAM", "segment": "Consumer", "pay_type": "DEBIT", "region": "Central America",
        "category_id": 5, "department": "Fan Shop", "week": weeks[rng.integers(0, 30, n)]})
    df[cost_data.TARGET] = rng.normal(0.9, 0.4, n).clip(0.5, 3.75)
    df["split"] = np.where(df["week"] <= "2015-05-25", "train",
                           np.where(df["week"] <= "2015-07-06", "validation", "test"))
    return df


def test_check_additivity_passes_for_exact_values_and_flags_errors():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 4))
    base = np.full(50, 3.0)
    pred = values.sum(axis=1) + 3.0
    assert shap_explain.check_additivity(values, base, pred)["ok"] is True
    bad = values.copy()
    bad[7, 2] += 5.0
    res = shap_explain.check_additivity(bad, base, pred)
    assert res["ok"] is False and res["max_abs_error"] > 4.9


def test_global_importance_separates_level_effects_from_within_product_variation():
    rng = np.random.default_rng(1)
    n_prod, n_wk = 6, 100
    products = np.repeat(np.arange(n_prod), n_wk)
    weeks = np.tile(np.arange(n_wk), n_prod)
    level = products * 5.0                                       # constant inside each product
    signal = rng.normal(scale=2.0, size=len(products))           # varies inside each product
    noise = rng.normal(scale=0.05, size=len(products))
    values = np.column_stack([level, signal, noise])
    g = shap_explain.global_importance(values, ["level", "signal", NOISE], weeks, products=products,
                                       n_boot=200).set_index("feature")
    assert bool(g.loc["level", "above_noise"]) and bool(g.loc["signal", "above_noise"])
    assert not bool(g.loc[NOISE, "above_noise"])
    assert g.loc["level", "within_product_abs"] < 1e-9 and not bool(g.loc["level", "within_above_noise"])
    assert bool(g.loc["signal", "within_above_noise"]) and g.loc["signal", "within_product_abs"] > 1.0
    assert g.index[0] == "level"                                  # sorted by mean |SHAP|


def test_global_importance_requires_the_noise_column():
    try:
        shap_explain.global_importance(np.ones((10, 2)), ["a", "b"], np.arange(10), n_boot=50)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_local_explanation_is_sorted_and_adds_up():
    v = np.array([0.5, -3.0, 1.0])
    tab, summ = shap_explain.local_explanation(v, [10, 20, 30], ["a", "b", "c"], base=4.0, raw_pred=2.5)
    assert tab["feature"].tolist() == ["b", "c", "a"]
    assert np.isclose(summ["base_value"] + summ["sum_shap"], summ["model_prediction"])


def test_pick_rows_returns_largest_and_median_error():
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.0, 2.0, 6.0, 1.5, 0.0])      # errors 0, 1, 5, 0.5, 1
    picks = shap_explain.pick_rows(y, pred)
    assert picks["largest_error"] == 2
    assert picks["median_error"] in (1, 4)           # the two errors equal to the median value


def test_shap_frame_has_one_column_per_feature_and_matches_rows():
    meta = pd.DataFrame({"product_id": [1, 2, 3], "week": pd.to_datetime(["2015-01-05"] * 3), "actual": [1, 2, 3]})
    f = shap_explain.shap_frame(meta, np.ones((3, 2)), np.array([2.0, 2.0, 2.0]), np.array([4.0, 4.0, 4.0]), ["a", "b"])
    assert list(f.columns) == ["product_id", "week", "actual", "shap_a", "shap_b", "base_value", "model_prediction"]
    assert len(f) == 3


def test_compute_shap_plumbing_with_stand_in_libraries():
    X = np.arange(12, dtype=float).reshape(4, 3)

    class M:
        def predict(self, X):
            return np.full(len(X), 7.0)

        def get_booster(self):
            return Booster()

    class Booster:
        def predict(self, dm, pred_contribs=False):
            assert pred_contribs is True
            return np.column_stack([np.ones((len(dm), 3)), np.full(len(dm), 2.0)])

    fake_shap = types.ModuleType("shap")

    class TE:
        def __init__(self, model):
            self.expected_value = np.array([3.0])

        def shap_values(self, X, check_additivity=True):
            return np.zeros((len(X), 3))

    fake_shap.TreeExplainer = TE
    with mock.patch.dict(sys.modules, {"shap": fake_shap}):
        v, b, r = run_shap.compute_shap("random_forest", M(), X)
    assert v.shape == (4, 3) and np.allclose(b, 3.0) and np.allclose(r, 7.0)

    fake_xgb = types.ModuleType("xgboost")
    fake_xgb.DMatrix = lambda X: X

    with mock.patch.dict(sys.modules, {"xgboost": fake_xgb}):
        v, b, r = run_shap.compute_shap("xgboost", M(), X)
    assert v.shape == (4, 3) and np.allclose(v, 1.0) and np.allclose(b, 2.0)

    seen = {}
    fake_cb = types.ModuleType("catboost")

    class Pool:
        def __init__(self, data, cat_features=None):
            seen["cat"] = cat_features

    fake_cb.Pool = Pool

    class C(M):
        def get_feature_importance(self, pool, type=None):
            assert type == "ShapValues"
            return np.column_stack([np.full((4, 3), 0.5), np.full(4, 1.5)])

    with mock.patch.dict(sys.modules, {"catboost": fake_cb}):
        v, b, r = run_shap.compute_shap("catboost", C(), X, cat_features=["market"])
    assert np.allclose(v, 0.5) and np.allclose(b, 1.5) and seen["cat"] == ["market"]


def test_compute_shap_unknown_model_raises():
    try:
        run_shap.compute_shap("magic", object(), np.zeros((2, 2)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_sample_rows_is_sorted_unique_reproducible_and_capped():
    a, b = run_shap.sample_rows(1000, 100), run_shap.sample_rows(1000, 100)
    assert len(a) == 100 and (a == b).all() and (np.diff(a) > 0).all()
    assert len(run_shap.sample_rows(50, 100)) == 50 and len(run_shap.sample_rows(50, None)) == 50


def test_run_one_end_to_end_with_an_exact_linear_explainer():
    feat = make_feat()
    model, va, cols = run_shap.fit_demand(feat, "random_forest", factory=lambda n, s: LinearRegression())
    tr = explain.add_noise(feat[feat["split"] == "train"], 10_000)
    mean_tr = tr[cols].to_numpy(dtype=float).mean(axis=0)

    def linear_shap(name, mdl, X, cat_features=None):
        X = np.asarray(X, dtype=float)
        values = mdl.coef_ * (X - mean_tr)                       # exact SHAP values of a linear model
        base = np.full(len(X), float(mdl.intercept_ + mdl.coef_ @ mean_tr))
        return values, base, mdl.predict(X)

    X = va[cols].to_numpy(dtype=float)
    with mock.patch.object(run_shap, "compute_shap", linear_shap):
        add, glob, local, frame = run_shap.run_one("demand", "random_forest", model, va, X, cols,
                                                   va["units"].to_numpy(), max_rows=20, n_boot=100)
    assert add["ok"] and add["max_abs_error"] < 1e-6
    assert len(frame) == 20 and NOISE in set(glob["feature"]) and "within_above_noise" in glob.columns
    assert set(local["case"]) == {"largest_error", "median_error"}
    for _, g in local.groupby("case"):
        assert np.isclose(g["base_value"].iloc[0] + g["shap"].sum(), g["model_prediction"].iloc[0])
    assert {"shap_" + c for c in cols} <= set(frame.columns)


def test_fit_cost_returns_arrays_for_xgboost_style_and_string_frame_for_catboost():
    df = make_cost_lines(2000)
    rf = lambda n, s: RandomForestRegressor(n_estimators=5, random_state=s, n_jobs=1)
    model, va, X, cols = run_shap.fit_cost(df, "xgboost", factory=rf)
    assert X.shape == (len(va), len(cols)) and cols[-1] == NOISE

    class Fake:
        def fit(self, X, y):
            self.X = X
            return self

        def predict(self, X):
            return np.zeros(len(X))

    model, va, X, cols = run_shap.fit_cost(df, "catboost", factory=lambda n, s: Fake())
    assert isinstance(X, pd.DataFrame) and not pd.api.types.is_numeric_dtype(X["market"])
    assert list(X.columns) == list(cost_data.FEATURES) + [NOISE]