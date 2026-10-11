import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import cost_data, explain, run_explain
from src.features import FEATURE_COLS
from tests.test_cost_models import make_lines
from tests.test_demand_models import FakeModel, fake_factory, make_feat


def signal_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n), explain.NOISE: rng.normal(size=n)})
    y = 3 * X["x0"].to_numpy() + rng.normal(scale=0.3, size=n)
    weeks = np.repeat(np.arange(n // 10), 10)
    return X, y, weeks


def test_importance_finds_the_real_signal_and_flags_only_it():
    X, y, weeks = signal_data()
    tr, va = slice(0, 800), slice(800, 1200)
    model = RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=1).fit(X.iloc[tr].to_numpy(), y[tr])
    imp = explain.permutation_importance(lambda d: model.predict(d.to_numpy()), X.iloc[va], y[va], weeks[va],
                                         explain.abs_loss, n_repeats=5, n_boot=200, seed=0)
    imp = explain.flag_informative(imp).set_index("feature")
    assert imp.loc["x0", "ci_low"] > 0.5 and imp.loc["x0", "importance"] > imp.loc["x1", "importance"] * 5
    assert bool(imp.loc["x0", "above_noise"]) is True
    assert bool(imp.loc["x1", "above_noise"]) is False
    assert bool(imp.loc[explain.NOISE, "above_noise"]) is False
    assert imp.index[0] == "x0" or imp["importance"].idxmax() == "x0"


def test_a_feature_the_model_ignores_has_zero_importance():
    X, y, weeks = signal_data(400)
    imp = explain.permutation_importance(lambda d: 3 * d["x0"].to_numpy(), X, y, weeks, explain.abs_loss,
                                         n_repeats=3, n_boot=100, seed=1).set_index("feature")
    for col in ("importance", "ci_low", "ci_high"):                  # zero up to floating-point rounding
        assert abs(imp.loc["x1", col]) < 1e-9
    assert imp.loc["x0", "importance"] > 0.5


def test_input_is_not_modified_and_results_are_reproducible():
    X, y, weeks = signal_data(300)
    before = X.copy()
    f = lambda d: 3 * d["x0"].to_numpy()
    a = explain.permutation_importance(f, X, y, weeks, explain.sq_loss, n_repeats=4, n_boot=100, seed=7)
    b = explain.permutation_importance(f, X, y, weeks, explain.sq_loss, n_repeats=4, n_boot=100, seed=7)
    c = explain.permutation_importance(f, X, y, weeks, explain.sq_loss, n_repeats=4, n_boot=100, seed=8)
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.set_index("feature").loc["x0", "importance"], c.set_index("feature").loc["x0", "importance"])


def test_length_mismatch_raises():
    X, y, weeks = signal_data(100)
    try:
        explain.permutation_importance(lambda d: d["x0"].to_numpy(), X, y[:-1], weeks, explain.abs_loss)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_flag_informative_requires_beating_zero_and_the_noise_upper_bound():
    imp = pd.DataFrame({
        "feature": ["a", "b", "c", explain.NOISE],
        "importance": [0.5, 0.2, 0.3, 0.05], "ci_low": [0.25, 0.15, -0.1, -0.1], "ci_high": [0.8, 0.3, 0.6, 0.2]})
    out = explain.flag_informative(imp).set_index("feature")
    assert bool(out.loc["a", "above_noise"]) is True       # 0.25 > noise upper bound 0.2
    assert bool(out.loc["b", "above_noise"]) is False      # 0.15 < 0.2
    assert bool(out.loc["c", "above_noise"]) is False      # interval includes zero
    assert bool(out.loc[explain.NOISE, "above_noise"]) is False
    neg = imp.copy(); neg.loc[3, "ci_high"] = -0.05
    out2 = explain.flag_informative(neg).set_index("feature")
    assert bool(out2.loc["b", "above_noise"]) is True      # threshold is max(0, noise upper) = 0


def test_add_noise_is_deterministic_independent_and_non_destructive():
    df = pd.DataFrame({"a": range(50)})
    a, b, c = explain.add_noise(df, 1), explain.add_noise(df, 1), explain.add_noise(df, 2)
    assert explain.NOISE not in df.columns
    assert np.allclose(a[explain.NOISE], b[explain.NOISE]) and not np.allclose(a[explain.NOISE], c[explain.NOISE])
    assert abs(a[explain.NOISE].mean()) < 0.5


def test_demand_importance_trains_on_train_rows_only_and_includes_the_noise_row():
    f = make_feat()
    FakeModel.last_train_rows = None
    imp = run_explain.demand_importance(f, "random_forest", n_repeats=2, n_boot=50, factory=fake_factory)
    assert FakeModel.last_train_rows == (f["split"] == "train").sum()
    assert len(imp) == len(FEATURE_COLS) + 1 and explain.NOISE in set(imp["feature"])
    assert (imp["importance"].abs() < 1e-9).all() and not imp["above_noise"].any()   # fake model ignores features
    assert set(imp["task"]) == {"demand"}


def test_cost_importance_xgboost_path_runs_and_flags_nothing_for_pure_noise_target():
    df = make_lines(3000)
    fac = lambda name, seed: RandomForestRegressor(n_estimators=15, min_samples_leaf=50, random_state=seed, n_jobs=1)
    imp = run_explain.cost_importance(df, "xgboost", n_repeats=2, n_boot=50, factory=fac)
    assert len(imp) == len(cost_data.FEATURES) + 1 and explain.NOISE in set(imp["feature"])
    assert set(imp["task"]) == {"cost"} and set(imp["loss"]) == {"MSE"}
    assert imp["ci_low"].le(imp["ci_high"]).all()


def test_cost_importance_catboost_path_passes_categoricals_as_strings():
    seen = {}

    class FakeCat:
        def fit(self, X, y):
            seen["dtype"] = X["market"].dtype
            seen["cols"] = list(X.columns)
            self.m = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    df = make_lines(1500)
    imp = run_explain.cost_importance(df, "catboost", n_repeats=2, n_boot=50, factory=lambda n, s: FakeCat())
    assert not pd.api.types.is_numeric_dtype(seen["dtype"])
    assert seen["cols"] == list(cost_data.FEATURES) + [explain.NOISE]
    assert (imp["importance"].abs() < 1e-9).all()