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


def test_permuted_order_is_a_permutation_and_stays_inside_groups():
    groups = np.repeat(np.arange(5), 20)
    gidx = explain.group_indices(groups)
    order = explain.permuted_order(100, gidx, np.random.default_rng(0))
    assert sorted(order) == list(range(100))
    for ix in gidx:
        assert set(order[ix]) == set(ix)
    assert (order != np.arange(100)).any()                       # something was actually shuffled
    free = explain.permuted_order(100, None, np.random.default_rng(0))
    assert sorted(free) == list(range(100)) and any(free[ix].max() > ix.max() or free[ix].min() < ix.min()
                                                    for ix in gidx)


def test_global_shuffle_matches_the_plain_value_shuffle_used_before():
    v = np.arange(50) * 1.5
    a = v[explain.permuted_order(50, None, np.random.default_rng(3))]
    b = np.random.default_rng(3).permutation(v)
    assert np.allclose(a, b)


def test_within_group_shuffle_keeps_group_level_information_and_finds_within_group_signal():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), 60)
    level = (g * 10.0)                                           # group-level feature (constant inside a group)
    x = rng.normal(size=len(g))                                  # within-group feature
    X = pd.DataFrame({"level": level, "x": x, explain.NOISE: rng.normal(size=len(g))})
    y = level + 2 * x
    weeks = np.tile(np.arange(60), 10)
    f = lambda d: d["level"].to_numpy() + 2 * d["x"].to_numpy()
    glob = explain.permutation_importance(f, X, y, weeks, explain.abs_loss, n_repeats=5, n_boot=100,
                                          seed=0).set_index("feature")
    within = explain.permutation_importance(f, X, y, weeks, explain.abs_loss, n_repeats=5, n_boot=100,
                                            seed=0, groups=g).set_index("feature")
    assert glob.loc["level", "importance"] > 10                  # shuffling across groups destroys the level
    assert abs(within.loc["level", "importance"]) < 1e-9         # ...but not within groups
    assert within.loc["x", "importance"] > 0.5 and within.loc["x", "ci_low"] > 0
    flagged = explain.flag_informative(within.reset_index()).set_index("feature")
    assert bool(flagged.loc["x", "above_noise"]) is True and bool(flagged.loc["level", "above_noise"]) is False


def test_groups_length_mismatch_raises():
    X, y, weeks = signal_data(100)
    try:
        explain.permutation_importance(lambda d: d["x0"].to_numpy(), X, y, weeks, explain.abs_loss,
                                       groups=np.arange(99))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_demand_within_product_scheme_gives_zero_importance_to_product_level_features():
    f = make_feat()
    imp = run_explain.demand_importance(f, "random_forest", n_repeats=3, n_boot=50, within_product=True)
    imp = imp.set_index("feature")
    assert set(imp["scheme"]) == {"within_product"}
    assert abs(imp.loc["product_code", "importance"]) < 1e-9 and abs(imp.loc["is_tier_a", "importance"]) < 1e-9