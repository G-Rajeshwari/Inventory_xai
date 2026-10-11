import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import cost_data, cost_models, run_cost_models

T = cost_data.TARGET


def make_lines(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-05", periods=30, freq="7D")
    df = pd.DataFrame({
        "order_dt": weeks[rng.integers(0, 30, n)] + pd.to_timedelta(rng.integers(0, 6, n), unit="D"),
        "product_id": rng.choice([10, 11, 12], n),
        "qty": rng.integers(1, 6, n).astype(float),
        "price": rng.choice([20.0, 50.0, 100.0], n),
        "discount_rate": rng.choice([0.0, 0.1, 0.2], n),
        "sched_days": rng.choice([0, 2, 4], n).astype(float),
        "shipping_mode": rng.choice(["A", "B"], n), "market": "LATAM", "segment": "Consumer",
        "pay_type": "DEBIT", "region": "Central America", "category_id": 5, "department": "Fan Shop",
    })
    df["week"] = df["order_dt"].dt.normalize() - pd.to_timedelta(df["order_dt"].dt.weekday, unit="D")
    df[T] = rng.normal(0.9, 0.4, n).clip(0.5, 3.75)
    df["split"] = np.where(df["week"] <= "2015-05-25", "train",
                           np.where(df["week"] <= "2015-07-06", "validation", "test"))
    return df.sort_values("order_dt").reset_index(drop=True)


class FakeModel:
    calls = 0
    last_n = None

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        FakeModel.calls += 1
        FakeModel.last_n = len(X)
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean + 10.0)       # deliberately outside the training range


def fake_factory(name, seed):
    return FakeModel(seed)


def test_ordinal_coder_is_learned_on_train_and_maps_unseen_to_minus_one():
    df = make_lines()
    tr = df[df["split"] == "train"]
    coder = cost_models.OrdinalCoder().fit(tr)
    other = df.head(5).copy()
    other["product_id"] = [10, 11, 12, 999, 10]
    X = coder.transform(other)
    pid_col = len(cost_data.NUMERIC_FEATURES) + cost_data.CATEGORICAL_FEATURES.index("product_id")
    assert X[3, pid_col] == -1
    assert X[0, pid_col] == X[4, pid_col]
    assert X.shape == (5, len(cost_data.FEATURES))


def test_fit_uses_only_train_rows_and_clips_to_training_range():
    df = make_lines()
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    FakeModel.last_n = None
    pred = cost_models.fit_predict("random_forest", 0, tr, va, factory=fake_factory)
    assert FakeModel.last_n == len(tr)
    assert pred.max() <= tr[T].max() + 1e-12 and pred.min() >= tr[T].min() - 1e-12


def test_random_forest_is_reproducible():
    df = make_lines(1500)
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    a = cost_models.fit_predict("random_forest", 2, tr, va)
    b = cost_models.fit_predict("random_forest", 2, tr, va)
    assert np.allclose(a, b)


def test_catboost_gets_native_categoricals_and_no_output_files():
    captured = {}

    class FakeCat:
        def __init__(self, **kw):
            captured.update(kw)

    fake_module = types.ModuleType("catboost")
    fake_module.CatBoostRegressor = FakeCat
    with mock.patch.dict(sys.modules, {"catboost": fake_module}):
        cost_models.make_model("catboost", 4)
    assert captured["cat_features"] == cost_data.CATEGORICAL_FEATURES
    assert captured["random_seed"] == 4 and captured["allow_writing_files"] is False
    frame = cost_models.as_catboost_frame(make_lines(10))
    assert all(frame[c].dtype == object or str(frame[c].dtype).startswith("str")
               for c in cost_data.CATEGORICAL_FEATURES)


def test_baselines_use_train_only_and_unseen_product_falls_back_to_global_mean():
    df = make_lines()
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"].copy()
    va.loc[va.index[0], "product_id"] = 999
    b = run_cost_models.baseline_preds(tr, va)
    assert np.allclose(b["global_mean"][0], tr[T].mean())
    assert np.isclose(b["product_mean"][0][0], tr[T].mean())          # unseen product
    expect = tr[tr["product_id"] == 10][T].mean()
    row = np.where(va["product_id"].to_numpy() == 10)[0][0]
    assert np.isclose(b["product_mean"][0][row], expect)


def test_evaluate_applies_both_conditions_of_the_decision_rule():
    df = make_lines(6000)
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"].reset_index(drop=True)
    y = va[T].to_numpy()
    gm = tr[T].mean()
    oracle = y.copy()
    same = np.full(len(y), gm)                          # identical to the reference
    barely = gm + 0.001 * (y - gm)                      # MSE about 0.2% below the reference: under the 0.5% floor
    res = run_cost_models.evaluate(tr, va, {"oracle": [oracle], "same": [same], "barely": [barely]}, n_boot=300)
    r = res.set_index("model")
    assert bool(r.loc["oracle", "improves_on_reference"]) is True
    assert bool(r.loc["same", "improves_on_reference"]) is False
    assert abs(r.loc["same", "diff_MSE"]) < 1e-12
    assert 0 < r.loc["barely", "skill"] < run_cost_models.PRACTICAL_FLOOR
    assert r.loc["barely", "ci_high"] < 0                                     # clearly below zero, yet...
    assert bool(r.loc["barely", "improves_on_reference"]) is False            # ...fails the 0.5% floor
    assert abs(r.loc["global_mean", "MSE_vs_global_mean"] - 1.0) < 1e-12
    assert res["n_rows"].nunique() == 1


def test_dollar_r2_is_high_for_the_arithmetic_baseline_when_price_varies_a_lot():
    rng = np.random.default_rng(0)
    n = 4000
    price = rng.choice([10.0, 50.0, 500.0, 1500.0], n)
    ratio = rng.normal(0.9, 0.3, n).clip(0.5, 3.0)
    va = pd.DataFrame({"price": price, "discount_rate": 0.0, T: ratio,
                       "week": pd.to_datetime("2015-06-01") + pd.to_timedelta(rng.integers(0, 5, n) * 7, unit="D"),
                       "product_id": 1})
    tr = va.copy()
    res = run_cost_models.evaluate(tr, va, {}, n_boot=100).set_index("model")
    assert res.loc["global_mean", "dollar_R2"] > 0.8        # price arithmetic alone explains most dollars
    assert abs(res.loc["global_mean", "skill"]) < 1e-12      # ...yet the ratio itself has no skill

def test_run_model_caches_and_stays_on_validation():
    df = make_lines()
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(run_cost_models, "PRED_DIR", Path(tmp)):
            FakeModel.calls = 0
            p1 = run_cost_models.run_model("xgboost", 0, df, factory=fake_factory)
            p2 = run_cost_models.run_model("xgboost", 0, df, factory=fake_factory)
            assert FakeModel.calls == 1 and np.allclose(p1, p2)
            saved = pd.read_csv(Path(tmp) / "validation_xgboost_seed0.csv", parse_dates=["order_dt"])
    assert len(saved) == (df["split"] == "validation").sum()