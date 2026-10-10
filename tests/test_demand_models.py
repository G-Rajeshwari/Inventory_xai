import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import baselines, data, demand_models, features, run_demand_models, split

TIERS = pd.Series({1: "A_high_volume", 2: "B_low_volume", 3: "B_low_volume"})
SPLIT = {"train": ["2015-01-05", "2015-07-27", 0],
         "validation": ["2015-08-03", "2015-09-28", 0],
         "test": ["2015-10-05", "2015-11-02", 0]}


def make_feat(n_weeks=45, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-05", periods=n_weeks, freq="7D")
    rows = []
    for p, lam in ((1, 40), (2, 5), (3, 6)):
        for w in weeks:
            rows.append((p, w, int(rng.poisson(lam))))
    table = pd.DataFrame(rows, columns=["product_id", "week", "units"])
    f = features.build_features(table, TIERS)
    f = split.assign_split(f, SPLIT)
    return baselines.add_baselines(f, fit_splits=("train",))


class FakeModel:
    """Predicts the training mean. Records what it was trained on."""
    calls = 0
    last_train_rows = None

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        FakeModel.calls += 1
        FakeModel.last_train_rows = len(X)
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean) - 1000.0   # deliberately negative


def fake_factory(name, seed):
    return FakeModel(seed)


def test_unknown_model_raises():
    try:
        demand_models.make_model("magic", 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_predictions_are_clipped_at_zero():
    f = make_feat()
    pred = demand_models.fit_predict("random_forest", 0, f[f["split"] == "train"],
                                     f[f["split"] == "validation"], factory=fake_factory)
    assert (pred >= 0).all()


def test_fit_uses_only_the_rows_it_is_given_and_train_precedes_validation():
    f = make_feat()
    tr, va = f[f["split"] == "train"], f[f["split"] == "validation"]
    assert tr["week"].max() < va["week"].min()
    FakeModel.last_train_rows = None
    demand_models.fit_predict("random_forest", 0, tr, va, factory=fake_factory)
    assert FakeModel.last_train_rows == len(tr)


def test_random_forest_is_reproducible_for_a_seed():
    f = make_feat()
    tr, va = f[f["split"] == "train"], f[f["split"] == "validation"]
    a = demand_models.fit_predict("random_forest", 3, tr, va)
    b = demand_models.fit_predict("random_forest", 3, tr, va)
    assert np.allclose(a, b)


def test_xgboost_runs_and_is_reproducible():
    f = make_feat()
    tr, va = f[f["split"] == "train"], f[f["split"] == "validation"]
    a = demand_models.fit_predict("xgboost", 1, tr, va)
    b = demand_models.fit_predict("xgboost", 1, tr, va)
    assert len(a) == len(va) and np.allclose(a, b)


def test_run_model_caches_predictions_and_resumes():
    f = make_feat()
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(run_demand_models, "PRED_DIR", Path(tmp)):
            FakeModel.calls = 0
            p1 = run_demand_models.run_model("tabpfn", 0, f, factory=fake_factory)
            assert FakeModel.calls == 1
            p2 = run_demand_models.run_model("tabpfn", 0, f, factory=fake_factory)
            assert FakeModel.calls == 1          # second call came from the cache
            assert np.allclose(p1, p2)


def test_run_model_only_predicts_validation_rows():
    assert run_demand_models.EVAL_SPLIT == "validation"
    f = make_feat()
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(run_demand_models, "PRED_DIR", Path(tmp)):
            run_demand_models.run_model("tabpfn", 0, f, factory=fake_factory)
            saved = pd.read_csv(Path(tmp) / "validation_tabpfn_seed0.csv", parse_dates=["week"])
    va = f[f["split"] == "validation"]
    assert len(saved) == len(va)
    assert saved["week"].max() <= pd.Timestamp(SPLIT["validation"][1])


def test_evaluate_flags_a_clearly_better_model_and_not_a_copy_of_the_reference():
    f = make_feat()
    val = f[f["split"] == "validation"].reset_index(drop=True)
    perfect = val["units"].to_numpy(dtype=float)
    copy_ref = val["pred_train_mean"].to_numpy(dtype=float)
    res = run_demand_models.evaluate(val, {"oracle": [perfect], "copycat": [copy_ref]}, n_boot=500)
    pooled = res[res["group"] == "pooled"].set_index("model")
    assert bool(pooled.loc["oracle", "beats_train_mean"]) is True
    assert bool(pooled.loc["copycat", "beats_train_mean"]) is False
    assert abs(pooled.loc["copycat", "diff_vs_train_mean"]) < 1e-9
    assert abs(pooled.loc["train_mean", "MAE_vs_train_mean"] - 1.0) < 1e-9
    # every model is scored on the same number of rows within a group
    assert pooled["n_rows"].nunique() == 1


def test_evaluate_averages_over_seeds():
    f = make_feat()
    val = f[f["split"] == "validation"].reset_index(drop=True)
    y = val["units"].to_numpy(dtype=float)
    res = run_demand_models.evaluate(val, {"m": [y + 1, y + 3]}, n_boot=200)
    row = res[(res["group"] == "pooled") & (res["model"] == "m")].iloc[0]
    assert row["n_seeds"] == 2
    assert np.isclose(row["MAE"], 2.0)
    assert np.isclose(row["MAE_seed_std"], np.sqrt(2.0))