import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pandas as pd

from src import demand_models, run_demand_models, run_tabpfn_cat
from tests.test_demand_models import FakeModel, fake_factory, make_feat


def test_tabpfn_cat_marks_product_code_as_categorical():
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("tabpfn")
    fake_module.TabPFNRegressor = FakeRegressor
    with mock.patch.dict(sys.modules, {"tabpfn": fake_module}):
        demand_models.make_model("tabpfn_cat", 3, ["lag_1", "product_code", "week_of_year"])
    assert captured["categorical_features_indices"] == [1]
    assert captured["random_state"] == 3
    with mock.patch.dict(sys.modules, {"tabpfn": fake_module}):
        captured.clear()
        demand_models.make_model("tabpfn_cat", 0, ["lag_1"])
    assert captured["categorical_features_indices"] is None


def test_plain_tabpfn_does_not_mark_categoricals():
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("tabpfn")
    fake_module.TabPFNRegressor = FakeRegressor
    with mock.patch.dict(sys.modules, {"tabpfn": fake_module}):
        demand_models.make_model("tabpfn", 0, ["product_code"])
    assert "categorical_features_indices" not in captured


def test_tabpfn_cat_run_stays_on_validation_and_caches():
    f = make_feat()
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(run_demand_models, "PRED_DIR", Path(tmp)):
            FakeModel.calls = 0
            res = run_tabpfn_cat.run(f, factory=fake_factory)
            first_calls = FakeModel.calls
            assert first_calls == len(run_tabpfn_cat.VARIANTS) * len(run_tabpfn_cat.SEEDS)
            run_tabpfn_cat.run(f, factory=fake_factory)
            assert FakeModel.calls == first_calls          # second run came from the cache
            saved = pd.read_csv(Path(tmp) / "tabpfn_cat" / "validation_id_only_tabpfn_cat_seed0.csv",
                                parse_dates=["week"])
    val = f[f["split"] == "validation"]
    assert len(saved) == len(val) and saved["week"].max() <= val["week"].max()
    assert {"tabpfn_cat[full]", "tabpfn_cat[id_only]", "train_mean"} <= set(res["model"])
    assert res.groupby("group")["n_rows"].nunique().eq(1).all()