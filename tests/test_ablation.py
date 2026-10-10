import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import demand_models, features, run_ablation, run_demand_models
from tests.test_demand_models import FakeModel, fake_factory, make_feat


def test_variants_use_only_existing_non_target_columns():
    for name, cols in run_ablation.VARIANTS.items():
        assert len(cols) == len(set(cols)), name
        assert set(cols) <= set(features.FEATURE_COLS), name
        assert "units" not in cols, name
    assert run_ablation.VARIANTS["id_only"] == ["product_code"]
    assert run_ablation.VARIANTS["full"] == list(features.FEATURE_COLS)


def test_fit_predict_uses_only_the_requested_columns():
    f = make_feat()
    tr, va = f[f["split"] == "train"], f[f["split"] == "validation"]
    seen = {}

    class Spy(FakeModel):
        def fit(self, X, y):
            seen["n_cols"] = X.shape[1]
            return super().fit(X, y)

    demand_models.fit_predict("random_forest", 0, tr, va, factory=lambda n, s: Spy(s),
                              feature_cols=["product_code", "week_of_year"])
    assert seen["n_cols"] == 2
    demand_models.fit_predict("random_forest", 0, tr, va, factory=lambda n, s: Spy(s))
    assert seen["n_cols"] == len(features.FEATURE_COLS)


def test_ablation_caches_each_variant_separately_and_stays_on_validation():
    f = make_feat()
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(run_demand_models, "PRED_DIR", Path(tmp)):
            FakeModel.calls = 0
            run_ablation.get_preds("id_only", "tabpfn", 0, f, factory=fake_factory)
            run_ablation.get_preds("id_season", "tabpfn", 0, f, factory=fake_factory)
            assert FakeModel.calls == 2
            run_ablation.get_preds("id_only", "tabpfn", 0, f, factory=fake_factory)
            assert FakeModel.calls == 2     # cache hit
            saved = pd.read_csv(Path(tmp) / "ablation" / "validation_id_only_tabpfn_seed0.csv", parse_dates=["week"])
    assert saved["week"].max() <= f[f["split"] == "validation"]["week"].max()
    assert len(saved) == (f["split"] == "validation").sum()


def test_vs_full_is_zero_for_identical_predictions_and_detects_improvement():
    f = make_feat()
    val = f[f["split"] == "validation"].reset_index(drop=True)
    y = val["units"].to_numpy(dtype=float)
    base = np.full(len(val), y.mean())
    preds = {("m", "full"): [base], ("m", "same"): [base.copy()], ("m", "oracle"): [y.copy()]}
    t = run_ablation.vs_full_table(val, preds, n_boot=300).set_index(["group", "model"])
    assert abs(t.loc[("pooled", "m[same]"), "diff_vs_full"]) < 1e-12
    assert t.loc[("pooled", "m[oracle]"), "full_ci_high"] < 0
EOF