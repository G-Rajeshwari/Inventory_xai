import numpy as np
import pandas as pd

from src import baselines, data, features, metrics, split


def make_lines(n_weeks=40, products=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-01-05")
    rows = []
    for p in products:
        for w in range(n_weeks):
            for _ in range(rng.integers(1, 4)):
                day = start + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
                rows.append((p, day, int(rng.integers(1, 6))))
    df = pd.DataFrame(rows, columns=["product_id", "order_dt", "qty"])
    df["week"] = data.add_week_start(df["order_dt"])
    return df


def make_table(n_weeks=40):
    lines = make_lines(n_weeks)
    last = pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=n_weeks - 1)
    return data.build_weekly_table(lines, [1, 2], "2015-01-05", last)


TIERS = pd.Series({1: "A_high_volume", 2: "B_low_volume"})


def test_week_start_is_monday():
    s = pd.Series(pd.to_datetime(["2015-01-07 13:00", "2015-01-11 23:59", "2015-01-12 00:00"]))
    assert list(data.add_week_start(s).dt.weekday) == [0, 0, 0]
    assert data.add_week_start(s).iloc[1] == pd.Timestamp("2015-01-05")
    assert data.add_week_start(s).iloc[2] == pd.Timestamp("2015-01-12")


def test_weekly_table_is_dense_and_conserves_units():
    lines = make_lines()
    last = pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=39)
    t = data.build_weekly_table(lines, [1, 2], "2015-01-05", last)
    assert len(t) == 2 * 40
    assert t.groupby("product_id").size().eq(40).all()
    assert t["units"].sum() == lines["qty"].sum()


def test_features_never_use_the_current_or_future_week():
    """Changing units at weeks >= T must not change any feature in rows with week <= T."""
    t = make_table()
    base = features.build_features(t, TIERS)
    T = pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=33)
    t2 = t.copy()
    rng = np.random.default_rng(1)
    mask = t2["week"] >= T
    t2.loc[mask, "units"] = rng.integers(1000, 2000, size=mask.sum())
    changed = features.build_features(t2, TIERS)
    cols = ["product_id", "week"] + list(features.FEATURE_COLS)
    a = base[base["week"] <= T][cols].reset_index(drop=True)
    b = changed[changed["week"] <= T][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # sanity: the next week's lag_1 DOES change (otherwise the test proves nothing)
    nxt = T + pd.Timedelta(weeks=1)
    assert not np.allclose(
        base[base["week"] == nxt]["lag_1"].to_numpy(),
        changed[changed["week"] == nxt]["lag_1"].to_numpy(),
    )


def test_lag_values_are_correct():
    t = make_table()
    f = features.build_features(t, TIERS)
    row = f[(f["product_id"] == 1)].iloc[3]
    hist = t[(t["product_id"] == 1) & (t["week"] < row["week"])].sort_values("week")["units"].to_numpy()
    assert row["lag_1"] == hist[-1]
    assert row["lag_4"] == hist[-4]
    assert np.isclose(row["roll_mean_4"], hist[-4:].mean())
    assert np.isclose(row["roll_mean_26"], hist[-26:].mean())
    assert row["units"] == t[(t["product_id"] == 1) & (t["week"] == row["week"])]["units"].iloc[0]


def test_warmup_rows_are_dropped_and_no_nans():
    t = make_table()
    f = features.build_features(t, TIERS)
    assert f[list(features.FEATURE_COLS)].notna().all().all()
    assert f["week"].min() == pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=features.WARMUP_WEEKS)


def test_split_is_ordered_and_disjoint():
    t = make_table()
    f = features.build_features(t, TIERS)
    sp = {"train": ["2015-01-05", "2015-07-27", 0],
          "validation": ["2015-08-03", "2015-09-14", 0],
          "test": ["2015-09-21", "2015-10-05", 0]}
    f = split.assign_split(f, sp)
    assert set(f["split"]) <= {"train", "validation", "test"}
    last_train = f[f["split"] == "train"]["week"].max()
    first_val = f[f["split"] == "validation"]["week"].min()
    last_val = f[f["split"] == "validation"]["week"].max()
    first_test = f[f["split"] == "test"]["week"].min()
    assert last_train < first_val and last_val < first_test


def test_train_mean_baseline_uses_only_train_rows():
    t = make_table()
    f = features.build_features(t, TIERS)
    sp = {"train": ["2015-01-05", "2015-07-27", 0],
          "validation": ["2015-08-03", "2015-09-14", 0],
          "test": ["2015-09-21", "2015-10-05", 0]}
    f = split.assign_split(f, sp)
    b = baselines.add_baselines(f, fit_splits=("train",))
    expected = f[f["split"] == "train"].groupby("product_id")["units"].mean()
    got = b.groupby("product_id")["pred_train_mean"].first()
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    # changing validation/test units must not change the baseline
    f2 = f.copy()
    f2.loc[f2["split"] != "train", "units"] = 9999
    b2 = baselines.add_baselines(f2, fit_splits=("train",))
    assert np.allclose(b["pred_train_mean"], b2["pred_train_mean"])


def test_metrics_values():
    y = np.array([0, 2, 4])
    p = np.array([1, 2, 2])
    assert np.isclose(metrics.mae(y, p), 1.0)
    assert np.isclose(metrics.rmse(y, p), np.sqrt(5 / 3))
    assert np.isclose(metrics.wape(y, p), 3 / 6)
    assert np.isnan(metrics.wape([0, 0], [1, 1]))  # zero actuals handled, no crash
    assert np.isclose(metrics.r2(y, y), 1.0)


def test_bootstrap_is_reproducible_and_detects_difference():
    weeks = np.repeat(np.arange(30), 3)
    a = np.full(90, 1.0)
    b = np.full(90, 2.0)
    r1 = metrics.paired_week_bootstrap(weeks, a, b, seed=3)
    r2_ = metrics.paired_week_bootstrap(weeks, a, b, seed=3)
    assert r1 == r2_
    assert r1["ci_high"] < 0  # A clearly better than B
    same = metrics.paired_week_bootstrap(weeks, a, a, seed=3)
    assert same["ci_low"] <= 0 <= same["ci_high"]