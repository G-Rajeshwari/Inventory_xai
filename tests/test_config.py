import json

import pandas as pd

from src import config


def test_split_is_chronological_and_disjoint():
    s = json.loads((config.REPORTS / "stage2_split.json").read_text())
    train_end = pd.Timestamp(s["train"][1])
    val_start = pd.Timestamp(s["validation"][0])
    val_end = pd.Timestamp(s["validation"][1])
    test_start = pd.Timestamp(s["test"][0])
    assert train_end < val_start
    assert val_end < test_start
    assert s["train"][2] + s["validation"][2] + s["test"][2] == s["n_weeks"]


def test_eligible_products_file():
    df = pd.read_csv(config.REPORTS / "eligible_products.csv")
    assert len(df) == 54
    assert set(df["tier"]) == {"A_high_volume", "B_low_volume"}
    assert (df["train_zero_share"] <= config.ELIG_MAX_ZERO_SHARE).all()


def test_leak_lists_do_not_overlap_with_targets():
    assert "Order Item Quantity" not in config.LEAKY_FOR_DEMAND
    assert "Order Item Quantity" not in config.LEAKY_FOR_COST