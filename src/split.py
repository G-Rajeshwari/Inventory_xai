"""Chronological split labels. Test rows must not be used until the final run."""
import pandas as pd

from src import data


def assign_split(table: pd.DataFrame, split: dict = None) -> pd.DataFrame:
    """Add a 'split' column (train / validation / test) from stage2_split.json dates."""
    split = split or data.load_split()
    out = table.copy()
    t_end = pd.Timestamp(split["train"][1])
    v_end = pd.Timestamp(split["validation"][1])
    out["split"] = "test"
    out.loc[out["week"] <= v_end, "split"] = "validation"
    out.loc[out["week"] <= t_end, "split"] = "train"
    return out