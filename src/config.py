"""Single source of truth for frozen decisions (Stages 1-2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# Modelling window (structure changes from Oct 2017)
WINDOW_START = "2015-01-01"
WINDOW_END = "2017-09-30 23:59:59"
FIRST_WEEK = "2015-01-05"   # first Monday in window
LAST_WEEK = "2017-09-18"    # last Monday whose full week is in window

# Chronological split (exact dates: reports/stage2_split.json)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Product selection, from TRAINING weeks only (frozen)
ELIG_MAX_ZERO_SHARE = 0.20
ELIG_RECENT_WEEKS = 8
TIER_A_MIN_MEAN = 50        # units per week in training weeks

SEEDS = [0, 1, 2, 3, 4]

# Columns known only after the order: never inputs to demand models
LEAKY_FOR_DEMAND = [
    "Delivery Status", "Late_delivery_risk", "Days for shipping (real)",
    "shipping date (DateOrders)", "Order Status",
]
# Columns that build or reveal the cost target: never inputs to cost models
LEAKY_FOR_COST = [
    "Sales", "Order Item Total", "Order Item Discount", "Order Profit Per Order",
    "Benefit per order", "Order Item Profit Ratio", "Sales per customer",
]