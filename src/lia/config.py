"""Central configuration. Everything that is a business assumption lives here, not buried in code."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("LIA_HOME", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# MySQL feature store (docker compose provides it). Empty -> file mode (parquet in data/processed).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

SEED = 42

# ---- time-based split (no random shuffling: credit data drifts) ----
TRAIN_END = pd.Timestamp("2026-02-01")      # train: everything before
VAL_END = pd.Timestamp("2026-08-01")        # validation: Feb-Jul 2026
ACTIVE_FROM = pd.Timestamp("2026-09-01")    # loans originated from here have no outcome yet
#                                              test / monitoring window: Aug-Oct 2026 (drift episode)

# ---- business assumptions for the cost-based approval threshold ----
COST_OF_FUNDS_PCT = 7.5      # bank's funding cost; margin = rate - cost of funds
MARGIN_LIFE_FACTOR = 0.55    # average outstanding principal ~ 55% of the original amount
EAD_FACTOR = 0.65            # exposure at default, as a share of the original amount
MIN_LOSS_FACTOR = 0.10       # floor on loss-given-default share (recovery cannot exceed exposure)
REVIEW_BAND = 0.30           # +-30% around the threshold = "manual review" zone

# ---- fair lending: never use these as model inputs ----
PROTECTED = ["gender", "marital_status"]

# ---- drift thresholds (PSI industry convention) ----
PSI_WARN = 0.10
PSI_ALERT = 0.25

INSURANCE_INTERVAL = (0.10, 0.90)   # 80% prediction interval
