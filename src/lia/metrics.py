"""Metrics for the loan decision: ROC-AUC, PR-AUC, MCC and the money-based approval rule."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from . import config

GRID = np.round(np.arange(0.03, 0.80, 0.005), 3)


def best_mcc(y, p):
    scores = [matthews_corrcoef(y, p >= t) for t in GRID]
    i = int(np.argmax(scores))
    return float(scores[i]), float(GRID[i])


def economics(df: pd.DataFrame, recovery_ratio: float) -> pd.DataFrame:
    """Per-loan money at stake. margin = net interest earned if the loan is repaid;
    loss = exposure written off if it defaults (net of recoveries)."""
    amt = pd.to_numeric(df["loan_amount_inr"], errors="coerce")
    rate = pd.to_numeric(df["interest_rate_pct"], errors="coerce").fillna(12.0)
    years = (pd.to_numeric(df["tenure_months"], errors="coerce").fillna(48.0)) / 12
    margin = amt * (rate - config.COST_OF_FUNDS_PCT).clip(lower=1.0) / 100 * years * config.MARGIN_LIFE_FACTOR
    loss = amt * max(config.EAD_FACTOR - recovery_ratio, config.MIN_LOSS_FACTOR)
    return pd.DataFrame({"margin": margin.values, "loss": loss.values}, index=df.index)


def breakeven_pd(econ: pd.DataFrame) -> np.ndarray:
    """Approve while PD < margin / (margin + loss): the profit-neutral default probability of that loan."""
    return (econ["margin"] / (econ["margin"] + econ["loss"])).values


def net_value(y, approve, econ) -> float:
    y = np.asarray(y)
    approve = np.asarray(approve, bool)
    gain = (1 - y) * econ["margin"].values - y * econ["loss"].values
    return float((gain * approve).sum())


def best_global_threshold(y, p, econ):
    vals = [net_value(y, p < t, econ) for t in GRID]
    i = int(np.argmax(vals))
    return float(GRID[i]), float(vals[i])


def evaluate(y, p, econ, global_thr: float | None = None) -> dict:
    y = np.asarray(y)
    mcc, mcc_thr = best_mcc(y, p)
    out = {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "mcc_best": mcc, "mcc_threshold": mcc_thr,
        "value_approve_all": net_value(y, np.ones(len(y), bool), econ),
        "value_breakeven_rule": net_value(y, p < breakeven_pd(econ), econ),
    }
    if global_thr is not None:
        out["value_global_threshold"] = net_value(y, p < global_thr, econ)
        out["mcc_at_global_threshold"] = float(matthews_corrcoef(y, p >= global_thr))
        out["approval_rate_global"] = float((p < global_thr).mean())
    out["approval_rate_breakeven"] = float((p < breakeven_pd(econ)).mean())
    return out
