"""Scheduled champion/challenger retraining, gated by drift monitoring.
   python -m lia.retrain --kind all            # drift check first; retrain only if drift is 'alert' or the model is old
   python -m lia.retrain --kind loan --force   # always train a challenger
A challenger is promoted only if it beats the current champion on the most recent labelled window."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, registry
from .drift import run_drift
from .features import INS_FEATURES, LOAN_FEATURES, build_insurance_features, build_loan_features
from .metrics import breakeven_pd, economics, net_value
from .etl import load_clean
from sklearn.metrics import roc_auc_score

MAX_MODEL_AGE_DAYS = 90
HOLDOUT_MONTHS = 4


def _age_days(meta: dict) -> float:
    t = datetime.strptime(meta["version"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400


def retrain_loan(force: bool = False) -> dict:
    from .loan_train import train_loan
    art = registry.load_champion("loan")
    champ, meta = art["model"], art["meta"]
    drift = run_drift("loan", "recent")
    need = force or drift["overall"] == "alert" or _age_days(meta) > MAX_MODEL_AGE_DAYS
    out = {"kind": "loan", "drift": drift["overall"], "max_psi": drift["max_psi"], "retrained": False}
    if not need:
        registry.log_event("loan", "retrain_skipped", {"reason": f"drift={drift['overall']}, model fresh"})
        return {**out, "reason": "no trigger"}
    df = build_loan_features(load_clean("loan"))
    matured = df[df["default_flag"].notna()]
    last = matured["application_date"].max()
    hold_start = (last - pd.DateOffset(months=HOLDOUT_MONTHS)).normalize()
    res = train_loan(train_end=hold_start, val_end=config.ACTIVE_FROM, save=False, tuned_params=meta["best_params"], quick=True)
    chall, cm = res["model"], res["meta"]
    va = res["val"]
    y = va["default_flag"].astype(int).values
    econ = economics(va, cm["recovery_ratio"])
    def score(m):
        p = m.predict_proba(va)
        return roc_auc_score(y, p), net_value(y, p < breakeven_pd(econ), econ)
    (auc_c, val_c), (auc_n, val_n) = score(champ), score(chall)
    promote = auc_n >= auc_c - 0.003 and val_n >= val_c
    out.update(retrained=True, promoted=bool(promote), holdout=[str(hold_start.date()), str(last.date())],
               champion={"auc": auc_c, "value": val_c}, challenger={"auc": auc_n, "value": val_n})
    if promote:
        cm["train_window"] = [cm["train_window"][0], str(hold_start.date())]
        registry.save_champion("loan", chall, cm, event="promoted")
        res["reference"].to_parquet(config.MODELS / "loan_reference.parquet")
    else:
        registry.log_event("loan", "challenger_rejected", out)
    return out


def retrain_insurance(force: bool = False) -> dict:
    from .insurance_train import train_insurance
    art = registry.load_champion("insurance")
    champ, meta = art["model"], art["meta"]
    drift = run_drift("insurance", "recent")
    need = force or drift["overall"] == "alert" or _age_days(meta) > MAX_MODEL_AGE_DAYS
    out = {"kind": "insurance", "drift": drift["overall"], "max_psi": drift["max_psi"], "retrained": False}
    if not need:
        registry.log_event("insurance", "retrain_skipped", {"reason": f"drift={drift['overall']}, model fresh"})
        return {**out, "reason": "no trigger"}
    df = build_insurance_features(load_clean("insurance"))
    h = df[(df["plan_type"] == "Health") & (df["target_invalid"] == 0)]
    last = h["application_date"].max()
    hold_start = (last - pd.DateOffset(months=HOLDOUT_MONTHS)).normalize()
    res = train_insurance(train_end=hold_start, val_end=last + pd.Timedelta(days=1), save=False,
                          tuned_params=meta["best_params"], quick=True)
    chall, va = res["model"], res["val"]
    y = va["annual_premium_inr"].values
    rm = lambda m: float(np.sqrt(np.mean((m.predict(va[INS_FEATURES]) - y) ** 2)))
    rmse_c, rmse_n = rm(champ), rm(chall)
    promote = rmse_n < 0.98 * rmse_c
    out.update(retrained=True, promoted=bool(promote), holdout=[str(hold_start.date()), str(last.date())],
               champion={"rmse": rmse_c}, challenger={"rmse": rmse_n})
    if promote:
        cm = res["meta"]
        cm["train_window"] = [cm["train_window"][0], str(hold_start.date())]
        registry.save_champion("insurance", chall, cm, event="promoted")
        res["reference"].to_parquet(config.MODELS / "insurance_reference.parquet")
    else:
        registry.log_event("insurance", "challenger_rejected", out)
    return out


def main(kind: str = "all", force: bool = False) -> list[dict]:
    results = []
    if kind in ("loan", "all"):
        results.append(retrain_loan(force))
    if kind in ("insurance", "all"):
        results.append(retrain_insurance(force))
    for r in results:
        print("[retrain]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["loan", "insurance", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.kind, a.force)
