"""Data-drift monitoring: PSI + KS per feature, score drift, and label shift. Thresholds in config
(PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 act). Run:  python -m lia.drift --kind loan --current recent"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config

EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if len(ref) < 30 or len(cur) < 30:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (near-)constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + EPS
    b = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(((b - a) * np.log(b / a)).sum())


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 30 or len(cur) < 30:
        return float("nan")
    cats = sorted(set(ref) | set(cur))
    a = np.array([(ref == c).mean() + EPS for c in cats])
    b = np.array([(cur == c).mean() + EPS for c in cats])
    return float(((b - a) * np.log(b / a)).sum())


def status(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    return "alert" if psi >= config.PSI_ALERT else "watch" if psi >= config.PSI_WARN else "stable"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, num: list[str], cat: list[str], scores: tuple | None = None,
                 label_rates: tuple | None = None, name: str = "") -> dict:
    feats = []
    for c in num:
        p = psi_numeric(ref[c], cur[c])
        ks = ks_2samp(ref[c].dropna(), cur[c].dropna())
        feats.append({"feature": c, "type": "numeric", "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                      "ks_pvalue": float(ks.pvalue), "status": status(p),
                      "missing_ref": round(float(ref[c].isna().mean()), 4), "missing_cur": round(float(cur[c].isna().mean()), 4)})
    for c in cat:
        p = psi_categorical(ref[c], cur[c])
        feats.append({"feature": c, "type": "categorical", "psi": round(p, 4), "status": status(p)})
    out = {"name": name, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_reference": int(len(ref)), "n_current": int(len(cur)), "features": feats}
    if scores is not None:                                # model-output drift is the earliest warning
        sp = psi_numeric(pd.Series(scores[0]), pd.Series(scores[1]))
        out["score_drift"] = {"psi": round(sp, 4), "status": status(sp)}
    if label_rates is not None:
        out["label_rate"] = {"reference": round(label_rates[0], 4), "current": round(label_rates[1], 4)}
    ps = [f["psi"] for f in feats if not np.isnan(f["psi"])]
    out["n_alert"] = int(sum(f["status"] == "alert" for f in feats))
    out["n_watch"] = int(sum(f["status"] == "watch" for f in feats))
    out["max_psi"] = round(max(ps), 4) if ps else 0.0
    sd = out.get("score_drift", {}).get("status")
    out["overall"] = "alert" if (out["n_alert"] >= 1 or sd == "alert") else "watch" if (out["n_watch"] >= 1 or sd == "watch") else "stable"
    out["top_drifting"] = [f["feature"] for f in sorted(feats, key=lambda f: -(f["psi"] if not np.isnan(f["psi"]) else 0))[:5]]
    return out


def save_report(kind: str, rep: dict) -> None:
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / f"drift_{kind}_latest.json").write_text(json.dumps(rep, indent=2, default=float))
    hist = config.REPORTS / f"drift_{kind}_history.jsonl"
    with hist.open("a") as fh:
        fh.write(json.dumps({k: rep[k] for k in ("time", "name", "overall", "n_alert", "n_watch", "max_psi", "top_drifting")}) + "\n")
    if config.DATABASE_URL:                               # also keep an auditable trail in the feature store
        try:
            from .db import log_drift_to_db
            log_drift_to_db(kind, rep)
        except Exception as exc:                          # noqa: BLE001 - monitoring must never break the job
            print(f"[drift] could not log to database: {exc}")


def run_drift(kind: str, current: str = "recent", months: int = 3) -> dict:
    """kind: loan | insurance.  current: recent (latest N months of history) | new (the unlabelled new batch)."""
    from . import registry
    from .etl import load_clean
    from .features import INS_CAT, INS_NUM, LOAN_CAT, LOAN_NUM, build_insurance_features, build_loan_features

    art = registry.load_champion(kind)
    model, meta = art["model"], art["meta"]
    ref = pd.read_parquet(config.MODELS / f"{kind}_reference.parquet")
    build, num, cat = (build_loan_features, LOAN_NUM, LOAN_CAT) if kind == "loan" else (build_insurance_features, INS_NUM, INS_CAT)
    if current == "new":
        cur_df = build(load_clean(f"{kind}_new"))
    else:
        full = build(load_clean(kind))
        if kind == "insurance":
            full = full[full["plan_type"] == "Health"]
        cur_df = full[full["application_date"] >= full["application_date"].max() - pd.DateOffset(months=months)]
    num = [c for c in num if c != "t_years"]          # a calendar index always "drifts"; it is not a data-quality signal
    feats = meta["features"]
    sc_ref = model.predict_proba(ref) if kind == "loan" else model.predict(ref)
    sc_cur = model.predict_proba(cur_df) if kind == "loan" else model.predict(cur_df)
    labels = None
    if kind == "loan" and "default_flag" in cur_df and cur_df["default_flag"].notna().any():
        full = build(load_clean("loan"))
        labels = (float(full[full["application_date"] < config.TRAIN_END]["default_flag"].mean()), float(cur_df["default_flag"].mean()))
    rep = drift_report(ref[feats], cur_df[feats], num, cat, scores=(sc_ref, sc_cur), label_rates=labels,
                       name=f"{kind}:{current}")
    save_report(kind, rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["loan", "insurance"], default="loan")
    ap.add_argument("--current", choices=["recent", "new"], default="recent")
    a = ap.parse_args()
    r = run_drift(a.kind, a.current)
    print(f"[drift] {r['name']}: overall={r['overall']} alerts={r['n_alert']} watch={r['n_watch']} max PSI={r['max_psi']}")
    for f in sorted(r["features"], key=lambda f: -(f["psi"] if f["psi"] == f["psi"] else 0))[:6]:
        print(f"   {f['feature']:24s} PSI {f['psi']:.3f}  {f['status']}")
    if "score_drift" in r:
        print(f"   {'(model score)':24s} PSI {r['score_drift']['psi']:.3f}  {r['score_drift']['status']}")
