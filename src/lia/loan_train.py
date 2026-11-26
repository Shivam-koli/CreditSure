"""Loan eligibility: fair model comparison + calibrated PD + money-based approval rule + SHAP.
Run:  python -m lia.loan_train [--fast]"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from skopt import BayesSearchCV  # noqa: E402
from skopt.space import Integer, Real  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from . import config, registry, tracking  # noqa: E402
from .etl import load_clean  # noqa: E402
from .features import LOAN_CAT, LOAN_FEATURES, LOAN_NUM, build_loan_features, time_split  # noqa: E402
from .metrics import GRID, best_global_threshold, breakeven_pd, economics, evaluate, net_value  # noqa: E402
from .modeling import Explainer, PlattCalibrator, ScoreModel, make_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
SPACE = {
    "clf__n_estimators": Integer(100, 500), "clf__max_depth": Integer(2, 7),
    "clf__learning_rate": Real(0.02, 0.3, prior="log-uniform"), "clf__subsample": Real(0.6, 1.0),
    "clf__colsample_bytree": Real(0.5, 1.0), "clf__min_child_weight": Integer(1, 20),
    "clf__reg_lambda": Real(0.01, 20, prior="log-uniform"),
}


def _xgb(**kw):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=config.SEED, verbosity=0, **kw)


def candidates() -> dict:
    n, c = LOAN_NUM, LOAN_CAT
    return {
        "Logistic regression (GLM) + SMOTE": make_pipeline(LogisticRegression(C=0.5, max_iter=3000), n, c, True, True),
        "Logistic regression (GLM), no SMOTE": make_pipeline(LogisticRegression(C=0.5, max_iter=3000), n, c, False, True),
        "Random forest + SMOTE": make_pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=5, n_jobs=2,
                                                                      random_state=config.SEED), n, c, True),
        "Gradient boosting + SMOTE": make_pipeline(GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                                              subsample=0.8, random_state=config.SEED), n, c, True),
        "XGBoost + SMOTE": make_pipeline(_xgb(), n, c, True),
        "XGBoost, no SMOTE": make_pipeline(_xgb(), n, c, False),
    }


def tune_xgb(X, y, n_iter: int, cv: int = 3):
    search = BayesSearchCV(make_pipeline(_xgb(), LOAN_NUM, LOAN_CAT, False), SPACE, n_iter=n_iter,
                           cv=StratifiedKFold(cv, shuffle=True, random_state=config.SEED), scoring="roc_auc",
                           random_state=config.SEED, n_jobs=1)
    search.fit(X, y)
    return search


def recovery_ratio(train: pd.DataFrame) -> float:
    bad = train[train["default_flag"] == 1]
    r = (bad["recovery_amount_inr"] / bad["loan_amount_inr"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.mean()) if len(r) else 0.0


def train_loan(fast: bool = False, train_end=config.TRAIN_END, val_end=config.VAL_END, save: bool = True,
               tuned_params: dict | None = None, event: str = "trained", quick: bool = False) -> dict:
    tracking.init("loan-eligibility")
    df = build_loan_features(load_clean("loan"))
    matured = df[df["default_flag"].notna()].copy()
    tr, va, te = time_split(matured, train_end, val_end)
    te = te[te["application_date"] < config.ACTIVE_FROM]
    Xtr, ytr, Xva, yva = tr[LOAN_FEATURES], tr["default_flag"].astype(int), va[LOAN_FEATURES], va["default_flag"].astype(int)
    rr = recovery_ratio(tr)
    econ_va, econ_te = economics(va, rr), economics(te, rr)
    print(f"[loan] train {len(tr)} | val {len(va)} | test {len(te)} | default rate train {ytr.mean():.3f} | recovery ratio {rr:.2f}")

    cands = candidates()
    best_params, cv_auc = None, None
    t0 = time.time()
    if tuned_params is not None:                                       # retrain path: reuse tuned hyper-parameters
        cands["XGBoost (tuned, BayesSearchCV)"] = make_pipeline(_xgb(), LOAN_NUM, LOAN_CAT, False).set_params(**tuned_params)
        best_params = tuned_params
    else:
        with tracking.run("BayesSearchCV-tuning", tags={"event": event, "stage": "hpo"}):
            search = tune_xgb(Xtr, ytr, n_iter=4 if fast else 25)
            best_params, cv_auc = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}, float(search.best_score_)
            tracking.log_params({"n_iter": 4 if fast else 25, **best_params})
            tracking.log_metrics({"cv_auc": cv_auc})
        cands["XGBoost (tuned, BayesSearchCV)"] = search.best_estimator_
        print(f"[loan] BayesSearchCV done in {time.time() - t0:.0f}s, CV AUC {cv_auc:.4f}")

    if quick:     # retrain path: only the tuned champion family is refitted
        cands = {k: v for k, v in cands.items() if k.startswith("XGBoost (tuned")}
    rows, store = [], {}
    for name, pipe in cands.items():
        with tracking.run(name, tags={"event": event, "smote": "SMOTE" in name,
                                      "quick": quick, "fast": fast}):
            t1 = time.time()
            if not name.startswith("XGBoost (tuned") or tuned_params is not None:
                pipe.fit(Xtr, ytr)
            fit_s = time.time() - t1
            pv, pt = pipe.predict_proba(Xva)[:, 1], (pipe.predict_proba(te[LOAN_FEATURES])[:, 1] if len(te) else None)
            cal = PlattCalibrator().fit(pv, yva)
            pvc, ptc = cal.transform(pv), (cal.transform(pt) if pt is not None else None)
            thr, _ = best_global_threshold(yva, pvc, econ_va)
            mv = evaluate(yva, pvc, econ_va, thr)
            mt = evaluate(te["default_flag"].astype(int), ptc, econ_te, thr) if pt is not None else {}
            store[name] = dict(pipe=pipe, cal=cal, thr=thr, pvc=pvc, ptc=ptc)
            row = {"model": name, "fit_seconds": round(fit_s, 1),
                   **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}}
            rows.append(row)
            tracking.log_params({"smote": "SMOTE" in name, "threshold": thr})
            tracking.log_metrics({k: v for k, v in row.items() if k != "model"})
            print(f"[loan] {name:38s} val AUC {mv['roc_auc']:.4f} MCC {mv['mcc_best']:.3f} value(Rs cr) {mv['value_global_threshold'] / 1e7:6.2f}"
                  + (f" | test AUC {mt['roc_auc']:.4f} MCC@thr {mt['mcc_at_global_threshold']:.3f}" if mt else ""))
    comp = pd.DataFrame(rows)

    # ---- champion: business value first, but value differences of <2% are noise -> tie-break on validation AUC ----
    comp["val_best_value"] = comp[["val_value_global_threshold", "val_value_breakeven_rule"]].max(axis=1)
    near_best = comp[comp["val_best_value"] >= 0.98 * comp["val_best_value"].max()]
    champ_name = near_best.sort_values("val_roc_auc", ascending=False).iloc[0]["model"]
    ch = store[champ_name]
    row = comp[comp["model"] == champ_name].iloc[0]
    rule = "breakeven" if row["val_value_breakeven_rule"] > row["val_value_global_threshold"] else "global"
    print(f"[loan] champion: {champ_name} | decision rule: {rule} | global threshold {ch['thr']:.3f}")

    model = ScoreModel(ch["pipe"], ch["cal"], LOAN_FEATURES)
    expl = Explainer(ch["pipe"], LOAN_FEATURES)
    sample = (te if len(te) else va)[LOAN_FEATURES].sample(min(1500, len(te) if len(te) else len(va)), random_state=config.SEED)
    importance = expl.contributions(sample).abs().mean().sort_values(ascending=False)

    meta = {
        "name": champ_name, "rule": rule, "threshold": ch["thr"], "recovery_ratio": rr, "best_params": best_params,
        "cv_auc_tuned": cv_auc, "features": LOAN_FEATURES, "num": LOAN_NUM, "cat": LOAN_CAT,
        "train_window": [str(tr["application_date"].min().date()), str(tr["application_date"].max().date())],
        "val_window": [str(va["application_date"].min().date()), str(va["application_date"].max().date())],
        "metrics": {"val": {k: row[f"val_{k}"] for k in ("roc_auc", "pr_auc", "mcc_best", "brier")},
                    "test": {k: row.get(f"test_{k}") for k in ("roc_auc", "pr_auc", "mcc_best", "mcc_at_global_threshold",
                                                              "value_global_threshold", "value_breakeven_rule",
                                                              "value_approve_all", "approval_rate_global")}},
        "feature_importance": {k: float(v) for k, v in importance.head(15).items()},
        "rate_by_purpose": {k: float(v) for k, v in tr.groupby("loan_purpose")["interest_rate_pct"].median().items()},
        "assumptions": {"cost_of_funds_pct": config.COST_OF_FUNDS_PCT, "margin_life_factor": config.MARGIN_LIFE_FACTOR,
                        "ead_factor": config.EAD_FACTOR},
    }
    if save:
        config.REPORTS.mkdir(parents=True, exist_ok=True)
        config.FIGURES.mkdir(parents=True, exist_ok=True)
        comp.to_csv(config.REPORTS / "loan_model_comparison.csv", index=False)
        (config.REPORTS / "loan_model_comparison.json").write_text(json.dumps(comp.to_dict("records"), indent=2, default=float))
        config.MODELS.mkdir(parents=True, exist_ok=True)
        ref = tr[LOAN_FEATURES].sample(min(6000, len(tr)), random_state=config.SEED)
        _plots(comp, store, champ_name, yva, econ_va, te, econ_te, importance, ch["thr"], rule)
        meta = registry.save_champion("loan", model, meta, event)
        ref.to_parquet(config.MODELS / "loan_reference.parquet")
        with tracking.run(f"champion-{champ_name}", tags={"role": "champion", "event": event,
                                                          "registry_version": meta["version"]}):
            tracking.log_params({"rule": rule, "threshold": ch["thr"]})
            tracking.log_metrics({f"val_{k}": v for k, v in meta["metrics"]["val"].items()}
                                 | {f"test_{k}": v for k, v in meta["metrics"]["test"].items() if v is not None})
            tracking.log_sklearn_model(ch["pipe"], artifact_path="pipeline", registered_model_name="loan-champion")
    reference = tr[LOAN_FEATURES].sample(min(6000, len(tr)), random_state=config.SEED)
    return {"comparison": comp, "meta": meta, "model": model, "reference": reference, "val": va, "econ_val": econ_va}


def _plots(comp, store, champ, yva, econ_va, te, econ_te, importance, thr, rule):
    FIG = config.FIGURES
    plt.rcParams.update({"figure.facecolor": "#051F20", "axes.facecolor": "#051F20", "axes.edgecolor": "#235347",
                         "text.color": "#DAF1DE", "axes.labelcolor": "#DAF1DE", "xtick.color": "#9DBFA8",
                         "ytick.color": "#9DBFA8", "grid.color": "#163832"})
    ev_y, ev_key = (te["default_flag"].astype(int).values, "ptc") if len(te) else (yva.values, "pvc")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in store.items():
        f, t, _ = roc_curve(ev_y, s[ev_key])
        ax.plot(f, t, lw=2.2 if name == champ else 1.1, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#235347")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - " + ("test (Aug-Oct 2026)" if len(te) else "validation"))
    ax.legend(fontsize=6.5, facecolor="#051F20", edgecolor="#235347"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "loan_roc.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ev_econ = econ_te if len(te) else econ_va
    for nm, p, e, col in [("validation", store[champ]["pvc"], econ_va, "#8EB69B"),
                          ("test", store[champ]["ptc"], econ_te, "#E3C16F")]:
        if p is None:
            continue
        y = yva.values if nm == "validation" else te["default_flag"].astype(int).values
        ax.plot(GRID, [net_value(y, p < t, e) / 1e7 for t in GRID], color=col, label=nm)
    ax.axvline(thr, color="#DAF1DE", ls="--", label=f"chosen threshold {thr:.2f}")
    ax.set(xlabel="approve if PD < threshold", ylabel="net value (Rs crore)", title="Money-based approval threshold")
    ax.legend(facecolor="#051F20", edgecolor="#235347"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "loan_value_curve.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    imp = importance.head(12)[::-1]
    ax.barh(imp.index, imp.values, color="#8EB69B")
    ax.set(xlabel="mean |SHAP| (log-odds of default)", title=f"Global drivers - {champ}"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG / "loan_shap_importance.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="few tuning iterations (smoke test)")
    a = ap.parse_args()
    train_loan(fast=a.fast)
