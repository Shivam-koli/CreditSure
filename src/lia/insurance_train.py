"""Health premium prediction: linear baseline vs XGBoost (tuned with BayesSearchCV) + 80% error range.
Premium PREDICTION (row level), not time-series forecasting.   Run:  python -m lia.insurance_train [--fast]"""
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
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from skopt import BayesSearchCV  # noqa: E402
from skopt.space import Integer, Real  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from . import config, registry, tracking  # noqa: E402
from .etl import load_clean  # noqa: E402
from .features import INS_CAT, INS_FEATURES, INS_NUM, build_insurance_features, time_split  # noqa: E402
from .modeling import Explainer, PremiumModel, make_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
SPACE = {
    "clf__n_estimators": Integer(150, 800), "clf__max_depth": Integer(2, 6),
    "clf__learning_rate": Real(0.02, 0.25, prior="log-uniform"), "clf__subsample": Real(0.6, 1.0),
    "clf__colsample_bytree": Real(0.5, 1.0), "clf__min_child_weight": Integer(1, 30),
    "clf__reg_lambda": Real(0.01, 20, prior="log-uniform"),
}


def _xgb(**kw):
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", n_jobs=2, random_state=config.SEED, verbosity=0, **kw)


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "median_ape": float(np.median(np.abs(p - y) / y))}


def train_insurance(fast: bool = False, train_end=config.TRAIN_END, val_end=config.VAL_END, save: bool = True,
                    tuned_params: dict | None = None, event: str = "trained", quick: bool = False) -> dict:
    tracking.init("insurance-pricing")
    df = build_insurance_features(load_clean("insurance"))
    h = df[(df["plan_type"] == "Health") & (df["target_invalid"] == 0)].copy()
    tr, va, te = time_split(h, train_end, val_end)
    if len(te) == 0:                      # rolling retrain: no untouched test window, report on validation
        te = va
    print(f"[insurance] Health policies: train {len(tr)} | val {len(va)} | test {len(te)}")
    Xtr, ytr, Xva, yva = tr[INS_FEATURES], tr["annual_premium_inr"], va[INS_FEATURES], va["annual_premium_inr"]
    Xte, yte = te[INS_FEATURES], te["annual_premium_inr"]

    cands = {  # name -> (pipeline, uses log target)
        "Linear regression (raw target)": (make_pipeline(LinearRegression(), INS_NUM, INS_CAT, scale=True), False),
        "Linear regression (log target)": (make_pipeline(LinearRegression(), INS_NUM, INS_CAT, scale=True), True),
        "XGBoost (raw target)": (make_pipeline(_xgb(), INS_NUM, INS_CAT), False),
        "XGBoost (log target)": (make_pipeline(_xgb(), INS_NUM, INS_CAT), True),
    }
    t0 = time.time()
    if quick:
        cands = {k: v for k, v in cands.items() if k.startswith("Linear regression (raw")}
    if tuned_params is not None:
        tuned = make_pipeline(_xgb(), INS_NUM, INS_CAT).set_params(**tuned_params)
        best_params, cv_rmse = tuned_params, None
    else:
        with tracking.run("BayesSearchCV-tuning", tags={"event": event, "stage": "hpo"}):
            search = BayesSearchCV(make_pipeline(_xgb(), INS_NUM, INS_CAT), SPACE, n_iter=4 if fast else 25,
                                   cv=KFold(3, shuffle=True, random_state=config.SEED), scoring="neg_root_mean_squared_error",
                                   random_state=config.SEED, n_jobs=1)
            search.fit(Xtr, np.log(ytr))                                    # tuned on the log scale (log-RMSE)
            tuned = search.best_estimator_
            best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
            cv_rmse = float(-search.best_score_)
            tracking.log_params({"n_iter": 4 if fast else 25, **best_params})
            tracking.log_metrics({"cv_log_rmse": cv_rmse})
        print(f"[insurance] BayesSearchCV done in {time.time() - t0:.0f}s (CV log-RMSE {cv_rmse:.4f})")
    cands["XGBoost (log target, tuned)"] = (tuned, True)

    rows, fitted = [], {}
    for name, (pipe, log) in cands.items():
        with tracking.run(name, tags={"event": event, "log_target": log, "quick": quick, "fast": fast}):
            pipe.fit(Xtr, np.log(ytr) if log else ytr)
            inv = (lambda z: np.exp(z)) if log else (lambda z: z)
            mv, mt = _metrics(yva, inv(pipe.predict(Xva))), _metrics(yte, inv(pipe.predict(Xte)))
            fitted[name] = (pipe, log)
            row = {"model": name, **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}}
            rows.append(row)
            tracking.log_params({"log_target": log})
            tracking.log_metrics({k: v for k, v in row.items() if k != "model"})
            print(f"[insurance] {name:34s} val RMSE {mv['rmse']:8,.0f} R2 {mv['r2']:.3f} | test RMSE {mt['rmse']:8,.0f} R2 {mt['r2']:.3f} MAE {mt['mae']:7,.0f}")
    comp = pd.DataFrame(rows)

    xgb_rows = comp[comp["model"].str.startswith("XGBoost")]
    champ = xgb_rows.sort_values("val_rmse").iloc[0]["model"]
    base = comp[comp["model"].str.startswith("Linear")].sort_values("val_rmse").iloc[0]["model"]   # best linear on validation
    pipe, log = fitted[champ]

    # ---- 80% error range: quantile XGBoost on log-premium + conformal (CQR) widening from the validation window ----
    qpipe = make_pipeline(_xgb(objective="reg:quantileerror", quantile_alpha=np.array([*config.INSURANCE_INTERVAL][:1] + [0.5] + [*config.INSURANCE_INTERVAL][1:])),
                          INS_NUM, INS_CAT)
    qpipe.set_params(clf__n_estimators=300, clf__max_depth=3, clf__learning_rate=0.05, clf__min_child_weight=10)  # shallow: stable quantiles
    qpipe.fit(Xtr, np.log(ytr))
    qv = qpipe.predict(Xva)
    lv = np.log(yva.values)
    scores = np.maximum(qv[:, 0] - lv, lv - qv[:, 2])
    alpha = config.INSURANCE_INTERVAL[0] * 2

    def conformal(sc):
        return float(np.quantile(sc, min(1.0, (1 - alpha) * (1 + 1 / len(sc)))))
    cqr = {"all": conformal(scores)}
    for g in ("Yes", "No"):                                  # Mondrian (group-wise) calibration on the validation window
        mask = (Xva["smoker"] == g).values
        if mask.sum() >= 50:
            cqr[g] = conformal(scores[mask])
    model = PremiumModel(pipe, qpipe, cqr, INS_FEATURES, log)
    lo, hi = model.predict_interval(Xte)
    cover = float(((yte.values >= lo) & (yte.values <= hi)).mean())
    width = float(np.median((hi - lo) / model.predict(Xte)))
    cover_by = {g: float(((yte.values[m] >= lo[m]) & (yte.values[m] <= hi[m])).mean())
                for g in ("Yes", "No") for m in [(Xte["smoker"] == g).values] if m.sum()}
    print(f"[insurance] champion {champ} (baseline {base}) | 80% interval coverage on test {cover:.1%} (by smoker: {cover_by}), median relative width {width:.0%}")

    m_champ = comp[comp["model"] == champ].iloc[0]
    m_base = comp[comp["model"] == base].iloc[0]
    expl = Explainer(pipe, INS_FEATURES)
    importance = expl.contributions(Xte.sample(min(1500, len(Xte)), random_state=config.SEED)).abs().mean().sort_values(ascending=False)
    meta = {
        "name": champ, "baseline": base, "log_target": log, "best_params": best_params, "cqr_width": cqr,
        "features": INS_FEATURES, "num": INS_NUM, "cat": INS_CAT, "segment": "Health",
        "train_window": [str(tr["application_date"].min().date()), str(tr["application_date"].max().date())],
        "metrics": {"test": {"rmse": m_champ["test_rmse"], "mae": m_champ["test_mae"], "r2": m_champ["test_r2"],
                             "baseline_rmse": m_base["test_rmse"], "baseline_r2": m_base["test_r2"],
                             "rmse_improvement_vs_linear": 1 - m_champ["test_rmse"] / m_base["test_rmse"],
                             "interval_coverage_80": cover, "interval_median_rel_width": width,
                             "interval_coverage_by_smoker": cover_by},
                    "val": {"rmse": m_champ["val_rmse"], "r2": m_champ["val_r2"]}},
        "feature_importance": {k: float(v) for k, v in importance.head(12).items()},
    }
    if save:
        config.FIGURES.mkdir(parents=True, exist_ok=True)
        config.MODELS.mkdir(parents=True, exist_ok=True)
        comp.to_csv(config.REPORTS / "insurance_model_comparison.csv", index=False)
        tr[INS_FEATURES].sample(min(6000, len(tr)), random_state=config.SEED).to_parquet(config.MODELS / "insurance_reference.parquet")
        _plots(model, base_model=fitted[base], te=te, lo=lo, hi=hi, importance=importance, champ=champ)
        meta = registry.save_champion("insurance", model, meta, event)
        with tracking.run(f"champion-{champ}", tags={"role": "champion", "event": event,
                                                      "registry_version": meta["version"]}):
            tracking.log_params({"baseline": base, "log_target": log})
            tracking.log_metrics({f"test_{k}": v for k, v in meta["metrics"]["test"].items()}
                                 | {f"val_{k}": v for k, v in meta["metrics"]["val"].items()})
            # The champion is PremiumModel = point pipeline + quantile pipeline + Mondrian-CQR widths (plain
            # dict, not an sklearn estimator). Log the two pipelines that ARE sklearn-compatible individually;
            # the composite is reconstructed from models/*.joblib + registry.json at serve time, not from here.
            tracking.log_sklearn_model(pipe, artifact_path="point_pipeline", registered_model_name="insurance-champion-point")
            tracking.log_sklearn_model(qpipe, artifact_path="quantile_pipeline")
    reference = tr[INS_FEATURES].sample(min(6000, len(tr)), random_state=config.SEED)
    return {"comparison": comp, "meta": meta, "model": model, "reference": reference, "val": va}


def _plots(model, base_model, te, lo, hi, importance, champ):
    plt.rcParams.update({"figure.facecolor": "#051F20", "axes.facecolor": "#051F20", "axes.edgecolor": "#235347",
                         "text.color": "#DAF1DE", "axes.labelcolor": "#DAF1DE", "xtick.color": "#9DBFA8",
                         "ytick.color": "#9DBFA8", "grid.color": "#163832"})
    y = te["annual_premium_inr"].values
    p = model.predict(te[INS_FEATURES])
    bp, blog = base_model
    pb = np.exp(bp.predict(te[INS_FEATURES])) if blog else bp.predict(te[INS_FEATURES])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, pred, ttl, col in [(axs[0], pb, "Linear baseline", "#E3C16F"), (axs[1], p, "XGBoost (champion)", "#8EB69B")]:
        ax.scatter(y, pred, s=5, alpha=.45, color=col)
        ax.plot([0, y.max()], [0, y.max()], "--", color="#235347")
        ax.set(xlabel="actual premium (Rs)", ylabel="predicted premium (Rs)", title=ttl, xlim=(0, np.quantile(y, .995)), ylim=(0, np.quantile(y, .995)))
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(config.FIGURES / "insurance_pred_vs_actual.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    imp = importance.head(10)[::-1]
    ax.barh(imp.index, imp.values, color="#8EB69B")
    ax.set(xlabel="mean |SHAP| (log premium)", title="What drives the premium"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(config.FIGURES / "insurance_shap_importance.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    train_insurance(fast=ap.parse_args().fast)
