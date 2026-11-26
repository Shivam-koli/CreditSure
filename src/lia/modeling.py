"""Shared modelling pieces: preprocessing, pipelines, probability calibration, SHAP explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def make_preprocessor(num: list[str], cat: list[str], scale: bool) -> ColumnTransformer:
    """Imputation lives here (not in cleaning) so it is fitted on the training rows only."""
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                 ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", Pipeline(cat_steps), cat)])


def make_pipeline(clf, num, cat, smote: bool = False, scale: bool = False):
    pre = make_preprocessor(num, cat, scale)
    if smote:   # SMOTE sits inside the pipeline -> it only ever sees the training fold
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=config.SEED)), ("clf", clf)])
    return Pipeline([("pre", pre), ("clf", clf)])


class PlattCalibrator:
    """Maps raw scores to calibrated probabilities (logistic regression on the logit of the score).
    Needed because SMOTE / class weights distort probabilities and the money rule needs real PDs."""

    def fit(self, p, y):
        z = self._logit(p).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def transform(self, p):
        return self.lr.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]


class ScoreModel:
    """Fitted pipeline + calibrator. predict_proba returns the calibrated probability of the positive class."""

    def __init__(self, pipe, calibrator: PlattCalibrator | None, features: list[str]):
        self.pipe, self.calibrator, self.features = pipe, calibrator, features

    def raw(self, X):
        return self.pipe.predict_proba(X[self.features])[:, 1]

    def predict_proba(self, X):
        p = self.raw(X)
        return self.calibrator.transform(p) if self.calibrator else p


# ----------------------------------------------------------------------------------------------
# SHAP: explanations aggregated back to the original (business) features
# ----------------------------------------------------------------------------------------------
def origin_feature(name: str, features: list[str]) -> tuple[str, bool]:
    """'num__missingindicator_cibil_score' -> ('cibil_score', True); 'cat__loan_purpose_Home Loan' -> ('loan_purpose', False)."""
    n = name.split("__", 1)[-1]
    indicator = n.startswith("missingindicator_")
    if indicator:
        n = n[len("missingindicator_"):]
    best = max((f for f in features if n == f or n.startswith(f + "_")), key=len, default=n)
    return best, indicator


class Explainer:
    def __init__(self, pipe, features: list[str]):
        import shap
        self.pipe, self.features = pipe, features
        self.pre, self.clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        self.names = list(self.pre.get_feature_names_out())
        self.origin = [origin_feature(n, features) for n in self.names]
        self.linear = isinstance(self.clf, LogisticRegression)
        self.tree = None if self.linear else shap.TreeExplainer(self.clf)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row contribution of each ORIGINAL feature, in model margin units (log-odds for classifiers)."""
        Xt = self.pre.transform(X[self.features])
        if self.linear:
            vals = Xt * self.clf.coef_[0]
        else:
            vals = self.tree.shap_values(Xt)
            if isinstance(vals, list):
                vals = vals[-1]
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[:, :, -1]
        df = pd.DataFrame(vals, columns=[o[0] for o in self.origin])
        return df.T.groupby(level=0).sum().T[self.features]

    def base_value(self) -> float:
        if self.linear:
            return float(self.clf.intercept_[0])
        ev = self.tree.expected_value
        return float(np.ravel(ev)[-1])


class PremiumModel:
    """Point prediction + 80% interval. Models work on log(premium); outputs are in rupees.
    The interval is a quantile-XGBoost band widened/shrunk by conformal calibration done SEPARATELY for
    smokers and non-smokers (Mondrian CQR), because their premium noise is very different."""

    def __init__(self, point, q_pipe, cqr, features, log: bool, group_col: str = "smoker"):
        self.point, self.q_pipe, self.cqr, self.features, self.log, self.group_col = point, q_pipe, cqr, features, log, group_col

    def predict(self, X) -> np.ndarray:
        p = self.point.predict(X[self.features])
        return np.exp(p) if self.log else p

    def _adjust(self, X) -> np.ndarray:
        if not isinstance(self.cqr, dict):
            return np.full(len(X), float(self.cqr))
        g = X[self.group_col].astype(object).fillna("No")
        return g.map(lambda v: self.cqr.get(v, self.cqr["all"])).astype(float).values

    def predict_interval(self, X):
        q = self.q_pipe.predict(X[self.features])            # columns: lo, median, hi (log scale)
        adj = self._adjust(X)
        lo, hi = q[:, 0] - adj, q[:, 2] + adj
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        pt = self.point.predict(X[self.features])
        lo, hi = np.minimum(lo, pt), np.maximum(hi, pt)      # the interval must contain the point estimate
        return np.exp(lo), np.exp(hi)
