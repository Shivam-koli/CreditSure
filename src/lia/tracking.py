"""Thin, fail-soft MLflow wrapper: experiment tracking + model registry for every training run
(loan and insurance), without an always-on extra container.

Why SQLite and not a running `mlflow server`: MLflow 3.x's plain filesystem store is deprecated
("maintenance mode") in favour of a database backend, and `mlflow server` now starts several worker
processes plus background job schedulers (trace archival, issue detection, online scoring) meant for
a shared team deployment - unnecessary weight and a stability risk on a single-user, zero-cost local
project that has already hit Docker disk/resource limits once (see README). A `sqlite:///...`
tracking URI gives the same runs/params/metrics/artifacts/Model Registry, written directly by the
training process with no server to keep alive. To browse it, run on demand (not a 24/7 container):
    docker compose --profile tools run --rm mlflow-ui     # http://localhost:5000
    # or, without Docker:  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

Every function here fails soft: if the tracking store can't be reached or written (e.g. a locked
SQLite file from a concurrent run), a training run still completes and simply isn't logged. Metrics
you can already see in reports/*.csv and models/registry.json are the source of truth; MLflow is an
additional, queryable history of every attempt, not a dependency of the pipeline.
"""
from __future__ import annotations

import contextlib
import os

from . import config

_ENABLED = True
_mlflow = None


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow' / 'mlflow.db'}")


def init(experiment: str) -> None:
    """Call once per script. Safe to call again (e.g. loan_train then insurance_train in one process)."""
    global _ENABLED, _mlflow
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # a local portfolio project phones no one home
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")     # quiet the coding-agent banner in plain logs
    try:
        import mlflow
        (config.ROOT / "mlflow").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(_tracking_uri())
        client = mlflow.MlflowClient()
        if client.get_experiment_by_name(experiment) is None:
            # explicit artifact_location -> logged models land under ROOT/mlflow/artifacts, next to
            # the sqlite db, regardless of the training script's CWD (mlflow's own default is ./mlruns
            # relative to CWD, which would scatter artifacts depending on where `python -m lia.X` is run from)
            artifacts = (config.ROOT / "mlflow" / "artifacts").resolve()
            client.create_experiment(experiment, artifact_location=f"file:{artifacts}")
        mlflow.set_experiment(experiment)
        _mlflow = mlflow
        _ENABLED = True
    except Exception as e:  # pragma: no cover - defensive: training must not depend on this
        _ENABLED = False
        print(f"[mlflow] tracking disabled ({e.__class__.__name__}: {e}); continuing without it")


@contextlib.contextmanager
def run(name: str, tags: dict | None = None):
    if not _ENABLED:
        yield None
        return
    try:
        with _mlflow.start_run(run_name=name) as r:
            if tags:
                with contextlib.suppress(Exception):
                    _mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield r
    except Exception as e:  # pragma: no cover
        print(f"[mlflow] run '{name}' not logged ({e.__class__.__name__}: {e})")
        yield None


def log_params(params: dict) -> None:
    if not (_ENABLED and params):
        return
    with contextlib.suppress(Exception):
        _mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict) -> None:
    if not (_ENABLED and metrics):
        return
    with contextlib.suppress(Exception):
        _mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})


# mlflow's sklearn flavour serializes with skops, which refuses to *load* a pickle containing a type
# it doesn't recognize as safe, UNLESS that type was explicitly trusted at save time. Our pipelines can
# contain any of these (XGBoost's and imblearn's own classes aren't in skops' default trusted set) -
# found by actually trying to log each candidate type, not by reading the skops docs in the abstract.
_SKOPS_TRUSTED = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier", "xgboost.sklearn.XGBRegressor",
                  "imblearn.pipeline.Pipeline", "imblearn.over_sampling._smote.base.SMOTE", "sklearn.tree._tree.Tree"]


def log_sklearn_model(model, artifact_path: str, registered_model_name: str | None = None) -> None:
    if not _ENABLED:
        return
    with contextlib.suppress(Exception):
        import mlflow.sklearn
        kwargs = dict(registered_model_name=registered_model_name, skops_trusted_types=_SKOPS_TRUSTED)
        try:  # mlflow >= ~2.13 prefers `name=`; older versions only have `artifact_path=`
            mlflow.sklearn.log_model(model, name=artifact_path, **kwargs)
        except TypeError:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path, **kwargs)
