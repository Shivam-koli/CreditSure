import numpy as np
import pandas as pd

from lia.drift import drift_report, psi_categorical, psi_numeric
from lia.metrics import breakeven_pd, economics, net_value


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi_numeric(a, pd.Series(rng.normal(0.8, 1, 5000))) > 0.25
    assert psi_categorical(pd.Series(["a"] * 70 + ["b"] * 30), pd.Series(["a"] * 30 + ["b"] * 70)) > 0.25


def test_drift_report_flags_alert():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "c": rng.choice(["u", "v"], 2000)})
    cur = pd.DataFrame({"x": rng.normal(1.0, 1, 2000), "c": rng.choice(["u", "v"], 2000)})
    rep = drift_report(ref, cur, ["x"], ["c"])
    assert rep["overall"] == "alert" and rep["top_drifting"][0] == "x"


def test_money_rule_matches_breakeven():
    df = pd.DataFrame({"loan_amount_inr": [100000.0], "interest_rate_pct": [13.5], "tenure_months": [36]})
    e = economics(df, recovery_ratio=0.25)
    be = breakeven_pd(e)[0]
    # at exactly the break-even PD the expected value of approving is zero
    assert abs((1 - be) * e["margin"].iloc[0] - be * e["loss"].iloc[0]) < 1e-6
    assert net_value([0], [True], e) > 0 and net_value([1], [True], e) < 0 and net_value([1], [False], e) == 0
