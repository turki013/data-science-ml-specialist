"""A/B experiment on the bundled fixture: clean assignment, a SIMULATED treatment effect, and a guardrail that moves
the wrong way.

The senior point is not "the test was significant". It is that significance on the primary metric
is only half the decision: the guardrail, the power estimate and the variance reduction (CUPED)
all belong in the same paragraph as the p-value.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dstk import experiments as E
from dstk.stats import ab_sample_size_two_proportions

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def ab() -> pd.DataFrame:
    return pd.read_csv(DATA / "experiment_ab.csv")


@pytest.fixture(scope="module")
def arms(ab):
    return ab[ab["group"] == "control"], ab[ab["group"] == "treatment"]


def test_assignment_is_balanced_and_srm_is_not_triggered(ab, arms):
    control, treatment = arms
    assert len(control) == len(treatment) == 2000
    assert ab["user_id"].is_unique
    assert E.srm_check([len(control), len(treatment)], [1, 1])["srm_detected"] is False


def test_primary_metric_has_a_significant_lift_with_an_interval(arms):
    control, treatment = arms
    r = E.two_proportion_test(int(control["converted"].sum()), len(control),
                              int(treatment["converted"].sum()), len(treatment))
    print(f"control {r['p1']:.4f} treatment {r['p2']:.4f} diff {r['diff']:+.4f} "
          f"CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p_value']:.4f}")
    assert r["p_value"] < 0.05 and r["ci_lo"] > 0        # planted +2pp treatment effect (synthetic data)
    assert 0.015 < r["diff"] < 0.035


def test_guardrail_moves_the_wrong_way_and_must_be_reported(arms):
    control, treatment = arms
    ratio = treatment["support_tickets"].mean() / control["support_tickets"].mean()
    print(f"support tickets per user: treatment/control = {ratio:.2f}")
    assert ratio > 1.15, "the fixture plants a guardrail regression; a review that ignores it is incomplete"


def test_cuped_reduces_variance_without_moving_the_mean(ab, arms):
    control, treatment = arms
    raw_gap = treatment["week1_minutes"].mean() - control["week1_minutes"].mean()
    adjusted, theta, reduction = E.cuped_adjust(ab["week1_minutes"].to_numpy(float),
                                                ab["pre_spend"].to_numpy(float))
    frame = ab.assign(week1_minutes_adj=adjusted)
    adj_gap = (frame.loc[frame["group"] == "treatment", "week1_minutes_adj"].mean()
               - frame.loc[frame["group"] == "control", "week1_minutes_adj"].mean())
    print(f"CUPED theta {theta:.4f}, variance reduction {reduction:.1%}, gap {raw_gap:+.3f} -> {adj_gap:+.3f}")
    assert theta == pytest.approx(0.08, abs=0.02)        # the covariate is pre-period spend
    assert reduction > 0.5                              # variance reduction is the point of CUPED
    assert adjusted.mean() == pytest.approx(ab["week1_minutes"].mean(), abs=1e-6)
    # CUPED is allowed to move the point estimate a little: it removes the small covariate imbalance
    # between the arms (0.9 SAR of pre_spend here). A LARGE move would mean the covariate is post-treatment.
    assert adj_gap == pytest.approx(raw_gap, abs=0.15)


def test_power_estimate_says_the_fixture_is_borderline_and_that_matters(ab, arms):
    control, treatment = arms
    need = ab_sample_size_two_proportions(control["converted"].mean(), treatment["converted"].mean())
    print(f"80% power would have needed ~{need} users per arm; the fixture has {len(control)}")
    assert 1500 <= need <= 3200
    # The observed p-value is below 0.05 while n is under the 80%-power estimate: possible, and exactly
    # why an under-powered result should be read as "promising", not "settled".
    assert need > len(control) * 0.9


def test_multiple_looks_would_need_correction(arms):
    control, treatment = arms
    p_conv = E.two_proportion_test(int(control["converted"].sum()), len(control),
                                   int(treatment["converted"].sum()), len(treatment))["p_value"]
    p_tickets = E.two_proportion_test(int(treatment["support_tickets"].gt(0).sum()), len(treatment),
                                      int(control["support_tickets"].gt(0).sum()), len(control))["p_value"]
    adj, reject = E.holm_bonferroni([p_conv, p_tickets])
    assert reject[0] is True or adj[0] < 0.05 * 2, "the primary metric should survive a two-metric correction"
    assert np.all(np.asarray(adj) >= np.asarray([p_conv, p_tickets]) - 1e-12)
