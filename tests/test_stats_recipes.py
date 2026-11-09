import math

import numpy as np
import pytest

from dstk import stats as S


def test_wilson_interval_known_values():
    lo, hi = S.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(0.2775, abs=2e-3)
    lo, hi = S.wilson_interval(50, 100)
    assert lo == pytest.approx(1 - hi, abs=1e-9)  # symmetric around 0.5
    assert 0.40 < lo < 0.42 and 0.58 < hi < 0.60
    with pytest.raises(ValueError):
        S.wilson_interval(5, 0)


def test_bootstrap_ci_covers_true_mean_is_reproducible_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 400)
    est, lo, hi = S.bootstrap_ci(x, np.mean, seed=1)
    assert lo < 10 < hi and lo < est < hi
    assert S.bootstrap_ci(x, np.mean, seed=1) == (est, lo, hi)
    _, lo2, hi2 = S.bootstrap_ci(x[:40], np.mean, seed=1)
    assert (hi2 - lo2) > (hi - lo)


def test_bootstrap_ci_handles_nan_and_empty():
    est, lo, hi = S.bootstrap_ci([1.0, 2.0, np.nan, 3.0], np.median)
    assert est == 2.0
    with pytest.raises(ValueError):
        S.bootstrap_ci([np.nan])


def test_paired_comparison_verdicts():
    rng = np.random.default_rng(3)
    b = rng.uniform(0.2, 0.5, 60)
    assert S.paired_comparison(b + rng.normal(0, 0.001, 60), b)["verdict"] == "no clear difference"
    assert S.paired_comparison(b - 0.08 + rng.normal(0, 0.01, 60), b)["verdict"] == "A better"
    assert S.paired_comparison(b + 0.08 + rng.normal(0, 0.01, 60), b)["verdict"] == "B better"


def test_paired_comparison_small_noisy_sample_is_inconclusive():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.3, 0.1, 10), rng.normal(0.3, 0.1, 10)
    r = S.paired_comparison(a, b)
    assert r["verdict"] == "no clear difference" and r["n"] == 10
    assert r["wins_a"] + r["wins_b"] + r["ties"] == 10


def test_paired_comparison_identical_inputs_and_shape_check():
    x = np.linspace(0.1, 0.5, 20)
    r = S.paired_comparison(x, x)
    assert r["verdict"] == "no clear difference" and r["sign_p"] == 1.0 and r["wilcoxon_p"] == 1.0
    with pytest.raises(ValueError):
        S.paired_comparison([1, 2], [1, 2, 3])


def test_within_threshold_report_counts_and_intervals():
    err = [0.05, 0.10, 0.20, 0.30, 0.40, -0.12]
    rep = S.within_threshold_report(err, thresholds=(0.15, 0.25))
    assert rep[0.15]["k"] == 3 and rep[0.15]["n"] == 6
    assert rep[0.25]["k"] == 4
    assert rep[0.15]["ci_lo"] < rep[0.15]["share"] < rep[0.15]["ci_hi"]


def test_error_summary_median_is_robust_to_outlier():
    s = S.error_summary([0.1, 0.1, 0.1, 0.1, 5.0])
    assert s["median"] == pytest.approx(0.1) and s["mean"] > 1.0 and s["n"] == 5


def test_ab_sample_size_matches_textbook_value():
    n = S.ab_sample_size_two_proportions(0.10, 0.12)
    assert 3800 <= n <= 3900
    assert S.ab_sample_size_two_proportions(0.10, 0.11) > n
    with pytest.raises(ValueError):
        S.ab_sample_size_two_proportions(0.1, 0.1)
