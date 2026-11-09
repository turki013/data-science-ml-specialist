import numpy as np
import pytest

from dstk import experiments as E
from dstk import timeseries as T


def test_seasonal_naive_and_metrics_by_hand():
    y = [1, 2, 3, 4, 5, 6, 7]
    assert T.seasonal_naive(y, 5, season=1).tolist() == [7, 7, 7, 7, 7]
    assert T.seasonal_naive(y, 5, season=3).tolist() == [5, 6, 7, 5, 6]
    assert T.wape([100, 100], [90, 120]) == pytest.approx(100 * 30 / 200)
    assert T.smape([100], [100]) == 0.0 and T.smape([0], [0]) == 0.0
    # naive in-sample MAE of 1..7 is 1, so MASE equals the forecast MAE
    assert T.mase([8, 9], [7, 7], y) == pytest.approx((1 + 2) / 2)
    with pytest.raises(ValueError):
        T.mase([1], [1], [5, 5, 5])


def test_rolling_origin_splits_are_expanding_and_never_overlap_future():
    sp = T.rolling_origin_splits(100, initial=60, horizon=10)
    assert len(sp) == 4
    for tr, te in sp:
        assert tr.max() < te.min() and te.size == 10 and tr[0] == 0
    assert [len(tr) for tr, _ in sp] == [60, 70, 80, 90]
    with pytest.raises(ValueError):
        T.rolling_origin_splits(10, initial=10, horizon=5)


def test_two_proportion_test_detects_real_difference_and_not_noise():
    r = E.two_proportion_test(100, 1000, 140, 1000)
    assert r["p_value"] < 0.01 and r["ci_lo"] > 0
    r0 = E.two_proportion_test(100, 1000, 104, 1000)
    assert r0["p_value"] > 0.5 and r0["ci_lo"] < 0 < r0["ci_hi"]


def test_srm_check_fires_on_broken_assignment():
    assert E.srm_check([5000, 5020], [1, 1])["srm_detected"] is False
    assert E.srm_check([5000, 5600], [1, 1])["srm_detected"] is True


def test_cuped_reduces_variance_without_shifting_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 20, 5000)
    y = 0.8 * x + rng.normal(0, 5, 5000)
    adj, theta, vr = E.cuped_adjust(y, x)
    assert theta == pytest.approx(0.8, abs=0.05) and vr > 0.8
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-6)


def test_multiple_testing_corrections_by_hand():
    p = [0.01, 0.04, 0.03, 0.005]
    adj, rej = E.holm_bonferroni(p)
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]
    bh, rej_bh = E.benjamini_hochberg(p)
    assert bh.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert rej_bh.tolist() == [True, True, True, True]
