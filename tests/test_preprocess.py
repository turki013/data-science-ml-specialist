import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from dstk.preprocess import QuantileWinsorizer, fit_iqr_bounds, outlier_flags, outlier_report, robust_zscore


def test_winsorizer_learns_bounds_on_train_only_and_keeps_rows():
    train = pd.DataFrame({"x": np.arange(100.0)})
    test = pd.DataFrame({"x": [-500.0, 50.0, 10_000.0]})
    w = QuantileWinsorizer(0.05, 0.95).fit(train)
    lo, hi = w.lo_[0], w.hi_[0]
    out = w.transform(test)
    assert out.shape == (3, 1) and out[0, 0] == lo and out[2, 0] == hi and out[1, 0] == 50.0
    assert w.lo_[0] == lo and w.hi_[0] == hi  # transform must not alter learned bounds


def test_winsorizer_inside_pipeline_cross_validates():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    X.iloc[0, 0] = 1e6
    y = X["b"] * 2 + rng.normal(scale=0.1, size=200)
    scores = cross_val_score(make_pipeline(QuantileWinsorizer(), Ridge()), X, y, cv=5, scoring="r2")
    assert scores.mean() > 0.9


def test_iqr_bounds_flags_and_report_use_train_statistics():
    train = pd.DataFrame({"x": list(range(1, 101))})
    other = pd.DataFrame({"x": [1, 50, 100, 1000]})
    b = fit_iqr_bounds(train, ["x"])
    flags = outlier_flags(other, b)
    assert flags["x"].tolist() == [False, False, False, True]
    rep = outlier_report(train, other, ["x"])
    assert rep.loc["x", "other_n"] == 1 and rep.loc["x", "train_n"] == 0


def test_robust_zscore_is_not_distorted_by_one_extreme_value():
    x = pd.Series([10, 11, 9, 10, 12, 10, 1000.0])
    z = robust_zscore(x)
    assert z.iloc[-1] > 100 and abs(z.iloc[:-1]).max() < 3
    assert (robust_zscore(pd.Series([5.0, 5.0, 5.0])) == 0).all()
