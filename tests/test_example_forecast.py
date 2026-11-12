"""Forecasting on the bundled daily-sales fixture: the split is the model's only honest defence.

These tests encode the cheapest senior habits: never randomly split a time series, always backtest
with an expanding window, and always put a naive/seasonal-naive baseline next to the fancy model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dstk import leakage as L
from dstk import timeseries as T

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
HORIZON, STEP, SEASON = 30, 30, 7


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    df = pd.read_csv(DATA / "sales_daily.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def test_the_series_is_time_ordered_with_weekly_seasonality(sales):
    assert len(sales) == 731 and sales["date"].is_monotonic_increasing      # 2024-01-01 .. 2025-12-31
    weekday_mean = sales.groupby(sales["date"].dt.dayofweek)["sales"].mean()
    assert weekday_mean[[4, 5]].min() > weekday_mean[[0, 1, 6]].max() * 1.15   # Fri/Sat peak
    assert sales["promo"].mean() < 0.10                                       # promos are rare


def test_random_split_of_a_time_series_is_rejected(sales):
    shuffled = sales.sample(frac=1.0, random_state=0)
    train, test = shuffled.iloc[:560], shuffled.iloc[560:]
    with pytest.raises(AssertionError, match="not before"):
        L.assert_temporal_order(train["date"], test["date"])
    L.assert_temporal_order(sales["date"].iloc[:560], sales["date"].iloc[560:])


def test_backtest_windows_expand_and_never_look_past_the_origin(sales):
    splits = T.rolling_origin_splits(len(sales), initial=len(sales) - 90, horizon=HORIZON, step=STEP)
    assert len(splits) == 3
    assert [len(tr) for tr, _ in splits] == [641, 671, 701]
    for tr, te in splits:
        assert tr.max() < te.min() and te.size == HORIZON


def test_seasonal_naive_beats_naive_in_every_backtest_window(sales):
    y = sales["sales"].to_numpy()
    rows = []
    for tr, te in T.rolling_origin_splits(len(y), initial=len(y) - 90, horizon=HORIZON, step=STEP):
        history, actual = y[tr], y[te]
        f_seasonal = T.seasonal_naive(history, HORIZON, season=SEASON)
        f_naive = T.seasonal_naive(history, HORIZON, season=1)
        # ONE scale for both forecasts: MASE's denominator is the in-sample error of the SEASONAL naive
        # benchmark. Recomputing it per forecast (season=1 for one, season=7 for the other) silently
        # changes the benchmark and can flip the ranking -- a classic, easy-to-miss review finding.
        rows.append((T.mase(actual, f_seasonal, history, season=SEASON),
                     T.mase(actual, f_naive, history, season=SEASON),
                     T.wape(actual, f_seasonal), T.wape(actual, f_naive)))
    seasonal, naive = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print("MASE seasonal", np.round(seasonal, 3), "naive", np.round(naive, 3))
    assert (seasonal < naive).all(), "the weekly season is the whole signal: a naive forecast must lose"
    assert seasonal.mean() < 0.8 * naive.mean()
    assert all(w_sn < w_nv for *_, w_sn, w_nv in rows)


def test_the_fixture_is_reported_on_the_metrics_a_review_will_ask_for(sales):
    y = sales["sales"].to_numpy()
    history, actual = y[:600], y[600:]
    forecast = T.seasonal_naive(history, len(actual), season=SEASON)
    mase = T.mase(actual, forecast, history, season=SEASON)
    print(f"MASE {mase:.3f}  WAPE {T.wape(actual, forecast):.2f}%  sMAPE {T.smape(actual, forecast):.2f}%")
    assert 0.5 < mase < 1.5          # beats the seasonal-naive in-sample scale, but not by magic
    assert 0 < T.wape(actual, forecast) < 15
    assert 0 < T.smape(actual, forecast) < 15


def test_smape_and_wape_disagree_when_actuals_are_near_zero():
    """Two "percentage errors" that are not interchangeable: sMAPE explodes near zero, WAPE does not.
    Quoting one without the other is how a forecast report becomes unauditable."""
    y_true, y_pred = [0.5, 1000.0], [2.0, 900.0]
    assert T.smape(y_true, y_pred) > 5 * T.wape(y_true, y_pred)
