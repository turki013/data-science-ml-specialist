"""Example "project hooks" on a synthetic housing-style problem.

HOW TO USE
    1. Copy this file to `project_hooks.py` at your repository root (next to pytest.ini).
    2. Replace each function with code that loads YOUR split, features, model and frozen test set.
    3. Run `pytest`. Every test runs against these example hooks AND your project hooks
       (parametrized as `example` / `project`). A hook you do not implement makes its tests skip.

HOOK CONTRACT (all optional)
    get_split()            -> dict(train, test, feature_cols, target, task, group_col=None,
                                   time_col=None, temporal_split=False, max_overlap_share=0.0)
    get_cv()               -> dict(splits=[(train_idx, val_idx), ...], groups=None|array, times=None|array)
    get_raw_sample()       -> DataFrame of RAW serving-time inputs (no target)
    bulk_features(df)      -> DataFrame, the vectorized TRAINING feature path
    single_features(rec)   -> dict, the SERVING feature path for one record
    load_pipeline()        -> fitted estimator with .predict(raw_df)
    get_frozen_test()      -> (X_raw DataFrame, y array); never changes between runs
    expected_frozen_fingerprint() -> str, a HARDCODED sha256 literal of the frozen test set
                           (canonical_fingerprint). Never compute it from the same file at check time.
    baseline_predict(X)    -> predictions of the reference baseline
    per_item_loss(y, pred) -> array of per-item losses, lower is better
    regression_config()    -> dict(max_mean_loss=float, min_median_gain=float)
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from pathlib import Path

SEED = 7
DISTRICTS = ["A", "B", "C", "D"]
RAW = ["size", "rooms", "district"]
TARGET = "price"


def _simulate(n_owners: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d_eff = dict(zip(DISTRICTS, [0.0, 0.15, 0.35, -0.1]))
    rows = []
    for o in range(n_owners):
        owner_eff = rng.normal(0, 0.08)
        for _ in range(int(rng.integers(3, 8))):
            size = float(rng.uniform(80, 400))
            rooms = int(rng.integers(2, 8))
            district = str(rng.choice(DISTRICTS))
            log_p = 11.0 + 0.004 * size + 0.04 * rooms + d_eff[district] + owner_eff + rng.normal(0, 0.12)
            rows.append((f"o{o}", size, rooms, district, float(np.exp(log_p)),
                         pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 400)))))
    return pd.DataFrame(rows, columns=["owner", "size", "rooms", "district", TARGET, "listed_on"])


def bulk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized TRAINING path."""
    out = df[RAW].copy()
    out["size_per_room"] = out["size"] / out["rooms"]
    out["log_size"] = np.log(out["size"])
    return out


def single_features(record: dict) -> dict:
    """Independent SERVING path for one record (kept separate on purpose so parity is a real test)."""
    size, rooms = float(record["size"]), float(record["rooms"])
    return {"size": size, "rooms": rooms, "district": record["district"],
            "size_per_room": size / rooms, "log_size": math.log(size)}


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["district"])],
                            remainder="passthrough")
    return Pipeline([("features", FunctionTransformer(bulk_features)),
                     ("prep", pre),
                     ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                                              random_state=SEED))])


@lru_cache(maxsize=1)
def get_split() -> dict:
    df = _simulate(260, SEED)
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df["owner"]))
    return {"train": df.iloc[tr_i].reset_index(drop=True), "test": df.iloc[te_i].reset_index(drop=True),
            "feature_cols": RAW, "target": TARGET, "task": "regression", "group_col": "owner",
            "time_col": "listed_on", "temporal_split": False, "max_overlap_share": 0.0}


@lru_cache(maxsize=1)
def get_cv() -> dict:
    tr = get_split()["train"]
    splits = list(GroupKFold(5).split(tr, groups=tr["owner"]))
    return {"splits": splits, "groups": tr["owner"].to_numpy(), "times": None}


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    s = get_split()
    return _build_pipeline().fit(s["train"][RAW], s["train"][TARGET])


def get_raw_sample() -> pd.DataFrame:
    return get_split()["test"][RAW].head(200).reset_index(drop=True)


FROZEN_TEST_PATH = Path(__file__).resolve().parent / "data" / "frozen_test.csv"
# Reference value, committed as a literal. It was produced ONCE by `python examples/make_frozen_test.py`
# (canonical_fingerprint over RAW columns + target). If this test fails, the frozen set changed: investigate
# the change; update this constant only as a deliberate, reviewed decision.
FROZEN_TEST_SHA256 = "4eec933127143de1ffef6c7502adedbb745ad52f59d2a36673d6e33dbff97833"


@lru_cache(maxsize=1)
def _frozen() -> tuple:
    df = pd.read_csv(FROZEN_TEST_PATH, float_precision="round_trip")
    return df[RAW].reset_index(drop=True), df[TARGET].to_numpy()


def get_frozen_test() -> tuple:
    return _frozen()


def expected_frozen_fingerprint() -> str:
    return FROZEN_TEST_SHA256


@lru_cache(maxsize=1)
def _baseline() -> DummyRegressor:
    s = get_split()
    return DummyRegressor(strategy="median").fit(s["train"][RAW], s["train"][TARGET])


def baseline_predict(X: pd.DataFrame) -> np.ndarray:
    return _baseline().predict(X)


def per_item_loss(y_true, y_pred) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.abs(y_pred - y_true) / y_true  # absolute percentage error


def regression_config() -> dict:
    # Set max_mean_loss a little above the accepted model's measured mean loss so that real
    # regressions fail the test but run-to-run noise does not.
    return {"max_mean_loss": 0.20, "min_median_gain": 0.02}
