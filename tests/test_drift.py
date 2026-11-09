import numpy as np
import pandas as pd
import pytest

from dstk import drift as D


def test_psi_is_small_for_same_distribution_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert D.psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert D.psi(ref, rng.normal(0.5, 1, 5000)) > 0.10
    assert D.psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_psi_status_thresholds():
    assert D.psi_status(0.05) == "stable"
    assert D.psi_status(0.15) == "moderate shift"
    assert D.psi_status(0.40) == "major shift"


def test_psi_handles_constant_reference_and_unseen_categories():
    assert D.psi([1.0] * 100, [1.0] * 100) == pytest.approx(0.0, abs=1e-9)
    assert D.psi([1.0] * 100, [2.0] * 100) > 0.25
    assert D.psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0.0, abs=1e-9)
    assert D.psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["c"] * 50) > 0.25


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        D.psi([], [1, 2, 3])


def test_psi_report_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000),
                        "c": rng.choice(["x", "y"], 2000)})
    cur = pd.DataFrame({"a": rng.normal(1.5, 1, 2000), "b": rng.normal(0, 1, 2000),
                        "c": rng.choice(["x", "y"], 2000)})
    rep = D.psi_report(ref, cur, ["a", "b"], ["c"])
    assert rep.iloc[0]["feature"] == "a" and rep.iloc[0]["status"] == "major shift"
    assert rep[rep["feature"] == "b"].iloc[0]["status"] == "stable"


def test_ks_report_flags_shift():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": rng.normal(1, 1, 1000), "b": rng.normal(0, 1, 1000)})
    rep = D.ks_report(ref, cur, ["a", "b"])
    assert rep.iloc[0]["feature"] == "a" and rep.iloc[0]["p_value"] < 0.001
