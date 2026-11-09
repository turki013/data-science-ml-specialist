"""Model-selection rule, split-leakage gap and Senior/Lite mode trigger."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from dstk import mode as M
from dstk.selection import RULE_TEXT, finalize_selection, select_model
from dstk.splitgap import assert_no_reversed_gap, split_leakage_gap


# ----------------------------------------------------------------------------------- selection rule
def _losses(seed=0, n=300):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.4, n)
    return base, rng


def test_rule_prefers_the_simpler_model_when_the_best_is_not_clearly_better():
    base, rng = _losses()
    oof = {"ridge": base + 0.0,
           "gbr": base - 0.002 + rng.normal(0, 0.05, base.size),        # slightly lower mean, noisy, not clear
           "dummy": base + 0.25}
    sel = select_model(oof, {"dummy": 0, "ridge": 1, "gbr": 3})
    assert sel["best_cv"] in {"gbr", "ridge"}
    assert sel["rule_choice"] == "ridge" and "dummy" not in sel["equivalent_to_best"]
    assert "Among best and its equivalents" in sel["rule"] and sel["rule"] == RULE_TEXT


def test_rule_chooses_the_complex_model_when_it_is_clearly_better():
    base, rng = _losses(1)
    oof = {"ridge": base, "gbr": base - rng.uniform(0.04, 0.10, base.size)}
    sel = select_model(oof, {"ridge": 1, "gbr": 3})
    assert sel["best_cv"] == "gbr" and sel["equivalent_to_best"] == ["gbr"] and sel["rule_choice"] == "gbr"
    assert sel["comparisons_vs_best"]["ridge"]["verdict"] == "A better"


def test_deviation_from_the_rule_requires_a_written_reason():
    base, rng = _losses(2)
    oof = {"ridge": base, "gbr": base - 0.001 + rng.normal(0, 0.05, base.size)}
    sel = select_model(oof, {"ridge": 1, "gbr": 3})
    assert sel["rule_choice"] == "ridge"
    ok = finalize_selection(sel, "ridge")
    assert ok["deviation"] is None and ok["final_choice"] == "ridge"
    with pytest.raises(ValueError, match="deviation_reason is required"):
        finalize_selection(sel, "gbr")
    dev = finalize_selection(sel, "gbr", "needs native missing-value handling in production")
    assert dev["final_choice"] == "gbr" and dev["deviation"].startswith("needs native")
    with pytest.raises(ValueError):
        finalize_selection(sel, "unknown")


def test_selection_input_validation():
    with pytest.raises(ValueError):
        select_model({"a": [1, 2, 3], "b": [1, 2]}, {"a": 1, "b": 2})
    with pytest.raises(ValueError):
        select_model({"a": [1, 2, 3], "b": [1, 2, 3]}, {"a": 1})


# ----------------------------------------------------------------------------------- split leakage gap
def _abs_loss(y, p):
    return np.abs(np.asarray(y) - np.asarray(p))


def test_gap_is_large_and_positive_when_siblings_leak_across_a_random_split():
    rng = np.random.default_rng(0)
    n_groups, per = 80, 6
    centers = rng.normal(size=(n_groups, 3))
    groups = np.repeat(np.arange(n_groups), per)
    X = centers[groups] + rng.normal(scale=0.01, size=(groups.size, 3))      # siblings are near-identical rows
    y = rng.normal(size=n_groups)[groups]                                     # group-specific target
    res = split_leakage_gap(KNeighborsRegressor(1), X, y, groups, _abs_loss, n_repeats=4)
    assert res["gap"] > 0 and res["gap_pct"] > 100
    assert_no_reversed_gap(res)


def test_gap_is_near_zero_without_group_structure_and_not_flagged():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=600)
    res = split_leakage_gap(Ridge(), X, y, np.arange(600), _abs_loss, n_repeats=5)
    assert abs(res["gap_pct"]) < 15
    assert_no_reversed_gap(res, tolerance=0.2)


def test_reversed_gap_is_flagged():
    fake = {"random_mean_loss": 0.14, "group_mean_loss": 0.12}
    with pytest.raises(AssertionError, match="direction is reversed"):
        assert_no_reversed_gap(fake, tolerance=0.05)
    assert_no_reversed_gap({"random_mean_loss": 0.14, "group_mean_loss": 0.139}, tolerance=0.05)


# ----------------------------------------------------------------------------------- mode trigger
@pytest.mark.parametrize("msg", [
    "Senior mode",
    "senior mode: review this repo",
    "/Senior review this repo",
    "  Senior Mode - train a model on this csv",
    "سينيور مود راجع المشروع",
    "تشوف السكيلز الجديدة؟ سينور مود راجع ريبو وحل كل المشاكل",          # real message: command after a question mark
    "شكراً.\nSenior mode\nابني النموذج",
    "وضع سينيور",
    "\"Senior mode\" please",
    "Manage this project end-to-end",
    "أبغاك تدير المشروع من الألف إلى الياء",
    "I need an end-to-end data science project for my graduation",
    "أدر المشروع كامل",
])
def test_senior_mode_triggers(msg):
    assert M.detect_mode(msg) == "senior"


@pytest.mark.parametrize("msg", [
    "train a model on this csv",
    "make it production-ready and be careful",
    "what is senior mode?",
    "I use /Senior in documentation",
    "I am not in senior mode today",
    "explain end-to-end encryption",
    "اشرح لي وش هو سينيور مود",
    "ابغى دقة عالية وتحليل صارم",
])
def test_lite_stays_the_default(msg):
    assert M.detect_mode(msg) == "lite"


def test_mode_persists_and_can_be_switched_off():
    assert M.detect_mode("continue with the model", current="senior") == "senior"
    assert M.detect_mode("Lite mode", current="senior") == "lite"
    assert M.detect_mode("/lite", current="senior") == "lite"
    assert M.detect_mode("تمام. وضع خفيف", current="senior") == "lite"
    assert M.detect_mode("Senior mode", current="lite") == "senior"
