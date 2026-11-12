"""Guards must be quiet on clean work.

A detector that fires on a healthy pipeline is worse than no detector: it teaches people to ignore
warnings. This file covers two things:

* **No false alarms:** a clean split, a clean parity path, a clean experiment and a clean report pass the
  same asserts the fixtures use to prove a leak *is* caught.
* **Critical inputs:** empty frames, NaN/Inf, an all-missing column and a non-default index produce either a
  documented empty result or a clear `ValueError` — never a wrong number, a silent all-NaN transform, or an
  `IndexError` from three layers down.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from dstk import contracts, drift, leakage, preprocess, ranking, report_lint, results_io, splitgap, stats
from dstk.preprocess import QuantileWinsorizer


# --------------------------------------------------------------------------- no false alarms on clean data
def _clean_split(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"entity": np.repeat(np.arange(n // 4), 4),
                       "f": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(df, groups=df["entity"]))
    return df.iloc[tr_i], df.iloc[te_i], df


def test_clean_split_passes_every_split_guard():
    train, test, df = _clean_split()
    leakage.assert_no_row_overlap(train, test, ["f", "y"])
    leakage.assert_no_group_overlap(train["entity"], test["entity"])
    leakage.assert_folds_respect_groups(GroupKFold(4).split(df, groups=df["entity"]), df["entity"])
    assert leakage.row_overlap(train, test, ["f", "y"]) == 0
    assert leakage.duplicate_report(df, ["entity"])["n_duplicates"] == len(df) - df["entity"].nunique()


def test_temporal_guard_accepts_aligned_and_boundary_equal_splits():
    t = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
    leakage.assert_temporal_order(t.iloc[:15], t.iloc[15:])
    # last train time == first test time is a legitimate boundary: allow_equal=True must not fire
    leakage.assert_temporal_order(t.iloc[:16], t.iloc[15:], allow_equal=True)
    with pytest.raises(AssertionError):
        leakage.assert_temporal_order(t.iloc[:16], t.iloc[15:])


def test_overlap_tolerance_is_honoured_not_ignored():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"a": [1.0, 9.0, 8.0, 7.0]})          # 1 of 4 rows duplicated, 25%
    leakage.assert_no_row_overlap(train, test, ["a"], max_share=0.25)
    with pytest.raises(AssertionError):
        leakage.assert_no_row_overlap(train, test, ["a"], max_share=0.0)


def test_reversed_gap_guard_is_quiet_when_the_two_splits_agree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + rng.normal(scale=0.5, size=120)
    result = splitgap.split_leakage_gap(LinearRegression(), X, y, np.arange(120),
                                        lambda a, b: np.abs(a - b), n_repeats=3)
    assert abs(result["gap"]) < 0.05 * result["random_mean_loss"] + 1e-9
    splitgap.assert_no_reversed_gap(result, tolerance=0.05)   # equal splits must not trigger the alarm


def test_psi_and_ks_are_quiet_on_identical_frames():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"v": rng.normal(size=400), "c": rng.choice(["a", "b"], 400)})
    assert drift.psi(ref["v"], ref["v"]) == pytest.approx(0.0, abs=1e-9)
    assert drift.psi_categorical(ref["c"], ref["c"]) == pytest.approx(0.0, abs=1e-9)
    report = drift.psi_report(ref, ref.copy(), ["v"], ["c"])
    assert list(report["status"]) == ["stable", "stable"]
    ks = drift.ks_report(ref, ref.copy(), ["v"]).iloc[0]
    assert ks["ks_stat"] == pytest.approx(0.0) and ks["p_value"] > 0.99


def test_linter_is_quiet_on_a_clean_report(tmp_path):
    res = tmp_path / "results.json"
    results_io.write_results(res, "holdout", {"mape_pct": 12.34, "n": 400}, source="train.py")
    md = tmp_path / "report.md"
    md.write_text("MAPE was 12.34% on 400 rows; the baseline was close, so we report no clear difference.\n",
                  encoding="utf-8")
    out = report_lint.lint_file(str(md), str(res))
    assert out["ok"] and out["integrity"]["ok"] and not out["unverified_numbers"]


def test_winsorizer_is_a_no_op_for_values_inside_the_bounds():
    train = pd.DataFrame({"x": np.arange(100.0)})
    fitted = QuantileWinsorizer(0.05, 0.95).fit(train)
    inside = fitted.transform(pd.DataFrame({"x": [10.0, 50.0, 90.0]}))
    assert inside["x"].tolist() == [10.0, 50.0, 90.0]        # clean values must not be moved


# --------------------------------------------------------------------------- critical inputs
def test_empty_frames_give_empty_results_not_crashes():
    empty = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
    assert leakage.duplicate_report(empty) == {"n_rows": 0, "n_duplicates": 0, "share": 0.0}
    assert leakage.row_overlap(empty, empty, ["a"]) == 0
    leakage.assert_no_row_overlap(empty, empty, ["a"])          # nothing to fail on
    assert leakage.text_group_ids(pd.Series([], dtype=object)).size == 0
    assert preprocess.outlier_report(empty, empty, ["a"]).shape[0] == 1
    assert drift.ks_report(empty, empty, []).shape[0] == 0


def test_all_missing_column_fails_loudly_when_fitting_bounds():
    with pytest.raises(ValueError, match="no non-missing values"):
        QuantileWinsorizer().fit(pd.DataFrame({"x": [np.nan, np.nan, np.nan]}))
    with pytest.raises(ValueError, match="no non-missing values"):
        QuantileWinsorizer().fit(pd.DataFrame({"ok": [1.0, 2.0, 3.0], "gone": [np.nan] * 3}))


def test_nan_heavy_inputs_return_the_documented_value_or_a_clear_error():
    assert stats.bootstrap_ci([1.0, np.nan, 3.0])[0] == pytest.approx(2.0)   # NaNs dropped
    with pytest.raises(ValueError, match="no finite values"):
        stats.bootstrap_ci([np.nan, np.nan])
    with pytest.raises(ValueError, match="no finite values"):
        stats.paired_comparison([np.nan], [np.nan])
    with pytest.raises(ValueError, match="empty sample"):
        drift.psi([], [1.0, 2.0])
    assert preprocess.robust_zscore(pd.Series([5.0, 5.0, 5.0])).tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(preprocess.robust_zscore(pd.Series([np.nan, np.nan])).iloc[0])  # no fake zero


def test_auc_scan_explains_why_it_refused():
    with pytest.raises(ValueError, match="got 0 distinct label value"):
        leakage.single_feature_auc_scan(pd.DataFrame({"x": pd.Series(dtype=float)}), np.array([]))
    with pytest.raises(ValueError, match="got 3 distinct label value"):
        leakage.single_feature_auc_scan(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([0, 1, 2]))
    # a constant feature is skipped, not flagged: the scan returns no rows rather than a fake 0.5 AUC
    scan = leakage.single_feature_auc_scan(pd.DataFrame({"const": [1.0] * 20}), np.array([0, 1] * 10))
    assert scan.empty


def test_empty_ranking_scores_zero_instead_of_being_skipped():
    out = ranking.evaluate_rankings({"u": []}, {"u": {"i1": 1.0, "i2": 1.0}}, ks=(5,), n_boot=50)
    assert out["n_evaluated"] == 1 and out["n_skipped"] == 0
    assert out["metrics"]["P@5"]["mean"] == 0.0 and out["metrics"]["MRR"]["mean"] == 0.0


def test_non_default_index_is_respected_but_does_not_change_fingerprints_or_row_labels():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    assert leakage.canonical_fingerprint(frame) == leakage.canonical_fingerprint(frame.reset_index(drop=True))
    flags = preprocess.outlier_flags(frame, {"a": (0.0, 2.5)})
    assert flags.index.tolist() == [10, 20, 30]
    errors = contracts.validate_frame(frame.assign(a=[1.0, 99.0, 3.0]),
                                      {"columns": {"a": {"max": 5}}}).errors
    assert any("rows [20]" in e for e in errors), errors  # the message points at the real index label


def test_contract_on_an_empty_frame_reports_the_row_count():
    report = contracts.validate_frame(pd.DataFrame({"a": pd.Series(dtype=float)}),
                                      {"columns": {"a": {"dtype": "number"}}, "min_rows": 5})
    assert not report.ok and any("only 0 rows" in e for e in report.errors)


def test_results_verification_never_raises_on_a_missing_file(tmp_path):
    out = results_io.verify_results(tmp_path / "nope.json")
    assert out["ok"] is False and out["has_stamp"] is False and "does not exist" in out["reason"]
