"""The bundled churn fixture is a real file with two PLANTED leaks, repeated customers and missing
values. These tests run the guards on that file, the way they would run on a project.

The lesson they encode: a leaky model can look 30 AUC points better than the honest one, and the
only thing that separates them is an audit, not a better algorithm.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dstk import leakage as L
from dstk.contracts import validate_frame

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
HONEST = ["tenure_months", "monthly_spend", "plan", "region", "support_calls_90d"]
LEAKY_EXTRA = ["support_calls_after_cancel", "cancel_recorded"]

SCHEMA = {
    "columns": {
        "customer_id": {"dtype": "str", "nullable": False},
        "tenure_months": {"dtype": "number", "nullable": False, "min": 0, "max": 120},
        "monthly_spend": {"dtype": "number", "min": 0, "max": 5000},
        "plan": {"dtype": "str", "nullable": False, "allowed": ["basic", "pro", "enterprise"]},
        "region": {"dtype": "str", "nullable": False, "allowed": ["riyadh", "jeddah", "dammam", "abha"]},
        "churned": {"dtype": "int", "nullable": False, "allowed": [0, 1]},
    },
    "min_rows": 500,
}


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    df = pd.read_csv(DATA / "churn_leaky.csv")
    df["cancel_recorded"] = df["cancelled_at"].notna().astype(int)   # the leak, as a usable column
    return df


@pytest.fixture(scope="module")
def split(churn):
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(churn, groups=churn["customer_id"]))
    return churn.iloc[tr], churn.iloc[te]


def _auc(train: pd.DataFrame, test: pd.DataFrame, cols) -> float:
    num = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    model = Pipeline([("prep", prep), ("gb", HistGradientBoostingClassifier(max_iter=120, random_state=0))])
    model.fit(train[cols], train["churned"])
    return float(roc_auc_score(test["churned"], model.predict_proba(test[cols])[:, 1]))


def test_grain_is_repeated_customers_so_the_split_must_be_grouped(churn):
    assert churn["customer_id"].nunique() < len(churn)
    assert L.duplicate_report(churn)["n_duplicates"] == 0        # no exact duplicate rows ...
    # ... and yet half the rows share a customer with another row: entity leakage, not row duplication.
    assert 0.3 < L.duplicate_report(churn, ["customer_id"])["share"] < 0.7


def test_grouped_split_has_no_customer_on_both_sides(split):
    train, test = split
    L.assert_no_group_overlap(train["customer_id"], test["customer_id"])


def test_missing_values_are_real_and_must_be_imputed_inside_the_pipeline(churn):
    assert 0 < churn["monthly_spend"].isna().mean() < 0.15
    assert churn["monthly_spend"].isna().sum() > 0


def test_single_feature_scan_flags_both_planted_leaks(churn):
    scan = L.single_feature_auc_scan(
        churn[["support_calls_after_cancel", "cancel_recorded", "tenure_months"]], churn["churned"], top=3)
    by = dict(zip(scan["feature"], scan["auc"]))
    assert by["cancel_recorded"] == pytest.approx(1.0)          # a timestamp that exists only for churners
    assert by["support_calls_after_cancel"] > 0.95              # recorded after the outcome
    assert not bool(scan.set_index("feature").loc["tenure_months", "flag"])


def test_leaky_model_looks_far_better_than_the_honest_one(split):
    train, test = split
    honest = _auc(train, test, HONEST)
    leaky = _auc(train, test, HONEST + LEAKY_EXTRA)
    print(f"honest AUC {honest:.3f}  with leaks {leaky:.3f}")
    assert honest < 0.85, "the honest signal is weak on purpose; a high value means the fixture changed"
    assert leaky > 0.97, "the planted leak stopped leaking: the fixture must be fixed, not the test"
    assert leaky - honest > 0.20, "this gap is the whole lesson: leakage buys AUC that does not exist"


def test_data_contract_gate_passes_the_fixture_and_catches_bad_rows(churn):
    assert validate_frame(churn, SCHEMA).ok
    bad = churn.copy()
    bad.loc[0, "tenure_months"] = -3
    bad.loc[1, "plan"] = "platinum"
    report = validate_frame(bad, SCHEMA)
    assert not report.ok
    assert any("tenure_months" in e for e in report.errors)
    assert any("platinum" in e for e in report.errors)
